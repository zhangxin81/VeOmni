# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Patch configuration for DeepseekV4 GPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.deepseek_v4.deepseek_v4_gpu_patch_gen_config -o veomni/models/transformers/deepseek_v4/generated --diff

Patches:
1. ``DeepseekV4Experts`` — drops upstream ``@use_experts_implementation`` so
   VeOmni controls MoE dispatch through ``veomni_moe_experts_forward``. The
   eager path preserves DeepSeek-V4's SwiGLU clamp semantics.
2. ``DeepseekV4SparseMoeBlock.forward`` reports the actual top-k expert indices
   (including hash-routed bootstrap layers) to the MoE load-balance monitor.
3. ``DeepseekV4ForCausalLM.forward`` — OpSlot guard for fused cross-entropy
   (``veomni_causal_lm_loss``), OpSlot guard for load-balancing loss, and
   ``MoeCausalLMOutputWithLogProbs`` so RL/PPO callers can read per-token
   log-probs / entropy alongside the loss.
4. Register ``get_parallel_plan`` on ``DeepseekV4ForCausalLM``.

DeepSeek-V4 uses partial/interleaved RoPE plus compressor-specific attention,
so this config intentionally does not replace RoPE or attention with Liger / FA
kernels. RMSNorm/SwiGLU swaps are also left eager: the shared expert and routed
expert paths use DeepSeek-V4-specific signatures / clamp behavior that generic
Liger modules do not model.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import load_balancing_loss_func
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs
from veomni.utils.moe_monitor import record_router_indices


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_gpu.py",
    description="DeepseekV4 with VeOmni fused-MoE + OpSlot fused-loss patches",
)

config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from transformers.utils import logging
    from veomni.ops.dispatch import OpSlot
    logger = logging.get_logger(__name__)
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    """
)


# ================================================================
# Patch: DeepseekV4Experts
# 1. Drop upstream ``@use_experts_implementation`` — it dispatches to
#    grouped-mm implementations outside VeOmni's MoE OpSlot control.
# 2. Keep routed experts on the eager path because current VeOmni fused MoE
#    kernels implement plain SiLU-gate SwiGLU and do not apply DeepSeek-V4's
#    ``swiglu_limit`` clamp.
# ================================================================
@config.replace_class(
    "DeepseekV4Experts",
    description="Drop @use_experts_implementation and add VeOmni fused MoE dispatch",
)
class PatchedDeepseekV4Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # --- Patch.2 ---
        if veomni_moe_experts_forward.use_non_eager_impl:
            logger.warning_once(
                "DeepSeek-V4 routed experts require swiglu_limit clamp semantics; "
                "current VeOmni fused MoE kernels do not implement that clamp, "
                "so DeepSeek-V4 experts run eagerly."
            )
        # --- Patch.2 ---

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up


# ================================================================
# Patch: DeepseekV4SparseMoeBlock.forward
# 1. Report top-k expert indices to the MoE load-balance monitor. The router
#    modules return logits/weights/indices, so the block is the first common
#    point that sees both hash-routed and learned-routed expert choices.
# ================================================================
@config.override_method(
    "DeepseekV4SparseMoeBlock.forward",
    description="Report DeepseekV4 top-k indices to the MoE load-balance monitor",
)
def deepseek_v4_sparse_moe_block_forward_patched(
    self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
) -> torch.Tensor:
    batch, seq_len, hidden_dim = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.view(-1, hidden_dim)
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    # --- Patch.1 ---
    record_router_indices(self.gate, indices)
    # --- Patch.1 ---
    routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
    return routed + self.shared_experts(residual)


# ================================================================
# Patch: DeepseekV4ForCausalLM.forward
# 1. OpSlot guard for fused cross-entropy loss; falls back to the eager HF loss
#    wrapper when no fused kernel is bound.
# 2. OpSlot guard for load-balancing loss and safe aux-loss composition when
#    router logits are unavailable.
# 3. Return ``MoeCausalLMOutputWithLogProbs`` so per-token log-probs / entropy
#    are constructor fields visible to FSDP2 and RL consumers.
# ================================================================
@config.override_method(
    "DeepseekV4ForCausalLM.forward",
    description="OpSlot guards for fused CE and load-balancing loss in DeepseekV4ForCausalLM.forward",
)
def deepseek_v4_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    model_kwargs = dict(kwargs)
    model_kwargs["output_router_logits"] = output_router_logits
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **model_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    # --- Patch.2 ---
    aux_loss = None
    if output_router_logits:
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None and loss is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
    # --- Patch.2 ---

    # --- Patch.3 ---
    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )
    # --- Patch.3 ---


# ================================================================
# Patch: DeepseekV4ForCausalLM.get_parallel_plan
# 1. Register VeOmni EP parallel plan on the patchgen-generated class.
# ================================================================
@config.override_method(
    "DeepseekV4ForCausalLM.get_parallel_plan",
    description="Register DeepseekV4 expert parallel plan for v5 generated modeling",
)
def deepseek_v4_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
