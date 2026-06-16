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
Patch configuration for DeepseekV4 NPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.deepseek_v4.deepseek_v4_npu_patch_gen_config -o veomni/models/transformers/deepseek_v4/generated --diff

NPU reuses the structural DeepSeek-V4 patches from the GPU config. We do not
swap V4's partial/interleaved RoPE on NPU because the generic NPU RoPE helper
expects Q/K pair signatures and full-head rotary semantics.
"""

from veomni.patchgen.patch_spec import PatchConfig

from .deepseek_v4_gpu_patch_gen_config import (
    PatchedDeepseekV4Experts,
    deepseek_v4_forcausallm_forward_patched,
    deepseek_v4_get_parallel_plan_patched,
    deepseek_v4_sparse_moe_block_forward_patched,
)


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_npu.py",
    description="DeepseekV4 with NPU-compatible VeOmni fused-MoE + OpSlot fused-loss patches",
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

config.replace_class(
    "DeepseekV4Experts",
    replacement=PatchedDeepseekV4Experts,
    description="Drop @use_experts_implementation and add VeOmni fused MoE dispatch",
)

config.override_method(
    "DeepseekV4SparseMoeBlock.forward",
    replacement=deepseek_v4_sparse_moe_block_forward_patched,
    description="Report DeepseekV4 top-k indices to the MoE load-balance monitor",
)

config.override_method(
    "DeepseekV4ForCausalLM.forward",
    replacement=deepseek_v4_forcausallm_forward_patched,
    description="OpSlot guards for fused CE and load-balancing loss in DeepseekV4ForCausalLM.forward",
)

config.override_method(
    "DeepseekV4ForCausalLM.get_parallel_plan",
    replacement=deepseek_v4_get_parallel_plan_patched,
    description="Register DeepseekV4 expert parallel plan for v5 generated modeling",
)
