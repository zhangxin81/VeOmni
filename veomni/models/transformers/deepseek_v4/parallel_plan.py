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


def get_parallel_plan():
    from ....distributed.parallel_state import get_parallel_state

    # DeepSeek-V4 routed experts apply ``swiglu_limit`` clamps before SwiGLU.
    # Current VeOmni fused/EP MoE kernels implement plain SiLU-gate SwiGLU, so
    # the patchgen model keeps experts eager. EP would slice expert weights
    # without providing the eager path's all-to-all token dispatch, so fail
    # explicitly if a caller tries to enable DeepSeek-V4 EP.
    if get_parallel_state().ep_enabled:
        raise NotImplementedError(
            "DeepSeek-V4 expert parallelism is not supported until VeOmni fused MoE kernels "
            "implement DeepSeek-V4 swiglu_limit clamp semantics. Use ep_size=1."
        )
    return None
