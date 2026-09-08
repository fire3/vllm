# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def release_sparse_mla_capture_scratch() -> None:
    """Drop captured sparse-MLA scratch after its CUDA graphs are destroyed.

    The Triton DSV4 sparse-MLA operator keeps one exact-capacity scratch entry
    per captured graph (addresses are baked into the graph). Those entries
    must follow the graph lifetime, so every runner teardown / profiling
    teardown path calls this after dropping the graph manager. Import is lazy:
    engines that never loaded the operator have nothing to release.
    """
    try:
        from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
            release_all_capture_scratch,
        )

        release_all_capture_scratch()
    except ImportError:
        # The DeepSeek V4 model module (or its Triton backend) is not part of
        # this engine; nothing is cached to release.
        pass


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    release_sparse_mla_capture_scratch()

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
