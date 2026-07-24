# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from benchmarks.kernels.deepseek_v4.a03_factories import (
    _make_logits_inputs,
    _run_topk,
)


def test_logits_inputs_reject_causal_mask_before_cuda_allocation() -> None:
    with pytest.raises(ValueError, match="require causal=false"):
        _make_logits_inputs(
            {
                "num_queries": 1,
                "num_keys": 1,
                "causal": True,
            }
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_run_topk_keeps_contiguous_output_when_topk_exceeds_logits_width() -> None:
    logits = torch.randn((4, 2), device="cuda", dtype=torch.float32)
    row_starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    row_ends = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32)
    output = torch.empty((4, 4), device="cuda", dtype=torch.int32)

    actual = _run_topk(logits, row_starts, row_ends, output).sort(dim=1).values
    expected = torch.tensor(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, 0],
            [-1, -1, 0, 1],
            [-1, -1, 0, 1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)
