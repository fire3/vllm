# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


def test_compute_slot_mappings_applies_padding_mask():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([2],),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64, device=device)
    is_padding = torch.tensor(
        [False, True, False, False, True, False, False, False],
        dtype=torch.bool,
        device=device,
    )

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded=8,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert slot_mappings.cpu().tolist() == [
        [32, PAD_SLOT_ID, 34, 48, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID]
    ]


@pytest.mark.skipif(
    not current_platform.is_device_capability(120),
    reason="fused block-table preparation is enabled only on SM120",
)
@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
def test_fused_gather_and_slot_mapping_matches_fallback_under_cuda_graph(
    cp_size: int,
    cp_rank: int,
):
    device = torch.device("cuda")

    def make_block_tables() -> BlockTables:
        tables = BlockTables(
            block_sizes=[16, 32, 8],
            max_num_reqs=8,
            max_num_batched_tokens=64,
            max_num_blocks_per_group=[8, 8, 8],
            device=device,
            kernel_block_sizes=[16, 16, 8],
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_interleave=2,
        )
        for req_idx in range(4):
            tables.append_block_ids(
                req_index=req_idx,
                new_block_ids=(
                    [10 + req_idx, 20 + req_idx],
                    [30 + req_idx],
                    [40 + req_idx, 50 + req_idx],
                ),
                overwrite=True,
            )
        tables.apply_staged_writes()
        return tables

    fused = make_block_tables()
    fallback = make_block_tables()
    fallback._use_fused_gather_slot_mapping = False

    idx_mapping = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 4, 9, 16], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [0, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6],
        dtype=torch.int64,
        device=device,
    )
    is_padding = torch.zeros(64, dtype=torch.bool, device=device)
    is_padding[[2, 7, 15]] = True

    def run(tables: BlockTables):
        return tables.gather_block_tables_and_compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_reqs_padded=8,
            num_tokens_padded=64,
            is_padding=is_padding,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(fused)
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fused_block_tables, fused_slot_mappings = run(fused)

    idx_mapping.copy_(torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device))
    graph.replay()
    fallback_block_tables, fallback_slot_mappings = run(fallback)
    torch.accelerator.synchronize()

    for fused_table, fallback_table in zip(
        fused_block_tables, fallback_block_tables, strict=True
    ):
        assert torch.equal(fused_table, fallback_table)
        assert torch.count_nonzero(fused_table[4:]) == 0
    assert torch.equal(fused_slot_mappings, fallback_slot_mappings)
