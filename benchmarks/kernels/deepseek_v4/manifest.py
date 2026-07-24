# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static DSv4/DSpark Triton kernel manifest scanner.

This module intentionally uses only the Python standard library. It scans
source text with ``ast`` and reports definitions and launches without importing
vLLM, torch, or triton.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCAN_ROOTS = (
    "vllm/model_executor/layers/quantization/utils",
    "vllm/models/deepseek_v4",
    "vllm/v1/attention/backends",
    "vllm/v1/spec_decode",
    "vllm/v1/worker/block_table.py",
    "vllm/v1/worker/gpu/block_table.py",
    "vllm/v1/worker/gpu/buffer_utils.py",
    "vllm/v1/worker/gpu/input_batch.py",
    "vllm/v1/worker/gpu/spec_decode",
    "vllm/v1/worker/gpu/sample",
    "vllm/v1/worker/gpu/utils.py",
)

BLIND_SPOT_NOTES = (
    "Static AST scan only; runtime dispatch, shape guards, CUDA Graph capture, "
    "and target reachability are not inferred.",
    "Only Triton-style kernel[grid](...) launches are reported; wrapper calls, "
    "custom op dispatch, and compiled extension calls are outside scope.",
    "Alias resolution is conservative: direct imports, import aliases, simple "
    "name assignments, and conditional local name aliases are tracked.",
    "Subscript calls are retained only when they resolve to a discovered "
    "Triton definition, an approved seed, or a kernel-named symbol.",
    "Conditional context records enclosing if/elif conditions as source text; "
    "it does not prove whether a branch executes.",
)


@dataclass(frozen=True)
class Seed:
    id: str
    phase: str
    symbol: str
    source_hint: str
    candidate_summary: str

    def to_json(self) -> dict[str, str]:
        return {
            "id": self.id,
            "phase": self.phase,
            "symbol": self.symbol,
            "source_hint": self.source_hint,
            "candidate_summary": self.candidate_summary,
        }


SEEDS = (
    Seed(
        "A01",
        "A",
        "_w8a8_triton_block_scaled_mm",
        "fp8_utils.py:768",
        "Compare existing SM120 CUTLASS and FlashInfer block-scale paths; "
        "prove default dispatch before adding shape dispatch.",
    ),
    Seed(
        "A02",
        "A",
        "_deepseek_v4_sm12x_fp8_einsum_kernel",
        "fp8_einsum.py:16",
        "Try FlashInfer cutlass FP8 BMM; use grouped CUTLASS only if layout "
        "or scale conversion does not erase the gain.",
    ),
    Seed(
        "A03",
        "A",
        "_fp8_mqa_logits_kernel",
        "sm12x_mqa.py:53",
        "Compare Triton, batched GEMM with epilogue, and fused CUDA or CuTe "
        "DSL for prefill indexer score.",
    ),
    Seed(
        "A04",
        "A",
        "_fp8_paged_mqa_logits_rowwise_kernel",
        "sm12x_mqa.py:295",
        "Fuse paged load, head-weighted ReLU, and score output for decode "
        "without materializing layout.",
    ),
    Seed(
        "A05",
        "A",
        "_fp8_paged_mqa_logits_kernel",
        "sm12x_mqa.py:185",
        "Keep as trace-gated fallback for misaligned or conditional edge shapes.",
    ),
    Seed(
        "A06",
        "A",
        "_tf32_hc_prenorm_gemm_kernel",
        "sm12x_mqa.py:606",
        "Determine whether this is hot-path or startup; compare cuBLASLt or "
        "CUTLASS only if hot.",
    ),
    Seed(
        "B01",
        "B",
        "_fused_q_kv_rmsnorm_kernel",
        "fused_qk_rmsnorm.py:8",
        "Compare two FlashInfer rmsnorm_cute launches with a single-launch "
        "dual-width CuTe DSL implementation.",
    ),
    Seed(
        "B02",
        "B",
        "_fused_inv_rope_fp8_quant_per_head",
        "fused_inv_rope_fp8_quant.py:17",
        "Implement inverse GPT-J RoPE plus per-128 FP8 quant in CuTe DSL or "
        "CUDA while preserving layout semantics.",
    ),
    Seed(
        "B03",
        "B",
        "_save_partial_states_kernel",
        "save_partial_states.py:48",
        "Use native CUDA copy or fuse into the compressor consumer; measure "
        "standalone and chain impact.",
    ),
    Seed(
        "B04",
        "B",
        "_fused_kv_compress_norm_rope_insert_indexer_attn",
        "fused_compress_quant_cache.py:302",
        "Build fused CuTe DSL or CUDA op for the indexer FP8 cache path.",
    ),
    Seed(
        "B05",
        "B",
        "_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn",
        "fused_compress_quant_cache.py:479",
        "Build FP4 indexer cache op while preserving nibble and UE8M0 layout.",
    ),
    Seed(
        "B06",
        "B",
        "_fused_kv_compress_norm_rope_insert_sparse_attn",
        "fused_compress_quant_cache.py:112",
        "Trace-prove whether CUDA head-512 path is inactive on target before "
        "considering a port.",
    ),
    Seed(
        "C01",
        "C",
        "_compute_global_topk_indices_and_lens_kernel",
        "cache_utils.py:460",
        "Fuse with persistent top-k output or global slot mapping, or move to "
        "native CUDA.",
    ),
    Seed(
        "C02",
        "C",
        "_build_c128a_topk_metadata_kernel",
        "sparse_mla.py:361",
        "Generate FlashInfer metadata directly and avoid intermediate tensor writes.",
    ),
    Seed(
        "C03",
        "C",
        "_compute_prefill_metadata_kernel",
        "sparse_swa.py:654",
        "Replace with native CUDA metadata builder.",
    ),
    Seed(
        "C04",
        "C",
        "_compute_swa_indices_and_lens_kernel",
        "sparse_swa.py:685",
        "Use native CUDA or fusion for causal SWA.",
    ),
    Seed(
        "C05",
        "C",
        "_compute_dspark_noncausal_swa_indices_kernel",
        "sparse_swa.py:752",
        "Cover fixed-width DSpark non-causal SWA for draft positions 0 to 7.",
    ),
    Seed(
        "C06",
        "C",
        "_build_prefill_chunk_metadata_kernel",
        "indexer.py:935",
        "Fuse chunk metadata generation with C02 or C03 where valid.",
    ),
    Seed(
        "C07",
        "C",
        "_prepare_uniform_decode_kernel",
        "indexer.py:46",
        "Trace-gate and fuse with decode metadata updates if hot.",
    ),
    Seed(
        "C08",
        "C",
        "quantize_and_insert_k_kernel",
        "cache_utils.py:27",
        "Confirm whether CuTe DSL or native path already replaces it.",
    ),
    Seed(
        "C09",
        "C",
        "_dequantize_and_gather_k_kernel",
        "cache_utils.py:219",
        "Compare against dequant_gather_k_cutedsl and dispatch only winning shapes.",
    ),
    Seed(
        "C10",
        "C",
        "_combine_topk_swa_indices_kernel",
        "cache_utils.py:567",
        "Fuse with C01 or C04 if chain data supports it.",
    ),
    Seed(
        "C11",
        "C",
        "_build_flashinfer_mixed_sparse_indices_kernel",
        "cache_utils.py:784",
        "Generate the FlashInfer consumer layout directly.",
    ),
    Seed(
        "D01",
        "D",
        "_prepare_dflash_inputs_kernel",
        "dflash/speculator.py:417",
        "Tune request-level input preparation without changing DSpark semantics.",
    ),
    Seed(
        "D02",
        "D",
        "_gumbel_sample_kernel",
        "sample/gumbel.py:162",
        "Reuse FlashInfer sampling only if RNG and input semantics match; "
        "otherwise keep Philox contract in native CUDA.",
    ),
    Seed(
        "D08",
        "D",
        "_compute_local_logits_stats_kernel",
        "rejection_sampler_utils.py:186",
        "Use native CUDA or CUB reduction.",
    ),
    Seed(
        "D09",
        "D",
        "_compute_cumulative_log_p_kernel",
        "rejection_sampler_utils.py:295",
        "Trace-gate and share statistics with D08 where possible.",
    ),
    Seed(
        "D10",
        "D",
        "_compute_local_residual_mass_kernel",
        "rejection_sampler_utils.py:370",
        "Trace-gate residual reduction.",
    ),
    Seed(
        "D11",
        "D",
        "_rejection_kernel",
        "rejection_sampler_utils.py:459",
        "Implement standard speculative rejection sampling in native CUDA.",
    ),
    Seed(
        "D12",
        "D",
        "_resample_kernel",
        "rejection_sampler_utils.py:656",
        "Coordinate with residual stats while preserving RNG.",
    ),
    Seed(
        "D13",
        "D",
        "_insert_resampled_kernel",
        "rejection_sampler_utils.py:805",
        "Fuse with final sampled-token writeback if chain data supports it.",
    ),
    Seed(
        "E01",
        "E",
        "_prepare_prefill_inputs_kernel",
        "input_batch.py:201",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E02",
        "E",
        "_prepare_pos_seq_lens_kernel",
        "input_batch.py:261",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E03",
        "E",
        "_combine_sampled_and_draft_tokens_kernel",
        "input_batch.py:326",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E04",
        "E",
        "_get_num_sampled_and_rejected_kernel",
        "input_batch.py:430",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E05",
        "E",
        "_post_update_kernel",
        "input_batch.py:479",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E06",
        "E",
        "_post_update_num_computed_tokens_kernel",
        "input_batch.py:581",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E07",
        "E",
        "_expand_idx_mapping_kernel",
        "input_batch.py:613",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E08",
        "E",
        "_gather_block_tables_kernel",
        "block_table.py:205",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E09",
        "E",
        "_compute_slot_mappings_kernel",
        "block_table.py:245",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "E10",
        "E",
        "_apply_write_kernel",
        "buffer_utils.py:274",
        "Optimize as vLLM-native bookkeeping only if DSv4 trace shows cost.",
    ),
    Seed(
        "F01",
        "F",
        "_fused_indexer_q_rope_quant_kernel",
        "common/ops/fused_indexer_q.py:71",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F02",
        "F",
        "_fused_indexer_q_rope_mxfp4_kernel",
        "common/ops/fused_indexer_q.py:180",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F03",
        "F",
        "_prepare_megamoe_inputs_kernel",
        "nvidia/ops/prepare_megamoe.py:16",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F04",
        "F",
        "_copy_page_indices_kernel",
        "v1/attention/backends/flashinfer.py:2230",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F05",
        "F",
        "_trtllm_prefill_attn_kvfp8_dequant",
        "v1/attention/backends/flashinfer.py:104",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F06",
        "F",
        "_compressed_slot_mapping_kernel",
        "v1/attention/backends/mla/compressor_utils.py:9",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F07",
        "F",
        "_flatten_sampled_kernel",
        "v1/worker/gpu/spec_decode/rejection_sampler.py:24",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F08",
        "F",
        "copy_and_expand_dflash_inputs_kernel",
        "v1/spec_decode/utils.py:458",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F09",
        "F",
        "_fused_mtp_input_rmsnorm_kernel",
        "common/ops/fused_mtp_input_rmsnorm.py:44",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "F10",
        "F",
        "_mtp_shared_head_rmsnorm_kernel",
        "common/ops/fused_mtp_input_rmsnorm.py:101",
        "Prove inactive if target backend already replaces it; otherwise "
        "ledger as adjacent conditional kernel.",
    ),
    Seed(
        "G01",
        "G",
        "_silu_mul_quant_fp8_packed_kernel",
        "quantization/utils/fp8_utils.py:153",
        "Compare the packed UE8M0 Triton path with the existing native fused "
        "SiLU quantizer on model-faithful SM120 DeepGEMM workspace shapes.",
    ),
)


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _iter_scan_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()
    for root in SCAN_ROOTS:
        path = repo_root / root
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("*.py"))
    return sorted(files, key=lambda item: item.relative_to(repo_root).as_posix())


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ast.dump(node, annotate_fields=False)


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _node_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    return None


def _decorator_kind(decorator: ast.AST) -> str | None:
    name = _node_name(decorator)
    if name in {"triton.jit", "jit"} or name.endswith(".triton.jit"):
        return "triton.jit"
    if name in {"triton.autotune", "autotune"} or name.endswith(".triton.autotune"):
        return "triton.autotune"
    return None


def _canonical_name(name: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current.rsplit(".", 1)[-1]


class TritonScanner(ast.NodeVisitor):
    def __init__(self, repo_root: Path, path: Path, source: str) -> None:
        self.repo_root = repo_root
        self.path = path
        self.source = source
        self.relpath = _repo_relative(path, repo_root)
        self.aliases: dict[str, str] = {}
        self.conditional_aliases: dict[str, list[tuple[str, list[str]]]] = {}
        self.definitions: list[dict[str, Any]] = []
        self.launches: list[dict[str, Any]] = []
        self.context: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            self.aliases[local] = alias.name

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name

    def visit_Assign(self, node: ast.Assign) -> None:
        value_name = _node_name(node.value)
        if value_name is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._record_alias(target.id, value_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value_name = _node_name(node.value) if node.value is not None else None
        if value_name is not None and isinstance(node.target, ast.Name):
            self._record_alias(node.target.id, value_name)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.context.append(_source_segment(self.source, node.test))
        for child in node.body:
            self.visit(child)
        self.context.pop()
        if node.orelse:
            self.context.append(f"else of {self._if_head(node)}")
            for child in node.orelse:
                self.visit(child)
            self.context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        decorators = [
            kind
            for item in node.decorator_list
            if (kind := _decorator_kind(item)) is not None
        ]
        if decorators:
            self.definitions.append(
                {
                    "symbol": node.name,
                    "path": self.relpath,
                    "line": node.lineno,
                    "decorators": sorted(set(decorators)),
                    "conditional_context": list(self.context),
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Subscript):
            raw_name = _node_name(node.func.value)
            if raw_name is not None:
                name = raw_name.rsplit(".", 1)[-1]
                conditional_aliases = self.conditional_aliases.get(name, [])
                if conditional_aliases:
                    for symbol, context in conditional_aliases:
                        self._append_launch(node, raw_name, symbol, context)
                else:
                    symbol = _canonical_name(raw_name, self.aliases)
                    self._append_launch(node, raw_name, symbol, list(self.context))
        self.generic_visit(node)

    def _if_head(self, node: ast.If) -> str:
        return _source_segment(self.source, node.test)

    def _record_alias(self, target: str, value_name: str) -> None:
        symbol = _canonical_name(value_name, self.aliases)
        if self.context:
            self.conditional_aliases.setdefault(target, []).append(
                (symbol, list(self.context))
            )
        else:
            self.aliases[target] = symbol

    def _append_launch(
        self,
        node: ast.Call,
        raw_name: str,
        symbol: str,
        context: list[str],
    ) -> None:
        self.launches.append(
            {
                "symbol": symbol,
                "path": self.relpath,
                "line": node.lineno,
                "launch_expr": _source_segment(self.source, node.func),
                "raw_callee": raw_name,
                "via_alias": raw_name.rsplit(".", 1)[-1] != symbol,
                "conditional_context": context,
            }
        )


def scan_repo(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    definitions: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    fingerprint = hashlib.sha256()

    for path in _iter_scan_files(repo_root):
        source = path.read_text(encoding="utf-8")
        relpath = _repo_relative(path, repo_root)
        fingerprint.update(relpath.encode())
        fingerprint.update(b"\0")
        fingerprint.update(source.encode())
        fingerprint.update(b"\0")
        try:
            tree = ast.parse(source, filename=relpath)
        except SyntaxError as exc:
            definitions.append(
                {
                    "symbol": "<syntax-error>",
                    "path": relpath,
                    "line": exc.lineno or 0,
                    "decorators": [],
                    "conditional_context": [],
                    "error": str(exc),
                }
            )
            continue
        scanner = TritonScanner(repo_root, path, source)
        scanner.visit(tree)
        definitions.extend(scanner.definitions)
        launches.extend(scanner.launches)

    seed_by_symbol = {seed.symbol: seed for seed in SEEDS}
    def_symbols = {item["symbol"] for item in definitions}
    candidate_symbols = def_symbols | set(seed_by_symbol)
    launches = [
        item
        for item in launches
        if item["symbol"] in candidate_symbols or "kernel" in item["symbol"].lower()
    ]
    launch_symbols = {item["symbol"] for item in launches}
    direct_launch_symbols = {
        item["symbol"] for item in launches if not item["via_alias"]
    }
    alias_launch_symbols = {item["symbol"] for item in launches if item["via_alias"]}

    unmatched = [seed.to_json() for seed in SEEDS if seed.symbol not in def_symbols]
    unseeded_defs = [
        item for item in definitions if item["symbol"] not in seed_by_symbol
    ]
    unseeded_launches = [
        item for item in launches if item["symbol"] not in seed_by_symbol
    ]
    seed_status = []
    for seed in SEEDS:
        has_definition = seed.symbol in def_symbols
        has_direct_launch = seed.symbol in direct_launch_symbols
        has_alias_launch = seed.symbol in alias_launch_symbols
        seed_status.append(
            {
                **seed.to_json(),
                "definition_found": has_definition,
                "direct_launch_found": has_direct_launch,
                "alias_launch_found": has_alias_launch,
                "launch_found": seed.symbol in launch_symbols,
            }
        )

    return {
        "schema_version": 1,
        "repo_root": repo_root.as_posix(),
        "source_scan_fingerprint_sha256": fingerprint.hexdigest(),
        "scan_roots": list(SCAN_ROOTS),
        "seed_count": len(SEEDS),
        "seeds": [seed.to_json() for seed in SEEDS],
        "definitions": sorted(
            definitions, key=lambda item: (item["symbol"], item["path"], item["line"])
        ),
        "launches": sorted(
            launches, key=lambda item: (item["symbol"], item["path"], item["line"])
        ),
        "seed_status": seed_status,
        "unmatched_seeds": unmatched,
        "unseeded_definitions": sorted(
            unseeded_defs,
            key=lambda item: (item["symbol"], item["path"], item["line"]),
        ),
        "unseeded_launches": sorted(
            unseeded_launches,
            key=lambda item: (item["symbol"], item["path"], item["line"]),
        ),
        "blind_spot_notes": list(BLIND_SPOT_NOTES),
    }


def check_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    for seed in manifest["seed_status"]:
        if not seed["definition_found"]:
            errors.append(f"{seed['id']} missing definition for {seed['symbol']}")
            continue
        if not seed["direct_launch_found"] and not seed["alias_launch_found"]:
            errors.append(
                f"{seed['id']} missing launch for {seed['symbol']} "
                "(no direct launch or resolved alias launch)"
            )
    return errors


def write_manifest(manifest: dict[str, Any], output: Path | None) -> None:
    data = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(data, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(data, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if seed definitions or launch evidence are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = scan_repo(args.repo_root)
    write_manifest(manifest, args.output)
    if args.check:
        errors = check_manifest(manifest)
        if errors:
            for error in errors:
                print(error)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
