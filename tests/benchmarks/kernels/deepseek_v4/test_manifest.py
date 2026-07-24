# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from benchmarks.kernels.deepseek_v4 import manifest


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _repo_with_manifest_module(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    source = Path(manifest.__file__).read_text(encoding="utf-8")
    _write(repo / "benchmarks/kernels/deepseek_v4/manifest.py", source)
    return repo


def test_seed_manifest_has_all_approved_entries() -> None:
    ids = [seed.id for seed in manifest.SEEDS]
    assert len(ids) == 57
    assert ids[:6] == ["A01", "A02", "A03", "A04", "A05", "A06"]
    assert ids[-11:] == [
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "F07",
        "F08",
        "F09",
        "F10",
        "G01",
    ]
    assert {seed.symbol for seed in manifest.SEEDS} >= {
        "_fused_kv_compress_norm_rope_insert_indexer_attn",
        "_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn",
        "_fused_kv_compress_norm_rope_insert_sparse_attn",
    }


def test_scans_decorators_direct_aliases_and_conditional_aliases(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/fused.py",
        """
from vllm.triton_utils import triton
from somewhere import _imported_kernel as imported_alias

@triton.jit
def direct_kernel(x):
    return

@triton.autotune(configs=[], key=[])
@triton.jit
def tuned_kernel(x):
    return

def launch_direct(x):
    direct_kernel[(1,)](x)

def launch_simple_alias(x):
    alias = direct_kernel
    alias[(1,)](x)

def launch_import_alias(x):
    imported_alias[(1,)](x)

def launch_conditional(x, flag):
    if flag:
        kernel = direct_kernel
    elif x:
        kernel = tuned_kernel
    else:
        kernel = imported_alias
    kernel[(1,)](x)
""",
    )

    result = manifest.scan_repo(repo)
    defs = {item["symbol"]: item for item in result["definitions"]}
    assert defs["direct_kernel"]["decorators"] == ["triton.jit"]
    assert defs["tuned_kernel"]["decorators"] == [
        "triton.autotune",
        "triton.jit",
    ]

    launches = result["launches"]
    assert any(
        item["symbol"] == "direct_kernel" and not item["via_alias"] for item in launches
    )
    assert any(
        item["symbol"] == "direct_kernel" and item["via_alias"] for item in launches
    )
    assert any(item["symbol"] == "_imported_kernel" for item in launches)
    assert any(
        item["symbol"] == "direct_kernel" and item["conditional_context"] == ["flag"]
        for item in launches
    )
    assert any(
        item["symbol"] == "tuned_kernel"
        and item["conditional_context"] == ["else of flag", "x"]
        for item in launches
    )


def test_conditional_alias_pattern_matches_fused_compress_launcher(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
        """
from vllm.triton_utils import triton

def wrapper(head_dim, use_fp4_cache):
    if head_dim == 512:
        kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
    elif use_fp4_cache:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
    else:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
    kernel[(1,)]()

@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn():
    return

@triton.jit
def _fused_kv_compress_norm_rope_insert_indexer_attn():
    return

@triton.jit
def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn():
    return
""",
    )
    result = manifest.scan_repo(repo)
    alias_launches = {
        item["symbol"] for item in result["launches"] if item["via_alias"]
    }
    assert alias_launches == {
        "_fused_kv_compress_norm_rope_insert_sparse_attn",
        "_fused_kv_compress_norm_rope_insert_indexer_attn",
        "_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn",
    }


def test_ignores_non_kernel_subscript_calls(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/xpu/mtp.py",
        """
def ordinary_calls(self, indices):
    values = set[str](indices)
    return self.layers[0](values)
""",
    )

    result = manifest.scan_repo(repo)

    assert result["launches"] == []


def test_output_is_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/a.py",
        """
from vllm.triton_utils import triton

@triton.jit
def kernel_a():
    return

def launch():
    alias = kernel_a
    alias[(1,)]()
""",
    )
    first = manifest.scan_repo(repo)
    second = manifest.scan_repo(repo)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert (
        first["source_scan_fingerprint_sha256"]
        == (second["source_scan_fingerprint_sha256"])
    )


def test_check_semantics_accept_alias_launch_for_missing_direct_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/seeded.py",
        """
from vllm.triton_utils import triton

@triton.jit
def seed_kernel():
    return

def launch():
    alias = seed_kernel
    alias[(1,)]()
""",
    )
    monkeypatch.setattr(
        manifest,
        "SEEDS",
        (manifest.Seed("T01", "T", "seed_kernel", "seeded.py:4", "test"),),
    )
    result = manifest.scan_repo(repo)
    assert manifest.check_manifest(result) == []


def test_check_semantics_fail_missing_definition_and_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/seeded.py",
        """
from vllm.triton_utils import triton

@triton.jit
def seed_kernel():
    return
""",
    )
    monkeypatch.setattr(
        manifest,
        "SEEDS",
        (
            manifest.Seed("T01", "T", "seed_kernel", "seeded.py:4", "test"),
            manifest.Seed("T02", "T", "missing_kernel", "missing.py:1", "test"),
        ),
    )
    result = manifest.scan_repo(repo)
    errors = manifest.check_manifest(result)
    assert errors == [
        "T01 missing launch for seed_kernel "
        "(no direct launch or resolved alias launch)",
        "T02 missing definition for missing_kernel",
    ]


def test_cli_writes_output_and_check_exits_zero_for_alias_launch(
    tmp_path: Path,
) -> None:
    repo = _repo_with_manifest_module(tmp_path)
    _write(
        repo / "vllm/models/deepseek_v4/common/ops/seeded.py",
        """
from vllm.triton_utils import triton

@triton.jit
def seed_kernel():
    return

def launch():
    alias = seed_kernel
    alias[(1,)]()
""",
    )
    test_module = repo / "run_cli_test.py"
    _write(
        test_module,
        """
from pathlib import Path
from benchmarks.kernels.deepseek_v4 import manifest

manifest.SEEDS = (
    manifest.Seed("T01", "T", "seed_kernel", "seeded.py:4", "test"),
)
result = manifest.scan_repo(Path.cwd())
manifest.write_manifest(result, Path("manifest.json"))
raise SystemExit(1 if manifest.check_manifest(result) else 0)
""",
    )

    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(test_module)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    assert data["seed_status"][0]["alias_launch_found"]


def test_repo_integration_manifest_has_expected_seed_count() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    result = manifest.scan_repo(repo_root)
    assert result["seed_count"] == 57
    assert manifest.check_manifest(result) == []
    assert "source_scan_fingerprint_sha256" in result
    assert not {"set", "layers"} & {item["symbol"] for item in result["launches"]}
    assert any(
        item["symbol"] == "_fused_kv_compress_norm_rope_insert_indexer_attn"
        for item in result["definitions"]
    )
