#!/usr/bin/env bash
# Prepare $ROOT/.deps for vLLM's CMake build.
#
# vLLM's CMake pulls ~10 Git repositories through FetchContent into
# $ROOT/.deps (setup.py passes -DFETCHCONTENT_BASE_DIR=$ROOT/.deps). When the
# build runs with FETCHCONTENT_FULLY_DISCONNECTED=ON (see cu130_build_wheel.sh),
# CMake never downloads anything -- it expects every dependency to already be
# present as .deps/<name>-src, checked out to the exact ref the .cmake files
# pin. This script does that download up front, so the later configure/build is
# fully offline.
#
# Layout and refs mirror what CMake's FetchContent/ExternalProject would
# produce, so the same .deps directory also works without FULLY_DISCONNECTED:
#   .deps/<name>-src          checkout of the pinned GIT_TAG (plus submodules)
#   .deps/<name>-subbuild/    ExternalProject bookkeeping (auto-created by
#                             CMake on the first online configure)
#
# The dependency table below must be kept in sync with:
#   CMakeLists.txt                      (cutlass)
#   cmake/external_projects/*.cmake     (deepgemm flashmla fmha_sm100 qutlass
#                                        tml_fa4 triton_kernels vllm-flash-attn)
#   cmake/cpu_extension.cmake           (onednn arm_compute)
#
# Usage:
#   bash scripts/build/prepare_deps.sh [--gpu|--cpu|--all] [--deps-dir DIR]
#                                      [--retries N] [--mirror URL]
#                                      [--fix-stamps] [--check] [-q]
#
#   Scope (default: --gpu, the CUDA build dependencies):
#     --gpu  only what a CUDA build needs
#     --cpu  only the CPU backend dependencies (onednn, arm_compute)
#     --all  both
#
#   --fix-stamps  Repair ExternalProject clone stamps under .deps/*-subbuild:
#                 if <name>-populate-gitinfo.txt is newer than
#                 <name>-populate-gitclone-lastrun.txt, ExternalProject deletes
#                 <name>-src and re-clones on every configure. Touching the
#                 stamps prevents that. Needed only when the build is NOT fully
#                 disconnected.
#
#   --check       Verify that .deps already satisfies CMake (refs + submodules)
#                 without downloading anything.
#
#   Mirroring:
#     GitHub HTTPS clones often stall on restricted networks. Set
#     VLLM_GIT_MIRROR (or --mirror) to a mirror prefix, e.g.:
#       VLLM_GIT_MIRROR=https://gitclone.com/github.com/
#       VLLM_GIT_MIRROR=https://ghfast.top/https://github.com/
#     Every github.com URL (and submodule fetch) is rewritten via git's
#     url.<mirror>.insteadOf, so no .gitmodules edits are needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_DIR="${DEPS_DIR:-${REPO_ROOT}/.deps}"
RETRIES="${VLLM_DEPS_RETRIES:-5}"
MIRROR="${VLLM_GIT_MIRROR:-}"
SCOPE="gpu"
FIX_STAMPS=0
CHECK_ONLY=0
QUIET=0
VLLM_FLASH_ATTN_ALL_SUBMODULES="${VLLM_FLASH_ATTN_ALL_SUBMODULES:-0}"

# Kill hung HTTPS transfers instead of waiting forever: abort if the link
# stays under LOW_SPEED_LIMIT bytes/s for LOW_SPEED_TIME seconds.
export GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-1000}"
export GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-120}"
export GIT_TERMINAL_PROMPT=0

# ---------------------------------------------------------------------------
# Dependency table. Keep in sync with the cmake files listed in the header.
#   reg <name> <group> <url> <tag> <submodules>
#   <submodules>: none | all | "path1 path2" (matches cmake GIT_SUBMODULES)
# ---------------------------------------------------------------------------
declare -A REPO_URL REPO_TAG REPO_SUBS REPO_GROUP

reg() { # name group url tag subs
  REPO_GROUP["$1"]="$2"; REPO_URL["$1"]="$3"; REPO_TAG["$1"]="$4"; REPO_SUBS["$1"]="$5"
}

reg cutlass        gpu "https://github.com/nvidia/cutlass.git"                  "v4.4.2"                                   none
reg deepgemm       gpu "https://github.com/deepseek-ai/DeepGEMM.git"            "a6b593d2826719dcf4892609af7b84ee23aaf32a" "third-party/cutlass third-party/fmt"
reg flashmla       gpu "https://github.com/vllm-project/FlashMLA"               "a8f794d1251cbfd88a5011445dd5582289c727e4" all
reg fmha_sm100     gpu "https://github.com/vllm-project/MSA.git"                "2e63ec37a0fc29bc20f39cd1a52e0f5affc33a73" all
reg qutlass        gpu "https://github.com/IST-DASLab/qutlass.git"              "830d2c4537c7396e14a02a46fbddd18b5d107c65" all
reg tml_fa4        gpu "https://github.com/vllm-project/tml-fa4.git"            "b206834606ed5b5f21f8eed6b0683f528ea9cf7d" all
reg triton_kernels gpu "https://github.com/triton-lang/triton.git"              "v3.5.1"                                   none

# vllm-flash-attn's submodules: csrc/cutlass is enough for a CUDA build
# (csrc/composable_kernel and third_party/aiter are AMD-only). CMake's online
# FetchContent initializes all of them; set VLLM_FLASH_ATTN_ALL_SUBMODULES=1
# to mirror that for a ROCm build.
flash_attn_subs="csrc/cutlass"
if [[ "${VLLM_FLASH_ATTN_ALL_SUBMODULES}" == "1" ]]; then
  flash_attn_subs="all"
fi
reg vllm-flash-attn gpu "https://github.com/vllm-project/flash-attention.git" "caaa4eb59845388a20b1f435ecaafb4bd9517ad8" "$flash_attn_subs"

if [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  onednn_tag="9c5be1cc59e368aebf0909e6cf20f981ea61462a"   # pinned ACL backport
else
  onednn_tag="v3.10"
fi
reg onednn         cpu "https://github.com/oneapi-src/oneDNN.git"               "$onednn_tag"                               none
reg arm_compute    cpu "https://github.com/ARM-software/ComputeLibrary.git"     "v52.6.0"                                   none

GPU_DEPS="cutlass deepgemm flashmla fmha_sm100 qutlass tml_fa4 triton_kernels vllm-flash-attn"
CPU_DEPS="onednn arm_compute"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say() { # message
  [[ "$QUIET" == "1" ]] && return 0
  printf '%s\n' "$*"
}

is_hash() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }

git_args() { # echo the git command line with the mirror injected
  local -a args=(git)
  [[ -n "$MIRROR" ]] && args+=(-c "url.${MIRROR}.insteadOf=https://github.com/")
  printf '%s\0' "${args[@]}"
}

# Run a git command, retrying with backoff. The mirror (if set) is injected so
# it also applies to submodule fetches.
run_git() {
  local -a args
  mapfile -d '' args < <(git_args)
  local attempt=1 rc
  while :; do
    if "${args[@]}" "$@" 2>&1 | sed 's/^/      /'; then
      return 0
    fi
    rc=${PIPESTATUS[0]}
    if (( attempt >= RETRIES )); then
      say "      git failed after ${RETRIES} attempts: $*"
      return "$rc"
    fi
    local backoff=$(( 2 ** attempt ))
    (( backoff > 30 )) && backoff=30
    say "      attempt ${attempt}/${RETRIES} failed, retrying in ${backoff}s ..."
    sleep "$backoff"
    attempt=$(( attempt + 1 ))
  done
}

clone_repo() { # url tag dir
  local url="$1" tag="$2" dir="$3"
  if is_hash "$tag"; then
    # Pinned SHA: full fetch of just that commit (shallow fetches of an
    # arbitrary commit are often unsupported by mirrors).
    run_git init -q "$dir" || return 1
    run_git -C "$dir" remote add origin "$url" || return 1
    run_git -C "$dir" fetch --progress origin "$tag" || return 1
    run_git -C "$dir" checkout -q FETCH_HEAD || return 1
  else
    run_git clone --depth 1 --single-branch --branch "$tag" --progress "$url" "$dir" || return 1
  fi
}

init_submodules() { # src subs
  local src="$1" subs="$2"
  [[ "$subs" == "none" ]] && return 0
  [[ -f "$src/.gitmodules" ]] || return 0
  # Full (non-shallow) submodule checkout: --depth 1 can leave the pinned
  # gitlink commit outside the shallow history and force extra fetches.
  local -a args=(submodule update --init --recursive --progress)
  [[ "$subs" != "all" ]] && args+=($subs)
  ( cd "$src" && run_git "${args[@]}" )
}

ensure_repo() { # name
  local name="$1"
  local url="${REPO_URL[$name]}" tag="${REPO_TAG[$name]}" subs="${REPO_SUBS[$name]}"
  local src="${DEPS_DIR}/${name}-src"

  say "== ${name}  ${url}  @ ${tag}"
  if [[ -d "$src/.git" ]]; then
    say "      already downloaded at ${src} (reusing)"
  else
    say "      cloning into ${src} ..."
    local tmp
    tmp="$(mktemp -d "${DEPS_DIR}/.prepare.XXXXXX")" || return 1
    if ! clone_repo "$url" "$tag" "$tmp"; then
      rm -rf "$tmp"
      say "      FAILED: ${name}"
      return 1
    fi
    mv "$tmp" "$src" || return 1
  fi

  # Pin to the exact ref (fetch only if missing locally).
  if ! ( cd "$src" && git rev-parse --verify -q "$tag^{commit}" >/dev/null 2>&1 ); then
    say "      fetching pinned ref ${tag} ..."
    ( cd "$src" && run_git fetch --progress origin "$tag" ) || return 1
  fi
  ( cd "$src" && run_git checkout -q --detach "$tag" ) || return 1

  init_submodules "$src" "$subs" || return 1
}

check_repo() { # name  (read-only; returns 0 when OK)
  local name="$1"
  local url="${REPO_URL[$name]}" tag="${REPO_TAG[$name]}" subs="${REPO_SUBS[$name]}"
  local src="${DEPS_DIR}/${name}-src"

  printf '== %-16s %s @ %s\n' "$name" "$url" "$tag"
  [[ -d "$src" ]] || { echo "      MISSING: ${src}"; return 1; }
  [[ -d "$src/.git" ]] || { echo "      NO_GIT:  ${src} (not a git checkout)"; return 1; }
  local head expected
  head="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
  # Resolve the pinned ref to a commit so tags (v4.4.2, v3.5.1) and hashes
  # compare the same way: a detached checkout of the tag's commit is valid.
  expected="$(git -C "$src" rev-parse -q --verify "$tag^{commit}" 2>/dev/null || true)"
  if [[ "$head" != "$expected" ]]; then
    echo "      WRONG_REF: HEAD=${head:0:12}..., expected ${tag} (${expected:0:12}...)"
    return 1
  fi
  if [[ "$subs" != "none" ]] && [[ -f "$src/.gitmodules" ]]; then
    # Only check the submodules this script initializes (for vllm-flash-attn
    # that is just csrc/cutlass on CUDA; the AMD ones may legitimately be
    # uninitialized). "all" checks every submodule.
    local -a sargs=()
    [[ "$subs" != "all" ]] && sargs=($subs)
    if git -C "$src" submodule status "${sargs[@]}" 2>/dev/null | grep -q '^[-+]'; then
      echo "      SUBMODULE_NOT_READY: some submodules are not initialized"
      return 1
    fi
  fi
  echo "      OK"
  return 0
}

fix_stamps() { # repair ExternalProject clone stamps so online configure does
  # not delete the prepared src dirs and re-clone them
  local fixed=0
  for stamp in "${DEPS_DIR}"/*-subbuild/*-populate-prefix/src/*-populate-stamp; do
    [[ -d "$stamp" ]] || continue
    local gitinfo lastrun download
    gitinfo="$(ls "$stamp"/*-gitinfo.txt 2>/dev/null | head -1 || true)"
    lastrun="$(ls "$stamp"/*-gitclone-lastrun.txt 2>/dev/null | head -1 || true)"
    download="$(ls "$stamp"/*-download 2>/dev/null | head -1 || true)"
    if [[ -n "$gitinfo" ]] && { [[ -z "$lastrun" ]] || [[ "$gitinfo" -nt "$lastrun" ]]; }; then
      [[ -n "$lastrun" ]] && touch "$lastrun"
      [[ -n "$download" ]] && touch "$download"
      say "      fixed clone stamp: ${stamp}"
      fixed=$(( fixed + 1 ))
    fi
  done
  say "Fixed ${fixed} clone stamp(s)."
}

usage() {
  sed -n '2,/^# -\{20,\}/p' "${BASH_SOURCE[0]}" |
    sed -n 's/^# \{0,1\}//p' |
    sed '$d'
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)               SCOPE="gpu" ;;
    --cpu)               SCOPE="cpu" ;;
    --all)               SCOPE="all" ;;
    --deps-dir)          DEPS_DIR="$2"; shift ;;
    --retries)           RETRIES="$2"; shift ;;
    --mirror)            MIRROR="$2"; shift ;;
    --fix-stamps)        FIX_STAMPS=1 ;;
    --check)             CHECK_ONLY=1 ;;
    -q|--quiet)          QUIET=1 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$DEPS_DIR"
DEPS_DIR="$(cd -- "$DEPS_DIR" && pwd)"

case "$SCOPE" in
  gpu) names=( $GPU_DEPS ) ;;
  cpu) names=( $CPU_DEPS ) ;;
  all) names=( $GPU_DEPS $CPU_DEPS ) ;;
esac

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "Checking ${#names[@]} dep(s) under ${DEPS_DIR} (read-only)..."
  ok=0; bad=0
  for n in "${names[@]}"; do
    if check_repo "$n"; then ok=$(( ok + 1 )); else bad=$(( bad + 1 )); fi
  done
  echo
  echo "Check done: ${ok} OK, ${bad} not ready."
  [[ "$bad" -eq 0 ]]
  exit $?
fi

if [[ "$FIX_STAMPS" == "1" ]]; then
  fix_stamps
  exit 0
fi

say "Fetching ${#names[@]} repo(s) into ${DEPS_DIR} (${RETRIES} attempts, mirror='${MIRROR:-direct}')"
say
ok=0; fail=0
for n in "${names[@]}"; do
  if ensure_repo "$n"; then ok=$(( ok + 1 )); else fail=$(( fail + 1 )); fi
done
say
say "Done: ${ok} ready, ${fail} failed."
if (( fail > 0 )); then
  say "Some repos failed. Re-run this script (already-downloaded repos are"
  say "skipped), point VLLM_GIT_MIRROR/--mirror at another mirror, or check"
  say "your network."
  exit 1
fi
say
say "Next: build vLLM offline. cu130_build_wheel.sh already sets"
say "FETCHCONTENT_FULLY_DISCONNECTED=ON and will reuse ${DEPS_DIR} as-is."
