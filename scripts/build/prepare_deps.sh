#!/usr/bin/env bash
# Prepare $ROOT/.deps for vLLM's CMake build.
#
# vLLM's CMake pulls several Git repositories through FetchContent into
# $ROOT/.deps (setup.py passes -DFETCHCONTENT_BASE_DIR=$ROOT/.deps). When the
# build runs with FETCHCONTENT_FULLY_DISCONNECTED=ON (see cu130_build_wheel.sh),
# CMake never downloads anything -- it expects every dependency to already be
# present as .deps/<name>-src, checked out to the exact ref the current checkout
# pins.
#
# This script discovers those repos/tags from the current checkout's CMake
# files, so switching branches or release tags automatically changes what gets
# fetched. There is no hand-maintained version table to keep in sync.
#
# Layout and refs mirror what CMake's FetchContent/ExternalProject would
# produce, so the same .deps directory also works without FULLY_DISCONNECTED:
#   .deps/<name>-src          checkout of the pinned GIT_TAG (plus submodules)
#   .deps/<name>-subbuild/    ExternalProject bookkeeping (auto-created by
#                             CMake on the first online configure)
#
# Usage:
#   bash scripts/build/prepare_deps.sh [--gpu|--cpu|--all] [--deps-dir DIR]
#                                      [--retries N] [--mirror URL]
#                                      [--device cuda|rocm]
#                                      [--list] [--fix-stamps] [--check] [-q]
#
#   Scope (default: --gpu, the CUDA build dependencies):
#     --gpu  only what a CUDA build needs
#     --cpu  only the CPU backend dependencies (onednn, arm_compute)
#     --all  both
#
#   --device  GPU backend whose deps should be resolved from CMake conditionals.
#             Default: cuda. Today this only affects triton_kernels.
#
#   --list       Print the deps resolved from the current checkout and exit.
#
#   --fix-stamps  Repair ExternalProject clone stamps under .deps/*-subbuild:
#                 if <name>-populate-gitinfo.txt is newer than
#                 <name>-populate-gitclone-lastrun.txt, ExternalProject deletes
#                 <name>-src and re-clones on every configure. Touching the
#                 stamps prevents that. Needed only when the build is NOT fully
#                 disconnected.
#
#   --check       Verify that .deps already satisfies CMake (remote URL, ref,
#                 and submodules) without downloading anything.
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
TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
FIX_STAMPS=0
CHECK_ONLY=0
LIST_ONLY=0
QUIET=0
VLLM_FLASH_ATTN_ALL_SUBMODULES="${VLLM_FLASH_ATTN_ALL_SUBMODULES:-0}"

# Kill hung HTTPS transfers instead of waiting forever: abort if the link
# stays under LOW_SPEED_LIMIT bytes/s for LOW_SPEED_TIME seconds.
export GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-1000}"
export GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-120}"
export GIT_TERMINAL_PROMPT=0

declare -A REPO_URL REPO_TAG REPO_SUBS REPO_GROUP REPO_SOURCE
declare -a GPU_DEPS CPU_DEPS

say() {
  [[ "$QUIET" == "1" ]] && return 0
  printf '%s\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

strip_matching_quotes() {
  local s
  s="$(trim "$1")"
  if [[ "${#s}" -ge 2 ]]; then
    if [[ "${s:0:1}" == '"' && "${s: -1}" == '"' ]]; then
      s="${s:1:${#s}-2}"
    elif [[ "${s:0:1}" == "'" && "${s: -1}" == "'" ]]; then
      s="${s:1:${#s}-2}"
    fi
  fi
  printf '%s' "$s"
}

is_hash() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]]
}

reg() { # name group url tag subs source
  local name="$1" group="$2" url="$3" tag="$4" subs="$5" source="$6"
  [[ -n "$url" ]] || die "${name}: missing GIT_REPOSITORY in ${source}"
  [[ -n "$tag" ]] || die "${name}: missing GIT_TAG in ${source}"
  REPO_GROUP["$name"]="$group"
  REPO_URL["$name"]="$url"
  REPO_TAG["$name"]="$tag"
  REPO_SUBS["$name"]="$subs"
  REPO_SOURCE["$name"]="$source"
  case "$group" in
    gpu) GPU_DEPS+=("$name") ;;
    cpu) CPU_DEPS+=("$name") ;;
    *) die "unknown dependency group '${group}' for ${name}" ;;
  esac
}

cmake_set_values() { # file var
  local file="$1" var="$2"
  awk -v var="$var" '
    {
      line = $0
      if (line ~ "^[[:space:]]*set\\([[:space:]]*" var "([[:space:]]|\\))") {
        sub("^[[:space:]]*set\\([[:space:]]*" var "[[:space:]]*", "", line)
        sub("\\)[[:space:]]*$", "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
        print line
      }
    }
  ' "$file"
}

cmake_nth_set_value() { # file var which(1|2|...|last)
  local file="$1" var="$2" which="${3:-last}"
  mapfile -t _vals < <(cmake_set_values "$file" "$var")
  [[ "${#_vals[@]}" -gt 0 ]] || return 1
  if [[ "$which" == "last" ]]; then
    printf '%s' "${_vals[-1]}"
  else
    local idx=$(( which - 1 ))
    [[ "$idx" -ge 0 && "$idx" -lt "${#_vals[@]}" ]] || return 1
    printf '%s' "${_vals[$idx]}"
  fi
}

cmake_nth_block() { # file call dep which(1|2|...|last)
  local file="$1" call="$2" dep="$3" which="${4:-last}"
  awk -v call="$call" -v dep="$dep" -v want="$which" '
    function matches_dep(block, normalized) {
      normalized = block
      gsub(/[[:space:]]+/, " ", normalized)
      return normalized ~ "^[[:space:]]*" call "[[:space:]]*\\([[:space:]]*" dep "([[:space:]]|\\))"
    }
    BEGIN {
      want_last = (want == "last")
      seen = 0
      in_block = 0
      depth = 0
      last_block = ""
    }
    {
      if (!in_block) {
        if ($0 !~ "^[[:space:]]*" call "[[:space:]]*\\(") {
          next
        }
        in_block = 1
        depth = 0
        current = ""
      }

      line = $0
      current = current line ORS

      probe = line
      opens = gsub(/\(/, "(", probe)
      closes = gsub(/\)/, ")", probe)
      depth += opens - closes

      if (in_block && depth <= 0) {
        in_block = 0
        if (matches_dep(current)) {
          seen++
          if (want_last) {
            last_block = current
          } else if (seen == want + 0) {
            print current
            exit
          }
        }
      }
    }
    END {
      if (want_last && last_block != "") {
        print last_block
      }
    }
  ' "$file"
}

cmake_block_arg_text() { # block key
  local block="$1" key="$2"
  printf '%s' "$block" | awk -v key="$key" '
    BEGIN { capture = 0 }
    {
      line = $0
      if (!capture) {
        if (line ~ "^[[:space:]]*" key "([[:space:]]|$)") {
          sub("^[[:space:]]*" key "[[:space:]]*", "", line)
          capture = 1
        } else {
          next
        }
      } else if (line ~ "^[[:space:]]*[A-Z_]+([[:space:]]|$)") {
        exit
      }

      if (line ~ "^[[:space:]]*#") {
        next
      }
      sub(/[[:space:]]*#.*$/, "", line)
      sub(/[[:space:]]*\)[[:space:]]*$/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (line != "") {
        print line
      }
    }
  ' | paste -sd' ' -
}

resolve_cmake_var() { # file var
  local file="$1" var="$2"
  case "$var" in
    TRITON_GIT|TRITON_KERNELS_TAG)
      if [[ "$TARGET_DEVICE" == "rocm" ]]; then
        cmake_nth_set_value "$file" "$var" 1
      else
        cmake_nth_set_value "$file" "$var" last
      fi
      ;;
    *)
      cmake_nth_set_value "$file" "$var" last
      ;;
  esac
}

resolve_cmake_tokens() { # file raw_tokens
  local file="$1" raw="$2"
  local -a parts out
  local part resolved
  read -r -a parts <<<"$raw"
  for part in "${parts[@]}"; do
    part="$(strip_matching_quotes "$part")"
    if [[ "$part" =~ ^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$ ]]; then
      resolved="$(resolve_cmake_var "$file" "${BASH_REMATCH[1]}" || true)"
      part="$(strip_matching_quotes "$resolved")"
    fi
    [[ -n "$part" ]] && out+=("$part")
  done
  printf '%s' "${out[*]}"
}

register_from_fetchcontent() { # name group file subs_mode(auto|none|<explicit>) [call] [which]
  local name="$1" group="$2" file="$3" subs_mode="$4" call="${5:-auto}" which="${6:-last}"
  local block url_raw tag_raw subs_raw url tag subs="$subs_mode"

  case "$call" in
    declare)
      block="$(cmake_nth_block "$file" "FetchContent_Declare" "$name" "$which")"
      ;;
    populate)
      block="$(cmake_nth_block "$file" "FetchContent_Populate" "$name" "$which")"
      ;;
    auto)
      block="$(cmake_nth_block "$file" "FetchContent_Declare" "$name" "$which")"
      if [[ -z "$block" ]]; then
        block="$(cmake_nth_block "$file" "FetchContent_Populate" "$name" "$which")"
      fi
      ;;
    *)
      die "${name}: unsupported block selector '${call}'"
      ;;
  esac

  [[ -n "$block" ]] || die "${name}: could not find FetchContent block in ${file}"

  url_raw="$(cmake_block_arg_text "$block" GIT_REPOSITORY || true)"
  tag_raw="$(cmake_block_arg_text "$block" GIT_TAG || true)"
  url="$(resolve_cmake_tokens "$file" "$url_raw")"
  tag="$(resolve_cmake_tokens "$file" "$tag_raw")"

  if [[ "$subs_mode" == "auto" ]]; then
    subs_raw="$(cmake_block_arg_text "$block" GIT_SUBMODULES || true)"
    if [[ -n "$subs_raw" ]]; then
      subs="$(resolve_cmake_tokens "$file" "$subs_raw")"
    else
      subs="auto"
    fi
  fi

  reg "$name" "$group" "$url" "$tag" "$subs" "$file"
}

discover_gpu_deps() {
  register_from_fetchcontent \
    cutlass gpu "${REPO_ROOT}/CMakeLists.txt" auto declare last
  register_from_fetchcontent \
    deepgemm gpu "${REPO_ROOT}/cmake/external_projects/deepgemm.cmake" auto populate last
  register_from_fetchcontent \
    flashkda gpu "${REPO_ROOT}/cmake/external_projects/flashkda.cmake" auto declare last
  register_from_fetchcontent \
    flashmla gpu "${REPO_ROOT}/cmake/external_projects/flashmla.cmake" auto declare last
  register_from_fetchcontent \
    fmha_sm100 gpu "${REPO_ROOT}/cmake/external_projects/fmha_sm100.cmake" auto declare last
  register_from_fetchcontent \
    qutlass gpu "${REPO_ROOT}/cmake/external_projects/qutlass.cmake" auto populate last
  register_from_fetchcontent \
    tml_fa4 gpu "${REPO_ROOT}/cmake/external_projects/tml_fa4.cmake" auto declare last
  register_from_fetchcontent \
    triton_kernels gpu "${REPO_ROOT}/cmake/external_projects/triton_kernels.cmake" auto declare last

  local fa_subs="csrc/cutlass"
  if [[ "${VLLM_FLASH_ATTN_ALL_SUBMODULES}" == "1" || "${TARGET_DEVICE}" == "rocm" ]]; then
    fa_subs="auto"
  fi
  register_from_fetchcontent \
    vllm-flash-attn gpu "${REPO_ROOT}/cmake/external_projects/vllm_flash_attn.cmake" \
    "${fa_subs}" declare last
}

discover_cpu_deps() {
  register_from_fetchcontent \
    arm_compute cpu "${REPO_ROOT}/cmake/cpu_extension.cmake" auto populate last

  # cpu_extension.cmake has three oneDNN declarations:
  #   1. local SOURCE_DIR override
  #   2. aarch64 online fetch
  #   3. non-aarch64 online fetch
  local onednn_decl=3
  case "$(uname -m)" in
    aarch64|arm64) onednn_decl=2 ;;
  esac
  register_from_fetchcontent \
    oneDNN cpu "${REPO_ROOT}/cmake/cpu_extension.cmake" auto declare "${onednn_decl}"
}

discover_deps() {
  case "$SCOPE" in
    gpu) discover_gpu_deps ;;
    cpu) discover_cpu_deps ;;
    all)
      discover_gpu_deps
      discover_cpu_deps
      ;;
    *) die "unknown scope '${SCOPE}'" ;;
  esac
}

print_manifest() {
  local -a names=("$@")
  local n
  printf 'Resolved %d dep(s) from current checkout (device=%s):\n' \
    "${#names[@]}" "${TARGET_DEVICE}"
  for n in "${names[@]}"; do
    printf '  %-16s [%s]\n' "$n" "${REPO_GROUP[$n]}"
    printf '      repo: %s\n' "${REPO_URL[$n]}"
    printf '      tag : %s\n' "${REPO_TAG[$n]}"
    printf '      subs: %s\n' "${REPO_SUBS[$n]}"
    printf '      from: %s\n' "${REPO_SOURCE[$n]#${REPO_ROOT}/}"
  done
}

git_args() {
  local -a args=(git)
  [[ -n "$MIRROR" ]] && args+=(-c "url.${MIRROR}.insteadOf=https://github.com/")
  printf '%s\0' "${args[@]}"
}

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

sync_origin_url() { # src url
  local src="$1" url="$2" current
  current="$(git -C "$src" remote get-url origin 2>/dev/null || true)"
  if [[ -z "$current" ]]; then
    say "      adding origin remote ${url}"
    ( cd "$src" && run_git remote add origin "$url" ) || return 1
  elif [[ "$current" != "$url" ]]; then
    say "      updating origin remote:"
    say "        old: ${current}"
    say "        new: ${url}"
    ( cd "$src" && run_git remote set-url origin "$url" ) || return 1
  fi
}

init_submodules() { # src subs
  local src="$1" subs="$2"
  [[ "$subs" == "none" ]] && return 0
  [[ -f "$src/.gitmodules" ]] || return 0

  local -a args=(submodule update --init --recursive --progress)
  if [[ "$subs" != "auto" ]]; then
    read -r -a sub_paths <<<"$subs"
    args+=("${sub_paths[@]}")
  fi
  ( cd "$src" && run_git "${args[@]}" )
}

ensure_repo() { # name
  local name="$1"
  local url="${REPO_URL[$name]}" tag="${REPO_TAG[$name]}" subs="${REPO_SUBS[$name]}"
  local src="${DEPS_DIR}/${name}-src"

  say "== ${name}  ${url}  @ ${tag}"
  if [[ -e "$src" && ! -d "$src/.git" ]]; then
    say "      FAILED: ${src} exists but is not a git checkout"
    return 1
  fi

  if [[ -d "$src/.git" ]]; then
    say "      already downloaded at ${src} (reusing)"
    sync_origin_url "$src" "$url" || return 1
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

check_repo() { # name (read-only; returns 0 when OK)
  local name="$1"
  local url="${REPO_URL[$name]}" tag="${REPO_TAG[$name]}" subs="${REPO_SUBS[$name]}"
  local src="${DEPS_DIR}/${name}-src"
  local remote head expected

  printf '== %-16s %s @ %s\n' "$name" "$url" "$tag"
  [[ -d "$src" ]] || { echo "      MISSING: ${src}"; return 1; }
  [[ -d "$src/.git" ]] || { echo "      NO_GIT:  ${src} (not a git checkout)"; return 1; }

  remote="$(git -C "$src" remote get-url origin 2>/dev/null || true)"
  if [[ "$remote" != "$url" ]]; then
    echo "      WRONG_REMOTE: origin=${remote:-<missing>}, expected ${url}"
    return 1
  fi

  head="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
  expected="$(git -C "$src" rev-parse -q --verify "$tag^{commit}" 2>/dev/null || true)"
  if [[ -z "$expected" ]]; then
    echo "      MISSING_REF: local checkout does not know ${tag}"
    return 1
  fi
  if [[ "$head" != "$expected" ]]; then
    echo "      WRONG_REF: HEAD=${head:0:12}..., expected ${tag} (${expected:0:12}...)"
    return 1
  fi

  if [[ "$subs" != "none" ]] && [[ -f "$src/.gitmodules" ]]; then
    local -a sargs=()
    if [[ "$subs" != "auto" ]]; then
      read -r -a sargs <<<"$subs"
    fi
    if git -C "$src" submodule status "${sargs[@]}" 2>/dev/null | grep -q '^[-+]'; then
      echo "      SUBMODULE_NOT_READY: some submodules are not initialized"
      return 1
    fi
  fi

  echo "      OK"
  return 0
}

fix_stamps() {
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
  sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" |
    sed -n 's/^# \{0,1\}//p' |
    sed '$d'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)               SCOPE="gpu" ;;
    --cpu)               SCOPE="cpu" ;;
    --all)               SCOPE="all" ;;
    --deps-dir)          DEPS_DIR="$2"; shift ;;
    --retries)           RETRIES="$2"; shift ;;
    --mirror)            MIRROR="$2"; shift ;;
    --device)            TARGET_DEVICE="$2"; shift ;;
    --list)              LIST_ONLY=1 ;;
    --fix-stamps)        FIX_STAMPS=1 ;;
    --check)             CHECK_ONLY=1 ;;
    -q|--quiet)          QUIET=1 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

case "$TARGET_DEVICE" in
  cuda|rocm) ;;
  *) die "--device must be 'cuda' or 'rocm' (got '${TARGET_DEVICE}')" ;;
esac

mkdir -p "$DEPS_DIR"
DEPS_DIR="$(cd -- "$DEPS_DIR" && pwd)"

discover_deps

case "$SCOPE" in
  gpu) names=( "${GPU_DEPS[@]}" ) ;;
  cpu) names=( "${CPU_DEPS[@]}" ) ;;
  all) names=( "${GPU_DEPS[@]}" "${CPU_DEPS[@]}" ) ;;
esac

if [[ "$LIST_ONLY" == "1" ]]; then
  print_manifest "${names[@]}"
  exit 0
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  print_manifest "${names[@]}"
  echo
  echo "Checking ${#names[@]} dep(s) under ${DEPS_DIR} (read-only)..."
  ok=0
  bad=0
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

print_manifest "${names[@]}"
say
say "Fetching ${#names[@]} repo(s) into ${DEPS_DIR} (${RETRIES} attempts, mirror='${MIRROR:-direct}')"
say
ok=0
fail=0
for n in "${names[@]}"; do
  if ensure_repo "$n"; then ok=$(( ok + 1 )); else fail=$(( fail + 1 )); fi
done
say
say "Done: ${ok} ready, ${fail} failed."
if (( fail > 0 )); then
  say "Some repos failed. Re-run this script (already-downloaded repos are"
  say "reused), point VLLM_GIT_MIRROR/--mirror at another mirror, or check"
  say "your network."
  exit 1
fi
say
say "Next: build vLLM offline. cu130_build_wheel.sh already sets"
say "FETCHCONTENT_FULLY_DISCONNECTED=ON and will reuse ${DEPS_DIR} as-is."
