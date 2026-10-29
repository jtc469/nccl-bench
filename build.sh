#!/bin/bash
#
# Build the benchmark binaries into external/build/ for a given system.
#
# Usage: ./build.sh [SYSTEM] [extra make args...]
#   e.g. ./build.sh scc
#        ./build.sh scc NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
#

set -e

SYSTEM=$1
shift || true

if [[ -z "$SYSTEM" || ! -f "config/${SYSTEM}.env" ]]; then
    echo "Usage: $0 [SYSTEM] [extra make args...]"
    echo "Systems: $(ls config | sed 's/\.env$//' | tr '\n' ' ')"
    exit 1
fi

source "config/${SYSTEM}.env"

# Only load modules on systems that have them
if command -v module >/dev/null 2>&1; then
    module purge

    for mod in "${MODULES[@]}"; do
        module load "$mod"
    done
fi


# Get the source. Refuse to build over a different checkout,
# e.g. stock NVIDIA nccl-tests, which can't write csv output
if [[ ! -d external ]]; then
    git clone ${TESTS_BRANCH:+--branch "$TESTS_BRANCH"} "$TESTS_REPO" external
else
    REMOTE=$(git -C external remote get-url origin)
    BRANCH=$(git -C external rev-parse --abbrev-ref HEAD)

    if [[ "$REMOTE" != "$TESTS_REPO" ]] \
        || [[ -n "$TESTS_BRANCH" && "$BRANCH" != "$TESTS_BRANCH" ]]; then
        echo "external/ is ${REMOTE} (${BRANCH})"
        echo "but ${SYSTEM} needs ${TESTS_REPO} (${TESTS_BRANCH:-default branch})"
        echo "Move or remove external/ and re-run."
        exit 1
    fi

    git -C external pull --ff-only
fi


# Point make at module-provided CUDA/NCCL if not already set
NCCL_HOME=${NCCL_HOME:-$EBROOTNCCL}

MAKE_ARGS=(-j"$(nproc)")
[[ -n "$CUDA_HOME" ]] && MAKE_ARGS+=(CUDA_HOME="$CUDA_HOME")
[[ -n "$NCCL_HOME" ]] && MAKE_ARGS+=(NCCL_HOME="$NCCL_HOME")

make -C external "${MAKE_ARGS[@]}" "$@"


# Record which build was used, for the report
BUILD_INFO=external/build/build-info.txt

printf '%s\n' \
    "system: $SYSTEM" \
    "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "host: $(hostname)" \
    "repo: $TESTS_REPO" \
    "branch: $(git -C external rev-parse --abbrev-ref HEAD)" \
    "commit: $(git -C external rev-parse HEAD)" \
    "make args: ${MAKE_ARGS[*]} $*" \
    "modules: ${MODULES[*]}" \
    > "$BUILD_INFO"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1 >> "$BUILD_INFO"
elif command -v hipcc >/dev/null 2>&1; then
    hipcc --version | head -n 1 >> "$BUILD_INFO"
fi

echo "Built into external/build/"
cat "$BUILD_INFO"
