#!/usr/bin/env bash
# Build brae_interFoam and the ProcComm tests for Rosie's V100 nodes.
# Run on dh-node3 in ONE Slurm batch allocation (not on a login node or the DGX).
#   sbatch -p batch --nodelist=dh-node3 -N1 -n1 -c 32 --mem=64G -t 01:00:00 \
#     --wrap='SRC=$HOME/brae-mgpu/src NJ=16 $HOME/brae-mgpu/src/tools/rosie/build_node3.sh'
# NJ controls ninja parallelism; default 16. The 32 CPU / 64 GB request leaves
# headroom for nvcc host compilation on nodes shared with the batch scheduler.
# All paths below are overridable for testing; defaults follow ~/brae-mgpu.
set -Eeuo pipefail
umask 022

ROOT=${ROOT:-$HOME/brae-mgpu}
SRC=${SRC:-$ROOT/src}
BUILD=${BUILD:-$ROOT/build70c}
ENV_PREFIX=${ENV_PREFIX:-${BRAE_CONDA_ENV:-$ROOT/env}}
CONDA=${CONDA:-$ROOT/conda}
NCCL_ROOT=${NCCL_ROOT:-$ROOT/nccl/nvidia/nccl}
RT=${RT:-$ROOT/cuda12rt}
CUDA=${CUDA:-/usr/local/cuda-12.9}
NJ=${NJ:-16}
SKIP_NETWORK_CHECK=${SKIP_NETWORK_CHECK:-0}
RECONFIGURE=0
BUILD_SPIKES=1

usage() {
    cat <<'EOF'
Usage: build_node3.sh [--reconfigure] [--no-spikes]
Environment: CUDA_FLAGS_EXTRA (default --compress-mode=none: nvcc 12.8+ compresses fatbins by default, which drivers older than 550 cannot load), SRC, ROOT, BUILD, ENV_PREFIX, CONDA, NCCL_ROOT, RT, CUDA, NJ, GCC_VER (conda gcc major, 14 for CUDA 12.9; nvcc 12.1 needs <= 12).
  --reconfigure  remove only CMake's cache/configuration, preserving objects
  --no-spikes    skip the optional nccl_while_graph standalone test programs
Incremental by default. SRC can be any brae source checkout (including a
worktree containing the final merged solver). A failed build leaves its log in
$BUILD/build.log; rerun the same command after correcting the error.
EOF
}
while (($#)); do
    case "$1" in
        --reconfigure) RECONFIGURE=1 ;;
        --no-spikes) BUILD_SPIKES=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
[[ $NJ =~ ^[1-9][0-9]*$ ]] || { echo "NJ must be a positive integer" >&2; exit 2; }
[[ -f $SRC/CMakeLists.txt && -x $CUDA/bin/nvcc ]] || {
    echo "need $SRC/CMakeLists.txt and $CUDA/bin/nvcc on dh-node3" >&2; exit 1;
}
for command_name in curl objdump ldd sort tar; do
    command -v "$command_name" >/dev/null || { echo "missing $command_name" >&2; exit 1; }
done

# Conda's sysroot is the key compatibility constraint: a newer host sysroot can
# silently introduce GLIBC_2.32+ references which the DGX (glibc 2.31) rejects.
if [[ ! -x $CONDA/bin/conda ]]; then
    if [[ $SKIP_NETWORK_CHECK != 1 ]]; then
        curl -fsSL --max-time 60 -o /dev/null -I \
            https://conda.anaconda.org/conda-forge/linux-64/repodata.json.zst || {
            echo "NO NETWORK to conda-forge" >&2; exit 2;
        }
    fi
    curl -fsSL --max-time 300 -o "$ROOT/miniforge.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$ROOT/miniforge.sh" -b -p "$CONDA"
    rm -f "$ROOT/miniforge.sh"
fi
if [[ ! -x $ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++ ]]; then
    "$CONDA/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels \
        gcc_linux-64=${GCC_VER:-14} gxx_linux-64=${GCC_VER:-14} sysroot_linux-64=2.28 \
        openmpi scotch cmake=3.28 ninja make zlib
fi
# Pin on every invocation, including an env made earlier without the pin.
"$CONDA/bin/conda" install -y -p "$ENV_PREFIX" -c conda-forge --override-channels \
    sysroot_linux-64=2.28

export PATH="$ENV_PREFIX/bin:$CUDA/bin:$PATH"
CXX=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++
CC=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export OMPI_CXX=$CXX OMPI_CC=$CC OPAL_PREFIX=$ENV_PREFIX
printf 'host %s glibc ' "$(hostname)"
ldd --version | awk 'NR == 1 { print $NF }'
echo "g++: $($CXX -dumpversion) cmake: $(cmake --version | awk 'NR == 1') mpicxx: $(command -v mpicxx)"
nvcc --version | awk 'END { print "nvcc:", $0 }'
[[ -f $ENV_PREFIX/include/scotch.h && -e $ENV_PREFIX/lib/libscotch.so ]] || {
    echo "scotch missing in $ENV_PREFIX" >&2; exit 4;
}
[[ -f $NCCL_ROOT/include/nccl.h && -e $NCCL_ROOT/lib/libnccl.so.2 ]] || {
    echo "NCCL include/nccl.h and lib/libnccl.so.2 are required under $NCCL_ROOT" >&2; exit 4;
}
[[ -e $NCCL_ROOT/lib/libnccl.so ]] || ln -s libnccl.so.2 "$NCCL_ROOT/lib/libnccl.so"

mkdir -p "$RT" "$BUILD"
cp -a "$CUDA"/lib64/libcudart.so* "$CUDA"/lib64/libnvrtc.so* \
    "$CUDA"/lib64/libnvrtc-builtins.so* "$CUDA"/lib64/libnvJitLink.so* "$RT/"
# Compat is deliberately absent from every link and rpath flag. It contains a
# proprietary forward-compatibility libcuda and belongs in LD_LIBRARY_PATH only
# on the DGX, never in a binary's build-time dependency search path.
LINK_FLAGS="-Wl,-rpath,$ENV_PREFIX/lib -Wl,-rpath,$NCCL_ROOT/lib -Wl,-rpath,$RT -L$ENV_PREFIX/lib"
if ((RECONFIGURE)); then
    rm -f "$BUILD/CMakeCache.txt"
    rm -rf "$BUILD/CMakeFiles"
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBRAE_WITH_NVSHMEM=OFF \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
    -DMPI_CXX_COMPILER="$ENV_PREFIX/bin/mpicxx" -DMPIEXEC_EXECUTABLE="$ENV_PREFIX/bin/mpiexec" \
    -DBRAE_WITH_NCCL=ON -DNCCL_ROOT="$NCCL_ROOT" "-DBRAE_NCCL_TEST_NP=1;2;4;8" \
    -DCMAKE_INCLUDE_PATH="$ENV_PREFIX/include" -DCMAKE_LIBRARY_PATH="$ENV_PREFIX/lib" \
    "-DCMAKE_CXX_FLAGS=-I$ENV_PREFIX/include" "-DCMAKE_CUDA_FLAGS=-I$ENV_PREFIX/include ${CUDA_FLAGS_EXTRA---compress-mode=none}" \
    "-DCMAKE_EXE_LINKER_FLAGS=$LINK_FLAGS"

targets=(brae_interFoam test_proc_comm test_proc_comm_graph test_proc_topology proc_topology_tool)
echo "Building ${targets[*]} with NJ=$NJ"
# Keep the entire log, not just grep's filtered lines, for diagnosis on the node.
set +e
cmake --build "$BUILD" -j "$NJ" --target "${targets[@]}" 2>&1 | tee "$BUILD/build.log"
build_rc=${PIPESTATUS[0]}
set -e
((build_rc == 0)) || { echo "build failed: see $BUILD/build.log" >&2; exit "$build_rc"; }
for binary in "${targets[@]}"; do
    [[ -x $BUILD/$binary ]] || { echo "MISSING $binary (see $BUILD/build.log)" >&2; exit 5; }
done

spikes=()
if ((BUILD_SPIKES)); then
    spike_dir=$SRC/tools/spikes/nccl_while_graph
    [[ -f $spike_dir/Makefile ]] || { echo "missing $spike_dir/Makefile" >&2; exit 5; }
    make -C "$spike_dir" clean
    make -C "$spike_dir" -j "$NJ" ARCH=70 HOSTCXX="$CXX" NCCL_ROOT="$NCCL_ROOT" \
        CUDA_HOME="$CUDA" MPI_HOME="$ENV_PREFIX" \
        EXTRA="-Xlinker -rpath,$ENV_PREFIX/lib -Xlinker -rpath,$NCCL_ROOT/lib -Xlinker -rpath,$RT -L$ENV_PREFIX/lib"
    spikes=("$spike_dir/nccl_spike" "$spike_dir/nccl_spike_mpi")
fi

binaries=()
for binary in "${targets[@]}"; do binaries+=("$BUILD/$binary"); done
binaries+=("${spikes[@]}")
status=0
for binary in "${binaries[@]}"; do
    [[ -x $binary ]] || { echo "missing built binary: $binary" >&2; status=1; continue; }
    max_glibc=$(objdump -T "$binary" | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)*' | sort -Vu | awk 'END { print }')
    max_glibc=${max_glibc:-GLIBC_none}
    echo "$binary: newest GLIBC symbol ${max_glibc}"
    if [[ $max_glibc != GLIBC_none ]]; then
        version=${max_glibc#GLIBC_}
        if [[ $(printf '2.31\n%s\n' "$version" | sort -V | awk 'END { print }') != 2.31 ]]; then
            echo "ERROR: $binary needs $max_glibc (> 2.31)" >&2; status=1
        fi
    fi
    echo "ldd 'not found' entries for $binary:"
    not_found=$(LD_LIBRARY_PATH="$ENV_PREFIX/lib:$NCCL_ROOT/lib:$RT:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$binary" | grep -F 'not found' || true)
    if [[ -n $not_found ]]; then
        printf '%s\n' "$not_found"
        echo "ERROR: unresolved runtime library above (compat must NOT be needed on dh-node3)" >&2
        status=1
    else
        echo "  (none)"
    fi
    dynamic=$(readelf -d "$binary")
    if grep -Fq "$ROOT/compat" <<<"$dynamic"; then
        echo "ERROR: compat directory leaked into dynamic tags of $binary" >&2; status=1
    fi
    rpaths=$dynamic
    rpaths=$(grep -E 'RPATH|RUNPATH' <<<"$rpaths" || true)
    printf '%s\n' "$rpaths"
    for required_path in "$ENV_PREFIX/lib" "$NCCL_ROOT/lib" "$RT"; do
        if ! grep -Fq "$required_path" <<<"$rpaths"; then
            echo "ERROR: required rpath $required_path missing from $binary" >&2; status=1
        fi
    done
done
((status == 0)) || exit "$status"
echo "BUILD OK: $BUILD (source $SRC; incremental NJ=$NJ)"
