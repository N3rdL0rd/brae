#!/bin/bash
# Build brae_interFoam for ONE V100 (sm_70) with CUDA 12.x. No root, no GPU needed to compile.
#   ./build_v100.sh [SRC=<repo containing this script>] [BUILD=$SRC/build_v100]
# Build the OLD binary (git main, host-loop solvers on CUDA 12) and the NEW one (the CUDA-12 branch) from two checkouts:
#   git worktree add ../brae-old main        ; ./build_v100.sh ../brae-old ../brae-old/build_v100
#   ./build_v100.sh <the branch checkout>    ; (or pass the two binaries to run_v100_bench.sh)
# Environment (defaults follow ~/code/maxi_job/brae_build_rosie.sh):
#   CUDA_MODULE=cuda/12.9  OMPI_MODULE=openmpi/4.1.4   module names; unset/empty to skip `module load`
#   CUDA_HOME=...   use this toolkit instead (puts $CUDA_HOME/bin first, passes -DCMAKE_CUDA_COMPILER)
#   HOSTCXX=g++-14  host compiler for nvcc 12.x (gcc <= 14 is supported by 12.9; -DCMAKE_CUDA_HOST_COMPILER)
#   ARCH=70  NJ=16  OF_ROOT=$HOME/OpenFOAM (Scotch from OF_ROOT/ThirdParty-v2606, or SCOTCH_INC/SCOTCH_LIB)
# The recipe from the Cuda12Enable work, if it differs (extra cmake flags), goes in EXTRA_CMAKE="-D..."
set -eo pipefail
SRC=${1:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
BUILD=${2:-$SRC/build_v100}
NJ=${NJ:-16}
ARCH=${ARCH:-70}
OF_ROOT=${OF_ROOT:-$HOME/OpenFOAM}
[ -f "$SRC/CMakeLists.txt" ] || { echo "no brae source in $SRC" >&2; exit 1; }

if [ -z "${CUDA_HOME:-}" ]; then
    command -v module >/dev/null 2>&1 || source /etc/profile
    module load ${CUDA_MODULE-cuda/12.9} ${OMPI_MODULE-openmpi/4.1.4}
else
    export PATH=$CUDA_HOME/bin:$PATH
    [ -z "${OMPI_MODULE-x}" ] || { command -v module >/dev/null 2>&1 || source /etc/profile; module load ${OMPI_MODULE-openmpi/4.1.4} || true; }
fi
echo "nvcc: $(command -v nvcc)  $(nvcc --version | tail -n 1)"
nvcc --version | grep -q "release 12\." || { echo "this kit expects CUDA 12.x (V100 is not supported by CUDA 13)" >&2; exit 1; }
echo "gcc: $(gcc -dumpversion)  host for nvcc: ${HOSTCXX:-default}  cmake: $(cmake --version | head -n 1)  mpicc: $(command -v mpicc)"

SCOTCH_INC=${SCOTCH_INC:-$(ls -d "$OF_ROOT"/ThirdParty-v2606/platforms/*/scotch_*/include 2>/dev/null | head -n 1)}
SCOTCH_LIB=${SCOTCH_LIB:-$(dirname "$(ls "$OF_ROOT"/ThirdParty-v2606/platforms/*/lib/libscotch.so 2>/dev/null | head -n 1)")}
if [ ! -f "$SCOTCH_INC/scotch.h" ]; then SCOTCH_INC=/usr/include/scotch; SCOTCH_LIB=/usr/lib64; fi
[ -f "$SCOTCH_INC/scotch.h" ] || { echo "Scotch headers not found (set SCOTCH_INC / SCOTCH_LIB)" >&2; exit 1; }
echo "scotch: $SCOTCH_INC  $SCOTCH_LIB"

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DBRAE_WITH_NVSHMEM=OFF \
      ${CUDA_HOME:+-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc} ${HOSTCXX:+-DCMAKE_CUDA_HOST_COMPILER=$HOSTCXX} \
      "-DCMAKE_INCLUDE_PATH=$SCOTCH_INC" "-DCMAKE_LIBRARY_PATH=$SCOTCH_LIB" \
      "-DCMAKE_CXX_FLAGS=-I$SCOTCH_INC" "-DCMAKE_CUDA_FLAGS=-I$SCOTCH_INC" "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$SCOTCH_LIB" ${EXTRA_CMAKE:-} \
    || { echo "cmake failed (no Ninja on this machine? remove '-G Ninja' from the cmake line)" >&2; exit 1; }
cmake --build "$BUILD" -j "$NJ" --target brae_interFoam 2>&1 | tee "$BUILD/build.log" | grep -a -E "error|Error|Linking|FAILED" | cut -c1-240 || true
[ -x "$BUILD/brae_interFoam" ] || { echo "brae_interFoam was not built; see $BUILD/build.log" >&2; exit 1; }
echo "built: $BUILD/brae_interFoam"
