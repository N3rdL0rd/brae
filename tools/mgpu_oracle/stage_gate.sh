#!/bin/bash
# The operator gate of the multi-GPU explicit stages: brae's per-rank STAGE DUMP (BRAE_MGPU_STAGE_DUMP) against
# OpenFOAM v2606's own dump of the same quantities (tools/dumpInterFoam stageDump.H, BRAE_STAGE_DUMP) at np = 1, 2, 4, 8.
#
#   tools/mgpu_oracle/stage_gate.sh <brae_interFoam> <case> <np> [method=scotch] [tolerance]
#       case    tank | tankGraded | open            np  1 (serial) | 2 | 4 | 8       method  scotch | simple | hier
#   -> $OUT_ROOT/<case>_np<N>[_<method>]/   (OUT_ROOT default /tmp/mgpu_stage)
#
# What it does: builds the mesh and the initial alpha, runs OpenFOAM ONE step (serial, or decomposePar + mpirun -parallel)
# so that a time directory holds a real state (U, phi, alpha, p_rgh), copies that case twice, then
#   * OpenFOAM:  BRAE_STAGE_DUMP=1 dumpInterFoam [-parallel] (startFrom latestTime)   -> s*.dump in the time directory
#   * brae:      BRAE_MGPU_STAGE_DUMP=<copy> brae_interFoam -device [-parallel]  (one process per rank, mpirun
#                --oversubscribe, MPI-host ProcComm; one GPU)                  -> s*.dump per rank
# and compares the two trees with compare_fields.py (mapped onto the global mesh through the processor addressing).
# Both sides read the SAME files, so no linear solve and no limiter is involved: the numbers are operator differences.
# The tolerance (maxrel, default 1e-11) is the exit status; the table is always printed.
# env: ORACLE_ROOT (default /mnt/storage/brae-scratch/oracle; setup.sh builds it), OUT_ROOT, KEEP=1 keeps the trees.
set -eo pipefail
BRAE=${1:?usage: stage_gate.sh <brae_interFoam> <case> <np> [method] [tol]}
CASE=${2:?case}
NP=${3:?np}
METHOD=${4:-scotch}
TOL=${5:-1e-11}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ORACLE_ROOT=${ORACLE_ROOT:-/mnt/storage/brae-scratch/oracle}
OUT_ROOT=${OUT_ROOT:-/tmp/mgpu_stage}
SUF=""; [ "$NP" = 1 ] || { [ "$METHOD" = scotch ] || SUF="_$METHOD"; }
OUT=$OUT_ROOT/${CASE}_np${NP}${SUF}
FIELDS=sAlpha.dump,sPhi.dump,sNHatf.dump,sSigmaK.dump,sStf.dump,sRho.dump,sSnGradRho.dump,sPhic.dump,sPhir.dump,sAlphaPhiUn.dump
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "no dump tool: ORACLE_ROOT=$ORACLE_ROOT $HERE/setup.sh" >&2; exit 1; }
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
SOLVER=$ORACLE_ROOT/bin/dumpInterFoam
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"

cp -r "$HERE/cases/$CASE" state
( cd state
  rm -f system/decomposeParDict.*
  blockMesh > ../log.blockMesh 2>&1; topoSet > ../log.topoSet 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig
  setFields > ../log.setFields 2>&1 )
if [ "$NP" != 1 ]; then
    DICT=$HERE/cases/$CASE/system/decomposeParDict.$METHOD$NP
    [ -f "$DICT" ] || { echo "no $DICT" >&2; exit 1; }
    cp "$DICT" state/system/decomposeParDict
    ( cd state && decomposePar -force > ../log.decompose 2>&1 )
fi
# one real step: the state the stages are taken on
export BRAE_STOP_ITER=1 BRAE_WRITE_ITERS=1
if [ "$NP" = 1 ]; then ( cd state && "$SOLVER" > ../log.step1 2>&1 )
else ( cd state && mpirun --bind-to none -np "$NP" "$SOLVER" -parallel > ../log.step1 2>&1 ); fi
unset BRAE_STOP_ITER BRAE_WRITE_ITERS
cp -r state of; cp -r state bd
# both start from the step-1 time directory
sed -i 's/^startFrom .*/startFrom       latestTime;/' of/system/controlDict bd/system/controlDict

# OpenFOAM: the stages
export BRAE_STAGE_DUMP=1
if [ "$NP" = 1 ]; then ( cd of && "$SOLVER" > ../log.of 2>&1 )
else ( cd of && mpirun --bind-to none -np "$NP" "$SOLVER" -parallel > ../log.of 2>&1 ); fi
unset BRAE_STAGE_DUMP

# brae: the same stages (start from the latest time, as OpenFOAM's -latestTime)
export BRAE_MGPU_STAGE_DUMP=$OUT/bd
if [ "$NP" = 1 ]; then "$BRAE" -case bd -device > log.brae 2>&1
else mpirun --bind-to none -np "$NP" "$BRAE" -case bd -parallel -device > log.brae 2>&1; fi
unset BRAE_MGPU_STAGE_DUMP
grep -a "stage dump" log.brae | head -n 8

T=$(ls -d state/[0-9]* 2>/dev/null | xargs -n1 basename | sort -g | tail -n 1)
[ -n "$T" ] && [ "$T" != 0 ] || T=$(ls -d state/processor0/[0-9]* | xargs -n1 basename | sort -g | tail -n 1)
echo "state time $T; case $CASE np $NP $METHOD"
set +e
python3 "$HERE/compare_fields.py" of bd --times "$T" --fields "$FIELDS" --tol "$TOL" > compare.md
RC=$?
set -e
cat compare.md
[ "${KEEP:-0}" = 1 ] || rm -rf state
exit $RC
