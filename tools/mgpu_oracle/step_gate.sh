#!/bin/bash
# The whole-step gate of the multi-GPU path: ONE time step of brae -parallel -device against OpenFOAM v2606 -parallel's own
# on the same decomposition and the same state (U, p_rgh, phi, alpha at the first restart time), final fields compared with
# compare_fields.py. Momentum, the three pressure correctors (distributed AMG-PCG vs OpenFOAM's GAMG at its tolerances: the
# FLOOR below), the alpha step and the step collectives all sit between the two states.
#
#   tools/mgpu_oracle/step_gate.sh <brae_interFoam> <gate dir> [tolU tolP tolPhi tolAlpha]
#       <gate dir> = a tools/mgpu_oracle/stage_gate.sh output (KEEP=1): <dir>/of and <dir>/bd hold the decomposed state
#   Default floors (maxrel to max|A|; measured np=2 tank: U 8.6e-05, p_rgh 6.5e-05, phi 1.2e-04, alpha 1.9e-07, which are
#   brae's own single-GPU numbers against OpenFOAM's: U 2.6e-05, p_rgh 6.6e-05, phi 6.9e-05, alpha 1.9e-07):
#       U 1e-3, p_rgh 5e-4, phi 5e-3, alpha 1e-6
#   BRAE_CONTROL_PROC_NO_EXCHANGE=1 (halos never refreshed) must FAIL it: run with CONTROL=1 and the exit status is inverted.
# env: ORACLE_ROOT, KEEP=1 keeps the working copies (<gate dir>/step_of, step_bd).
set -eo pipefail
BRAE=${1:?usage: step_gate.sh <brae_interFoam> <gate dir> [tolU tolP tolPhi tolAlpha]}
DIR=$(readlink -f "${2:?gate dir}")
TU=${3:-1e-3}; TP=${4:-5e-4}; TPHI=${5:-5e-3}; TA=${6:-1e-6}
# STEPS=3|10 floors (measured np 2 / np 4, 3 steps; 10 steps: U 1.25e-3, p 9e-7 / 1.9e-6, phi 3.0e-3 / 3.2e-3 against OpenFOAM: U 6.0e-4 / 9.0e-4, p_rgh 9e-6 / 7e-6, phi 1.2e-3 / 1.8e-3, alpha 1.9e-7 / 6.3e-7)
[ "${STEPS:-1}" = 3 -o "${STEPS:-1}" = 10 ] && { TU=${3:-3e-3}; TP=${4:-5e-4}; TPHI=${5:-1e-2}; TA=${6:-1e-5}; }
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ORACLE_ROOT=${ORACLE_ROOT:-/mnt/storage/brae-scratch/oracle}
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
NP=$(ls -d "$DIR"/of/processor* | wc -l)
cd "$DIR"
rm -rf step_of step_bd
# STEPS=1 (default): one step; STEPS=3: three (endTime 0.0006 ends the third step of the tank, dt 1.44e-4, 1.73e-4, 2.07e-4)
STEPS=${STEPS:-1}; ENDT=0.0002; [ "$STEPS" = 3 ] && ENDT=0.0006; [ "$STEPS" = 10 ] && ENDT=0.002
for side in of bd; do
    cp -r $side step_$side
    for p in step_$side/processor*; do rm -f $p/*/*.dump; done
    sed -i "s/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         $ENDT;/; s/^writeInterval .*/writeInterval   1;/" step_$side/system/controlDict
done
unset BRAE_STAGE_PRESSURE BRAE_MGPU_STAGE_DUMP BRAE_STAGE_DUMP BRAE_DUMP_ITER
( cd step_of && mpirun --bind-to none -np "$NP" "$ORACLE_ROOT/bin/dumpInterFoam" -parallel > ../log.step_of 2>&1 )
# a run that reached "End" on every rank has stepped; an exit status from the launcher's teardown (observed at np 8:
# a glibc abort in an Open MPI/PMIx thread AFTER every rank printed End) is not a failed step
( cd step_bd && mpirun --bind-to none -np "$NP" "$BRAE" -case . -parallel -device > ../log.step_bd 2>&1 ) || true
[ "$(grep -c '^End: t = ' log.step_bd)" = "$NP" ] || { echo "FAIL: not every rank reached End (log.step_bd)"; exit 1; }
T=$(ls step_bd/processor0 | grep -E '^[0-9.e+-]+$' | sort -g | tail -n 1)
python3 "$HERE/compare_fields.py" step_of step_bd --times "$T" --fields U,p_rgh,phi,alpha.water \
    --tol-field U=$TU --tol-field p_rgh=$TP --tol-field phi=$TPHI --tol-field alpha.water=$TA --tol 1e9 > log.step_cmp 2>&1 || true
cat log.step_cmp
rc=0
for spec in U:$TU p_rgh:$TP phi:$TPHI alpha.water:$TA; do
    f=${spec%%:*}; tol=${spec##*:}
    worst=$(awk -F'|' -v f="$f" '$4 ~ " "f" $" && $5 ~ /internal|boundary/ {print $8+0}' log.step_cmp | sort -g | tail -n 1)
    [ -n "$worst" ] || { echo "no row for $f"; rc=1; continue; }
    awk -v w="$worst" -v t="$tol" 'BEGIN{exit !(w <= t)}' || { echo "EXCEEDS $f: $worst > $tol"; rc=1; }
done
[ -n "$KEEP" ] || rm -rf step_of step_bd
exit $rc
