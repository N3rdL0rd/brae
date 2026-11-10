#!/usr/bin/env bash
# One exclusive DGX-1 (8x V100) allocation, start to finish, as the batch job itself -- so the node is released the
# moment the script ends, whatever happened:
#   sbatch -p dgx -N1 --gres=gpu:v100:8 --exclusive -t 03:00:00 --job-name=g4brae \
#     --output=$HOME/brae-mgpu/results/session.%j.log --wrap='$HOME/brae-mgpu/src/tools/rosie/session_dgx.sh'
# In order, each step only if the ones it needs passed:
#   0. the node: driver, GPUs. A driver older than 535 could not load this build when it was tried (docs/multi-gpu.md);
#      the np = 1 check below is the test, and the session ends there if it fails.
#   1. fresh extractions of the four staged G4 cases (a staged case is one run)
#   2. two-step smoke checks: np 1, then np 2 on MPI-host, then np 2/4/8 on NCCL (MPI-host where NCCL fails)
#   3. PARITY: PARITY_STEPS fixed steps with every solve converged at np 1/2/4/8, each compared with np 1
#      (tools/mgpu_oracle/compare_fields.py; needs numpy -- without it the kept cases are compared afterwards)
#   4. the 60-step timing runs at np 1/2/4/8 on the case's own tolerances
# Everything lands in $ROOT/results/session_<tag>/; summary.txt is the table to copy back.
set -uo pipefail
umask 022
ROOT=${ROOT:-$HOME/brae-mgpu}
SRC=${SRC:-$ROOT/src}
export BRAE_ROOT=${BRAE_ROOT:-$ROOT/build70c}
export MPIEXEC=${MPIEXEC:-$ROOT/env/bin/mpirun}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RES=$ROOT/results/session_$TAG
CASES=$ROOT/cases/session_$TAG
PARITY_STEPS=${PARITY_STEPS:-5}
NPS=${NPS:-"1 2 4 8"}
# the first of NPS is the reference of the parity step; CASE_PREFIX and MIN_GPUS exist for a laptop rehearsal
CASE_PREFIX=${CASE_PREFIX:-g4_np}
MIN_GPUS=${MIN_GPUS:-8}
REF=${NPS%% *}
CHECK_TIMEOUT=${CHECK_TIMEOUT:-420}
PARITY_TIMEOUT=${PARITY_TIMEOUT:-1200}
TIMING_TIMEOUT=${TIMING_TIMEOUT:-1200}
# seconds after which no new step is started (the allocation's limit less the slowest step)
DEADLINE=${DEADLINE:-9300}
RUN=$SRC/tools/rosie/run_dgx.sh
T0=$(date +%s)
mkdir -p "$RES" "$CASES"
say() { printf '%s [%5ds] %s\n' "$(date -u +%T)" "$(($(date +%s) - T0))" "$*" | tee -a "$RES/session.log"; }
note() { printf '%s\n' "$*" >> "$RES/summary.txt"; }
late() { (($(date +%s) - T0 > DEADLINE)); }
# step <name> <timeout> <run_dgx options...>: 0 if the run and its summary passed
step() {
    local name=$1 limit=$2; shift 2
    if late; then say "SKIP $name: past the session deadline"; note "$name SKIPPED(deadline)"; return 1; fi
    say "START $name: $*"
    "$RUN" "$@" --timeout "$limit" --results "$RES/$name" > "$RES/$name.out" 2>&1
    local rc=$?
    local line; line=$(grep -h '^case=' "$RES/$name.out" | tail -n 1)
    say "END   $name rc=$rc ${line:-$(tail -n 2 "$RES/$name.out" | tr '\n' ' ')}"
    note "$name rc=$rc ${line}"
    if ((rc != 0)); then
        for f in "$RES/$name"/ranks/rank.*.log "$RES/$name/mpirun.log"; do
            [[ -s $f ]] && { echo "--- tail $f"; tail -n 25 "$f"; } >> "$RES/$name.failure.txt"
        done
    fi
    return $rc
}

say "session $TAG on $(hostname); source $(cat "$SRC/.synced_commit" 2>/dev/null || echo unknown); binary $(sha256sum "$BRAE_ROOT/brae_interFoam" | cut -c1-16)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | tee -a "$RES/session.log"
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
note "host=$(hostname) driver=$DRIVER gpus=$NGPU tag=$TAG"
((NGPU >= MIN_GPUS)) || { say "STOP: $NGPU GPUs visible, $MIN_GPUS expected"; exit 3; }
((${DRIVER%%.*} >= 535)) || say "driver $DRIVER is older than 535: the np = 1 check decides whether this build loads here"

say "extracting the staged cases into $CASES"
for n in $NPS; do
    tar -C "$CASES" -xf "$ROOT/$CASE_PREFIX$n.tar.zst" || { say "STOP: could not extract $CASE_PREFIX$n"; exit 4; }
done

# 2. smoke checks
declare -A COMM PCASE
if [[ " $NPS " == *" 1 "* ]]; then
    step check_np1 "$CHECK_TIMEOUT" --case "$CASES/${CASE_PREFIX}1" --np 1 --check || { say "STOP: np = 1 does not run on this node"; exit 5; }
    COMM[1]=nccl
fi
for n in $NPS; do
    ((n > 1)) || continue
    if ((n == 2)); then
        # the MPI-host arm first at np 2: the two backends failed differently when this was last tried
        step check_np2_mpi "$CHECK_TIMEOUT" --case "$CASES/${CASE_PREFIX}2" --np 2 --comm mpi --check && COMM[2]=mpi
    fi
    if step check_np${n}_nccl "$CHECK_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm nccl --check; then
        COMM[$n]=nccl
    elif ((n > 2)) && step check_np${n}_mpi "$CHECK_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm mpi --check; then
        COMM[$n]=mpi
    fi
    [[ -n ${COMM[$n]:-} ]] || say "np = $n runs on neither backend: left out of the parity and timing steps"
done
if [[ -z ${COMM[2]:-} ]]; then
    # one more arm for the triage: the host-driven solver loops, NCCL's own log up
    NCCL_DEBUG=INFO step check_np2_nccl_hostloops "$CHECK_TIMEOUT" --case "$CASES/${CASE_PREFIX}2" --np 2 --comm nccl --check --no-cond-graph
fi

# 3. parity with converged solves
PY_OK=0; python3 -c 'import numpy' 2>/dev/null && PY_OK=1
for n in $NPS; do
    [[ -n ${COMM[$n]:-} ]] || continue
    step parity_np$n "$PARITY_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm "${COMM[$n]}" --parity "$PARITY_STEPS" || continue
    PCASE[$n]=$CASES/$CASE_PREFIX$n.parity_${COMM[$n]}
    ((n != REF)) || continue
    if ((PY_OK)) && [[ -n ${PCASE[$REF]:-} ]]; then
        python3 "$SRC/tools/mgpu_oracle/compare_fields.py" "${PCASE[$REF]}" "${PCASE[$n]}" --times last \
            --fields U,p_rgh,phi,alpha.water > "$RES/parity_np$n.md" 2>&1
        say "parity np $REF vs np $n:"; grep -E 'internal' "$RES/parity_np$n.md" | tee -a "$RES/session.log"
        note "parity_np$n $(grep -E '\| *U *\| *internal' "$RES/parity_np$n.md" | awk -F'|' '{print "U maxrel" $8}') $(grep -E '\| *p_rgh *\| *internal' "$RES/parity_np$n.md" | awk -F'|' '{print "p_rgh maxrel" $8}') $(grep -E '\| *phi *\| *internal' "$RES/parity_np$n.md" | awk -F'|' '{print "phi maxrel" $8}')"
    else
        say "parity np $n: no comparison here (numpy $PY_OK, np $REF case ${PCASE[$REF]:-missing}); the cases are kept"
    fi
done

# 4. timings on the case's own tolerances; the placement tool's map where it gives one
for n in $NPS; do
    [[ -n ${COMM[$n]:-} ]] || continue
    if ((n > 1)) && [[ ${COMM[$n]} == nccl ]]; then
        step time_np$n "$TIMING_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm nccl --topo && continue
        # a failed placement probe leaves the case unused; a failed run does not, and then this refuses to start
        ALLOW_CASE_REUSE=0 step time_np${n}_notopo "$TIMING_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm nccl
    else
        step time_np$n "$TIMING_TIMEOUT" --case "$CASES/$CASE_PREFIX$n" --np "$n" --comm "${COMM[$n]}"
    fi
done
say "done; summary:"
tee -a "$RES/session.log" < "$RES/summary.txt"
