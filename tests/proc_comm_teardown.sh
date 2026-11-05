#!/bin/bash
# Repeated np=8 process teardown: the abort side channel must have exactly one live
# nonblocking allgather, and communicator destruction must leave no request behind.
# A nonzero rank (including glibc's heap-integrity abort) fails the whole launch.
set -uo pipefail
TEST=${1:?usage: $0 <test_proc_comm_teardown>}
LAUNCHES=${LAUNCHES:-20}
MPIEXEC=${MPIEXEC:-mpirun}
OUT=$(mktemp -d /tmp/proc_comm_teardown.XXXXXX)
trap '[ -n "${KEEP_OUT:-}" ] || rm -rf "$OUT"' EXIT
export MALLOC_CHECK_=3
for i in $(seq 1 "$LAUNCHES"); do
    export MALLOC_PERTURB_=$((40 + i))
    log=$OUT/$(printf '%02d' "$i").log
    mode=
    if ((i % 2 == 0)); then mode=serialized; fi
    timeout 180 "$MPIEXEC" --oversubscribe -np 8 \
        -x MALLOC_CHECK_ -x MALLOC_PERTURB_ "$TEST" $mode >"$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        cat "$log"
        echo "FAIL: np=8 teardown launch $i/$LAUNCHES exited $rc; log: $log" >&2
        KEEP_OUT=1
        exit 1
    fi
    rm -f "$log"
done
echo "PASS: $LAUNCHES consecutive np=8 teardown launches"
