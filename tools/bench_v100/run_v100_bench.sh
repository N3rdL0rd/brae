#!/bin/bash
# Benchmark brae interFoam on ONE GPU (a V100): the G4 tank case, endTime 0.008 (89 steps), three runs alternating
# the OLD build (git main, on CUDA 12 the device WHILE solvers are off: host loops) and the NEW build.
#   ./run_v100_bench.sh <sketch dir> <old brae_interFoam> <new brae_interFoam> [outdir=./v100_bench_<host>_<date>]
#   sketch dir = ~/code/sketches/g4_tank_refill_interfoam (needs run_brae.sh, compare_reference.sh, case/, reference/)
# Env: GPU=0 (index, via CUDA_VISIBLE_DEVICES)  RUNS=3  END=0.008  SCRATCH=<outdir>/work (case copies, ~1 GB each: DELETED
# after the md5 check unless KEEP=1; set SCRATCH to a disk with space, not a small tmpfs)
# Results: table on stdout, same in <outdir>/summary.txt; logs, clocks and topology in <outdir>/. No root.
set -u
SKETCH=${1:?usage: $0 <sketch dir> <old binary> <new binary> [outdir]}
OLD=${2:?old binary}; NEW=${3:?new binary}
OUT=${4:-$PWD/v100_bench_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}
RUNS=${RUNS:-3}; END=${END:-0.008}; export CUDA_VISIBLE_DEVICES=${GPU:-0}
SCRATCH=${SCRATCH:-$OUT/work}
for f in "$SKETCH/run_brae.sh" "$SKETCH/compare_reference.sh" "$OLD" "$NEW"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
mkdir -p "$OUT" "$SCRATCH"; OUT=$(cd "$OUT" && pwd)

{ date; hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi; nvidia-smi topo -m
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE 2>&1
  echo "--- md5 of the binaries"; md5sum "$OLD" "$NEW"; } > "$OUT/system.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,clocks.max.mem,persistence_mode --format=csv -i "${CUDA_VISIBLE_DEVICES%%,*}" | tee "$OUT/gpu.csv"
# 1 Hz clock/power/throttle trace over the whole benchmark
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv -l 1 -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$OUT/clocks_trace.csv" 2>/dev/null &
TRACE=$!
trap 'kill $TRACE 2>/dev/null' EXIT

phase() { grep -a -E "^ +$1 " "$2" | head -n 1 | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/'; }
printf '%-4s %-4s %12s %14s %12s %10s  %s\n' run bin "ALL PHASES" "pressure ms" "alpha ms" "exit" "End line" > "$OUT/table.txt"
declare -A SUM
i=0
for r in $(seq 1 "$RUNS"); do
  for tag in old new; do
    BIN=$OLD; [ $tag = new ] && BIN=$NEW
    W=$SCRATCH/${tag}_$r
    echo "=== run $r/$RUNS $tag: $BIN"
    ( cd "$SKETCH" && ./run_brae.sh "$END" "$W" "$BIN" ) > "$OUT/run_${tag}_$r.out" 2>&1
    rc=$?
    cp "$W/log.brae" "$OUT/log_${tag}_$r.brae" 2>/dev/null
    all=$(grep -a "ALL PHASES" "$W/log.brae" 2>/dev/null | head -n 1 | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/')
    pr=$(phase "5 pressure correctors" "$W/log.brae"); [ -n "$pr" ] || pr=$(grep -a "pressure correctors" "$W/log.brae" | head -n 1 | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/')
    al=$(grep -a "1 alpha step" "$W/log.brae" 2>/dev/null | head -n 1 | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/')
    endl=$(grep -a '^End' "$W/log.brae" 2>/dev/null | cut -c1-100)
    printf '%-4s %-4s %12s %14s %12s %10s  %s\n' "$r" "$tag" "${all:-?}" "${pr:-?}" "${al:-?}" "$rc" "$endl" | tee -a "$OUT/table.txt"
    # identity vs the repository's reference (informational: V100 may differ from the RTX 5090 reference in the last bits)
    ( cd "$SKETCH" && ./compare_reference.sh "$W" ) > "$OUT/compare_${tag}_$r.txt" 2>&1
    echo "     compare_reference: $(tail -n 1 "$OUT/compare_${tag}_$r.txt")"
    # md5 of the final fields, for old vs new
    ( cd "$W" && md5sum "$END"/{U,p,p_rgh,phi,alpha.water,alphaPhi0.water} 2>/dev/null ) > "$OUT/md5_${tag}_$r.txt"
    [ "${KEEP:-0}" = 1 ] || rm -rf "$W"
  done
done
kill $TRACE 2>/dev/null; trap - EXIT

{
echo; echo "== $(sed -n 2p "$OUT/gpu.csv")"
cat "$OUT/table.txt"
echo
for tag in old new; do
  awk -v t=$tag '$2==t && $3!="?" {n++; s+=$3; p+=$4; if(min==""||$3<min)min=$3} END{ if(n) printf "%-4s mean ALL PHASES %.1f ms/step (best %.1f)  mean pressure %.1f ms/step over %d runs\n", t, s/n, min, p/n, n}' "$OUT/table.txt"
done
echo
echo "identity vs reference/ (informational only; V100 results may differ in the last bits):"
for f in "$OUT"/compare_*.txt; do echo "  $(basename "$f" .txt): $(tail -n 1 "$f")"; done
echo "old vs new final fields (the two builds must agree with each other):"
if cmp -s "$OUT/md5_old_1.txt" "$OUT/md5_new_1.txt"; then echo "  run 1: old == new, byte identical"; else echo "  run 1: old DIFFERS from new (see md5_old_1.txt / md5_new_1.txt)"; fi
for tag in old new; do for r in $(seq 2 "$RUNS"); do cmp -s "$OUT/md5_${tag}_1.txt" "$OUT/md5_${tag}_$r.txt" || echo "  $tag run $r differs from $tag run 1 (non-deterministic?)"; done; done
echo "clocks: $OUT/clocks_trace.csv (sm clock / throttle reasons over the run), $OUT/system.txt (nvidia-smi -q clocks, topo -m)"
} | tee "$OUT/summary.txt"
rmdir "$SCRATCH" 2>/dev/null
echo "send back: tar czf $(basename "$OUT").tgz -C $(dirname "$OUT") $(basename "$OUT")"
