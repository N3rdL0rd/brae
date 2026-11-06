#!/usr/bin/env bash
# Run one staged brae_interFoam measurement inside ONE srun step on dh-dgx1-3.
#   tools/rosie/run_dgx.sh --case $HOME/brae-mgpu/cases/g4_np2 --np 2 --topo
#   tools/rosie/run_dgx.sh --case ... --np 8 --topo --no-cond-graph --timeout 3600
# Laptop smoke test (ranks share the RTX 5090; NOT a performance measurement):
#   BRAE_MODE=laptop BRAE_ROOT=/mnt/storage/brae-scratch/RunPath/build \
#     MPIEXEC=/usr/lib64/openmpi/bin/mpirun \
#     LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/usr/local/cuda/lib64 \
#     tools/rosie/run_dgx.sh --case .../tank_np2 --np 2 --comm mpi --check
# One staged case is one run: outputs and functionObject state change it. A
# repeat measurement must use a fresh extraction or a filesystem copy.
set -Eeuo pipefail
umask 022

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BRAE_ROOT=${BRAE_ROOT:-${BUILD:-}}
MPIEXEC=${MPIEXEC:-}
CASE=
NP=
RESULTS=
BACKEND=nccl
CHECK=0
TOPO=0
GPU_MAP=${BRAE_GPU_MAP:-}
COND_GRAPH=${BRAE_COND_GRAPH:-}
TIMEOUT_SECONDS=${TIMEOUT:-1800}
GPU_SAMPLE_SECONDS=${GPU_SAMPLE_SECONDS:-1}
VERIFY_LIBS=1
ALLOW_CASE_REUSE=${ALLOW_CASE_REUSE:-0}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}

usage() {
    cat <<'EOF'
Usage: run_dgx.sh --case DIR --np N [options]
  --comm nccl|mpi      communication backend (default nccl)
  --check             copy the case's dictionaries and run exactly two fixed steps
  --topo              obtain BRAE_GPU_MAP with proc_topology_tool --topology probe
  --gpu-map 0,3,...   explicit comma-separated node-local rank -> GPU map
  --no-cond-graph     force BRAE_COND_GRAPH=0 (default: unset/driver decision)
  --cond-graph        force BRAE_COND_GRAPH=1 (only with a compatible driver)
  --timeout SECONDS   hard timeout, including mesh load and final write (1800)
  --results DIR       result directory (default BRAE_RESULTS or ROOT/results/run_*)
  --no-lib-check      skip ldd check (not recommended)
Environment: BRAE_MODE=dgx|laptop (default dgx), BRAE_ROOT (build directory),
  BRAE_BINARY, TOPOLOGY_TOOL, MPIEXEC, NVIDIA_SMI, BRAE_RESULTS,
  ALLOW_CASE_REUSE=1 (acknowledge overwriting/restarting an existing run).
DGX runtime paths use ROOT=$HOME/brae-mgpu: compat/12-9, env/lib,
nccl/nvidia/nccl/lib and cuda12rt in that order on LD_LIBRARY_PATH. Compat is
not a link-time path. All GPUs remain visible to every rank; rank binding is
performed by brae through BRAE_GPU_MAP (or rank i -> GPU i without a map).
--check never edits the input case: it copies system/, substitutes only the
timing/write controls and retains all function objects. The two-step run is a
smoke test, not a scaling measurement; no OpenFOAM install is needed on the DGX.
Adaptive input is not truncated.
EOF
}
while (($#)); do
    case "$1" in
        --case) CASE=$2; shift ;;
        --np) NP=$2; shift ;;
        --comm) BACKEND=$2; shift ;;
        --check) CHECK=1 ;;
        --topo) TOPO=1 ;;
        --gpu-map) GPU_MAP=$2; TOPO=0; shift ;;
        --no-cond-graph) COND_GRAPH=0 ;;
        --cond-graph) COND_GRAPH=1 ;;
        --timeout) TIMEOUT_SECONDS=$2; shift ;;
        --results) RESULTS=$2; shift ;;
        --no-lib-check) VERIFY_LIBS=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
BRAE_MODE=${BRAE_MODE:-dgx}
[[ $BRAE_MODE == dgx || $BRAE_MODE == laptop ]] || { echo "BRAE_MODE must be dgx or laptop" >&2; exit 2; }
[[ $BACKEND == nccl || $BACKEND == mpi ]] || { echo "--comm must be nccl or mpi" >&2; exit 2; }
[[ $NP =~ ^[1-9][0-9]*$ ]] || { echo "--np must be a positive integer" >&2; exit 2; }
[[ $TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] || { echo "--timeout must be a positive integer" >&2; exit 2; }
[[ $GPU_SAMPLE_SECONDS =~ ^[1-9][0-9]*$ ]] || { echo "GPU_SAMPLE_SECONDS must be an integer" >&2; exit 2; }
[[ -d $CASE ]] || { echo "staged case does not exist: $CASE" >&2; exit 1; }
CASE=$(cd -- "$CASE" && pwd)
[[ -f $CASE/system/controlDict && -f $CASE/system/fvSolution ]] || {
    echo "$CASE is not an OpenFOAM case" >&2; exit 1;
}
if ((NP == 1)); then
    [[ -f $CASE/0/alpha.water && -d $CASE/constant/polyMesh ]] || {
        echo "np=1 needs the serial staged case with 0/ and constant/polyMesh" >&2; exit 1;
    }
else
    for ((rank = 0; rank < NP; ++rank)); do
        [[ -d $CASE/processor$rank/constant/polyMesh ]] || {
            echo "missing processor$rank/constant/polyMesh; stage with stage_case.sh --np $NP" >&2; exit 1;
        }
        [[ -f $CASE/processor$rank/0/alpha.water ]] || {
            echo "missing initialized processor$rank/0/alpha.water" >&2; exit 1;
        }
    done
fi
if [[ -z $BRAE_ROOT ]]; then
    if [[ $BRAE_MODE == laptop ]]; then
        BRAE_ROOT=${BRAE_BINARY%/*}
    else
        BRAE_ROOT=$HOME/brae-mgpu/build70c
    fi
fi
BRAE_BINARY=${BRAE_BINARY:-$BRAE_ROOT/brae_interFoam}
TOPOLOGY_TOOL=${TOPOLOGY_TOOL:-$BRAE_ROOT/proc_topology_tool}
[[ -x $BRAE_BINARY ]] || { echo "no executable brae_interFoam: $BRAE_BINARY" >&2; exit 1; }
((TOPO == 0)) || [[ -x $TOPOLOGY_TOOL ]] || { echo "--topo needs $TOPOLOGY_TOOL" >&2; exit 1; }
[[ -n $MPIEXEC ]] || MPIEXEC=$(command -v mpirun || true)
[[ -x $MPIEXEC ]] || { echo "MPIEXEC must be the mpirun absolute path" >&2; exit 1; }
for command_name in timeout python3 ldd readlink; do
    command -v "$command_name" >/dev/null || { echo "missing command $command_name" >&2; exit 1; }
done

# Parse the case's scalar time controls, rather than guessing a two-step cutoff
# from a Courant-limited dt. --check is deliberately conservative and explicit.
mapfile -t TIME_VALUES < <(python3 - "$CASE/system/controlDict" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for key in ("endTime", "deltaT", "adjustTimeStep"):
    match = re.search(r"(?m)^\s*" + key + r"\s+([^;]+);", text)
    if not match:
        raise SystemExit(f"missing controlDict entry: {key}")
    print(match.group(1).split("//")[0].strip())
PY
)
[[ ${#TIME_VALUES[@]} -eq 3 ]] || { echo "could not read time controls" >&2; exit 1; }
END_TIME=${TIME_VALUES[0]}
DELTA_T=${TIME_VALUES[1]}
ADJUST_DT=${TIME_VALUES[2]}
if ((CHECK)); then
    # Rank-local mesh/fields are hard-linked; shared dictionaries are copied
    # because /tmp and the case can be on different filesystems. Only system/
    # timing controls change; the input case remains pristine.
    CASE_INPUT=$CASE
    CASE=$(mktemp -d "$(dirname "$CASE")/.brae-check-${NP}.XXXXXX")
    # shellcheck disable=SC2329 # called by the combined EXIT trap
    cleanup_check_case() { rm -rf "$CASE"; }
    trap cleanup_check_case EXIT
    for item in "$CASE_INPUT"/*; do
        name=${item##*/}
        [[ $name == system || $name == constant || $name == postProcessing ]] && continue
        cp -al "$item" "$CASE/$name"
    done
    cp -a "$CASE_INPUT/system" "$CASE/system"
    cp -a "$CASE_INPUT/constant" "$CASE/constant"
    python3 - "$CASE/system/controlDict" "$DELTA_T" <<'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
end_time = format(2.25 * float(sys.argv[2]), ".17g")
values = {"deltaT": sys.argv[2], "adjustTimeStep": "no", "writeControl": "timeStep",
          "writeInterval": "2", "purgeWrite": "1", "endTime": end_time}
for key, value in values.items():
    pattern = r"(?m)^(\s*" + re.escape(key) + r"\s+)[^;]+;"
    text, count = re.subn(pattern, r"\g<1>" + value + ";", text, count=1)
    if count != 1:
        raise SystemExit(f"{path}: missing timing entry {key}")
path.write_text(text)
PY
    END_TIME=$(python3 -c 'import sys; print(format(2.25*float(sys.argv[1]), ".17g"))' "$DELTA_T")
fi
EXPECTED_STEPS=$(python3 - "$END_TIME" "$DELTA_T" "$ADJUST_DT" <<'PY'
import sys
end, dt, adjustable = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3].lower()
if dt <= 0:
    raise SystemExit("deltaT must be positive")
if adjustable not in ("yes", "true", "on", "1"):
    ratio = end / dt
    nearest = round(ratio - 0.25)
    if nearest < 1 or abs(ratio - (nearest + 0.25)) > 1e-8 * max(1.0, ratio):
        raise SystemExit("fixed timing case must have endTime/dt = N + 0.25 (stage_case.sh preset)")
    # Mirror the solver's integer budget int(4*ratio + .5).
    print(nearest)
else:
    print(0)
PY
)

# No Slurm allocation variables may leak into mpirun's resource assignment.
# srun supplies one process (the launcher); the launcher creates NP local ranks.
if [[ $BRAE_MODE == dgx ]]; then
    while IFS= read -r name; do unset "$name"; done < <(env | grep -E '^SLURM_' | cut -d= -f1)
    export OMPI_MCA_ras='^slurm'
    export OMPI_MCA_plm=isolated OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1
    ROOT=${ROSIE_ROOT:-$HOME/brae-mgpu}
    export PATH="$ROOT/env/bin:$PATH" OPAL_PREFIX="$ROOT/env"
    # COMPAT_DIR="" runs without the forward-compatibility libraries (nodes whose kernel driver is older than 535, e.g. dh-dgx1-2 with 525, refuse them: error 803)
    COMPAT_DIR=${COMPAT_DIR-$ROOT/compat/12-9}
    # drivers older than 530 default to eager module loading, which fails on the whole-solver fatbin
    export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
    export LD_LIBRARY_PATH="${COMPAT_DIR:+$COMPAT_DIR:}$ROOT/env/lib:$ROOT/nccl/nvidia/nccl/lib:$ROOT/cuda12rt:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    # Never inherit a job's Slurm context in the laptop regression either.
    while IFS= read -r name; do unset "$name"; done < <(env | grep -E '^SLURM_' | cut -d= -f1)
    export OMPI_MCA_ras='^slurm'
    # Fedora's OpenMPI is intentionally not always on the default path.
    if [[ -x /usr/lib64/openmpi/bin/mpirun && $MPIEXEC == */openmpi/bin/mpirun ]]; then
        export PATH="/usr/lib64/openmpi/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    export OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export BRAE_COMM=$BACKEND BRAE_INTER_PHASE_TIME=1 BRAE_GS_HOST_SMOOTHER=${BRAE_GS_HOST_SMOOTHER:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if [[ -n $COND_GRAPH ]]; then export BRAE_COND_GRAPH=$COND_GRAPH; else unset BRAE_COND_GRAPH || true; fi

if ((VERIFY_LIBS)); then
    missing=$(ldd "$BRAE_BINARY" | grep -F 'not found' || true)
    [[ -z $missing ]] || { printf 'Unresolved brae_interFoam libraries:\n%s\n' "$missing" >&2; exit 1; }
fi

MPI_OPTIONS=(-np "$NP" --host "localhost:$NP" --bind-to none)
if [[ $BRAE_MODE == laptop ]]; then
    MPI_OPTIONS+=(--oversubscribe)
    if [[ $BACKEND == nccl ]]; then
        export BRAE_NCCL_ALLOW_SHARED_GPU=1 NCCL_MULTI_RANK_GPU_ENABLE=1
        echo "WARNING: laptop NCCL ranks share a GPU; bring-up only, not scaling evidence" >&2
    fi
fi

RESULTS=${RESULTS:-${BRAE_RESULTS:-$BRAE_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)_np${NP}_${BACKEND}}}
if [[ -e $RESULTS ]]; then echo "refusing to reuse result directory: $RESULTS" >&2; exit 1; fi
mkdir -p "$RESULTS/ranks"
RESULTS=$(cd -- "$RESULTS" && pwd)
if ((TOPO)); then
    [[ $BRAE_MODE == dgx ]] || { echo "--topo requires physical GPU topology; not valid on the shared laptop" >&2; exit 2; }
    [[ $BACKEND == nccl ]] || { echo "--topo is intended for --comm nccl" >&2; exit 2; }
    TOPOLOGY_LOG=$RESULTS/topology.log
    if ! "$TOPOLOGY_TOOL" --topology probe --np "$NP" --case "$CASE" --map-only > "$RESULTS/topology-map.txt" 2> "$TOPOLOGY_LOG"; then
        echo "topology probe failed; see $TOPOLOGY_LOG" >&2; exit 1
    fi
    GPU_MAP=$(<"$RESULTS/topology-map.txt")
fi
if [[ -n $GPU_MAP ]]; then
    [[ $GPU_MAP =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid GPU map: $GPU_MAP" >&2; exit 1; }
    IFS=, read -r -a GPU_IDS <<< "$GPU_MAP"
    [[ ${#GPU_IDS[@]} -eq NP ]] || { echo "GPU map has ${#GPU_IDS[@]} entries, expected $NP" >&2; exit 1; }
    export BRAE_GPU_MAP=$GPU_MAP
else
    unset BRAE_GPU_MAP || true
fi

command -v "$NVIDIA_SMI" >/dev/null || { echo "missing nvidia-smi: $NVIDIA_SMI" >&2; exit 1; }
if [[ $ALLOW_CASE_REUSE != 1 ]]; then
    output_root=$CASE/processor0
    ((NP > 1)) || output_root=$CASE
    if { compgen -G "$output_root/[0-9]*" | grep -Ev '/0$' >/dev/null || [[ -d $CASE/postProcessing ]]; }; then
        echo "case already has output; extract a fresh copy (or explicitly ALLOW_CASE_REUSE=1)" >&2; exit 1
    fi
fi
{
    date -u +%FT%TZ
    hostname
    printf 'case=%s np=%s backend=%s map=%s condGraph=%s mode=%s timeout=%s\n' \
        "$CASE" "$NP" "$BACKEND" "${GPU_MAP:-rank}" "${COND_GRAPH:-auto}" "$BRAE_MODE" "$TIMEOUT_SECONDS"
    "$NVIDIA_SMI" --query-gpu=index,pci.bus_id,name,uuid,driver_version,memory.total --format=csv,noheader
    "$NVIDIA_SMI" topo -m || true
    "$NVIDIA_SMI" nvlink -s || true
    printf 'binary=%s\n' "$BRAE_BINARY"
    sha256sum "$BRAE_BINARY"
    ldd "$BRAE_BINARY"
} > "$RESULTS/system.txt" 2>&1

nvidia_sampler() {
    "$NVIDIA_SMI" --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu \
        --format=csv,noheader,nounits -l "$GPU_SAMPLE_SECONDS" > "$RESULTS/nvidia-smi.csv" 2> "$RESULTS/nvidia-smi.err"
}
nvidia_sampler & sampler_pid=$!
# shellcheck disable=SC2329 # invoked indirectly by the EXIT trap
cleanup() {
    if kill -0 "$sampler_pid" 2>/dev/null; then kill "$sampler_pid" 2>/dev/null || true; fi
    wait "$sampler_pid" 2>/dev/null || true
}
trap 'cleanup; [[ ${CHECK:-0} != 1 ]] || cleanup_check_case' EXIT

# Direct local redirection is portable across OpenMPI 4/5. The launcher writes
# startup/job diagnostics separately to mpirun.log.
export ROSIE_RANK_LOG_DIR=$RESULTS/ranks
solver_options=(-case "$CASE" -device)
((NP == 1)) || solver_options+=(-parallel)
command_line=("$MPIEXEC" "${MPI_OPTIONS[@]}" "$HERE/rank_wrapper.sh" \
    "$BRAE_BINARY" "${solver_options[@]}")
START=$(python3 -c 'import time; print(time.time())')
set +e
timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" "${command_line[@]}" > "$RESULTS/mpirun.log" 2>&1
run_rc=$?
set -e
END=$(python3 -c 'import time; print(time.time())')
WALL_SECONDS=$(python3 - "$START" "$END" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
)
logs=()
for ((rank = 0; rank < NP; ++rank)); do
    log=$RESULTS/ranks/rank.$rank.log
    if [[ ! -s $log ]]; then
        echo "missing rank $rank log under $RESULTS/ranks (launcher rc=$run_rc)" >&2
        ((run_rc != 0)) || run_rc=1
        exit "$run_rc"
    fi
    logs+=("$log")
done
cat > "$RESULTS/run.env" <<EOF
CASE='$CASE'
NP='$NP'
BACKEND='$BACKEND'
GPU_MAP='$GPU_MAP'
COND_GRAPH='$COND_GRAPH'
WALL_SECONDS='$WALL_SECONDS'
TIMEOUT_SECONDS='$TIMEOUT_SECONDS'
EXPECTED_STEPS='$EXPECTED_STEPS'
EOF
summary=0
if ((run_rc == 0)); then
    if python3 "$HERE/summarize_run.py" "$RESULTS/run.env" "${logs[@]}" > "$RESULTS/summary.env"; then
        summary=1
    else
        echo "solver returned zero but logs are incomplete/inconsistent" >&2
        run_rc=1
    fi
else
    echo "run failed/timeout rc=$run_rc; retaining all logs in $RESULTS" >&2
fi
{
    printf 'rc=%d wall_s=%s steps_requested=%s timeout_s=%s\n' "$run_rc" "$WALL_SECONDS" "$CHECK" "$TIMEOUT_SECONDS"
    if ((summary)); then grep '^ROSIE_SUMMARY=' "$RESULTS/summary.env"; fi
} | tee "$RESULTS/status.txt"
if ((summary)); then
    # Print exactly the human-readable summary, not shell quoting.
    python3 - "$RESULTS/summary.env" <<'PY'
import ast, sys
for line in open(sys.argv[1]):
    if line.startswith("ROSIE_SUMMARY="):
        print(ast.literal_eval(line.split("=", 1)[1]))
PY
fi
exit "$run_rc"
