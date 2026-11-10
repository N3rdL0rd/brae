# Rosie G4 multi-GPU run kit

Build `brae_interFoam` for **sm_70 / CUDA 12.9 / NCCL** on `dh-node3`, then run the
initialized 2,483,933-cell G4 tank on `dh-dgx1-3` at np = 1, 2, 4 and 8.
The lead performs all cluster operations. **One Slurm job/step at a time.** These
scripts never submit a job, contact a cluster, or modify the source sketch.

## Measured kit status (laptop, 2026-11-05)

- All four G4 decompositions and `.tar.zst` archives were generated with OpenFOAM
  v2606/scotch and their reference-cell ownership checked. The four compressed
  archives pass `zstd -t`. The source sketch and the main clone were not changed.
- The current runner passed two-step MPI-host runs at np = 1, 2 and 4 on the small
  55,296-cell oracle tank (ranks sharing the laptop RTX 5090). Per-rank logs,
  `ALL PHASES`, alpha range, final deltaT, binary field output and 1 s GPU samples
  were parsed and checked. `BRAE_COND_GRAPH=0` was also exercised at np = 4.
- This is **not** G4 correctness or V100 scaling evidence. No DGX solver run has
  been performed by this kit. The laptop binary used for runner tests was the
  read-only RunPath build; `build_node3.sh` deliberately takes the final merged
  source via `SRC` instead of baking in a particular worktree.
- `bash -n`, ShellCheck 0.11.0 and Python byte-compilation are run on the scripts.
  The conda/sm_70 build itself must run on `dh-node3`: the laptop is not that
  toolchain or architecture. Actual build memory/time are still to be measured.

## What is staged

`stage_case.sh` uses `case/0`, **not** `0.orig`: the former has a nonuniform binary
initialized `alpha.water`; the latter is uniform dry alpha. It creates
`g4_np{1,2,4,8}` under `/mnt/storage/brae-scratch/RosieKit/cases` and, with
`--tar`, one self-contained `g4_npN/` per `g4_npN.tar.zst`.

The default timing preset is **60 fixed steps, `deltaT = 1e-4`, binary output only
at the last step**. `endTime` is `(60 + 0.25)*deltaT`, so brae's half-step clock
condition and its `int(4*endTime/deltaT + .5)` iteration budget both admit exactly
60 steps. Actual simulated end time is 0.006. The runner rejects an early exit.
Use `--dt` to change the fixed step or `--adaptive` to retain Courant-limited dt;
adaptive `--steps` is a simulated-time budget, **not** an exact iteration count.

`pulseStop` is retained with `enabled false` in the fixed timing preset so it
cannot truncate the requested 60 steps. Its input `speedLive` FO remains enabled,
so global per-step liquid momentum is still collected. All other supported FOs
retain their original settings. `DISABLE_PULSE_STOP=0 stage_case.sh` restores the
early-stop physics preset; a genuinely early adaptive run then fails the expected
step check rather than being mislabeled as a complete timing measurement.

The original `pRefPoint (0 0 0.2)` lies on a mesh vertex; OpenFOAM rejects it under
decomposition as found on multiple domains. The scripts use OpenFOAM's
`writeCellCentres` to choose global cell **140530**, whose actual centre is:

```
pRefPoint (-0.0020019063045386958 -0.002001905456696628 0.20006092821695245);
```

This is **inside exactly one cell**, not an arbitrary perturbation. The same
cell centre is used at every np and checked through `cellProcAddressing`.
It changes the reference location relative to the old serial sketch; this is
intentional and must be stated when comparing with old results.

All original FOs remain. The `interface` (`surfaces`) FO has `enabled false`
because brae does not implement that type; it would run in an OpenFOAM reference
if left enabled. Global vol/surface integrals, expression fields, fixed-location
probes, `fieldMinMax` and `runTimeControl` have distributed implementations in
current brae. In particular, probe ownership, global reductions, per-rank state
and master-only files are supported; **unported FO types are ignored by brae**,
hence the explicit disabling here. This is not a claim that the G4 zero-degree
contact-angle result is already correct: that remains an open numerical issue.

Generated cell centres, `0.orig`, STL/triSurface inputs, old output, and the
redundant global mesh/`cellDist` in decomposed cases are excluded. Each parallel
case contains only `processorN` meshes and initialized fields plus shared
`constant` dictionaries. Binary mesh/fields keep transfer size reasonable.

### Exact final archive sizes (60-step preset)

| np | archive bytes | MiB | uncompressed staged bytes | MiB |
|---:|---:|---:|---:|---:|
| 1 | 288,505,846 | 275.14 | 562,305,818 | 536.26 |
| 2 | 212,393,056 | 202.53 | 438,361,632 | 418.05 |
| 4 | 212,220,015 | 202.37 | 437,891,768 | 417.60 |
| 8 | 208,711,844 | 199.02 | 425,705,531 | 405.98 |

Total upload: **921,830,761 bytes = 879.06 MiB (0.922 GB)**. Decomposed cases
are not byte-for-byte invariant across repeated Scotch calls; use these supplied
archives for all runs rather than mixing newly decomposed copies.

## 0. Laptop: regenerate only if necessary

```bash
cd ~/clones/brae-worktrees/RosieKit
# Do not use sudo; no output goes into the sketch or the home directory.
tools/rosie/stage_case.sh --np '1 2 4 8' --steps 60 --dt 1e-4 --tar
# Optional quicker test-case staging:
CASE=/path/to/initialized/case CASE_LABEL=tank \
  OUT=/mnt/storage/brae-scratch/RosieKit/test_cases \
  tools/rosie/stage_case.sh --np '2 4' --steps 2 --dt 1e-4 --tar
```

`OF_ENV` defaults to the laptop's OpenFOAM v2606 `of_env.sh`; override it when
needed. `CASE`, `OUT`, `TAR_DIR`, `ZST_LEVEL` are supported environment overrides.
The script refuses to overwrite an existing staged case. Move an older generated
case aside first; never run it in the source sketch.

## 1. Upload source, runtime and cases

Use the existing `~/brae-mgpu` conda installation recipe in `docs/multi-gpu.md`.
`compat/12-9` is a local NVIDIA proprietary runtime (about 300 MB), **not** part
of the kit; do not redistribute it. NCCL (wheel >= 2.28) and `cuda12rt` are
required. Put the final merged brae checkout at `~/brae-mgpu/src`.

```bash
# Laptop: substitute Rosie's actual login host for <rosie>.
scp /mnt/storage/brae-scratch/RosieKit/cases/g4_np{1,2,4,8}.tar.zst \
  <rosie>:~/brae-mgpu/
# On Rosie login (no compute allocation):
mkdir -p ~/brae-mgpu/cases
for n in 1 2 4 8; do
  tar -C ~/brae-mgpu/cases -xf ~/brae-mgpu/g4_np${n}.tar.zst
  zstd -q -t ~/brae-mgpu/g4_np${n}.tar.zst
 done
```

The build script is already under `src/tools/rosie`. Keep that directory intact:
`run_dgx.sh` calls `rank_wrapper.sh` and `summarize_run.py` beside itself.

## 2. One batch job: build on dh-node3

```bash
sbatch -p batch --nodelist=dh-node3 -N1 -n1 -c 32 --mem=64G -t 01:00:00 \
  --job-name=brae70 --output=$HOME/brae-mgpu/build70c.job.log \
  --wrap='SRC=$HOME/brae-mgpu/src NJ=16 $HOME/brae-mgpu/src/tools/rosie/build_node3.sh'
```

Wait for this job to finish and inspect `build70c.job.log` before doing anything
else. `NJ=16` is deliberate; increasing it raises concurrent nvcc memory use.
**Use `-c 32 --mem=64G` rather than the minimum 16 cores/32 GB** for a full solver
build: the laptop's reported approximately 2 min/24-core build is a useful lower
bound, not a measured V100-toolchain estimate. The first cold conda+full solver
build is budgeted **10–25 min** [INFERENCE], and the one-hour allocation leaves
ample headroom; incremental builds should be **1–5 min** [INFERENCE]. Do not run
16 concurrent nvcc jobs if the measured job approaches 32 GB. The build is
incremental; use `--reconfigure` only after a toolchain/configuration change.
No `rm -rf build70c` is needed for a rerun.

Required output:

- `brae_interFoam`, `test_proc_comm`, `test_proc_comm_graph`,
  `test_proc_topology`, `proc_topology_tool` in `~/brae-mgpu/build70c`.
- `build70c/build.log`, the optional NCCL spikes, and copied `libcudart`,
  `libnvrtc`, `libnvrtc-builtins`, `libnvJitLink` in `cuda12rt`.
- Every binary's newest GLIBC symbol <= 2.31, empty ldd `not found` lists, and
  **no compat directory in rpath**. Compat is only for runtime `LD_LIBRARY_PATH`.
- Save `build70c.job.log` and `build70c/build.log` to copy back.

## 3. One exclusive DGX allocation, sequential steps

Acquire one allocation as documented in `docs/multi-gpu.md`, for example:

```bash
sbatch -p dgx --nodelist=dh-dgx1-3 --gres=gpu:v100:8 --exclusive \
  --time=04:00:00 --job-name=g4brae --wrap='sleep 14400'
# Set J to the returned job id. Run one of the following steps at a time.
srun --jobid=$J --overlap -N1 -n1 --gres=gpu:v100:8 --cpu-bind=none \
  --time=04:00:00 --pty bash
# In that shell (the script itself unsets SLURM_ variables for mpirun):
cd $HOME/brae-mgpu/src
export BRAE_ROOT=$HOME/brae-mgpu/build70c
```

The runtime recipe is `compat/12-9:env/lib:nccl/nvidia/nccl/lib:cuda12rt:
/usr/lib/x86_64-linux-gnu`, `OPAL_PREFIX=$HOME/brae-mgpu/env`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `OMPI_MCA_ras=^slurm`, `--host localhost:N`,
`--bind-to none`, and no per-rank `CUDA_VISIBLE_DEVICES`. `run_dgx.sh` implements
this, with `OMPI_MCA_plm=isolated`, `btl=self,vader`, `pml=ob1` as in the tested
cluster test recipe. It requires direct `mpirun`, **not** `srun` as the solver
launcher. One `srun` shell is fine; do not nest a second `srun` per rank.

### 3a. Transport/topology tests before the tank

Run the test binaries with the same runtime environment and mpirun rules, using
`tools/proc_comm_nccl/run_proc_comm_tests.sh` or the explicit commands in its
README. NCCL np = 2, 4 and 8, graph tests, topology probe, and `test_proc_topology`
should pass first. Copy `proc_comm_results_*/` back. A transport failure is not a
solver timing result.

### 3b. Two-step tank smoke checks, one per np

Use the **same** uploaded 60-step cases. `--check` creates a temporary sibling
case, hard-links the rank-local read-only mesh/fields, copies only `system/` and
shared `constant/` dictionaries, and substitutes a two-step fixed timing/write
preset with Python. It removes the temporary case on exit and never changes the
input case. No OpenFOAM utilities or second mesh upload are needed on the DGX.

```bash
for n in 1 2 4 8; do
  tools/rosie/run_dgx.sh --case $HOME/brae-mgpu/cases/g4_np$n --np $n \
    --comm nccl --topo --check --timeout 1800 \
    --results $HOME/brae-mgpu/results/check_np$n
  # Inspect status before proceeding to the next n.
 done
```

For np=1 the serial case runs without `-parallel`. The summary requires exactly
two steps. Expected cost is **2–15 min per np** [INFERENCE], dominated by the
large binary mesh read and first AMG setup. A 60-minute job for all checks is
prudent.

### 3c. Full 60-step NCCL timing runs

Use a **fresh, unmodified extraction per run**. Each invocation writes fields,
FO state and `postProcessing` into its case. The runner rejects used cases and
existing result directories unless `ALLOW_CASE_REUSE=1` is explicitly supplied
for a deliberate restart (not valid as a fresh scaling measurement).

```bash
for n in 1 2 4 8; do
  tools/rosie/run_dgx.sh --case $HOME/brae-mgpu/cases/g4_np$n --np $n \
    --comm nccl --topo --timeout 3600 \
    --results $HOME/brae-mgpu/results/nccl_np$n
 done
```

`--topo` calls exactly
`proc_topology_tool --topology probe --np N --case DIR --map-only`. The selected
`BRAE_GPU_MAP` is saved in `run.env` and `system.txt`. On a DGX-1, this can avoid
unnecessary cross-socket links. Its `topology.log` and `topology-map.txt` are
saved beside the other results. `--gpu-map` is the manual override. Mapping is a
placement experiment, not a correctness guarantee; inspect the tool output.
For the default automatic graph decision omit graph flags. Add
`--no-cond-graph` for a host-driven comparison (`BRAE_COND_GRAPH=0`) and use a
separate fresh case/result directory. `--cond-graph` (`=1`) is only for a known
compatible driver; compat 12.9 is first in the runtime library path.

Before comparing backend variants, run an MPI-host baseline with `--comm mpi`.
On physical V100s ranks still bind to separate GPUs; laptop MPI oversubscription
is for smoke testing only and is not performance evidence.

**Timing estimates** [INFERENCE]: from the old single-V100 0.488 s/step reference,
60 solver steps cost about 29 s at np=1. Until measured, budget **5–20 min per
run including large-case startup/writes**, and **30–90 min for the full NCCL sweep**.
Distributed AMG-PCG may need 1.4–4x more iterations; this can remove ideal speedup.
`BRAE_INTER_PHASE_TIME=1` synchronizes at phase boundaries and adds instrumentation
overhead: all scaling comparisons in this kit use the same setting. Record an
uninstrumented repeat separately if absolute production timing is required.

### 3d. Output and failure triage

Each result directory contains:

| file | meaning |
|---|---|
| `ranks/rank.N.log` | direct stdout+stderr of each solver rank |
| `mpirun.log` | launcher diagnostics, kept separate |
| `summary.env` | parsed steps, slowest-rank `ALL PHASES` ms/step, total wall, startup+output, alpha min/max, final deltaT/time |
| `status.txt` | launcher/summary status; `rc=0` is required |
| `nvidia-smi.csv` | 1 s GPU utilization/memory/power/clocks/temperature samples |
| `system.txt` | driver/GPU UUID/PCI id/topology/NVLink, binary hash and ldd |
| `command.txt`, `run.env` | exact launch command, map, backend and expected step count |

`ms_per_step` is the **slowest rank's `ALL PHASES`**, the startup-excluded loop
measurement; it is not `wall_s/steps`. `startup_output_s = wall_s - steps*ms/step/1000`
can include mesh load, first-step AMG work outside the phase timer, and final I/O.
Do not report it as pure initialization time. All ranks must have equal step count,
final dt/time and alpha extrema, and fixed-dt runs must have the requested count.
A launcher return of zero is not sufficient if those checks fail. A timeout leaves
logs in place and returns nonzero. `timeout` is a hard cap, including final writes.

The `End` alpha range is a smoke/correctness indicator, **not** a proof of
physical correctness. Check `alpha=[min,max]`, unexpected early `pulseStop`, NaNs,
large divergence, per-rank AMG iterations, and G4 zero-degree contact-angle issues.

### 3e. The whole allocation as one job

`session_dgx.sh` runs sections 3b to 3c in order as the batch job itself, so the node is released when it ends: the
driver and GPU count, fresh extractions, the two-step checks (np 1, np 2 on MPI-host, np 2/4/8 on NCCL with MPI-host
as the fallback), the parity step below, then the 60-step timings. A step runs only if what it needs passed; every
run's logs and a `summary.txt` land in `results/session_<tag>/`.

```bash
sbatch -p dgx -N1 --gres=gpu:v100:8 --exclusive -t 03:00:00 --job-name=g4brae \
  --output=$HOME/brae-mgpu/results/session.%j.log \
  --wrap='$HOME/brae-mgpu/src/tools/rosie/session_dgx.sh'
```

**Parity** is `run_dgx.sh --parity N`: N fixed steps with every linear solve converged (tolerance 1e-12, relTol 0),
the case copy kept as `<case>.parity_<comm>`, and each np compared with np 1 by `compare_fields.py`. Compare
converged runs only: at the case's own tolerances OpenFOAM's np 1, 2 and 4 runs of a G4-like case differ from each
other by 7.8e-2 to 3.1e-1 of U (docs/multi-gpu.md), so a loose comparison says nothing. Laptop rehearsal on the staged
tank, np 2 against np 4, three steps: U 2.3e-8.

## 4. Correctness collection

For each successful np, retain `postProcessing/` (rank-0 global FO files) and the
last binary `processorN/<time>` output (or reconstruct it with OpenFOAM v2606).
Do not compare against old serial output made with the ambiguous axis reference;
use the same new pRefPoint and dt in the serial control.

Use the existing readers in `tools/mgpu_oracle` (from the final source tree):

```bash
python3 tools/mgpu_oracle/compare_fields.py SERIAL_CASE PARALLEL_CASE \
  --times all --fields alpha.water,U,p_rgh,p,phi
python3 tools/mgpu_oracle/compare_fo.py SERIAL_CASE/postProcessing \
  PARALLEL_CASE/postProcessing --last-row
```

They map per-rank fields through `cellProcAddressing`/`faceProcAddressing` and
compare parsed values, not text formatting. Record actual differences; do not
claim bitwise identity for different MPI reduction orders or NCCL graph modes.
For a stronger reference run OpenFOAM v2606 on the same staged case, then compare
its reconstructed output. `reconstructPar -newTimes` needs the rank-local meshes
and output fields retained until this is done.

## 5. Copy back and release

```bash
# On the DGX, after all sequential steps have completed:
tar -C $HOME/brae-mgpu -cf - results | zstd -3 -T0 -o $HOME/brae-mgpu/results.tar.zst
# Laptop:
scp <rosie>:~/brae-mgpu/results.tar.zst /mnt/storage/brae-scratch/RosieKit/
scp <rosie>:~/brae-mgpu/build70c.job.log /mnt/storage/brae-scratch/RosieKit/
scp <rosie>:~/brae-mgpu/build70c/build.log /mnt/storage/brae-scratch/RosieKit/
# Only after every intended run has finished:
scancel $J
```

On the laptop, `tools/rosie/collect_results.py results/nccl_np{1,2,4,8}` emits a
TSV scaling table. It refuses directories without a `COMPLETE` summary. Keep the
result archives and case archives together so a rerun cannot mix decompositions.

## Laptop-only runner test (optional)

```bash
BRAE_MODE=laptop BRAE_ROOT=/mnt/storage/brae-scratch/RunPath/build \
  MPIEXEC=/usr/lib64/openmpi/bin/mpirun \
  BRAE_RESULTS=/mnt/storage/brae-scratch/RosieKit/results/tank_np2 \
  tools/rosie/run_dgx.sh --case /mnt/storage/brae-scratch/RosieKit/test_cases/tank_np2 \
  --np 2 --comm mpi --check --timeout 240
```

Laptop mode does not use compat or `plm=isolated` (Fedora OpenMPI 5 lacks it).
It explicitly enables oversubscription. NCCL on one shared GPU additionally needs
`BRAE_NCCL_ALLOW_SHARED_GPU=1`, which laptop mode sets for `--comm nccl`; this is a
slow bring-up option, not a scaling measurement. No full G4 np=8 laptop run is
needed for this kit and was not used as validation.
