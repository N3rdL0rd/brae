# Multi-GPU `interFoam` (in progress)

One case across up to 8 GPUs of one node, one process per GPU, with OpenFOAM's own `-parallel` semantics: the mesh is
split by OpenFOAM's `decomposePar` (`processorN/` directories), processor patches are coupled interfaces, each rank
builds its own preconditioner, and reductions are global. The aim is a result that can be checked against OpenFOAM
`-parallel` on the same decomposition. Single-GPU runs are unchanged (byte for byte, checked on every commit).

**Status: not usable for a split case yet.** A decomposed case with processor faces stops at start-up with
`processor patches need the multi-GPU coupling, not yet implemented`. A one-rank `decomposePar` case (`-parallel`, np = 1)
runs and is byte-identical to the serial run. The operators, MULES and the momentum/pressure assembly across a processor
face are not ported yet; see [roadmap](roadmap.md).

## What is in

| piece | what it does | where |
|---|---|---|
| reader and roots | `PatchInfo`/`FvPatch` carry `myProcNo`, `neighbProcNo`, `referPatch`; the case's dictionaries and the rank's `processor<rank>/` mesh, fields, zones and output are separate roots | `src/applications/solvers/common/case_roots.cuh` |
| processor geometry | weights, delta, deltaCoeffs, nonOrthDeltaCoeffs and correction vectors from OpenFOAM's `processorFvPatch` formulas with one setup exchange; **0 ULP against OpenFOAM v2606** on tank, graded and open cases at np 2/4/8 | `src/parallel/processor_coupling.*`, [design note](multigpu-processor-interface.md) |
| device interface | processor faces are the last boundary faces of the device mesh; neighbour reads go through halo slots after the local cells (`nCellsExt = nC + nHalo`) | `src/cuda/device_processor.cuh` |
| `ProcComm` | halo exchange of cell and face fields, deterministic rank-order reductions (host and device-resident), broadcast, collective abort; **loopback** (N virtual ranks in one process), **MPI-host** and **NCCL** backends; capturable in CUDA graphs | `src/parallel/proc_comm.*` |
| topology | rank-to-GPU placement for a DGX-1's NVLink cube-mesh (exhaustive and deterministic), `nvidia-smi topo -m` parser, CPU/socket advice | `src/parallel/proc_topology.*`, `tools/proc_comm_nccl/` |
| distributed solvers | global reductions, processor interface in the matrix product, AMG-PCG with a global stopping decision (all ranks take the same iteration count), symmetric Gauss-Seidel with a halo frozen per sweep, per-rank AMG hierarchy cache | `src/matrices/lduMatrix/` |
| writer and function objects | per-rank `processor<rank>/<time>` output with `processor` patch entries (byte-identical to OpenFOAM's `decomposePar` output; `reconstructPar` works), the reference cell on one rank, global `deltaN` and end-of-run statistics, function objects reduced across ranks (rank-0 files) | `inter_writer_cpp.*`, `inter_function_objects.cuh`, `common/proc_globals.cuh` |

Measured so far: distributed AMG-PCG on a decomposed mesh needs more iterations than one GPU, since each rank's V-cycle
is block-local as in OpenFOAM (synthetic block splits: about 1.4-1.65× at np 2, 2× at np 4-8; OpenFOAM's `p_rgh` matrix
at np 2/4/8: 58/73/83 iterations against 45 monolithic). Predicted scaling on 8 V100 (an estimate from message counts and
latencies, not a measurement): 4.7-5.9× on a 2.5M-cell mesh.

## Options

Command line (`brae_interFoam`; the `brae` launcher forwards them):

| option | effect |
|---|---|
| `-parallel` | read the case as decomposed by `decomposePar`: dictionaries from the case, mesh, fields and output from `processor<rank>/` |
| `-rank N -nprocs M` | the rank and the process count, when not taken from the launcher. Otherwise read from `OMPI_COMM_WORLD_RANK/SIZE`, `PMI_RANK/SIZE` or `SLURM_PROCID/NTASKS`; the explicit flags win |

Refused by name under `-parallel`: `processorCyclic`, `cyclicAMI`/`cyclicACMI`, dynamic refinement, topology change and
moving meshes, MRF zones, `fvOptions`, any turbulence model but laminar.

Run-time environment:

| variable | effect |
|---|---|
| `BRAE_COMM` (or `BRAE_COMM_BACKEND`) | `loopback`, `mpi`, `nccl` or `auto` (default: NCCL when built, more than one rank, and the ranks of a node fit its GPUs; else MPI-host). `BRAE_COMM_NP` sets the loopback rank count |
| `BRAE_GPU_MAP="g0,g1,..."` | node-local rank `i` uses GPU `gi` (from the placement tool); applied before any CUDA context exists; an invalid map aborts the job |
| `BRAE_NCCL_ALLOW_SHARED_GPU=1` | let several NCCL ranks share one GPU (testing on a single-GPU machine; needs CUDA MPS for np = 8) |
| `BRAE_PROC_GRAPH=0` | run the distributed AMG-PCG / symGaussSeidel loops host-driven on the NCCL backend too (default there is the whole-loop CUDA graph; the other backends are always host-driven) |
| `BRAE_MGPU_GEOMETRY_ONLY=1` | test gate: steps the processor-patch refusals aside, couples the patches, writes a geometry dump and stops |

Build:

| CMake option | effect |
|---|---|
| `BRAE_WITH_NCCL` (default OFF) | build the NCCL backend; `-DNCCL_ROOT=<dir with include/nccl.h and lib/libnccl.so>` (or `NCCL_HOME`, `CMAKE_PREFIX_PATH`). NCCL 2.28 or newer is recommended; a graph communicator needs `graphUsageMode=0` (set by `ProcComm`) |
| `BRAE_NCCL_TEST_NP` | process counts of the NCCL tests, for example `"1;2;4;8"` (needs that many GPUs) |
| NVSHMEM | removed: use `BRAE_WITH_NCCL` (or the default MPI-host backend); there is no `BRAE_WITH_NVSHMEM` option |

CUDA graphs: on the NCCL backend the np > 1 AMG-PCG and symGaussSeidel loops run as ONE conditional-WHILE graph per rank
(the halo exchange and the deterministic allgather-based reductions inside the body on the graph-only communicator, the
stop decided on the device from the globally reduced residual, so every rank runs the same count). They are bit-identical
to the host-driven loops — same iteration counts, residual histories and solutions — which the MPI-host backend runs and
`BRAE_PROC_GRAPH=0` selects on NCCL; `BRAE_PCG_DEVICE=0` and `BRAE_GS_HOST_LOOP` also force the host loops. The decomposed
solve's graph cache is keyed on the same identity as the single-GPU one plus the processor interface and the halo plan.
`tests/test_dist_solvers.cu` compares the two arms under `BRAE_COMM=nccl` (7 PCG and 7 Gauss-Seidel configurations per
problem/decomposition, including maxIter caps, relTol, minIter and an already-converged start).
They need a driver that supports CUDA 12.4 or newer (conditional graph nodes); where that is missing —
the DGX-1 V100 nodes' driver 535 = CUDA 12.2 — `braeCondGraphsAvailable()` is false and every decomposed solve takes the
host-driven loops, as `BRAE_COND_GRAPH=0` forces. The NCCL communicators are process-lifetime and deliberately leaked at
exit: after WHILE graphs have run on the graph communicator, NCCL's `ncclCommDestroy`/`Abort` can wait forever for a
global quiescence (measured at np=2 on one shared GPU), so the captured graphs are destroyed first (a registry in
`proc_hooks.cuh`) and the communicators are left to the process teardown.


## Tools and tests

- `tools/mgpu_oracle/` — the OpenFOAM v2606 oracle: small decomposed cases, serial against `-parallel` runs, per-rank
  operator dumps, a field comparator, the frozen-state harness (`FROZEN.md`) and the geometry gate. `README.md` there
  describes `setup.sh` and `run_oracle.sh`.
- `tools/spikes/nccl_while_graph/` — NCCL inside a conditional graph `WHILE` body, with a per-GPU-pair sweep for a DGX-1.
- `tools/proc_comm_nccl/` — run instructions for the NCCL backend and the placement tool on 2, 4 and 8 GPUs.
- `tools/bench_v100/` — a single-V100 benchmark of the tank case.
- ctest: `proc_comm*` (loopback, MPI, NCCL, graph capture), `dist_solvers_np{2,4,8}`, `mgpu_proc_geometry`,
  `processor_patch_reader`, `mgpu_np1_byte_identical` (needs OpenFOAM v2606 and a GPU; skips otherwise).

What has not run: any real multi-GPU NCCL run across distinct physical GPUs, NVLink behaviour, and sm_70 execution of
this code.
