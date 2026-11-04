# Multi-GPU interFoam: the processor interface (geometry, predicates, device build)

Scope: what exists after the processor-geometry work (InterFoamStep WP1 and the build half of WP4). Operators and
solvers across a processor face are NOT ported; a normal `-parallel` run with processor faces still stops by name
("processor patches need the multi-GPU coupling, not yet implemented").

## Pieces

| piece | where |
|---|---|
| neighbour data once, bit-exact | `ProcComm::exchangeFaceData` (src/parallel/proc_comm.*), 9 doubles/face = `processorPolyPatch::initGeometry` (Cf, Sf, owner-cell centre) |
| OpenFOAM's formulas | `coupleProcessorPair` (fv_patch.cu): `delta = (Cf-Cn) - (nbrCf-nbrCn)`, `w = nbrCn/(nf & (Cf-Cn) + nbrCn)` with `nbrCn = nbrNf & (nbrCf - nbrCn)`, `nbrNf = nbrSf/(|nbrSf|+VSMALL)`, `deltaCoeffs = 1/|delta|`, `nonOrthDeltaCoeffs = 1/max(nf & delta, 0.05|delta|)`, `corrVec = nf - delta*nonOrthDeltaCoeffs` (processorFvPatch.C, basicFvGeometryScheme.C). NOT `1/|C_r - C_l|` of `computeProcDeltaCoeffs` |
| one collective per rank | `attachProcessorCoupling` (src/parallel/processor_coupling.*): exchange, `HaloPlan` (`buildPlanFromPatches`), coupled half of every patch, `nbrFaceCells = nC + slot` |
| FvPatch | `coupled`, `owner`, `weights`, `delta`, `nbrD`, `deltaCoeffs`, `nonOrthDeltaCoeffs`, `nonOrthCorrectionVectors`, `nbrFaceCells` (halo slots), `haloBase`, `nHaloCells` |
| device mesh | `DeviceMesh::nHalo`, `nCellsExt = nC + nHalo`, `nBndFacesPhys`; processor faces are the LAST boundary faces of the gather (bcType 8), processor face j = boundary face `nBndFacesPhys + j` |
| device interface | `buildProcessorInterfaces` (cyclic_interface.cuh) + `buildDeviceProcInterface` (device_processor.cuh): a `DeviceCyclic` whose `nbrCell[j] = nC + slot`, `twin = -1` (lambda sync = `ProcComm::exchangeFaces`), checked against the plan |
| gate | `BRAE_MGPU_GEOMETRY_ONLY=1` (inter_parallel.cuh `mgpuGeometryOnly()`): steps the four by-name refusals aside for a processor patch and makes braeInterFoam couple, build, dump and STOP (`runProcessorGeometryOnly`). Delete it with the operator work |

## The predicate split

`isCoupledInterfaceType` (cyclic, cyclicAMI, cyclicACMI, cyclicPeriodicAMI) decided ~160 sites, and what every one of
them means is "skip this patch in the ordinary boundary machinery: it is an interface held outside the boundary
gather". A processor patch must NOT take that arm: its faces stay boundary faces (bval route for the sum-class
operators) and only the neighbour-reading operators go through the interface. So:

* `isCyclicFamilyType(t)`: the old predicate, renamed, behaviour unchanged for every case (all sites migrated);
* `isProcessorType(t)`: `processor` only (processorCyclic is refused by name under -parallel);
* `isCoupledType(t)`: either; used where a site already tested `isCyclicFamilyType(t) || t == "processor"`.

Per site group (what a processor patch needs; H = handled now, D = deferred to the operator/driver agent, R = refused):

| site group | sites | processor patch needs | status |
|---|---|---|---|
| DeviceMesh boundary gather (device_mesh.cuh), DeviceBoundary gather/flatten (device_boundary.cuh) | 8 | stays IN the gather; bcType 8 | H (prefix `nBndFacesPhys`, `nHalo`) |
| device geometry builder (device_fv_geometry.cu) | 1 | the interface faces' Cf/Sf come from the mesh as for any boundary face | H |
| interface build (cyclic_interface.cuh, device_cyclic.cuh) | - | translational pair, neighbour = halo slot | H (`buildProcessorInterfaces`) |
| interFoam device driver boundary uploads/flatteners (inter_driver_device.cu: patch value upload loops, `bnd*` arrays, p_rgh/U/alpha device patches) | ~40 | processor faces are in `nBndFaces` after the physical prefix: loops over `nBndFacesPhys`, bval filled by halo interpolation | D |
| interFoam device fast-path gates (flux push `1679`, contact angle `1868`/`1946`, forcesFast `3651`, device p_rgh patches `4405`, adjustPhiDeferred `4952`) | ~8 | accept processor, keep refusing cyclic; per-face arrays aligned to `nBndFacesPhys` | D |
| interFoam device stencils/hooks (nHatStencil, stf/snGrad, MULES twin, forces) | ~10 | halo of alpha/grad(alpha); lambda `exchangeFaces` | D |
| device turbulence (device_inter_turbulence.cu, 7) | - | wall distance across ranks; k/eps/omega halos | R (laminar only under -parallel, inter_parallel.cuh) |
| host patch-field factory (fv_patch_field.cuh) | 2 | no `processor` branch: a processor entry in a field file has no reader | D |
| host fvc/fvm/MULES/limiter operators branching on `FvPatch::coupled` | ~180 `.coupled` | neighbour = `nbrFaceCells` halo slot, halo-extended host arrays | D (the `coupled` flag is set; a driver that does not extend its arrays must not run) |
| case translation coupled-patch block (inter_case_cpp.cu), setRDeltaT, kOmegaSST linearUpwind | 4 | by-name refusal | R behind the gate (open only for the geometry stage) |
| writer / field writer (foam_field_writer.cuh, inter_writer_cpp.cu) | 4 | `processor` patch entries with `value` | D (writer agent) |
| simpleFoam / rhoSimpleFoam / pimpleFoam drivers | ~30 | their old global-mesh decomposition path | unchanged (`isCoupledType` folds their `|| == "processor"`) |
| dynamic mesh (hexRef8, polyTopoChange, removeFaces, MRF, wall distance, LES delta, fvcSmooth, vol point interpolation) | ~12 | refused under -parallel (dynamicMeshDict, MRF, non-laminar) | R |
| tests/ | ~20 | cyclic-only fixtures | unchanged |

## Halo-extended storage (the cell-array length policy)

A cell array that a neighbour-reading kernel reads (the interface kernels, neighbour index `nC + slot`) must hold
`nCellsExt = nC + nHalo` entries. The policy is ONE rule in two places:

* `DeviceBuffer<T>` has a LENGTH (`size()`: what every launch, `copyTo`/`copyFrom` and size check sees, always the
  owned cells) and a CAPACITY (`capacity()`: what the pool block holds). They differ only for a halo-extended cell
  array: `size() == nC`, `capacity() == nC + nHalo`, tail zeroed at allocation. Moves, pool give-back
  (`capacity * sizeof`), `resize(n)` (drops the capacity, as before), `resize(n, cap)` (new) follow. So no
  `size() == nC` check anywhere in the tree changes meaning, and no D2H download returns the tail.
* `cellArrayLength(dm)` (device_mesh.cuh) = `max(nCells, nCellsExt)`; `makeCellBuffer(dm[, host])` and
  `resizeCellBuffer(buf, dm)` allocate with it. With `nHalo == 0` they reduce to `DeviceBuffer(n)` / `resize(n)`:
  same size, same pool class, same pointers, so the single-GPU run (and every graph/cache key) is untouched.
* `deviceCopy` (blas1.cu) copies the capacity, so an old-time copy of a halo-extended array keeps its tail
  (an exchange is still needed whenever the owned values changed after the last one).

Sites that allocate with the halo capacity (state in the driver, producers in the operators); a producer allocating
the output makes every consumer correct without its own change:

| where | what |
|---|---|
| inter_driver_device.cu, state | `dA`, `dAOld`, `dUx/y/z`, `dUox/y/z`, `dPrgh`, `dK` (uploaded from the host fields with `makeCellBuffer`) |
| device_fvc.cu producers | `deviceGaussGrad`, `deviceLeastSquaresGrad`, the fused Gauss / least-squares gradients (gradient outputs are read across a face), `deviceDiv` (K = -div(nHatf)) |
| device_simple.cu producers | `deviceReciprocalV` (rAU), the `H` component output of `deviceMatrixH`-form (`Hk`) |
| device_alpha_step.cu | `alpha2` (the corrector's complement, read by the compression flux) |
| device_inter_alpha_step.cu | the mixture's `alpha2`, `rho`, `mu`, `nu` |
| device_inter_peqn.cu | the corrected velocity `UX/UY/UZ` (read by the next `H`) |
| blas1.cu | `deviceCopy` |
