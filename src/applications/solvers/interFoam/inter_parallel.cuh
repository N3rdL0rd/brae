#pragma once
// What brae_interFoam says about a decomposed case (`-parallel`) before it builds a single field:
// the per-rank patch table, and the refusals, by name, of everything the multi-GPU path does not carry yet.
//
// Header-only and free of CUDA so a unit test can drive it on a PrimitiveMesh's PatchInfo list and a
// scratch directory. The rule for each refusal is the one the single-GPU code already follows: say which
// entry of which file, and why it is not run, rather than run a different case.
#include "case_roots.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

namespace parallelDetail {
inline bool fileExists(const std::string& p)
{
    return std::filesystem::exists(p) || std::filesystem::exists(p + ".gz");
}
}   // namespace parallelDetail

// THE GATE OF THE MULTI-GPU COUPLING. Every by-name refusal of a processor patch (refuseProcessorCoupling below,
// the case translation's coupled-patch block, setRDeltaT, the kOmegaSST closure) stays in force unless the
// environment says BRAE_MGPU_GEOMETRY_ONLY=1: then they step aside for a `processor` patch AND the driver stops
// right after the coupling geometry is built (braeInterFoam.cu: runProcessorGeometryOnly, which couples the
// patches over ProcComm, builds the device interface and writes the geometry dump) -- the operators across a
// processor face are not ported, so nothing past that point may run. Close the gate for good by deleting it
// when the operator and solver work lands; no case file can open it.
inline bool mgpuGeometryOnly()
{
    static const bool on = [] {
        const char* e = std::getenv("BRAE_MGPU_GEOMETRY_ONLY");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

// THE STAGE-DUMP GATE (BRAE_MGPU_STAGE_DUMP=<dir>): the driver couples the processor patches over ProcComm, builds the
// halo-extended device state and the processor interface, runs the FIRST step's explicit stages (interface normal and
// curvature, the surface-tension force, the high-order alpha flux) with the processor face operators, writes them per
// rank in the OpenFOAM dump format under <dir>/processor<rank>/<time>/ and stops (inter_stage_dump.cuh). The solver
// path past that point (limiter, momentum, pressure, collectives) is not ported, so nothing else may run: the same
// by-name refusals step aside for a `processor` patch as for BRAE_MGPU_GEOMETRY_ONLY, and only for this run.
inline std::string mgpuStageDumpDir()
{
    const char* e = std::getenv("BRAE_MGPU_STAGE_DUMP");
    return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string();
}
// either gate steps the by-name refusals of a processor patch aside
inline bool mgpuProcessorGate()
{
    return mgpuGeometryOnly() || !mgpuStageDumpDir().empty();
}

// One line per patch: kind, size and, for a processor patch, the ranks it joins. Printed for every rank so a
// decomposition can be read off the log of any of them.
inline std::string processorPatchTable(const CaseRoots& roots, const std::vector<PatchInfo>& patches)
{
    std::string s = "brae interFoam: -parallel, rank " + std::to_string(roots.rank) + " of "
                  + std::to_string(roots.nProcs) + ", reading " + roots.procDir + "\n";
    char buf[256];
    long nProc = 0, nProcFaces = 0;
    for (const PatchInfo& p : patches)
    {
        const bool proc = p.type == "processor" || p.type == "processorCyclic";
        if (proc)
        {
            std::snprintf(buf, sizeof buf, "    %-28s %-16s faces %8ld  rank %d <-> rank %d%s%s\n", p.name.c_str(),
                          p.type.c_str(), (long)p.size, (int)p.myProcNo, (int)p.neighbProcNo,
                          p.referPatch.empty() ? "" : "  referPatch ", p.referPatch.c_str());
            ++nProc;
            nProcFaces += p.size;
        }
        else
        {
            std::snprintf(buf, sizeof buf, "    %-28s %-16s faces %8ld\n", p.name.c_str(), p.type.c_str(), (long)p.size);
        }
        s += buf;
    }
    std::snprintf(buf, sizeof buf, "    %ld processor patch(es), %ld processor face(s)\n", nProc, nProcFaces);
    return s + buf;
}

// Everything `-parallel` does not run, by name. Applies at every np: np = 1 is the same program with one
// domain, and a case brae refuses on N domains is refused on one so that the np = 1 gate means something.
inline void refuseUnderParallel(const CaseRoots& roots, const std::vector<PatchInfo>& patches)
{
    using parallelDetail::fileExists;
    const std::string WHO = "-parallel: ";
    for (const PatchInfo& p : patches)
    {
        if (p.type == "processorCyclic")
        {
            throw std::runtime_error(
                WHO + "patch '" + p.name + "' is a processorCyclic (a cyclic cut by the decomposition, referPatch '"
                + p.referPatch + "'). Its halo reads would need the cyclic's rotation or translation, which is not "
                "ported. Decompose with `preservePatches (" + (p.referPatch.empty() ? std::string("<cyclic>") : p.referPatch)
                + ");` in system/decomposeParDict so the cyclic stays on one rank.");
        }
        if (p.type == "cyclicAMI" || p.type == "cyclicACMI" || p.type == "cyclicPeriodicAMI")
        {
            throw std::runtime_error(
                WHO + "patch '" + p.name + "' is " + p.type + ". An AMI's weights couple faces across ranks "
                "through a stencil that is not exchanged; cyclicAMI/cyclicACMI are not ported under -parallel.");
        }
    }
    const std::string& D = roots.dictDir;
    if (fileExists(D + "/constant/dynamicMeshDict"))
    {
        const FoamDict d = readDict(D + "/constant/dynamicMeshDict");
        const std::string dyn = d.wordOr("dynamicFvMesh", "");
        if (dyn != "staticFvMesh")
        {
            throw std::runtime_error(
                WHO + "constant/dynamicMeshDict names dynamicFvMesh `" + (dyn.empty() ? std::string("(none)") : dyn)
                + "`. Moving meshes, topology changes and dynamic refinement are not ported under -parallel; "
                "only `staticFvMesh` runs.");
        }
    }
    if (fileExists(D + "/constant/MRFProperties"))
    {
        const FoamDict d = readDict(D + "/constant/MRFProperties");
        for (const auto& e : d.subs)
        {
            if (e.second.switchOr("active", true))
            {
                throw std::runtime_error(WHO + "constant/MRFProperties has the active MRF zone `" + e.first
                                         + "`. MRF zones are not ported under -parallel.");
            }
        }
    }
    for (const char* rel : {"/constant/fvOptions", "/system/fvOptions"})
    {
        if (!fileExists(D + rel)) continue;
        const FoamDict root = readDict(D + rel);
        const FoamDict* wrapped = root.subDict("options");
        for (const auto& e : (wrapped ? *wrapped : root).subs)
        {
            if (e.second.switchOr("active", true))
            {
                throw std::runtime_error(WHO + std::string(rel + 1) + " has the active option `" + e.first + "` (type "
                                         + e.second.wordOr("type", "?") + "). fvOptions are not ported under -parallel.");
            }
        }
        break;
    }
    if (fileExists(D + "/constant/turbulenceProperties"))
    {
        const FoamDict d = readDict(D + "/constant/turbulenceProperties");
        const std::string sim = d.wordOr("simulationType", "");
        if (sim != "laminar")
        {
            throw std::runtime_error(
                WHO + "constant/turbulenceProperties has simulationType `" + sim
                + "`. Only laminar runs under -parallel: a turbulence model's wall distance and its k/epsilon/omega "
                "halos are not ported.");
        }
    }
}

// A processor patch with faces is half of a coupled interface that the multi-GPU path does not couple yet.
// np = 1 has none (decomposePar writes a processor patch only where a face is cut), so it runs as an
// ordinary case; a processor patch handed to the SERIAL loop would be solved as an ordinary wall.
inline void refuseProcessorCoupling(const CaseRoots& roots, const std::vector<PatchInfo>& patches)
{
    for (const PatchInfo& p : patches)
    {
        if (p.type != "processor" || p.size == 0) continue;
        if (roots.parallel)
        {
            if (mgpuProcessorGate()) continue;   // the gates: the driver builds the coupling and stops after its stage
            throw std::runtime_error(
                "-parallel: rank " + std::to_string(roots.rank) + " of " + std::to_string(roots.nProcs)
                + " has the processor patch '" + p.name + "' (" + std::to_string((long)p.size) + " faces, rank "
                + std::to_string((int)p.myProcNo) + " <-> rank " + std::to_string((int)p.neighbProcNo)
                + "): processor patches need the multi-GPU coupling, not yet implemented.");
        }
        throw std::runtime_error(
            "patch '" + p.name + "' is a processor patch (rank " + std::to_string((int)p.myProcNo)
            + " <-> rank " + std::to_string((int)p.neighbProcNo) + ") of a decomposed case, and a serial run would "
            "solve it as an ordinary wall. Run the decomposed case with -parallel; processor patches also need the "
            "multi-GPU coupling, not yet implemented.");
    }
}

}   // namespace interFoam
}   // namespace cpu
}   // namespace brae
