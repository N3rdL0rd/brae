// brae::ProcComm: backends (loopback, MPI-host, NCCL stub), halo plans, synthetic partitioner.
// The contract is documented in proc_comm.cuh.
#include "proc_comm.cuh"

#include <mpi.h>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>

namespace brae {

namespace {

// ------------------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------------------
std::atomic<std::size_t> g_allocs{0};
ProcComm* g_current = nullptr;
thread_local ProcComm* t_current = nullptr;

void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ProcComm cuda: ") + what + ": " + cudaGetErrorString(e));
}

template<class T> T* dalloc(std::size_t n)
{
    if (n == 0) return nullptr;
    T* p = nullptr;
    ck(cudaMalloc(&p, n*sizeof(T)), "cudaMalloc");
    ++g_allocs;
    return p;
}

template<class T> T* dupload(const std::vector<T>& v)
{
    T* p = dalloc<T>(v.size());
    if (p)
    {
        // pageable H2D may return before the DMA is done: complete it before another stream can read the memory
        ck(cudaMemcpyAsync(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread), "upload");
        ck(cudaStreamSynchronize(cudaStreamPerThread), "upload sync");
    }
    return p;
}

constexpr int kMaxComps = 16;      // components per pack/unpack launch (kernel parameter space)
constexpr int kMaxTotalComps = 64; // components per exchange
constexpr int kMaxRanks = 64;      // loopback ranks (kernel parameter space of the fold)

// Pack/unpack descriptors. Staging layout of a cell exchange, neighbour-major: the data of neighbour p with
// `count` cells occupies [C*off, C*(off+count)), component c of element j at C*off + c*count + j.
struct SendEntry { int off, count; };
struct RecvEntry { const double* src; int srcOff, count, dstSlot; };
struct FaceEntry { const double* src; int srcOff, count, start; };

struct CompList
{
    double* p[kMaxComps];
    long long cs[kMaxComps];
    int n;      // components in this launch
    int base;   // global index of component 0 of this launch
};

struct SlotList { const double* p[kMaxRanks]; int n; };

__global__ void packCellsK(const int* sendCells, const SendEntry* tab, CompList cl, int C, double* out)
{
    const SendEntry e = tab[blockIdx.y];
    const long long total = static_cast<long long>(e.count)*cl.n;
    for (long long t = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x)*blockDim.x)
    {
        const int c = static_cast<int>(t / e.count);
        const int j = static_cast<int>(t - static_cast<long long>(c)*e.count);
        out[static_cast<long long>(C)*e.off + static_cast<long long>(cl.base + c)*e.count + j] =
            cl.p[c][static_cast<long long>(sendCells[e.off + j])*cl.cs[c]];
    }
}

__global__ void unpackCellsK(const RecvEntry* tab, CompList cl, int C, int nC)
{
    const RecvEntry e = tab[blockIdx.y];
    const long long total = static_cast<long long>(e.count)*cl.n;
    for (long long t = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x)*blockDim.x)
    {
        const int c = static_cast<int>(t / e.count);
        const int j = static_cast<int>(t - static_cast<long long>(c)*e.count);
        cl.p[c][static_cast<long long>(nC + e.dstSlot + j)*cl.cs[c]] =
            e.src[static_cast<long long>(C)*e.srcOff + static_cast<long long>(cl.base + c)*e.count + j];
    }
}

__global__ void combineFacesK(const FaceEntry* tab, double* arr, int op)
{
    const FaceEntry e = tab[blockIdx.y];
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < e.count; i += gridDim.x*blockDim.x)
    {
        const double s = e.src[e.srcOff + i];
        double& a = arr[e.start + i];
        switch (op)
        {
            case 0: a = s; break;
            case 1: a = s < a ? s : a; break;
            case 2: a = s > a ? s : a; break;
            default: a = a + s; break;
        }
    }
}

// op: 0 sum, 1 min, 2 max, 3 bcast of slot `root`. Rank order 0..n-1: the same bits on every rank.
__global__ void foldSlotsK(double* out, SlotList sl, int n, int op, int root)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    {
        if (op == 3) { out[i] = sl.p[root][i]; continue; }
        double acc = sl.p[0][i];
        for (int r = 1; r < sl.n; ++r)
        {
            const double v = sl.p[r][i];
            if (op == 0) acc = acc + v;
            else if (op == 1) acc = v < acc ? v : acc;
            else acc = v > acc ? v : acc;
        }
        out[i] = acc;
    }
}

int opCode(ReduceOp op) { return op == ReduceOp::Sum ? 0 : op == ReduceOp::Min ? 1 : 2; }

void foldHost(double* out, const std::vector<const double*>& s, int n, int op, int root)
{
    for (int i = 0; i < n; ++i)
    {
        if (op == 3) { out[i] = s[root][i]; continue; }
        double acc = s[0][i];
        for (std::size_t r = 1; r < s.size(); ++r)
        {
            const double v = s[r][i];
            if (op == 0) acc = acc + v;
            else if (op == 1) acc = v < acc ? v : acc;
            else acc = v > acc ? v : acc;
        }
        out[i] = acc;
    }
}

int faceOpCode(FaceOp op)
{
    switch (op) { case FaceOp::Replace: return 0; case FaceOp::Min: return 1; case FaceOp::Max: return 2; default: return 3; }
}

struct FlatComps
{
    double* p[kMaxTotalComps];
    long long cs[kMaxTotalComps];
    int n = 0;
};

void flatten(const HaloField* f, int nFields, FlatComps& out)
{
    out.n = 0;
    for (int i = 0; i < nFields; ++i)
    {
        if (f[i].nComp < 1) throw std::invalid_argument("ProcComm: HaloField with nComp < 1");
        if (f[i].nComp > 1 && f[i].compStride == 0) throw std::invalid_argument("ProcComm: HaloField with nComp > 1 needs compStride");
        for (int k = 0; k < f[i].nComp; ++k)
        {
            if (out.n >= kMaxTotalComps) throw std::length_error("ProcComm: more than 64 components in one exchange");
            out.p[out.n] = f[i].data + static_cast<long long>(k)*f[i].compStride;
            out.cs[out.n] = f[i].cellStride;
            ++out.n;
        }
    }
}

void fillList(CompList& cl, const FlatComps& fc, int c0)
{
    cl.n = std::min(kMaxComps, fc.n - c0);
    cl.base = c0;
    for (int k = 0; k < cl.n; ++k) { cl.p[k] = fc.p[c0 + k]; cl.cs[k] = fc.cs[c0 + k]; }
}

int gridFor(long long total)
{
    return static_cast<int>(std::min<long long>((total + 255)/256, 1 << 20));
}

// Backend-independent device-side staging of a plan: index tables and the staging buffers.
struct StagePlan : PlanState
{
    int maxComps = 0;
    std::size_t sendCap = 0, recvCap = 0;      // doubles
    double* dSend = nullptr;                   // pack target (peers read it on loopback)
    double* dRecv = nullptr;                   // MPI: unpack source
    SendEntry* dSendTab = nullptr;
    RecvEntry* dRecvTab = nullptr;
    FaceEntry* dFaceTab = nullptr;
    int nNbr = 0, maxSend = 0, maxRecv = 0, maxFace = 0;
    const HaloPlanDesc* desc = nullptr;

    ~StagePlan() override
    {
        cudaFree(dSend); cudaFree(dRecv); cudaFree(dSendTab); cudaFree(dRecvTab); cudaFree(dFaceTab);
    }

    void init(const HaloPlanDesc& d, int maxC, bool wantRecv)
    {
        desc = &d;
        maxComps = maxC;
        nNbr = static_cast<int>(d.nbrs.size());
        sendCap = std::max<std::size_t>(static_cast<std::size_t>(maxC)*d.sendCells.size(), d.nProcFaces);
        recvCap = std::max<std::size_t>(static_cast<std::size_t>(maxC)*d.nHalo, d.nProcFaces);
        dSend = dalloc<double>(sendCap);
        if (wantRecv) dRecv = dalloc<double>(recvCap);
        std::vector<SendEntry> st(nNbr);
        for (int k = 0; k < nNbr; ++k)
        {
            st[k] = {d.nbrs[k].sendOffset, d.nbrs[k].sendCount};
            maxSend = std::max(maxSend, d.nbrs[k].sendCount);
            maxRecv = std::max(maxRecv, d.nbrs[k].recvCount);
            maxFace = std::max(maxFace, d.nbrs[k].faceCount);
        }
        dSendTab = dupload(st);
    }

    // rtab/ftab: per neighbour, filled by the backend (the source buffers differ).
    void setTables(const std::vector<RecvEntry>& rt, const std::vector<FaceEntry>& ft)
    {
        cudaFree(dRecvTab); cudaFree(dFaceTab);
        dRecvTab = dupload(rt);
        dFaceTab = dupload(ft);
    }

    void pack(const HaloPlan& plan, const FlatComps& fc, cudaStream_t s) const
    {
        if (nNbr == 0 || maxSend == 0) return;
        for (int c0 = 0; c0 < fc.n; c0 += kMaxComps)
        {
            CompList cl; fillList(cl, fc, c0);
            packCellsK<<<dim3(gridFor(static_cast<long long>(maxSend)*cl.n), nNbr), 256, 0, s>>>(
                plan.dSendCells(), dSendTab, cl, fc.n, dSend);
        }
        ck(cudaGetLastError(), "packCells launch");
    }

    void unpack(const HaloPlan& plan, const FlatComps& fc, cudaStream_t s) const
    {
        if (nNbr == 0 || maxRecv == 0) return;
        for (int c0 = 0; c0 < fc.n; c0 += kMaxComps)
        {
            CompList cl; fillList(cl, fc, c0);
            unpackCellsK<<<dim3(gridFor(static_cast<long long>(maxRecv)*cl.n), nNbr), 256, 0, s>>>(
                dRecvTab, cl, fc.n, plan.nCells());
        }
        ck(cudaGetLastError(), "unpackCells launch");
    }

    void combineFaces(double* arr, FaceOp op, cudaStream_t s) const
    {
        if (nNbr == 0 || maxFace == 0) return;
        combineFacesK<<<dim3(gridFor(maxFace), nNbr), 256, 0, s>>>(dFaceTab, arr, faceOpCode(op));
        ck(cudaGetLastError(), "combineFaces launch");
    }
};

void checkCapacity(const StagePlan& st, int C)
{
    if (C > st.maxComps)
        throw std::length_error("ProcComm: exchange of " + std::to_string(C) + " components exceeds the plan's maxComps "
                                + std::to_string(st.maxComps));
}

} // namespace

namespace detail {
struct PlanAccess
{
    static PlanState* state(const HaloPlan& p) { return p.state_.get(); }
    static void set(HaloPlan& p, std::unique_ptr<PlanState> s) { p.state_ = std::move(s); }
    static const ProcComm* owner(const HaloPlan& p) { return p.owner_; }
    static const HaloPlanDesc& desc(const HaloPlan& p) { return p.desc_; }
    static void upload(HaloPlan& p)
    {
        const HaloPlanDesc& d = p.desc_;
        p.dSendCells_ = dupload(d.sendCells);
        p.dFaceCells_ = dupload(d.faceCells);
        std::vector<int> fh(d.faceToSlot.size());
        for (std::size_t i = 0; i < fh.size(); ++i) fh[i] = d.nCells + d.faceToSlot[i];
        p.dFaceHalo_ = dupload(fh);
    }
};
} // namespace detail

// ------------------------------------------------------------------------------------------------------------
// HaloPlanDesc, builder
// ------------------------------------------------------------------------------------------------------------
void HaloPlanDesc::validate() const
{
    auto bad = [](const std::string& m) { throw std::invalid_argument("HaloPlanDesc: " + m); };
    if (nCells < 0 || nHalo < 0 || nProcFaces < 0) bad("negative size");
    int so = 0, rs = 0, fs = 0;
    for (std::size_t k = 0; k < nbrs.size(); ++k)
    {
        const HaloNeighbour& n = nbrs[k];
        if (n.rank < 0) bad("negative neighbour rank");
        for (std::size_t j = 0; j < k; ++j) if (nbrs[j].rank == n.rank) bad("neighbour rank listed twice");
        if (n.sendCount < 0 || n.recvCount < 0 || n.faceCount < 0) bad("negative count");
        if (n.sendOffset != so || n.recvStart != rs || n.faceStart != fs) bad("ranges are not contiguous in neighbour order");
        so += n.sendCount; rs += n.recvCount; fs += n.faceCount;
    }
    if (static_cast<std::size_t>(so) != sendCells.size()) bad("sendCells size != sum of sendCount");
    if (rs != nHalo) bad("sum of recvCount != nHalo");
    if (fs != nProcFaces) bad("sum of faceCount != nProcFaces");
    if (faceCells.size() != static_cast<std::size_t>(nProcFaces) || faceToSlot.size() != static_cast<std::size_t>(nProcFaces))
        bad("faceCells/faceToSlot size != nProcFaces");
    for (int c : sendCells) if (c < 0 || c >= nCells) bad("sendCells entry out of range");
    for (int c : faceCells) if (c < 0 || c >= nCells) bad("faceCells entry out of range");
    for (const HaloNeighbour& n : nbrs)
        for (int i = n.faceStart; i < n.faceStart + n.faceCount; ++i)
            if (faceToSlot[i] < n.recvStart || faceToSlot[i] >= n.recvStart + n.recvCount)
                bad("faceToSlot outside its neighbour's slot range");
}

HaloPlanDesc buildHaloPlanFromProcessorPatches(int nCells, const std::vector<ProcessorPatchDesc>& patches)
{
    HaloPlanDesc d;
    d.nCells = nCells;
    for (const ProcessorPatchDesc& p : patches)
    {
        if (p.faceCells.size() != p.nbrFaceCells.size())
            throw std::invalid_argument("buildHaloPlanFromProcessorPatches: faceCells and nbrFaceCells differ in length "
                                        "(fetch the neighbour's face cells first)");
        HaloNeighbour n;
        n.rank = p.nbrRank;
        n.sendOffset = static_cast<int>(d.sendCells.size());
        n.recvStart = d.nHalo;
        n.faceStart = d.nProcFaces;
        n.faceCount = static_cast<int>(p.faceCells.size());
        std::unordered_map<int, int> sent, slotOf;
        for (std::size_t i = 0; i < p.faceCells.size(); ++i)
        {
            const int mine = p.faceCells[i];
            if (sent.emplace(mine, static_cast<int>(sent.size())).second) d.sendCells.push_back(mine);
            const auto it = slotOf.emplace(p.nbrFaceCells[i], static_cast<int>(slotOf.size())).first;
            d.faceCells.push_back(mine);
            d.faceToSlot.push_back(n.recvStart + it->second);
        }
        n.sendCount = static_cast<int>(sent.size());
        n.recvCount = static_cast<int>(slotOf.size());
        d.nHalo += n.recvCount;
        d.nProcFaces += n.faceCount;
        d.nbrs.push_back(n);
    }
    d.validate();
    return d;
}

// ------------------------------------------------------------------------------------------------------------
// HaloPlan
// ------------------------------------------------------------------------------------------------------------
HaloPlan::HaloPlan(const ProcComm* owner, HaloPlanDesc d) : owner_(owner), desc_(std::move(d)) {}

HaloPlan::~HaloPlan()
{
    state_.reset();
    cudaFree(dSendCells_);
    cudaFree(dFaceCells_);
    cudaFree(dFaceHalo_);
}

// ------------------------------------------------------------------------------------------------------------
// ProcComm base
// ------------------------------------------------------------------------------------------------------------
ProcComm::ProcComm(Backend b, int rank, int size) : rank_(rank), size_(size), backend_(b) {}
ProcComm::~ProcComm()
{
    if (g_current == this) g_current = nullptr;
    if (t_current == this) t_current = nullptr;
}

ProcComm* ProcComm::current() { return t_current ? t_current : g_current; }
void ProcComm::setCurrent(ProcComm* c) { g_current = c; }
void ProcComm::setThreadCurrent(ProcComm* c) { t_current = c; }
std::size_t ProcComm::allocationCount() { return g_allocs.load(); }

void ProcComm::bindThread()
{
    if (device_ >= 0) ck(cudaSetDevice(device_), "cudaSetDevice");
    setThreadCurrent(this);
}

std::unique_ptr<HaloPlan> ProcComm::newPlan(HaloPlanDesc desc)
{
    desc.validate();
    std::unique_ptr<HaloPlan> p(new HaloPlan(this, std::move(desc)));
    detail::PlanAccess::upload(*p);
    return p;
}

PlanState* ProcComm::planState(const HaloPlan& p) { return detail::PlanAccess::state(p); }
void ProcComm::setPlanState(HaloPlan& p, std::unique_ptr<PlanState> s) { detail::PlanAccess::set(p, std::move(s)); }

void ProcComm::checkPlan(const HaloPlan& p) const
{
    if (detail::PlanAccess::owner(p) != this)
        throw std::logic_error("ProcComm: HaloPlan used with a communicator other than the one that made it");
}

std::unique_ptr<HaloPlan> ProcComm::makePlan(HaloPlanDesc desc, int maxComps)
{
    if (maxComps < 1 || maxComps > kMaxTotalComps) throw std::invalid_argument("ProcComm::makePlan: maxComps must be 1..64");
    if (size_ == 1)
    {
        if (!desc.nbrs.empty() || desc.nHalo != 0)
            throw std::invalid_argument("ProcComm::makePlan: np == 1 cannot have neighbours");
        desc.validate();
        return std::unique_ptr<HaloPlan>(new HaloPlan(this, std::move(desc)));   // no device memory
    }
    return doMakePlan(std::move(desc), maxComps);
}

std::unique_ptr<HaloPlan> ProcComm::buildPlanFromPatches(int nCells, std::vector<ProcessorPatchDesc> patches, int maxComps)
{
    std::vector<int> nbr;
    std::vector<std::vector<int>> send;
    for (const ProcessorPatchDesc& p : patches) { nbr.push_back(p.nbrRank); send.push_back(p.faceCells); }
    std::vector<std::vector<int>> recv = exchangeIntLists(nbr, send);
    for (std::size_t k = 0; k < patches.size(); ++k) patches[k].nbrFaceCells = std::move(recv[k]);
    return makePlan(buildHaloPlanFromProcessorPatches(nCells, patches), maxComps);
}

std::vector<std::vector<int>> ProcComm::exchangeIntLists(const std::vector<int>& nbrRanks,
                                                         const std::vector<std::vector<int>>& send)
{
    if (nbrRanks.size() != send.size()) throw std::invalid_argument("ProcComm::exchangeIntLists: size mismatch");
    if (size_ == 1)
    {
        if (!nbrRanks.empty()) throw std::invalid_argument("ProcComm::exchangeIntLists: np == 1 has no neighbours");
        return {};
    }
    return doExchangeInts(nbrRanks, send);
}

void ProcComm::collectiveAbort(const std::string& msg)
{
    if (size_ == 1) throw std::runtime_error(msg);
    doAbort(msg);
    throw std::logic_error("ProcComm::doAbort returned");
}

void ProcComm::exchangeCells(const HaloPlan& plan, std::initializer_list<double*> scalars, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    HaloField f[kMaxTotalComps];
    int n = 0;
    for (double* p : scalars)
    {
        if (n >= kMaxTotalComps) throw std::length_error("ProcComm: more than 64 fields in one exchange");
        f[n++] = HaloField{p, 1, 0, 1};
    }
    exchangeCells(plan, f, n, stream, ch);
}

void ProcComm::exchangeVector(const HaloPlan& plan, double* x, double* y, double* z, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f[3] = {{x, 1, 0, 1}, {y, 1, 0, 1}, {z, 1, 0, 1}};
    exchangeCells(plan, f, 3, stream, ch);
}

void ProcComm::exchangeCellsPacked(const HaloPlan& plan, double* data, int nComp, long long compStride, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f{data, nComp, compStride, 1};
    exchangeCells(plan, &f, 1, stream, ch);
}

void ProcComm::exchangeCellsAoS(const HaloPlan& plan, double* data, int nComp, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f{data, nComp, 1, nComp};
    exchangeCells(plan, &f, 1, stream, ch);
}

// ------------------------------------------------------------------------------------------------------------
// LOOPBACK backend
// ------------------------------------------------------------------------------------------------------------
namespace {

struct LbPlan : StagePlan {};

struct LbGroup
{
    int np;
    std::size_t slotCap;

    // abort-aware host rendezvous of the np rank threads
    std::mutex m;
    std::condition_variable cv;
    int arrived = 0;
    unsigned long gen = 0;
    bool aborted = false;
    std::string abortMsg;
    int abortRank = -1;

    std::vector<cudaEvent_t> ready, done;      // per rank, recorded on that rank's stream
    std::vector<double*> dSlot;                // per rank device slot of the device reductions
    std::vector<std::vector<double>> hSlot;    // per rank host slot of the host reductions
    std::vector<std::vector<std::vector<int>>> mail;   // [from][to]
    std::map<int, std::vector<LbPlan*>> reg;   // plan id -> per-rank plan, for makePlan()

    LbGroup(int n, std::size_t cap)
    : np(n), slotCap(cap), ready(n, nullptr), done(n, nullptr), dSlot(n, nullptr), hSlot(n),
      mail(n, std::vector<std::vector<int>>(n))
    {
        for (int r = 0; r < np; ++r)
        {
            ck(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming), "cudaEventCreate");
            ck(cudaEventCreateWithFlags(&done[r], cudaEventDisableTiming), "cudaEventCreate");
            dSlot[r] = dalloc<double>(slotCap);
        }
    }
    ~LbGroup()
    {
        for (int r = 0; r < np; ++r)
        {
            cudaEventDestroy(ready[r]);
            cudaEventDestroy(done[r]);
            cudaFree(dSlot[r]);
        }
    }

    void sync()
    {
        std::unique_lock<std::mutex> lk(m);
        if (aborted) throw ProcCommAbort(abortMsg, abortRank);
        const unsigned long g = gen;
        if (++arrived == np) { arrived = 0; ++gen; cv.notify_all(); return; }
        cv.wait(lk, [&] { return gen != g || aborted; });
        if (gen == g) throw ProcCommAbort(abortMsg, abortRank);   // aborted before the barrier was released
    }

    // First abort wins; sticky.
    void abort(int rank, const std::string& msg)
    {
        std::lock_guard<std::mutex> lk(m);
        if (!aborted) { aborted = true; abortMsg = msg; abortRank = rank; }
        cv.notify_all();
    }
    bool isAborted()
    {
        std::lock_guard<std::mutex> lk(m);
        return aborted;
    }
    std::string message() { std::lock_guard<std::mutex> lk(m); return abortMsg; }
    int originRank() { std::lock_guard<std::mutex> lk(m); return abortRank; }
};

class LoopbackComm final : public ProcComm
{
public:
    LoopbackComm(std::shared_ptr<LbGroup> g, int rank, int np) : ProcComm(Backend::Loopback, rank, np), g_(std::move(g))
    {
        if (np > 1) ck(cudaGetDevice(&device_), "cudaGetDevice");
    }
    bool graphCapturable() const override { return true; }
    bool abortPending() const override { return g_->isAborted(); }

protected:
    // Collective: every rank registers its plan, then reads the peers' staging pointers/offsets.
    std::unique_ptr<HaloPlan> doMakePlan(HaloPlanDesc desc, int maxComps) override
    {
        std::unique_ptr<HaloPlan> plan = newPlan(std::move(desc));
        auto st = std::make_unique<LbPlan>();
        LbPlan* sp = st.get();
        sp->init(detail::PlanAccess::desc(*plan), maxComps, false);
        setPlanState(*plan, std::move(st));
        const int id = planSeq_++;
        {
            std::lock_guard<std::mutex> lk(g_->m);
            auto& v = g_->reg[id];
            v.resize(g_->np, nullptr);
            v[rank_] = sp;
        }
        g_->sync();
        const HaloPlanDesc& d = *sp->desc;
        std::vector<RecvEntry> rt(d.nbrs.size());
        std::vector<FaceEntry> ft(d.nbrs.size());
        std::string err;
        for (std::size_t k = 0; k < d.nbrs.size() && err.empty(); ++k)
        {
            const HaloNeighbour& n = d.nbrs[k];
            if (n.rank >= g_->np) { err = "neighbour rank out of range"; break; }
            const LbPlan* peer = g_->reg[id][n.rank];
            const HaloNeighbour* pn = nullptr;
            for (const HaloNeighbour& q : peer->desc->nbrs) if (q.rank == rank_) pn = &q;
            if (!pn) { err = "rank " + std::to_string(n.rank) + " does not list rank " + std::to_string(rank_) + " as a neighbour"; break; }
            if (pn->sendCount != n.recvCount || pn->recvCount != n.sendCount || pn->faceCount != n.faceCount)
            {
                err = "plan mismatch between ranks " + std::to_string(rank_) + " and " + std::to_string(n.rank);
                break;
            }
            rt[k] = {peer->dSend, pn->sendOffset, n.recvCount, n.recvStart};
            ft[k] = {peer->dSend, pn->faceStart, n.faceCount, n.faceStart};
        }
        if (!err.empty()) g_->abort(rank_, "ProcComm::makePlan: " + err);   // the others throw at the sync below
        g_->sync();
        if (!err.empty()) throw ProcCommAbort(g_->message(), g_->originRank());
        sp->setTables(rt, ft);
        g_->sync();   // no rank exchanges before every rank's tables point at live peers
        return plan;
    }

    // pack -> ready -> [rendezvous] -> peers' staging read by my unpack -> done -> [rendezvous] -> wait done
    void doExchangeCells(const HaloPlan& plan, const HaloField* fields, int nFields, cudaStream_t s, CommChannel) override
    {
        auto* st = static_cast<LbPlan*>(planState(plan));
        FlatComps fc;
        flatten(fields, nFields, fc);
        if (fc.n == 0) return;
        checkCapacity(*st, fc.n);
        st->pack(plan, fc, s);
        rendezvous(s, neighbourRanks(*st), [&] { st->unpack(plan, fc, s); });
    }

    void doExchangeFaces(const HaloPlan& plan, double* arr, FaceOp op, cudaStream_t s, CommChannel) override
    {
        auto* st = static_cast<LbPlan*>(planState(plan));
        const int nf = plan.nProcFaces();
        if (nf > 0) ck(cudaMemcpyAsync(st->dSend, arr, static_cast<std::size_t>(nf)*sizeof(double), cudaMemcpyDeviceToDevice, s), "face pack");
        rendezvous(s, neighbourRanks(*st), [&] { st->combineFaces(arr, op, s); });
    }

    void doAllReduceDevice(double* d, int n, ReduceOp op, cudaStream_t s, CommChannel) override
    {
        deviceFold(d, n, opCode(op), 0, s);
    }
    void doBcastDevice(double* d, int n, int root, cudaStream_t s, CommChannel) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        deviceFold(d, n, 3, root, s);
    }

    void doAllReduceHost(double* h, int n, ReduceOp op) override { hostFold(h, n, opCode(op), 0); }
    void doBcastHost(double* h, int n, int root) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        hostFold(h, n, 3, root);
    }

    void doBarrier() override { g_->sync(); }

    std::vector<std::vector<int>> doExchangeInts(const std::vector<int>& nbr, const std::vector<std::vector<int>>& send) override
    {
        for (std::size_t k = 0; k < nbr.size(); ++k)
        {
            if (nbr[k] < 0 || nbr[k] >= size_) throw std::invalid_argument("ProcComm::exchangeIntLists: rank out of range");
            g_->mail[rank_][nbr[k]] = send[k];
        }
        g_->sync();
        std::vector<std::vector<int>> recv(nbr.size());
        for (std::size_t k = 0; k < nbr.size(); ++k) recv[k] = g_->mail[nbr[k]][rank_];
        g_->sync();
        for (std::size_t k = 0; k < nbr.size(); ++k) g_->mail[rank_][nbr[k]].clear();
        return recv;
    }

    [[noreturn]] void doAbort(const std::string& msg) override
    {
        g_->abort(rank_, msg);
        throw ProcCommAbort(g_->message(), g_->originRank());
    }

private:
    static std::vector<int> neighbourRanks(const StagePlan& st)
    {
        std::vector<int> r;
        for (const HaloNeighbour& n : st.desc->nbrs) r.push_back(n.rank);
        return r;
    }

    template<class Body> void rendezvous(cudaStream_t s, const std::vector<int>& peers, Body body)
    {
        ck(cudaEventRecord(g_->ready[rank_], s), "record ready");
        g_->sync();
        for (int p : peers) ck(cudaStreamWaitEvent(s, g_->ready[p], 0), "wait ready");
        body();
        ck(cudaEventRecord(g_->done[rank_], s), "record done");
        g_->sync();
        for (int p : peers) ck(cudaStreamWaitEvent(s, g_->done[p], 0), "wait done");
    }

    void deviceFold(double* d, int n, int op, int root, cudaStream_t s)
    {
        if (n < 0 || static_cast<std::size_t>(n) > g_->slotCap)
            throw std::length_error("ProcComm: device reduction of " + std::to_string(n) + " doubles exceeds the loopback reduceCapacity "
                                    + std::to_string(g_->slotCap));
        if (n == 0) return;
        SlotList sl;
        sl.n = g_->np;
        for (int r = 0; r < g_->np; ++r) sl.p[r] = g_->dSlot[r];
        std::vector<int> all(g_->np);
        for (int r = 0; r < g_->np; ++r) all[r] = r;
        ck(cudaMemcpyAsync(g_->dSlot[rank_], d, static_cast<std::size_t>(n)*sizeof(double), cudaMemcpyDeviceToDevice, s), "reduce stage");
        rendezvous(s, all, [&] {
            foldSlotsK<<<gridFor(n), 256, 0, s>>>(d, sl, n, op, root);
            ck(cudaGetLastError(), "fold launch");
        });
    }

    void hostFold(double* h, int n, int op, int root)
    {
        g_->hSlot[rank_].assign(h, h + n);
        g_->sync();
        std::string err;
        std::vector<const double*> s(g_->np);
        for (int r = 0; r < g_->np; ++r)
        {
            if (static_cast<int>(g_->hSlot[r].size()) != n) err = "host reduction size differs between ranks";
            s[r] = g_->hSlot[r].data();
        }
        if (!err.empty()) g_->abort(rank_, "ProcComm: " + err);
        else foldHost(h, s, n, op, root);
        g_->sync();   // slots may be rewritten only after everyone has read them (throws for everyone after an abort)
    }

    std::shared_ptr<LbGroup> g_;
    int planSeq_ = 0;
};

} // namespace

std::vector<std::unique_ptr<ProcComm>> ProcComm::createLoopback(int np, std::size_t reduceCapacity)
{
    if (np < 1 || np > kMaxRanks) throw std::invalid_argument("ProcComm::createLoopback: np must be 1..64");
    std::shared_ptr<LbGroup> g;
    if (np > 1) g = std::make_shared<LbGroup>(np, reduceCapacity);   // np == 1 allocates nothing
    std::vector<std::unique_ptr<ProcComm>> v;
    for (int r = 0; r < np; ++r) v.push_back(std::unique_ptr<ProcComm>(new LoopbackComm(g, r, np)));
    return v;
}

// MPI-HOST backend: added in the next commit.
std::unique_ptr<ProcComm> ProcComm::createMpiHost()
{
    throw std::runtime_error("ProcComm MPI-host backend: not implemented yet");
}

// ------------------------------------------------------------------------------------------------------------
// NCCL backend: NOT IMPLEMENTED. CommSolvers.md step C. What it has to provide behind this same interface:
//   makePlan      per-neighbour ncclSend/ncclRecv lists inside ONE ncclGroupStart/End on `stream` (pack kernel before,
//                 unpack after; staging is plain cudaMalloc memory)
//   allReduce     ncclAllGather of the n partials + fixed-rank-order local fold (same bits on every rank)
//   channels      two communicators: commGraph (graph-only, graphUsageMode = 0 / NCCL_GRAPH_MIXING_SUPPORT = 0, used
//                 only for CommChannel::Graph) and commEager (everything else); warm both up before any capture
//   abort         the collectives learn the abort out of band (a device-visible flag polled by the unpack/fold kernels)
// ------------------------------------------------------------------------------------------------------------
std::unique_ptr<ProcComm> ProcComm::createNccl()
{
    throw std::runtime_error("ProcComm NCCL backend: not implemented (CommSolvers.md step C); use loopback or mpi");
}

std::vector<std::unique_ptr<ProcComm>> ProcComm::create(const std::string& backendIn, int np)
{
    std::string b = backendIn;
    if (b.empty()) { const char* e = std::getenv("BRAE_COMM_BACKEND"); b = e ? e : "mpi"; }
    std::vector<std::unique_ptr<ProcComm>> v;
    if (b == "loopback")
    {
        if (np <= 0) { const char* e = std::getenv("BRAE_COMM_NP"); np = e ? std::atoi(e) : 1; }
        return createLoopback(np);
    }
    if (b == "mpi") { v.push_back(createMpiHost()); return v; }
    if (b == "nccl") { v.push_back(createNccl()); return v; }
    throw std::invalid_argument("ProcComm::create: unknown backend '" + b + "' (loopback | mpi | nccl)");
}

// ------------------------------------------------------------------------------------------------------------
// synthetic partitioner
// ------------------------------------------------------------------------------------------------------------
SyntheticPartition partitionBox(int nx, int ny, int nz, int np, const std::vector<int>& owner)
{
    const long long N = static_cast<long long>(nx)*ny*nz;
    if (nx < 1 || ny < 1 || nz < 1 || static_cast<long long>(owner.size()) != N)
        throw std::invalid_argument("partitionBox: owner size != nx*ny*nz");
    SyntheticPartition sp;
    sp.np = np; sp.nx = nx; sp.ny = ny; sp.nz = nz; sp.owner = owner;
    sp.cellGlobal.assign(np, {});
    std::vector<int> local(N);
    for (int g = 0; g < N; ++g)
    {
        if (owner[g] < 0 || owner[g] >= np) throw std::invalid_argument("partitionBox: owner out of range");
        local[g] = static_cast<int>(sp.cellGlobal[owner[g]].size());
        sp.cellGlobal[owner[g]].push_back(g);
    }
    for (int r = 0; r < np; ++r)
        if (sp.cellGlobal[r].empty()) throw std::invalid_argument("partitionBox: rank " + std::to_string(r) + " owns no cell");

    struct Face { long long key; int mine, nbr; };   // global ids
    std::map<std::pair<int, int>, std::vector<Face>> pairs;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int g = i + nx*(j + ny*k);
                const int nb[3] = {i + 1 < nx ? g + 1 : -1, j + 1 < ny ? g + nx : -1, k + 1 < nz ? g + nx*ny : -1};
                for (int dir = 0; dir < 3; ++dir)
                {
                    const int g2 = nb[dir];
                    if (g2 < 0 || owner[g] == owner[g2]) continue;
                    const long long key = static_cast<long long>(g)*3 + dir;   // g < g2: the lower id keys the face
                    pairs[{owner[g], owner[g2]}].push_back({key, g, g2});
                    pairs[{owner[g2], owner[g]}].push_back({key, g2, g});
                }
            }
    sp.patches.assign(np, {});
    sp.faceKey.assign(np, {});
    sp.haloGlobal.assign(np, {});
    sp.descs.assign(np, {});
    std::vector<std::vector<int>> nbrGlobal(np);   // per rank, per processor face: global id of the neighbour cell
    for (auto& kv : pairs)
    {
        const int r = kv.first.first, p = kv.first.second;
        std::vector<Face>& fs = kv.second;
        std::sort(fs.begin(), fs.end(), [](const Face& a, const Face& b) { return a.key < b.key; });
        ProcessorPatchDesc pd;
        pd.nbrRank = p;
        for (const Face& f : fs)
        {
            pd.faceCells.push_back(local[f.mine]);
            pd.nbrFaceCells.push_back(local[f.nbr]);
            sp.faceKey[r].push_back(f.key);
            nbrGlobal[r].push_back(f.nbr);
        }
        sp.patches[r].push_back(std::move(pd));
    }
    for (int r = 0; r < np; ++r)
    {
        sp.descs[r] = buildHaloPlanFromProcessorPatches(static_cast<int>(sp.cellGlobal[r].size()), sp.patches[r]);
        sp.haloGlobal[r].assign(sp.descs[r].nHalo, -1);
        for (int i = 0; i < sp.descs[r].nProcFaces; ++i) sp.haloGlobal[r][sp.descs[r].faceToSlot[i]] = nbrGlobal[r][i];
    }
    return sp;
}

std::vector<int> blockOwner(int nx, int ny, int nz, int px, int py, int pz)
{
    auto axis = [](int n, int P) {
        std::vector<int> b(n);
        for (int i = 0; i < n; ++i)
            for (int q = 0; q < P; ++q)
                if (i < static_cast<int>((static_cast<long long>(q + 1)*n)/P)) { b[i] = q; break; }
        return b;
    };
    const std::vector<int> bx = axis(nx, px), by = axis(ny, py), bz = axis(nz, pz);
    std::vector<int> o(static_cast<std::size_t>(nx)*ny*nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) o[i + nx*(j + static_cast<std::size_t>(ny)*k)] = bx[i] + px*(by[j] + py*bz[k]);
    return o;
}

std::vector<int> scatteredOwner(int nx, int ny, int nz, int np, int chunk)
{
    const int cx = (nx + chunk - 1)/chunk, cy = (ny + chunk - 1)/chunk, cz = (nz + chunk - 1)/chunk;
    std::vector<int> chunkRank(static_cast<std::size_t>(cx)*cy*cz);
    for (int k = 0; k < cz; ++k)
        for (int j = 0; j < cy; ++j)
            for (int i = 0; i < cx; ++i)
            {
                const unsigned h = static_cast<unsigned>(i)*73856093u ^ static_cast<unsigned>(j)*19349663u ^ static_cast<unsigned>(k)*83492791u;
                chunkRank[i + cx*(j + static_cast<std::size_t>(cy)*k)] = static_cast<int>((h >> 3) % static_cast<unsigned>(np));
            }
    std::vector<int> cnt(np, 0);
    for (int r : chunkRank) ++cnt[r];
    for (int r = 0; r < np && r < static_cast<int>(chunkRank.size()); ++r)
        if (cnt[r] == 0) chunkRank[r] = r;
    std::vector<int> o(static_cast<std::size_t>(nx)*ny*nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                o[i + nx*(j + static_cast<std::size_t>(ny)*k)] =
                    chunkRank[(i/chunk) + cx*((j/chunk) + static_cast<std::size_t>(cy)*(k/chunk))];
    return o;
}

} // namespace brae
