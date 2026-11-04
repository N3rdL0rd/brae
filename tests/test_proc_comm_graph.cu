// ProcComm CUDA-graph gate: on the LOOPBACK backend one step of an SPMD "solver" (batched halo exchange of
// cell fields + a processor-face exchange (min) + a device-resident allReduce sum and max + a device-resident
// bcast, with kernels between them) is captured into ONE joint CUDA graph for all virtual ranks and replayed
// 1000 times; the state of every rank is compared with an integer reference after every replay. All values are
// integers below 2^53, so the comparison is bitwise independent of fma/ordering.
//
//   test_proc_comm_graph                loopback np = 2,3,4,8
//   mpirun -np 2 test_proc_comm_graph mpi    the MPI-host backend must REFUSE to be captured (throws logic_error)
#include "proc_comm.cuh"

#include <mpi.h>

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>

using namespace brae;

namespace {

void cu(cudaError_t e, const char* w)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda ") + w + ": " + cudaGetErrorString(e));
}

#define REQUIRE(cond, ...)                                                                                          \
    do { if (!(cond)) {                                                                                             \
        char b_[512]; std::snprintf(b_, sizeof b_, __VA_ARGS__);                                                    \
        throw std::runtime_error(std::string(b_) + "  [" #cond "] at line " + std::to_string(__LINE__)); } } while (0)

constexpr double kMod = 1000003.0;

__global__ void wK(const double* u, double* w, int nC)
{
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < nC; c += gridDim.x*blockDim.x) w[c] = 2.0*u[c] + 1.0;
}
__global__ void faceK(const double* u, const int* faceCells, double* fa, int nf)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nf; i += gridDim.x*blockDim.x) fa[i] = u[faceCells[i]];
}
__global__ void sumK(const double* u, const double* w, const double* fa, int nC, int nH, int nf, double* partial, double* stamp)
{
    double s = 0, m = 0;
    for (int k = 0; k < nH; ++k) { s += u[nC + k]; const double v = w[nC + k]; m = v > m ? v : m; }
    for (int i = 0; i < nf; ++i) s += fa[i];
    partial[0] = s;
    partial[1] = m;
    stamp[0] = u[0];
}
__global__ void updateK(double* u, const int* glob, int nC, const double* partial, const double* stamp)
{
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < nC; c += gridDim.x*blockDim.x)
        u[c] = fmod(3.0*u[c] + partial[0] + partial[1] + static_cast<double>(glob[c] % 7) + stamp[0], kMod);
}

template<class T> struct DBuf
{
    T* p = nullptr;
    std::size_t n;
    explicit DBuf(std::size_t n_) : n(n_) { cu(cudaMalloc(&p, std::max<std::size_t>(n, 1)*sizeof(T)), "malloc"); cu(cudaMemsetAsync(p, 0, std::max<std::size_t>(n, 1)*sizeof(T), cudaStreamPerThread), "memset"); cu(cudaStreamSynchronize(cudaStreamPerThread), "memset sync"); }
    template<class V> explicit DBuf(const std::vector<V>& v) : DBuf(v.size())
    {
        if (!v.empty())
        {
            cu(cudaMemcpyAsync(p, v.data(), v.size()*sizeof(V), cudaMemcpyHostToDevice, cudaStreamPerThread), "upload");
            cu(cudaStreamSynchronize(cudaStreamPerThread), "upload sync");
        }
    }
    DBuf(const DBuf&) = delete;
    ~DBuf() { cudaFree(p); }
};

struct RankState
{
    std::unique_ptr<HaloPlan> plan;
    std::unique_ptr<DBuf<double>> u, w, fa, partial, stamp;
    std::unique_ptr<DBuf<int>> glob;
    cudaStream_t stream = nullptr;
};

int blocks(int n) { return std::max(1, std::min((n + 255)/256, 4096)); }

// One step, enqueued on the rank's stream. Identical code eager and captured.
void rankStep(ProcComm& c, RankState& st, cudaStream_t s)
{
    const HaloPlan& pl = *st.plan;
    const int nC = pl.nCells(), nH = pl.nHalo(), nf = pl.nProcFaces();
    wK<<<blocks(nC), 256, 0, s>>>(st.u->p, st.w->p, nC);
    faceK<<<blocks(nf), 256, 0, s>>>(st.u->p, pl.dFaceCells(), st.fa->p, nf);
    c.exchangeCells(pl, {st.u->p, st.w->p}, s, CommChannel::Graph);
    c.exchangeFaces(pl, st.fa->p, FaceOp::Min, s, CommChannel::Graph);
    sumK<<<1, 1, 0, s>>>(st.u->p, st.w->p, st.fa->p, nC, nH, nf, st.partial->p, st.stamp->p);
    c.allReduce(st.partial->p, 1, ReduceOp::Sum, s, CommChannel::Graph);
    c.allReduce(st.partial->p + 1, 1, ReduceOp::Max, s, CommChannel::Graph);
    c.bcastDevice(st.stamp->p, 1, c.size() - 1, s, CommChannel::Graph);
    updateK<<<blocks(nC), 256, 0, s>>>(st.u->p, st.glob->p, nC, st.partial->p, st.stamp->p);
    cu(cudaGetLastError(), "step kernels");
}

// reference: the same step on global integer arrays
void refStep(const SyntheticPartition& sp, std::vector<long long>& U)
{
    long long total = 0, M = 0;
    for (int r = 0; r < sp.np; ++r)
    {
        const HaloPlanDesc& d = sp.descs[r];
        long long s = 0, m = 0;
        for (int k = 0; k < d.nHalo; ++k)
        {
            const long long v = U[sp.haloGlobal[r][k]];
            s += v;
            m = std::max(m, 2*v + 1);
        }
        for (int i = 0; i < d.nProcFaces; ++i)
            s += std::min(U[sp.cellGlobal[r][d.faceCells[i]]], U[sp.haloGlobal[r][d.faceToSlot[i]]]);
        total += s;
        M = std::max(M, m);
    }
    const long long stamp = U[sp.cellGlobal[sp.np - 1][0]];
    for (std::size_t g = 0; g < U.size(); ++g)
        U[g] = static_cast<long long>(std::fmod(static_cast<double>(3*U[g] + total + M + static_cast<long long>(g % 7) + stamp), kMod));
}

void checkState(const SyntheticPartition& sp, const std::vector<std::unique_ptr<RankState>>& rs, const std::vector<long long>& U, const char* when)
{
    for (int r = 0; r < sp.np; ++r)
    {
        const int nC = rs[r]->plan->nCells();
        std::vector<double> h(nC);
        cu(cudaMemcpy(h.data(), rs[r]->u->p, nC*sizeof(double), cudaMemcpyDeviceToHost), "download");
        for (int c = 0; c < nC; ++c)
            REQUIRE(h[c] == static_cast<double>(U[sp.cellGlobal[r][c]]), "%s: rank %d cell %d: %.17g != %lld", when, r, c, h[c], U[sp.cellGlobal[r][c]]);
    }
}

struct Watchdog
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread t;
    Watchdog(const char* what, int seconds)
    {
        t = std::thread([this, what, seconds] {
            std::unique_lock<std::mutex> lk(m);
            if (!cv.wait_for(lk, std::chrono::seconds(seconds), [&] { return done; }))
            {
                std::printf("FAIL: watchdog: '%s' hung for %d s\n", what, seconds);
                std::fflush(stdout);
                std::_Exit(3);
            }
        });
    }
    ~Watchdog() { { std::lock_guard<std::mutex> lk(m); done = true; } cv.notify_all(); t.join(); }
};

int runGraphCase(const std::string& name, const SyntheticPartition& sp, int replays)
{
    const int np = sp.np;
    Watchdog wd(name.c_str(), 240);
    std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::createLoopback(np);
    std::vector<std::unique_ptr<RankState>> rs(np);
    std::vector<std::string> err(np);
    cudaEvent_t fork = nullptr;
    std::vector<cudaEvent_t> join(np, nullptr);
    cu(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "event");
    for (auto& e : join) cu(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<long long> U(sp.owner.size());
    for (std::size_t g = 0; g < U.size(); ++g) U[g] = static_cast<long long>((g*7919 + 13) % 1000003);

    std::vector<std::thread> th;
    for (int r = 0; r < np; ++r)
        th.emplace_back([&, r] {
            ProcComm& c = *comms[r];
            try
            {
                c.bindThread();
                rs[r].reset(new RankState);
                RankState& st = *rs[r];
                cu(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking), "stream");
                st.plan = c.makePlan(sp.descs[r], 4);
                const int nExt = st.plan->nExt(), nC = st.plan->nCells();
                st.u.reset(new DBuf<double>(nExt));
                st.w.reset(new DBuf<double>(nExt));
                st.fa.reset(new DBuf<double>(st.plan->nProcFaces()));
                st.partial.reset(new DBuf<double>(2));
                st.stamp.reset(new DBuf<double>(1));
                st.glob.reset(new DBuf<int>(sp.cellGlobal[r]));
                std::vector<double> u0(nC);
                for (int i = 0; i < nC; ++i) u0[i] = static_cast<double>((static_cast<std::size_t>(sp.cellGlobal[r][i])*7919 + 13) % 1000003);
                cu(cudaMemcpy(st.u->p, u0.data(), nC*sizeof(double), cudaMemcpyHostToDevice), "upload u");
                cu(cudaStreamSynchronize(cudaStreamPerThread), "sync");
                c.barrier();
                // eager warm-up step: wakes every code path (and every lazily created resource) before capture
                rankStep(c, st, st.stream);
                cu(cudaStreamSynchronize(st.stream), "sync");
                c.barrier();
                // ---- joint capture: rank 0 forks everybody into ONE capture ----
                if (r == 0)
                {
                    cu(cudaStreamBeginCapture(st.stream, cudaStreamCaptureModeGlobal), "begin capture");
                    cu(cudaEventRecord(fork, st.stream), "fork");
                }
                c.barrier();
                if (r != 0) cu(cudaStreamWaitEvent(st.stream, fork, 0), "join capture");
                rankStep(c, st, st.stream);
                if (r != 0) cu(cudaEventRecord(join[r], st.stream), "record join");
                c.barrier();
                if (r == 0)
                {
                    for (int q = 1; q < np; ++q) cu(cudaStreamWaitEvent(st.stream, join[q], 0), "wait join");
                    cu(cudaStreamEndCapture(st.stream, &graph), "end capture");
                    cu(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                }
                c.barrier();
            }
            catch (const std::exception& e)
            {
                err[r] = e.what();
                try { c.collectiveAbort(e.what()); } catch (...) {}
            }
        });
    for (auto& t : th) t.join();
    int fails = 0;
    for (int r = 0; r < np; ++r)
        if (!err[r].empty()) { ++fails; std::printf("FAIL %s rank %d: %s\n", name.c_str(), r, err[r].c_str()); }
    if (fails) return fails;
    try
    {
        refStep(sp, U);   // the eager step
        checkState(sp, rs, U, "eager step");
        size_t nNodes = 0;
        cu(cudaGraphGetNodes(graph, nullptr, &nNodes), "nodes");
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < replays; ++rep)
        {
            cu(cudaGraphLaunch(exec, rs[0]->stream), "graph launch");
            cu(cudaStreamSynchronize(rs[0]->stream), "graph sync");
            refStep(sp, U);
            const std::string when = "replay " + std::to_string(rep);
            checkState(sp, rs, U, when.c_str());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("PASS %s: %d replays bitwise correct, graph of %zu nodes (%.1f ms incl. verification)\n", name.c_str(), replays, nNodes, ms);
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %s: %s\n", name.c_str(), e.what());
        fails = 1;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    for (auto& r : rs) { cudaStreamSynchronize(r->stream); cudaStreamDestroy(r->stream); }
    cudaEventDestroy(fork);
    for (auto e : join) if (e) cudaEventDestroy(e);
    return fails;
}

// The MPI-host backend is host-synchronous: a call inside a capture must throw, not corrupt the capture.
int mpiRefuses(int argc, char** argv)
{
    Pstream::init(argc, argv);
    int fails = 0;
    {
        std::unique_ptr<ProcComm> c = ProcComm::createMpiHost();
        if (c->size() > 1)
        {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            double* d = nullptr;
            cu(cudaMalloc(&d, 8*sizeof(double)), "malloc");
            REQUIRE(!c->graphCapturable(), "MPI-host backend claims to be graph capturable");
            cu(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "begin capture");
            bool threw = false;
            try { c->allReduce(d, 1, ReduceOp::Sum, s); }
            catch (const std::logic_error&) { threw = true; }
            cudaGraph_t g = nullptr;
            cudaStreamEndCapture(s, &g);
            if (g) cudaGraphDestroy(g);
            cudaGetLastError();
            if (!threw) { std::printf("FAIL rank %d: MPI-host allReduce inside a capture did not throw\n", c->rank()); ++fails; }
            cudaFree(d);
            cudaStreamDestroy(s);
        }
    }
    const int total = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master()) std::printf("%s test_proc_comm_graph mpi: backend refuses capture\n", total == 0 ? "PASS" : "FAIL");
    Pstream::finalize();
    return total == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 1 && std::string(argv[1]) == "mpi") return mpiRefuses(argc, argv);
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0) { std::printf("SKIP: no CUDA device\n"); return 77; }
        int fails = 0;
        for (int np : {2, 3, 4, 8})
        {
            static const int fac[9][3] = {{0,0,0},{1,1,1},{2,1,1},{1,3,1},{2,2,1},{0,0,0},{0,0,0},{0,0,0},{2,2,2}};
            const int nx = 20, ny = 15, nz = 8;
            fails += runGraphCase("graph np=" + std::to_string(np) + " block",
                                  partitionBox(nx, ny, nz, np, blockOwner(nx, ny, nz, fac[np][0], fac[np][1], fac[np][2])), 1000);
            fails += runGraphCase("graph np=" + std::to_string(np) + " scattered",
                                  partitionBox(18, 15, 12, np, scatteredOwner(18, 15, 12, np, 3)), 1000);
        }
        std::printf("test_proc_comm_graph: %s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
