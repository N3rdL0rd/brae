#pragma once
// The solver-side view of ProcComm: which channel the hooks use. The matrix product's halo exchange
// (device_spmv.cu) and the reductions' allReduce (reductions.cu) enqueue on the CHANNEL set here, so a solver loop that
// is captured into a CUDA graph (a conditional WHILE body) runs them on the graph communicator (CommChannel::Graph;
// NCCL's graph-only communicator, graphUsageMode = 0) and everything else stays on the eager one. Loopback and MPI-host
// ignore the channel. Thread-local, default Eager.
#include "proc_comm.cuh"

namespace brae {

inline CommChannel& procChannelRef()
{
    thread_local CommChannel ch = CommChannel::Eager;
    return ch;
}
inline CommChannel procChannel() { return procChannelRef(); }

// RAII: ops issued by the hooks inside the scope use `ch`.
class ProcChannelScope
{
public:
    explicit ProcChannelScope(CommChannel ch) : saved_(procChannelRef()) { procChannelRef() = ch; }
    ~ProcChannelScope() { procChannelRef() = saved_; }
    ProcChannelScope(const ProcChannelScope&) = delete;
    ProcChannelScope& operator=(const ProcChannelScope&) = delete;
private:
    CommChannel saved_;
};

} // namespace brae
