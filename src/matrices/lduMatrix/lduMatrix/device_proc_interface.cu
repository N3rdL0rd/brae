#include "device_proc_interface.cuh"

#include <stdexcept>

namespace brae {

DeviceProcInterface::DeviceProcInterface(ProcComm& comm, const HaloPlan& plan)
:
    comm_(&comm),
    plan_(&plan)
{
    nGlobalCells_ = plan.nCells();
    if (comm.size() == 1) return;
    nGlobalCells_ = static_cast<long long>(comm.allReduce(static_cast<double>(plan.nCells()), ReduceOp::Sum));
    const int n = plan.nProcFaces();
    coeff_.resize(static_cast<std::size_t>(n));
    if (n > 0)
    {
        const std::vector<label> own(plan.desc().faceCells.begin(), plan.desc().faceCells.end());
        const std::vector<label> rank = pairOwnerRanks(own, nRanks_);
        rank_.copyFrom(rank);
    }
}

void DeviceProcInterface::setCoeff(const std::vector<scalar>& hostCoeff)
{
    if (comm_->size() == 1) return;
    if (hostCoeff.size() != static_cast<std::size_t>(plan_->nProcFaces()))
    {
        throw std::invalid_argument("DeviceProcInterface::setCoeff: expected one coefficient per processor face");
    }
    coeff_.copyFrom(hostCoeff);
}

DeviceProcView DeviceProcInterface::view() const
{
    DeviceProcView v;
    if (comm_->size() == 1) return v;
    v.comm = comm_;
    v.plan = plan_;
    v.nFaces = plan_->nProcFaces();
    v.faceCell = plan_->dFaceCells();
    v.nbrIdx = plan_->dFaceHaloCell();
    v.coeff = coeff_.data();
    v.rank = rank_.data();
    v.nRanks = nRanks_;
    v.nGlobalCells = nGlobalCells_;
    return v;
}

} // namespace brae
