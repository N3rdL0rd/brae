// ProcComm's MPI abort side channel at process/generation teardown. Every rank destroys its
// communicator before MPI_Finalize, so a pending nonblocking allgather must be completed, never
// mistaken for a completed MPI_Test and abandoned with another collective targeting its buffer.
#include "proc_comm.cuh"
#include "cf_pstream.cuh"
#include <mpi.h>


#include <chrono>
#include <cstdio>
#include <thread>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo();
    const int np = Pstream::nProcs();
    if (np != 8)
    {
        std::fprintf(stderr, "test_proc_comm_teardown requires np=8, got %d\n", np);
        Pstream::finalize();
        return 1;
    }
    const int generations = argc > 1 ? std::atoi(argv[1]) : 8;
    for (int g = 0; g < generations; ++g)
    {
        {
            auto comm = ProcComm::createMpiHost();
            double value = comm->allReduce(static_cast<double>(rank + g), ReduceOp::Sum);
            const double expected = static_cast<double>(np*(np - 1)/2 + np*g);
            if (value != expected)
            {
                std::fprintf(stderr, "rank %d generation %d: reduction %.17g != %.17g\n",
                             rank, g, value, expected);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            // Change which rank arrives last, so some polls see an incomplete allgather.
            std::this_thread::sleep_for(std::chrono::milliseconds((rank + g) % np));
        }
    }
    Pstream::finalize();
    if (rank == 0) std::printf("PASS: %d ProcComm generations at np=8\n", generations);
    return 0;
}
