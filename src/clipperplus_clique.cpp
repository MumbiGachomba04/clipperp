#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"
#include <mpi.h>

namespace clipperplus 
{

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph)
{
    int n = graph.size();
    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
    auto k_core_bound = graph.max_core_number() + 1;

    // Heuristic clique computed in serial
    std::vector<Node> heuristic_clique = find_heuristic_clique(graph);
    if (heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
        return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }

    // Nodes filtered based on core number . also done in serial
    std::vector<int> core_number = graph.get_core_numbers();
    std::vector<int> keep, keep_pos(n, -1);
    for (Node i = 0, j = 0; i < n; ++i) {
        if (core_number[i] + 1 >= heuristic_clique.size()) {
            keep.push_back(i);
            keep_pos[i] = j++;
        }
    }
    std::cout<< "Keep size : " << keep.size() << std::endl;

    Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
    M_pruned.diagonal().setOnes();

    Eigen::VectorXd u0 = Eigen::VectorXd::Ones(keep.size());
    for (auto v : heuristic_clique) {
        u0(keep_pos[v]) = 0;
    }
    u0.normalize();

    // Parallel computation of maximum clique
    MPI_Init(NULL); 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Distribute graph partitions 
    int rows_per_proc = keep.size() / num_procs;
    int rest = keep.size() % num_procs; // remainder
    
    std::vector<int> send_counts(num_procs, rows_per_proc);
    for (int i = 0; i < rest; i++) {
        send_counts[i]++; // Distribute remainder rows to first processes
    }

    std::vector<int> displacements(num_procs, 0); //Update row numbers based on displacements 
    for (int i = 1; i < num_procs; i++) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    int start_row = displacements[rank];
    int end_row = start_row + send_counts[rank];
    Eigen::MatrixXd local_M = M_pruned.block(start_row, 0, end_row - start_row, keep.size());

    // Run clique optimization on local partition
    std::vector<long> long_clique = clipperplus::clique_optimization(local_M, u0.segment(start_row, end_row - start_row), Params());
    std::vector<Node> local_clique(long_clique.begin(), long_clique.end());
    //std::vector<Node> local_clique = clipperplus::clique_optimization(local_M, u0.segment(start_row, end_row - start_row), Params());
    int local_clique_size = local_clique.size();

    // Gather maximum clique size across processes
    int global_max_clique_size;
    MPI_Allreduce(&local_clique_size, &global_max_clique_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Process 0 collects all results
    std::vector<Node> global_clique;
    if (rank == 0) {
        global_clique = local_clique;
    }

    // Broadcast the globally largest clique to all processes  
    // A bit redundant
    int clique_size = global_clique.size();
    MPI_Bcast(&clique_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_clique.resize(clique_size);
    MPI_Bcast(global_clique.data(), clique_size, MPI_INT, 0, MPI_COMM_WORLD);

    auto certificate = CERTIFICATE::NONE;
    if (global_clique.size() == k_core_bound) {
        certificate = CERTIFICATE::CORE_BOUND;
    } else if (global_clique.size() == chromatic_welsh) {
        certificate = CERTIFICATE::CHROMATIC_BOUND;
    }

    return {global_clique, certificate};
     MPI_Finalize();
}

}
