#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"
#include <mpi.h>
#include <metis.h>

namespace clipperplus 
{

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph)
{
    MPI_Init(NULL, NULL);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = graph.size();
    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
    auto k_core_bound = graph.max_core_number() + 1;

    // ** Step 1: Compute Heuristic Clique (Single Process) **
    std::vector<Node> heuristic_clique = find_heuristic_clique(graph);
    if (heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
        MPI_Finalize();
        return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }

    // ** Step 2: Filter Nodes Based on Core Numbers (Single Process) **
    std::vector<int> core_number = graph.get_core_numbers();
    std::vector<int> keep, keep_pos(n, -1);
    for (Node i = 0, j = 0; i < n; ++i) {
        if (core_number[i] + 1 >= heuristic_clique.size()) {
            keep.push_back(i);
            keep_pos[i] = j++;
        }
    }

    Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
    M_pruned.diagonal().setOnes();

    Eigen::VectorXd u0 = Eigen::VectorXd::Ones(keep.size());
    for (auto v : heuristic_clique) {
        u0(keep_pos[v]) = 0;
    }
    u0.normalize();

    // ** Step 3: Convert Graph to CSR Format (Only Rank 0) **
    std::vector<int> xadj, adjncy, vwgt;
    if (rank == 0) {
        int n_keep = keep.size();
        xadj.resize(n_keep + 1);
        adjncy.clear();

        xadj[0] = 0;
        for (int i = 0; i < n_keep; ++i) {
            for (auto neighbor : graph.neighbors(keep[i])) {
                auto it = std::find(keep.begin(), keep.end(), neighbor);
                if (it != keep.end()) {
                    adjncy.push_back(std::distance(keep.begin(), it));
                }
            }
            xadj[i + 1] = adjncy.size();
        }
    }

    // ** Step 4: METIS Partitioning (Only Rank 0) **
    std::vector<idx_t> partition(keep.size(), 0);
    if (rank == 0) {
        idx_t num_vertices = keep.size();
        idx_t num_parts = num_procs;
        idx_t objval;

        METIS_PartGraphKway(&num_vertices, 
                            nullptr, 
                            xadj.data(), adjncy.data(), 
                            nullptr, nullptr, nullptr, 
                            &num_parts, 
                            nullptr, nullptr, 
                            nullptr, 
                            &objval, 
                            partition.data());
    }

    // ** Step 5: Broadcast Partition Info to All Processes **
    MPI_Bcast(partition.data(), partition.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // ** Step 6: Extract Local Subgraph Based on Partitioning **
    std::vector<Node> local_nodes;
    for (size_t i = 0; i < keep.size(); ++i) {
        if (partition[i] == rank) {
            local_nodes.push_back(keep[i]);
        }
    }

    Graph local_graph = graph.induced(local_nodes);
    Eigen::MatrixXd local_M = local_graph.get_adj_matrix();

    // ** Step 7: Compute Local Clique **
    std::vector<long> long_clique = clipperplus::clique_optimization(local_M, u0, Params());
    std::vector<Node> local_clique(long_clique.begin(), long_clique.end());
    int local_clique_size = local_clique.size();

    // ** Step 8: Gather Maximum Clique Size Across Processes **
    int global_max_clique_size;
    MPI_Allreduce(&local_clique_size, &global_max_clique_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // ** Step 9: Rank 0 Collects the Maximum Clique **
    std::vector<Node> global_clique;
    if (rank == 0) {
        global_clique = local_clique;
    }

    // ** Step 10: Broadcast the Maximum Clique **
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

    MPI_Finalize();
    return {global_clique, certificate};
}

} 
