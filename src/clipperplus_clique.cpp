#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"


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

    MPI_Init(NULL, NULL);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> partition(keep.size(), 0);
    std::vector<int> xadj, adjncy;

    if (rank == 0)
    {
        // Convert graph to METIS format
        graph.to_csr(xadj, adjncy);
        
        // Partition the graph using METIS
        idx_t n_vertices = keep.size();
        idx_t n_constraints = 1;
        idx_t n_parts = num_procs;
        idx_t objval;
        std::vector<idx_t> part(n_vertices);

        METIS_PartGraphKway(&n_vertices, &n_constraints, xadj.data(), adjncy.data(),
                            NULL, NULL, NULL, &n_parts, NULL, NULL, NULL, &objval, part.data());

        partition.assign(part.begin(), part.end());
    }

    // Broadcast partition data
    MPI_Bcast(partition.data(), partition.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // Assign partitions to processes
    std::vector<Node> local_nodes;
    for (size_t i = 0; i < partition.size(); i++) {
        if (partition[i] == rank) {
            local_nodes.push_back(i);
        }
    }

    Eigen::MatrixXd local_M = M_pruned(local_nodes, local_nodes);
    Eigen::VectorXd local_u0 = u0(local_nodes);

    std::vector<long> long_clique = clipperplus::clique_optimization(local_M, local_u0, Params());
    std::vector<Node> local_clique(long_clique.begin(), long_clique.end());
    int local_clique_size = local_clique.size();

    int global_max_clique_size;
    MPI_Allreduce(&local_clique_size, &global_max_clique_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<Node> global_clique;
    if (rank == 0) {
        global_clique = local_clique;
    }
    
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
