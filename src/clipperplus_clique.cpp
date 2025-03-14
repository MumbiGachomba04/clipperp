#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"


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

    // ** Step 1: Compute Heuristic Clique **
    std::vector<Node> heuristic_clique = find_heuristic_clique(graph);
    if (heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
        MPI_Finalize();
        return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }

    // ** Step 2: Filter Nodes Based on Core Numbers **
    std::vector<int> core_number = graph.get_core_numbers();
    std::vector<int> keep, keep_pos(n, -1);
    for (Node i = 0, j = 0; i < n; ++i) {
        if (core_number[i] + 1 >= heuristic_clique.size()) {
            keep.push_back(i);
            keep_pos[i] = j++;
        }
    }

    // ** Step 3: Convert Graph to CSR Format (Only Rank 0) **
    std::vector<int> xadj, adjncy;
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
        idx_t num_parts = num_procs;  // One partition per process
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

    // ** Step 5: Rank 0 Iterates Over Each Partition Sequentially **
    std::vector<Node> max_clique;
    if (rank == 0) {
        for (int p = 0; p < num_procs; p++) {
            std::vector<Node> partition_nodes;
            for (size_t i = 0; i < keep.size(); ++i) {
                if (partition[i] == p) {
                    partition_nodes.push_back(keep[i]);
                }
            }

            // Extract induced subgraph for this partition
            Graph subgraph = graph.induced(partition_nodes);
            Eigen::MatrixXd local_M = subgraph.get_adj_matrix();
            Eigen::VectorXd u0 = Eigen::VectorXd::Ones(partition_nodes.size());
            u0.normalize();

            // Run clique optimization on this partition
            std::vector<long> long_clique = clipperplus::clique_optimization(local_M, u0, Params());
            std::vector<Node> local_clique(long_clique.begin(), long_clique.end());

            // Keep track of the largest clique found
            if (local_clique.size() > max_clique.size()) {
                max_clique = local_clique;
            }
        }
    }

    // ** Step 6: Rank 0 Determines Certificate and Finalizes MPI **
    auto certificate = CERTIFICATE::NONE;
    if (rank == 0) {
        if (max_clique.size() == k_core_bound) {
            certificate = CERTIFICATE::CORE_BOUND;
        } else if (max_clique.size() == chromatic_welsh) {
            certificate = CERTIFICATE::CHROMATIC_BOUND;
        }
    }

    MPI_Finalize();
    return {max_clique, certificate};
}

} // namespace clipperplus
