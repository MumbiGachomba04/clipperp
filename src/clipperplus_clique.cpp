#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"


namespace clipperplus 
{

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph)
{
    int n = graph.size();
    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
    auto k_core_bound = graph.max_core_number() + 1;
    
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

    if (keep.empty()) {
        return {heuristic_clique, CERTIFICATE::NONE};
    }
    
    Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
    M_pruned.diagonal().setOnes();

    Eigen::VectorXd u0 = Eigen::VectorXd::Ones(keep.size());
    for (auto v : heuristic_clique) {
        if (keep_pos[v] >= 0) {
            u0(keep_pos[v]) = 0;
        }
    }
    u0 += 0.01 * Eigen::VectorXd::Random(keep.size());
    u0.normalize();
    
    std::vector<long> long_clique = clipperplus::clique_optimization(M_pruned, u0, Params());
    std::vector<Node> optimal_clique;
    for (long idx : long_clique) {
        optimal_clique.push_back(keep[idx]);
    }
    
    if (optimal_clique.size() < heuristic_clique.size()) {
        optimal_clique = heuristic_clique;
    }
    
    // Partitioning after optimization
    std::vector<idx_t> xadj(optimal_clique.size() + 1, 0), adjncy;
    idx_t num_vertices = optimal_clique.size();
    
    for (idx_t i = 0; i < num_vertices; ++i) {
        for (idx_t j = 0; j < num_vertices; ++j) {
            if (M_pruned(i, j) > 0) {
                adjncy.push_back(j);
            }
        }
        xadj[i + 1] = adjncy.size();
    }
    
    if (adjncy.empty()) {
        return {optimal_clique, CERTIFICATE::NONE};
    }
    
    int num_parts = 4;
    std::vector<idx_t> partition(num_vertices, 0);
    idx_t objval;
    idx_t ncon = 1;
    int status = METIS_PartGraphKway(&num_vertices, 
                                     &ncon, 
                                     xadj.data(), adjncy.data(), 
                                     nullptr, nullptr, nullptr, 
                                     &num_parts, 
                                     nullptr, nullptr, 
                                     nullptr, 
                                     &objval, 
                                     partition.data());

    if (status != METIS_OK) {
        throw std::runtime_error("METIS partitioning failed!");
    }
    
    std::vector<Node> final_clique;
    for (int p = 0; p < num_parts; p++) {
        std::vector<Node> partition_nodes;
        for (size_t i = 0; i < optimal_clique.size(); ++i) {
            if (partition[i] == p) {
                partition_nodes.push_back(optimal_clique[i]);
            }
        }
        
        if (partition_nodes.size() > final_clique.size()) {
            final_clique = partition_nodes;
        }
    }
    
    auto certificate = CERTIFICATE::NONE;
    if (final_clique.size() == k_core_bound) {
        certificate = CERTIFICATE::CORE_BOUND;
    } else if (final_clique.size() == chromatic_welsh) {
        certificate = CERTIFICATE::CHROMATIC_BOUND;
    }

    return {final_clique, certificate};
}

}
