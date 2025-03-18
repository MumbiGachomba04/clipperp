#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"


namespace clipperplus 
{

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph)
{
    int n = graph.size();
    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
    auto k_core_bound = graph.max_core_number() + 1;
    std::cout << "step 1" << std::endl;
    
    std::vector<Node> heuristic_clique = find_heuristic_clique(graph);
    if (heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
        return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }
    std::cout << "step 2" << std::endl;
    
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
    std::cout << "step 3" << std::endl;
    
    Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
    M_pruned.diagonal().setOnes();

    
    
    std::vector<idx_t> xadj(keep.size() + 1, 0), adjncy;
    idx_t num_vertices = keep.size();

    for (idx_t i = 0; i < num_vertices; ++i) {
        for (idx_t j = 0; j < num_vertices; ++j) {
            if (M_pruned(i, j) > 0) {
                adjncy.push_back(j);
            }
        }
        xadj[i + 1] = adjncy.size();
    }

    if (adjncy.empty()) {
        return {heuristic_clique, CERTIFICATE::NONE};
    }
    std::cout << "step 4" << std::endl;

    
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

    std::cout << "step 5" << std::endl;
    
    std::vector<Node> max_clique;
    for (int p = 0; p < num_parts; p++) {
        std::vector<Node> partition_nodes;
        for (size_t i = 0; i < keep.size(); ++i) {
            if (partition[i] == p) {
                partition_nodes.push_back(keep[i]);
            }
        }

        if (partition_nodes.empty()) {
            continue;
        }

        Graph subgraph = graph.induced(partition_nodes);
        Eigen::MatrixXd local_M = subgraph.get_adj_matrix();
        Eigen::VectorXd u0 = Eigen::VectorXd::Random(partition_nodes.size());
        // for(auto v : heuristic_clique) {
        //      assert(keep_pos[v] >= 0);
        //      u0(keep_pos[v]) = 0;
        // }
        //u0.normalize();
        u0 = (u0.array() - 0.5) * 2;  // Normalize to range [-1,1]
        u0.normalize();     

        std::vector<long> long_clique = clipperplus::clique_optimization(local_M, u0, Params());
        std::vector<Node> local_clique;
        for (long idx : long_clique) {
            local_clique.push_back(partition_nodes[idx]);  // Map back to global indices
        }


        if (local_clique.size() > max_clique.size()) {
            max_clique = local_clique;
        }
    }
    std::cout << "step 6" << std::endl;
   
    auto certificate = CERTIFICATE::NONE;
    if (max_clique.size() == k_core_bound) {
        certificate = CERTIFICATE::CORE_BOUND;
    } else if (max_clique.size() == chromatic_welsh) {
        certificate = CERTIFICATE::CHROMATIC_BOUND;
    }

    return {max_clique, certificate};
}

} // namespace clipperplus
