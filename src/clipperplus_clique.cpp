#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"


namespace clipperplus 
{


std::pair<std::vector<Node>, CERTIFICATE> parallel_find_clique(const Graph &graph)
{
    int num_vertices = graph.size();
    int num_parts = 4;
    std::vector<idx_t> partition(num_vertices, 0);
    idx_t objval;
    idx_t ncon = 1;
    
    
    std::vector<idx_t> xadj, adjncy;

    //create csr matrix
    xadj.resize(num_vertices + 1, 0);
    
    for (int i = 0; i < num_vertices; ++i) {
        const auto &neighbors = graph.neighbors(i);
        xadj[i + 1] = xadj[i] + neighbors.size();
        adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
    }
    
    // Partition the graph 
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
        throw std::runtime_error("METIS partitioning failed");
    }
    
    // Create subgraphs 
    std::vector<Graph> subgraphs(num_parts);
    std::vector<std::vector<Node>> partition_nodes(num_parts);
    
    for (int i = 0; i < num_vertices; ++i) {
        partition_nodes[partition[i]].push_back(i);
    }
    
    for (int i = 0; i < num_parts; ++i) {
        subgraphs[i] = graph.induced(partition_nodes[i]);
    }
    
    std::vector<std::pair<std::vector<Node>, CERTIFICATE>> results(num_parts);
    
    for (int i = 0; i < num_parts; ++i) {
        results[i] = find_clique(subgraphs[i]);
    }
    
    // Find the largest clique among the parts
    auto best_result = *std::max_element(results.begin(), results.end(), 
        [](const auto &a, const auto &b) {
            return a.first.size() < b.first.size();
        });
    
    return best_result;
}


std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph)
{
    int n = graph.size();

    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
    auto k_core_bound = graph.max_core_number() + 1;

    auto heuristic_clique = find_heuristic_clique(graph);    
    if(heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
        return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }

    std::vector<int> core_number = graph.get_core_numbers();
    std::vector<int> keep, keep_pos(n, -1);
    for(Node i = 0, j = 0; i < n; ++i) {
        if(core_number[i] + 1 >= heuristic_clique.size()) {
            keep.push_back(i);
            keep_pos[i] = j++;
        }
    }

    Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
    M_pruned.diagonal().setOnes();

    Eigen::VectorXd u0 = Eigen::VectorXd::Ones(keep.size());

    for(auto v : heuristic_clique) {
        assert(keep_pos[v] >= 0);
        u0(keep_pos[v]) = 0;
    }
    u0.normalize();

    auto clique_optim_pruned = clipperplus::clique_optimization(M_pruned, u0, Params());
    std::vector<Node> optimal_clique;
    if(clique_optim_pruned.size() < heuristic_clique.size()) {
        optimal_clique = heuristic_clique;
    } else {
        for(auto v : clique_optim_pruned) {
            assert(v >= 0 && v < keep.size());
            optimal_clique.push_back(keep[v]);
        }
    }


    auto certificate = CERTIFICATE::NONE;
    if(optimal_clique.size() == k_core_bound) {
        certificate = CERTIFICATE::CORE_BOUND;
    } else if(optimal_clique.size() == chromatic_welsh) {
        certificate = CERTIFICATE::CHROMATIC_BOUND;
    }

    return {optimal_clique, certificate};
}
}
