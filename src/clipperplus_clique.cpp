/*
computes a maximal clique in graph, and certifies if it's maximum clique

Author: kaveh fathian (kavehfathian@gmail.com)
 */

#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"

namespace clipperplus 
{


std::pair<std::vector<Node>, CERTIFICATE> find_clique_dist(const Graph &local_graph, 
                                                                 const std::unordered_map<Node, int> &global_to_local, 
                                                                 MPI_Comm comm)
{   

    int rank;
    MPI_Comm_rank(comm, &rank);	
    int local_n = local_graph.size();

    auto chromatic_welsh = estimate_chormatic_number_welsh_powell(local_graph);
    auto k_core_bound = local_graph.max_core_number() + 1;

    auto heuristic_clique = find_heuristic_clique(local_graph);    

    int local_clique_size = heuristic_clique.size();
    int local_chromatic_welsh = chromatic_welsh;
    int local_kcore_bound = k_core_bound;
    
    int global_max_clique_size;
    int global_chromatic_welsh;
    int global_kcore_bound;    
    MPI_Allreduce(&local_clique_size, &global_max_clique_size, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_chromatic_welsh, &global_chromatic_welsh, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_kcore_bound, &global_kcore_bound, 1, MPI_INT, MPI_MAX, comm);

    if (local_clique_size == std::min({global_kcore_bound, global_chromatic_welsh})) {
    return {heuristic_clique, CERTIFICATE::HEURISTIC};
    }
    //if(heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh})) {
      //  return {heuristic_clique, CERTIFICATE::HEURISTIC};
    //}

    std::vector<int> core_number = local_graph.get_core_numbers();
    std::vector<int> keep, keep_pos(local_n, -1);
    for(Node i = 0, j = 0; i < local_n; ++i) {
        if(core_number[i] + 1 >= heuristic_clique.size()) {
            keep.push_back(i);
            keep_pos[i] = j++;
        }
    }

    Eigen::MatrixXd M_pruned = local_graph.get_adj_matrix()(keep, keep);
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
    
    std::vector<int> clique_sizes;
    int local_size = optimal_clique.size();
    MPI_Gather(&local_size, 1, MPI_INT, clique_sizes.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
        int max_clique_size = *std::max_element(clique_sizes.begin(), clique_sizes.end());
        std::cout << "Largest Clique Size: " << max_clique_size << std::endl;
    }



    return {optimal_clique, certificate};
}

} 
