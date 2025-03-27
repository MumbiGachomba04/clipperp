#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"




namespace clipperplus 
{


std::pair<std::vector<Node>, CERTIFICATE> parallel_find_clique(const Graph &graph)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_vertices = graph.size();
    int num_parts = size; 
    std::vector<idx_t> partition(num_vertices, 0);
    idx_t objval;
    idx_t ncon = 1;

    std::vector<std::pair<std::vector<Node>, CERTIFICATE>> local_result(1);

    if (rank == 0) {

        // Convert graph to CSR
        std::vector<idx_t> xadj(num_vertices + 1, 0);
        std::vector<idx_t> adjncy;
        for (int i = 0; i < num_vertices; ++i) {
            const auto &neighbors = graph.neighbors(i);
            xadj[i + 1] = xadj[i] + neighbors.size();
            adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
        }

        // METIS partitioning
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_UFACTOR] = 500;
        int status = METIS_PartGraphKway(&num_vertices,
                                         &ncon,
                                         xadj.data(), adjncy.data(),
                                         nullptr, nullptr, nullptr,
                                         &num_parts,
                                         nullptr, nullptr,
                                         options,
                                         &objval,
                                         partition.data());

        if (status != METIS_OK) {
            throw std::runtime_error("METIS partitioning failed");
        }

        // Prepare subgraphs & send to other procs
        for (int i = 0; i < num_parts; ++i) {
            std::vector<Node> nodes;
            for (int j = 0; j < num_vertices; ++j) {
                if (partition[j] == i) {
                    nodes.push_back(j);
                }
            }
            Graph subgraph = graph.induced(nodes);

            if (i == 0) {
                local_result[0] = find_clique(subgraph);
            } else {
               auto buffer = subgraph.graph_to_vector();
               int size = buffer.size();
               MPI_Send(&size, 1, MPI_INT, i, 444, MPI_COMM_WORLD);
               MPI_Send(buffer.data(), size, MPI_INT, i, 444, MPI_COMM_WORLD);
            }
        }
    } else {
       
        Graph subgraph;
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> buffer(size);
        MPI_Recv(buffer.data(), size, MPI_INT, 0 , 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        subgraph = subgraph.vector_to_graph(buffer);
        local_result[0] = find_clique(subgraph);
    }

    
    // get results ( only clique size  &  certificate)
    struct CliqueInfo {
        int clique_size;
        CERTIFICATE certificate;
    } local_info, best_info;

    if (rank == 0) {
        local_info.clique_size = local_result[0].first.size();
        local_info.certificate = local_result[0].second;
    } else {
        local_info.clique_size = local_result[0].first.size();
        local_info.certificate = local_result[0].second;
    }

    // find max clique size
    int best_size;
    MPI_Allreduce(&local_info.clique_size, &best_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Master collects best result
    if (rank == 0) {
        std::pair<std::vector<Node>, CERTIFICATE> best_result = local_result[0];

        for (int i = 1; i < size; ++i) {
            int recv_size;
            
            MPI_Recv(&recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int recv_cert;
            MPI_Recv(&recv_cert, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CERTIFICATE certificate = static_cast<CERTIFICATE>(recv_cert);

            if (recv_size == best_size) {
                // Optionally receive full clique
                std::vector<Node> clique(recv_size);
                MPI_Recv(clique.data(), recv_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                best_result = std::make_pair(clique, certificate);
            }
        }


        //return best_result;
    } else {
        // Workers send result to master
        MPI_Send(&local_info.clique_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int cert = static_cast<int>(local_info.certificate);
        MPI_Send(&cert, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (local_info.clique_size == best_size) {
            MPI_Send(local_result[0].first.data(), local_info.clique_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
       
    }
    int best_cert;
    if (rank == 0) {
        best_cert = static_cast<int>(best_result.second);
    }
    MPI_Bcast(&best_cert, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CERTIFICATE final_certificate = static_cast<CERTIFICATE>(best_cert);

    // Broadcast the clique nodes
    std::vector<Node> best_clique;
    if (rank == 0) {
        best_clique = best_result.first;
    }

    best_clique.resize(best_size);
    MPI_Bcast(best_clique.data(), best_size, MPI_INT, 0, MPI_COMM_WORLD);

    return {best_clique, final_certificate};

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
