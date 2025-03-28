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
    int num_parts = size;  // Each process handles one partition
    std::vector<idx_t> partition(num_vertices, 0);
    idx_t objval;
    idx_t ncon = 1;

    std::vector<idx_t> xadj(num_vertices + 1, 0);
    std::vector<idx_t> adjncy;

    // rank 0 partitions the graph
    if (rank == 0) {
        for (int i = 0; i < num_vertices; ++i) {
            const auto &neighbors = graph.neighbors(i);
            xadj[i + 1] = xadj[i] + neighbors.size();
            adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
        }

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_UFACTOR] = 500; 

        int status = METIS_PartGraphKway(&num_vertices, &ncon, xadj.data(), adjncy.data(),
                                         nullptr, nullptr, nullptr, &num_parts, 
                                         nullptr, nullptr, options, &objval, partition.data());

        if (status != METIS_OK) {
            throw std::runtime_error("METIS partitioning failed");
        }
    }

    // Broadcast partition to all processes . no need to send csr
    MPI_Bcast(partition.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process determines its subset of nodes
    std::vector<Node> local_nodes;
    for (int i = 0; i < num_vertices; ++i) {
        if (partition[i] == rank) {
            local_nodes.push_back(i);
        }
    }

    //  each process creates subgraph
    Graph local_graph = graph.induced(local_nodes);

    // Each process finds the max clique in its subgraph
    auto local_result = find_clique(local_graph);
    int local_clique_size = local_result.first.size();

    std::cout << "Rank: " << rank << "Local clique size: " << local_clique_size << std::endl;

    // Gather all clique sizes at rank 0 to find the max
    int best_size;
    MPI_Allreduce(&local_clique_size, &best_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<Node> best_clique;
    CERTIFICATE final_certificate;

    if (rank == 0) {
        best_clique = local_result.first;
        final_certificate = local_result.second;

        for (int i = 1; i < size; ++i) {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int recv_cert;
            MPI_Recv(&recv_cert, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CERTIFICATE certificate = static_cast<CERTIFICATE>(recv_cert);

            if (recv_size == best_size) {
                std::vector<Node> clique(recv_size);
                MPI_Recv(clique.data(), recv_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                best_clique = clique;
                final_certificate = certificate;
            }
        }
    } else {
        MPI_Send(&local_clique_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int cert = static_cast<int>(local_result.second);
        MPI_Send(&cert, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (local_clique_size == best_size) {
            MPI_Send(local_result.first.data(), local_clique_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Broadcast final best clique and certificate to all processes
    int best_cert;
    if (rank == 0) {
        best_cert = static_cast<int>(final_certificate);
    }
    MPI_Bcast(&best_cert, 1, MPI_INT, 0, MPI_COMM_WORLD);
    final_certificate = static_cast<CERTIFICATE>(best_cert);

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
