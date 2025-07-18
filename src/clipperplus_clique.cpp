#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"
#include <unordered_set>
#include <random>
#include <chrono>
#include <iostream>




namespace clipperplus 
{

std::pair<std::vector<Node>, CERTIFICATE> parallel_find_clique(const Graph &graph, int partitioning)
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
        std::cout<< "GRAPH SIZE : " << num_vertices << std::endl;
        std::cout<< "USE METIS?? : " << partitioning << std::endl;
        std::vector<idx_t> vwgt(num_vertices); // helps METIS avoid splitting densely connected cores, improving the chances that cliques remain intact within partitions.
        for (int i = 0; i < num_vertices; ++i) {
            const auto &neighbors = graph.neighbors(i);
            vwgt[i] = neighbors.size();
            xadj[i + 1] = xadj[i] + vwgt[i];
            adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
        }
  double st, en;
 
    if (partitioning == 1) {
// ---------------------METIS PARTITIONING---------------------------
     st= MPI_Wtime(); 
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_UFACTOR] = 500; 
        options[METIS_OPTION_SEED] = 42;

        int status = METIS_PartGraphKway(&num_vertices, &ncon, xadj.data(), adjncy.data(),
                                         vwgt.data(), nullptr, nullptr, &num_parts, 
                                         nullptr, nullptr, options, &objval, partition.data());
        // int status = METIS_PartGraphRecursive(&num_vertices, &ncon, xadj.data(), adjncy.data(),
        //                                  nullptr, nullptr, nullptr, &num_parts, 
        //                                  nullptr, nullptr, options, &objval, partition.data());
           std::cout << "OBJVAL: " << objval << std::endl;
        if (status != METIS_OK) {
            throw std::runtime_error("METIS partitioning failed");
        }
    en = MPI_Wtime(); 
    std::cout<< "Metis partitioning took ---------------- " << en-st << " seconds" << std::endl; 
// // ---------------------END OF METIS  PARTITIONING----------------
    }


else {
// ---------------------MANUAL PARTITIONING----------------
     st= MPI_Wtime(); 
        int local_size=num_vertices/num_parts;
        int rem = num_vertices%num_parts;
        int count= 1;
        int part_count=0;
        
        for (int i = 0; i < num_vertices; ++i) {
            if (count%local_size == 0){     
             part_count++;
             if (part_count==num_parts){ part_count=num_parts-1;}
            }
            partition[i] = part_count;

            count++;
        }
    en = MPI_Wtime();
        std::cout<< "Natural partitioning took " << en-st << " seconds" << std::endl; 
// ---------------------END OF MANUAL PARTITIONING-------------------
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

        int top_nodes;
        if (enable_overlap == true)
        {
            switch (overlap_mode)
            {
                case clipperplus::OverlapMode::GENERAL:
                {
                    // --------------------- GENERAL OVERLAPPING---------------------------
                    std::vector<int> degrees = graph.degrees();
                    int N = std::max(1, static_cast<int>(overlap_ratio * num_vertices));  // overlap_ratio top nodes

                    std::vector<int> top_degree_nodes;
                    for (int i = 0; i < N; ++i) {
                        auto max_it = std::max_element(degrees.begin(), degrees.end());
                        int max_idx = std::distance(degrees.begin(), max_it);
                        top_degree_nodes.push_back(max_idx);
                        degrees[max_idx] = -1; // mark as used
                    }

                    std::cout << "\n=== Rank: " << rank << " Top-" << N << " degree nodes (copied) ===\n";
                    for (int v : top_degree_nodes) {
                        int deg = graph.degree(v);
                        int original_partition = partition[v];

                        std::cout << "Node: " << v
                            << " | Degree: " << deg
                            << " | Original Part: " << original_partition
                            << " | Current Rank: " << rank
                            << std::endl;
                    }
                    std::cout << "==============================================" << std::endl;

                    for (int v : top_degree_nodes) {
                        if (std::find(local_nodes.begin(), local_nodes.end(), v) == local_nodes.end()) {
                            local_nodes.push_back(v);
                        }
                    }
                }
                    break;
                case clipperplus::OverlapMode::NEIGHBOR:
                {
                    // --------------------- NEIGHBORHOOD OVERLAPPING ----------------
                    // Set to store candidate overlap nodes:
                    // i.e., neighbors of local nodes that belong to other partitions
                    std::unordered_set<Node> candidate_overlap;
                    auto s1 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    for (Node v : local_nodes) {
                        for (Node n : graph.neighbors(v)) {
                            if (partition[n] != rank) {
                                // If the neighbor `n` belongs to a different partition,
                                // then it is a candidate for overlap
                                candidate_overlap.insert(n);
                            }
                        }
                    }

                    // Vector to store each candidate node along with its degree (number of neighbors)
                    std::vector<std::pair<Node, int>> overlap_with_degrees;

                    for (Node n : candidate_overlap) {
                        int degree = graph.neighbors(n).size();
                        // emplace_back constructs the pair directly in-place in the vector for performance
                        overlap_with_degrees.emplace_back(n, degree);
                    }

                    auto e1 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    elapsed = e1 - s1;
                    std::cout << "Rank: " << rank << " | Selection of related neighbor nodes: " << elapsed << " s" << std::endl;


                    auto s2 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    // Sort the candidate nodes in descending order by degree
                    // Higher-degree nodes are more likely to be part of the global clique
                    std::sort(overlap_with_degrees.begin(), overlap_with_degrees.end(),
                        [](const std::pair<Node, int>& a, const std::pair<Node, int>& b) {
                            return a.second > b.second;
                        });

                    auto e2 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    elapsed = e2 - s2;
                    std::cout << "Rank: " << rank << " | Reordering selected nodes by degrees: " << elapsed << " s" << std::endl;

                    // Select the top 10% of candidate nodes based on degree (at least 1 node)
                    top_nodes = std::max(1, static_cast<int>(overlap_with_degrees.size() * overlap_ratio));

                    auto s3 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    // Pick the top high-degree overlap candidates
                    for (int i = 0; (i < top_nodes) && (i < overlap_with_degrees.size()); ++i) {
                        local_nodes.push_back(overlap_with_degrees[i].first);
                    }
                    auto e3 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    elapsed = e3 - s3;
                    std::cout << "Rank: " << rank << " | Selection of nodes by overlap ratio: " << elapsed << " s" << std::endl;


                    auto e4 = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
                    elapsed = e4 - s1;
                    std::cout << "Rank: " << rank << " | END OF OVERLAPPING: " << elapsed << " s" << std::endl;
                }
                    break;
                default:
                    break;
            }
        }
        Graph local_graph = graph.induced(local_nodes);

        auto local_result = find_clique(local_graph);
        int local_clique_size = local_result.first.size();

        std::cout << "Rank: " << rank << " | Local nodes: " << local_nodes.size() << " | Overlap nodes: " << top_nodes << " | Local clique size: " << local_clique_size << std::endl;

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
        }
        else {
            MPI_Send(&local_clique_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            int cert = static_cast<int>(local_result.second);
            MPI_Send(&cert, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (local_clique_size == best_size) {
                MPI_Send(local_result.first.data(), local_clique_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }

        int best_cert;
        if (rank == 0) {
            best_cert = static_cast<int>(final_certificate);
        }
        MPI_Bcast(&best_cert, 1, MPI_INT, 0, MPI_COMM_WORLD);
        final_certificate = static_cast<CERTIFICATE>(best_cert);

        best_clique.resize(best_size);
        MPI_Bcast(best_clique.data(), best_size, MPI_INT, 0, MPI_COMM_WORLD);

        auto EALL = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
        elapsed = EALL - SALL;
        std::cout << "Rank: " << rank << " | FINAL TIME: " << elapsed << " s" << std::endl;

        return { best_clique, final_certificate };
    }

    std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph& graph)
    {
        int n = graph.size();

        auto chromatic_welsh = estimate_chormatic_number_welsh_powell(graph);
        auto k_core_bound = graph.max_core_number() + 1;

        auto heuristic_clique = find_heuristic_clique(graph);
        if (heuristic_clique.size() == std::min({ k_core_bound, chromatic_welsh })) {
            return { heuristic_clique, CERTIFICATE::HEURISTIC };
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
            assert(keep_pos[v] >= 0);
            u0(keep_pos[v]) = 0;
        }
        u0.normalize();

        auto clique_optim_pruned = clipperplus::clique_optimization(M_pruned, u0, Params());
        std::vector<Node> optimal_clique;
        if (clique_optim_pruned.size() < heuristic_clique.size()) {
            optimal_clique = heuristic_clique;
        }
        else {
            for (auto v : clique_optim_pruned) {
                assert(v >= 0 && v < keep.size());
                optimal_clique.push_back(keep[v]);
            }
        }

        auto certificate = CERTIFICATE::NONE;
        if (optimal_clique.size() == k_core_bound) {
            certificate = CERTIFICATE::CORE_BOUND;
        }
        else if (optimal_clique.size() == chromatic_welsh) {
            certificate = CERTIFICATE::CHROMATIC_BOUND;
        }

        return { optimal_clique, certificate };
    }
}