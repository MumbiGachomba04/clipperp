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
    idx_t options[METIS_NOPTIONS];  // Define the options array
    METIS_SetDefaultOptions(options);  // Initialize it with default values
    options[METIS_OPTION_UFACTOR] = 500;
    // Partition the graph 
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