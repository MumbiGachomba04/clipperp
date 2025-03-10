#pragma once

#include <Eigen/Dense>
#include <memory.h>
#include <vector>


namespace clipperplus
{

using Node = int;
using Edge = std::pair<Node, Node>;
using Neighborlist = std::vector<Node>;


class Graph
{
public:
    Graph() = default;
    Graph(Eigen::MatrixXd adj_matrix);
    // static Graph from_list(const std::vector<Neighborlist> &adj_list);

    int size() const;
    int degree(Node v) const;
    std::vector<int> degrees() const;

    const std::vector<Node> &neighbors(Node v) const;

    inline bool is_edge(Node u, Node v) const
    {
        return adj_matrix(u, v) != 0;
    }

    void merge(const Graph &g);
    Graph induced(const std::vector<Node> &nodes) const;

    int max_core_number() const;
    const std::vector<int> &get_core_numbers() const;
    const std::vector<Node> &get_core_ordering() const;
    const std::vector<Node> &get_local_to_global() const;
    const std::unordered_map<Node, int> &get_global_to_local() const;    
    const Eigen::MatrixXd &get_adj_matrix() const;
    
    void addGhostEdge(Node u, Node v, const std::vector<int> &part);
    void syncGhostNodes();

private:
    void calculate_kcores() const;

private:
    Eigen::MatrixXd adj_matrix;
    std::vector<Neighborlist> adj_list;

    mutable std::vector<Node> kcore_ordering;
    mutable std::vector<int> kcore;
    int num_procs;  
    int rank;       

    mutable std::vector<Node> local_to_global;                     
    mutable std::unordered_map<Node, int> global_to_local;         
    std::unordered_map<Node, int> ghost_nodes;
};


}
