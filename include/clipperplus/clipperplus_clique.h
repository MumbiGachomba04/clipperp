#pragma once

#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <mpi.h>

#include "clipperplus/clique_optimization.h"
#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/utils.h"


namespace clipperplus 
{

enum class CERTIFICATE 
{
    NONE,
    HEURISTIC,
    CORE_BOUND,
    CHROMATIC_BOUND
};

std::pair<std::vector<Node>, CERTIFICATE> find_clique_dist(const Graph &local_graph, 
                                                                 const std::unordered_map<Node, int> &global_to_local, 
                                                                 MPI_Comm comm);

} 
