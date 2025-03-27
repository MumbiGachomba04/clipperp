#pragma once

#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <mpi.h>
#include <metis.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

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

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph);
std::pair<std::vector<Node>, CERTIFICATE> parallel_find_clique(const Graph &graph);

} 
