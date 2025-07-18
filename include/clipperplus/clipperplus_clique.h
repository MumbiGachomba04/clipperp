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

enum class OverlapMode {
    GENERAL = 0,
    NEIGHBOR = 1
};

enum class PartitioningMode {
    MANUAL = 0,
    METIS = 1
};

extern PartitioningMode partitioning_mode;
extern double uFactor;
extern double seedValue;
extern bool enable_recursive;

extern bool enable_overlap;
extern OverlapMode overlap_mode;
extern double overlap_ratio;

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph);
std::pair<std::vector<Node>, CERTIFICATE> parallel_find_clique(const Graph &graph,int partitioning);

} 
