#pragma once

#include <parmetis.h>
#include <mpi.h>
#include <vector>
#include <Eigen/Dense>

namespace clipperplus{

void partitionGraphParMETIS(const Eigen::MatrixXd &adj_matrix, int num_procs, std::vector<int> &partition);

}

