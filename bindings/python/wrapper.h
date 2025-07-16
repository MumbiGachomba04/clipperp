/**
 * @file wrappers.h
 * @brief Wrapper for Clipperplus' C++ functions with parameters passed by reference
 */

#pragma once

#include <pybind11/pybind11.h>
#include "clipperplus/clipperplus_clique.h"
#include "clipperplus/clique_optimization.h"

class Wrapper {
  public:
    static std::tuple<long, std::vector<int>, int> clipperplus_clique_wrapper(const Eigen::MatrixXd& adj){
      int size;
      bool partitioning = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      std::vector<clipperplus::Node> clique;
      clipperplus::CERTIFICATE certificate;
      std::pair<std::vector<clipperplus::Node>, clipperplus::CERTIFICATE> result;
      MPI_Barrier(MPI_COMM_WORLD);
      double start_time = MPI_Wtime();
      if (size > 1){
      result = clipperplus::parallel_find_clique(adj,partitioning);
      }
      else {
        result = clipperplus::find_clique(adj);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      double end_time = MPI_Wtime();
      clique = result.first;
      certificate = result.second;
      
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) {

      std::cout << "Elapsed time: " << end_time-start_time << " seconds" << std::endl;
      }
      return std::make_tuple((long)clique.size(), clique, (int)certificate);
    }

    static std::vector<int> find_heuristic_clique_wrapper(
      const Eigen::MatrixXd& adj, 
      std::vector<int>& clique
    ){
      clique = clipperplus::find_heuristic_clique(adj);
      return clique;
    }
    
    static std::tuple<int, unsigned long, std::vector<long>> clique_optimization_wrapper(
      const Eigen::MatrixXd& M, 
      const Eigen::VectorXd& u0
    ){
      std::vector<long> clique = clipperplus::clique_optimization(M, u0, clipperplus::Params());
      return std::make_tuple(1, clique.size(), clique);
    }

};
