#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "clipperplus/clipperplus_graph.h"
#include "clipperplus/clipperplus_clique.h"
#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clique_optimization.h"

using namespace std;



Eigen::SparseMatrix<double> read_sparse_adjacency_matrix(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;

    // Read first line: size of the matrix
    int nrows = 0;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> nrows;
    }

    std::vector<Eigen::Triplet<double>> triplets;

    // Read edges
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int i, j;
        if (!(iss >> i >> j)) continue;

        triplets.emplace_back(i, j, 1.0);
        if (i != j) triplets.emplace_back(j, i, 1.0); // make symmetric
    }

    Eigen::SparseMatrix<double> mat(nrows, nrows);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}


// Reads an adjacency matrix from a text file into Eigen::MatrixXd
Eigen::MatrixXd read_adjacency_matrix(const std::string& filename) {
   std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open " + filename);

    std::vector<std::vector<double>> rows;
    std::string                      line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        rows.emplace_back(std::istream_iterator<double>{iss},
                          std::istream_iterator<double>{});
    }
    const std::size_t n = rows.size();
    if (n == 0) return {};

    Eigen::MatrixXd mat(n, n);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) mat(r, c) = rows[r][c];
    return mat;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <adjacency_matrix_file>  <0 for natural partitioning or 1 for using metis>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    bool use_metis =  std::stoi(argv[2]) != 0;
    MPI_Init(NULL,NULL);
    int numproc,rank;
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // Load adjacency matrix
    Eigen::MatrixXd adj = read_adjacency_matrix(filename);

    // Eigen::SparseMatrix<double> sparse_adj = read_sparse_adjacency_matrix(argv[1]);
    // Eigen::MatrixXd adj = Eigen::MatrixXd(sparse_adj);  // if clipperplus::Graph expects dense

    double start, end; 
    // Construct graph
    clipperplus::Graph G(adj);
    std::pair<std::vector<int>, clipperplus::CERTIFICATE> result;
    start= MPI_Wtime(); 
    if(numproc == 1) {
	   result  = find_clique(G);
    } 
    else {
        result  =  parallel_find_clique(G,use_metis);
    }
    end = MPI_Wtime(); 
    if (rank == 0)  {
    std::cout<< "clique finding took " << end-start << " seconds" << std::endl; 
    std::vector<int> clique = result.first;
    clipperplus::CERTIFICATE cert = result.second;

    std::cout << endl<< endl << "Heuristic clique of size " << clique.size() << std::endl;
    }
 

    MPI_Finalize();
    return 0;
}
