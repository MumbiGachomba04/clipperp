#include <iostream>
#include <sstream>
#include <fstream>
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
    std::ifstream infile(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        data.push_back(row);
    }

    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix(i, j) = data[i][j];

    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <adjacency_matrix_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    MPI_Init(NULL,NULL);
    int numproc;
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
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
        result  =  parallel_find_clique(G);
    }
    end = MPI_Wtime(); 
    std::cout<< "clique finding took " << end-start << " seconds" << std::endl; 
    std::vector<int> clique = result.first;
    clipperplus::CERTIFICATE cert = result.second;

    std::cout << endl<< endl << "Heuristic clique of size " << clique.size() << ": ";
    /*for (int v : clique) {
        std::cout << v << " ";
    }*/
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}
