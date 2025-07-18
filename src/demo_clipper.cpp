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
using namespace clipperplus;


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

    // ---------------------------------------------
    // Command line argument parser:
    /*  -p=0 : MANUAL partitioning mode
        -p=1 : METIS partitioning mode
        -r : Enable recursive metis
        -u=<value> : Set uFactor value
        -s=<value> : Set random seed value

        -o : Enable overlapping -> Default -n
        -g : Set overlap mode to GENERAL
        -n : Set overlap mode to NEIGHBOR
        -r=<value> : Set overlap ratio, e.g. -r=0.05 for %5
    */
    // ---------------------------------------------

    for (int i = 2; i < argc; ++i) 
    {
        std::string arg = argv[i];

        if (arg.rfind("-p=", 0) == 0 && arg.size() > 3 && isdigit(arg[3]))
        {
            int partitioningMode = arg[3] - '0';

            if (partitioningMode == 1) // 1 for metis partitioning 
            {
                partitioning_mode = PartitioningMode::METIS;
            }
            else
            {
                partitioning_mode = PartitioningMode::MANUAL;
            }
        }
        else if (arg.rfind("-p=", 0) == 0)
        {
            std::cerr << "Invalid partitioning mode format in: " << arg << std::endl;
            return 1;
        }
        else if (arg.rfind("-u=", 0) == 0) {
            try {
                uFactor = std::stod(arg.substr(3));
            }
            catch (...) {
                std::cerr << "Invalid uFactor format in: " << arg << std::endl;
                return 1;
            }
        }
        else if (arg.rfind("-s=", 0) == 0) {
            try {
                seedValue = std::stod(arg.substr(3));
            }
            catch (...) {
                std::cerr << "Invalid seed format in: " << arg << std::endl;
                return 1;
            }
        }
        else if (arg == "-r") 
        {
            enable_recursive = true;
        }
        else if (arg == "-o") 
        {
            enable_overlap = true;
            overlap_mode = OverlapMode::NEIGHBOR; // default
        }
        else if (arg == "-g") 
        {
            overlap_mode = OverlapMode::GENERAL;
        }
        else if (arg == "-n") 
        {
            overlap_mode = OverlapMode::NEIGHBOR;
        }
        else if (arg.rfind("-r=", 0) == 0) 
        {
            try 
            {
                overlap_ratio = std::stod(arg.substr(3));
            }
            catch (...) {
                std::cerr << "Invalid overlap ratio format in: " << arg << std::endl;
                return 1;
            }
        }
        else 
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    std::string filename = argv[1];
    int use_metis =  0;

    
    MPI_Init(NULL,NULL);
    int numproc,rank;
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (rank == 0)
    {
        use_metis= std::stoi(argv[2]) ;
    }
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

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime(); 

    if (rank == 0)  
    {
        std::cout<< "clique finding took " << end-start << " seconds" << std::endl; 
        std::vector<int> clique = result.first;
        clipperplus::CERTIFICATE cert = result.second;

        std::cout << endl<< endl << "Heuristic clique of size " << clique.size() << std::endl;
    }
 

    MPI_Finalize();
    return 0;
}
