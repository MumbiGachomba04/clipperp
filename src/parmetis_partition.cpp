#include "clipperplus/parmetis_partition.h"
#include <iostream>



namespace clipperplus {

void partitionGraphParMETIS(const Eigen::MatrixXd &adj_matrix, int num_procs, std::vector<int> &partition) {
   

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm = MPI_COMM_WORLD;	
    int num_vertices = adj_matrix.rows();
    partition.resize(num_vertices);

    std::vector<int> xadj, adjncy, adjwgt;
    int edge_count = 0;
    xadj.push_back(0);

    for (int i = 0; i < num_vertices; i++)
    {
        for (int j = 0; j < num_vertices; j++)
        {
            if (adj_matrix(i, j) != 0)
            {
                adjncy.push_back(j);
                adjwgt.push_back(1);
                edge_count++;
            }
        }
        xadj.push_back(edge_count);
    }

    int wgtflag = 0, numflag = 0, ncon = 1, options[3] = {1, 0, 0}, edgecut;

    ParMETIS_V3_PartKway(
        &num_vertices, xadj.data(), adjncy.data(), NULL, NULL,
        &wgtflag, &numflag, &ncon, &num_procs, NULL,
        NULL, NULL, &edgecut, partition.data(), &comm
    );
}
}
