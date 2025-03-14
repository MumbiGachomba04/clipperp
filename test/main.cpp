#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv); 
  std::cout<< "MPI START" << std::endl;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  MPI_Finalize();
}