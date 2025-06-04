cmake_minimum_required(VERSION 3.10)
project(clipperplus VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###############################################################################
# Options
###############################################################################

option(DEBUG_FLAG "Enable debugging output" ON)
option(DEBUG_TIMING_FLAG "Enable timing report" ON)
option(DEBUG_OPTIM_FLAG "Enable debugging clipper optimization" ON)

option(BUILD_PMC_HEU "Build PMC heuristic" ON)
option(BUILD_BINDINGS_MATLAB "Build MATLAB bindings" OFF)
option(BUILD_BINDINGS_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build testsuite" OFF)


###############################################################################
# Dependencies
###############################################################################

include(FetchContent)
cmake_policy(SET CMP0135 NEW)

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen Version: ${EIGEN3_VERSION_STRING} ${EIGEN3_VERSION}")

set(OpenMP_C_FLAGS "-qopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "gomp")
set(OpenMP_CXX_LIB_NAMES "gomp")
set(OpenMP_gomp_LIBRARY "/apps/GPP/GCC/13.2.0/lib/../lib64/libgomp.so")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)


set(METIS_DIR "/apps/GPP/EASYBUILD/software/METIS/5.1.0-GCCcore-13.2.0")
find_library(METIS_LIB metis PATHS ${METIS_DIR}/lib)
find_path(METIS_INCLUDE_DIR metis.h PATHS ${METIS_DIR}/include)
include_directories(${METIS_INCLUDE_DIR})

if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/e4fdb87e76b9fc4b01c54ad81aea19d6e994b994.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(BUILD_BINDINGS_PYTHON)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.9.2
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)


###############################################################################
# Targets
###############################################################################

set(CLLIPPER_PLUS_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/clipperplus_clique.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/clipperplus_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/clique_optimization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/clipperplus_heuristic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/demo_clipper.cpp
)

# source files
add_library(clipperplus SHARED)
target_sources(clipperplus PRIVATE ${CLLIPPER_PLUS_SOURCE_FILES})

# headers:
target_include_directories(clipperplus PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# link libraries:
target_link_libraries(clipperplus PUBLIC Eigen3::Eigen)
target_link_libraries(clipperplus PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(clipperplus PUBLIC MPI::MPI_CXX)
target_link_libraries(clipperplus PUBLIC ${METIS_LIB})
# debug flag  
if(DEBUG_FLAG)
        message(STATUS "Enabling debug flag")
        target_compile_definitions(clipperplus PRIVATE DEBUG)
endif()
if(DEBUG_OPTIM_FLAG)
        message(STATUS "Enabling debug optimization flag")
        target_compile_definitions(clipperplus PRIVATE DEBUG_OPTIM)
endif()
if(DEBUG_TIMING_FLAG)
        message(STATUS "Enabling debug timing flag")
        target_compile_definitions(clipperplus PRIVATE DEBUG_TIMING)
endif()

# version
set_target_properties(clipperplus PROPERTIES
  VERSION ${PROJECT_VERSION})
target_compile_definitions(clipperplus PUBLIC CLIPPERPLUS_VERSION="${PROJECT_VERSION}")


###############################################################################
# Extras
###############################################################################
if(BUILD_BINDINGS_MATLAB)
    message(STATUS "Attempting to build MATLAB bindings.")
    add_subdirectory(bindings/matlab)
endif()

if(BUILD_BINDINGS_PYTHON)
    message(STATUS "Building Python bindings.")
    add_subdirectory(bindings/python)
endif()


if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
