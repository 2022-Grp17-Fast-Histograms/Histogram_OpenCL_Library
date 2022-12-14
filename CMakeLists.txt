cmake_minimum_required(VERSION 3.0.0)
project(Histogram_OpenCL_Library VERSION 0.1.0)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Select output folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin")

# Add sources
add_executable(histogram_driver ${PROJECT_SOURCE_DIR}/src/histogram_driver.cpp ${PROJECT_SOURCE_DIR}/src/histogram.cpp)

# Add include and lib dependencies
target_include_directories(histogram_driver PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(histogram_driver PRIVATE ${PROJECT_SOURCE_DIR}/lib/OpenCL.lib)

# Copy kernel file to the output folder
if(NVIDIA)
	target_compile_definitions(histogram_driver PRIVATE NVIDIA)
    configure_file(${PROJECT_SOURCE_DIR}/src/histogram_kernel_nvidia.cl ${PROJECT_SOURCE_DIR}/bin/histogram_kernel.cl COPYONLY)
else()
    configure_file(${PROJECT_SOURCE_DIR}/src/histogram_kernel_intel.cl ${PROJECT_SOURCE_DIR}/bin/histogram_kernel.cl COPYONLY)
endif()

include(CPack)