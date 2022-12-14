# Histogram OpenCL Library

This project is a library done with OpenCL framework with the purpose of generating fast histograms from a given raw image data in YUV or NV12 formats.

The library generates histograms for the average and variance of pixel blocks according to the given configurations.

The library only needs the pointers to the raw image data.

The project also includes a driver example that loads a YUV image, perform the calculation of the histograms both on CPU and GPU (using the library), and then validates the results.

## Project Files

- histogram.hpp: Header file for the library
- histogram.cpp: Source file for the library
- histogram_kernel_intel.cl: Kernel file for the library for Intel GPU
- histogram_kernel_nvidia.cl: Kernel file for the library for NVidia GPU
- histogram_driver.hpp: Header file for the driver example
- histogram_driver.cpp: Source file for the driver example

## Building and running driver example

The project uses CMake as a build system for the driver example.

Below are instructions to build it and run it from the project root folder.

For Intel GPU:

```
cmake -B build
cmake --build ./build --target histogram_driver --config release
cd bin
histogram_driver.exe
```

For NVidia GPU:

```
cmake -D NVIDIA=true -B build
cmake --build ./build --target histogram_driver --config release
cd bin
histogram_driver.exe
```

The CMake build will make a copy of the correct kernel file to the bin directory according to the GPU selected.

The driver needs to be executed from the same directory as histogram_kernel.cl in order to run properly.
