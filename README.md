# Histogram OpenCL Library

This project is a library done with OpenCL framework with the purpose of generating fast histograms from a given raw image data in YUV or NV12 formats.

The library generates histograms for the average and variance of pixel blocks according to the given configurations.

The library only needs the pointers to the raw image data.

The project also includes a driver example that loads a YUV image, perform the calculation of the histograms both on CPU and GPU (using the library), and then validates the results.

## Project Files

- histogram.hpp: Header file for the library
- histogram.cpp: Source file for the library
- histogram_kernel.cl: Kernel file for the library
- histogram_driver.hpp: Header file for the driver example
- histogram_driver.cpp: Source file for the driver example

## Building and running driver example

The project uses CMake as a build system for the driver example.

Below are instructions to build it and run it from the project root folder.

```
cmake -B build
cmake --build ./Build --target histogram --config release
cd bin
histogram.exe
```
