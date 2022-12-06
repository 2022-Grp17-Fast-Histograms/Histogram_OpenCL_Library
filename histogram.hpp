#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <utility>
#include <CL/opencl.hpp>

class Histogram {
    public:
    Histogram();
    Histogram(int imgWidth, int imgHeight, int blockWidth, int blockHeight, int numOfBins, bool showErrors);
    void setupEnvironment();
    void printEnvironment();
    void writeInputBuffers(std::vector<int> imageVector);
    void writeInputBuffers(const void *ptr);
    
    void calculateHistograms(bool detailed);
    std::vector<float> getAverage(std::string channel);
    std::vector<float> getVariance(std::string channel);
    std::vector<int> getAverageHistogram(std::string channel);
    std::vector<float> getVarianceHistogram(std::string channel);
    double getElapsedTime(std::string channel);

    private:
    void calculateSizes();
    void createInputBuffers();
    void createOutputBuffers();
    int adjustDimension(int dimension, int blockDimension);

    // Control
    bool environmentSetUp;

    // Image and Block
    int imgWidth;
    int imgHeight;
    int blockWidth;
    int blockHeight;
    int numOfBins;

    // Channel Details
    int ySize;
    int uSize;
    int vSize;

    int yBlockWidth;
    int yBlockHeight;
    int yBlockSize;
    int yNumOfBlocks;

    int uBlockWidth;
    int uBlockHeight;
    int uBlockSize;
    int uNumOfBlocks;

    int vBlockWidth; 
    int vBlockHeight;
    int vBlockSize;
    int vNumOfBlocks;

    // Error
    bool showErrors;
    int clError;
    int histError;

    // Platform Devices Queue
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device defaultDevice;
    cl::Program program;
    cl::Context context;
    cl::CommandQueue commandQueue;

    // Kernels
    cl::Kernel histogramsKernel;
    cl::Kernel histogramsDetailKernel;

    // Ranges
    cl::NDRange yGlobalRange;
    cl::NDRange uGlobalRange;
    cl::NDRange vGlobalRange;
    cl::NDRange yLocalRange;
    cl::NDRange uLocalRange;
    cl::NDRange vLocalRange;
    cl::Event event;

    // Input Buffers
    cl::Buffer yImageBuffer;
    cl::Buffer uImageBuffer;
    cl::Buffer vImageBuffer;
    cl::Buffer numOfBinsBuffer;

    // Output Buffers
    cl::Buffer yAverageBuffer;
    cl::Buffer uAverageBuffer;
    cl::Buffer vAverageBuffer;
    cl::Buffer yVarianceBuffer;
    cl::Buffer uVarianceBuffer;
    cl::Buffer vVarianceBuffer;
    cl::Buffer yAverageHistBuffer;
    cl::Buffer uAverageHistBuffer;
    cl::Buffer vAverageHistBuffer;
    cl::Buffer yVarianceHistBuffer;
    cl::Buffer uVarianceHistBuffer;
    cl::Buffer vVarianceHistBuffer;

    // Output Vectors
    std::vector<float> yAverage;
    std::vector<float> uAverage;
    std::vector<float> vAverage;
    std::vector<float> yVariance;
    std::vector<float> uVariance;
    std::vector<float> vVariance;
    std::vector<int> yAverageBins;
    std::vector<int> uAverageBins;
    std::vector<int> vAverageBins;
    std::vector<float> yVarianceBins;
    std::vector<float> uVarianceBins;
    std::vector<float> vVarianceBins;

    // Timers
    double yElapsedTime;
    double uElapsedTime;
    double vElapsedTime;
};