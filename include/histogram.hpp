#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <utility>
#include <CL/opencl.hpp>

#ifdef NVIDIA
    typedef float varhist;
#else
    typedef int varhist;
#endif

class Histogram {
    public:
    enum class Format {
        YUV,
        NV12
    };

    enum class Color {
        Chromatic,
        Grayscale
    };

    enum class Channel {
        Y,
        U,
        V
    };

    enum class Detail {
        Exclude,
        Include
    };

    enum class ErrorLevel {
        NoError,
        ShowError
    };

    Histogram();
    Histogram(Format format, Color color, int imgWidth, int imgHeight, int blockWidth, int blockHeight, int numOfBins);
    Histogram(const Histogram &o);
    ~Histogram();

    void setupEnvironment();
    void printEnvironment();
    void writeInputBuffers(std::vector<int> imageVector);
    void writeInputBuffers(const void *ptr);
    void setImageSize(int imgWidth, int imgHeight);
    void setBlockSize(int blockWidth, int blockHeight);
    void setNumofBins(int numOfBins);
    void setErrorLevel(ErrorLevel errorLevel);
    
    void calculateHistograms();
    void calculateHistograms(Detail detail);
    std::vector<float> getAverage(Channel channel);
    std::vector<float> getVariance(Channel channel);
    std::vector<int> getAverageHistogram(Channel channel);
    std::vector<varhist> getVarianceHistogram(Channel channel);
    double getElapsedTime();

    private:
    void calculateSizes();
    void createInputBuffers();
    void createOutputBuffers();
    void createOutputVectors();
    int adjustDimension(int dimension, int blockDimension);

    // Control
    bool environmentSetUp;

    // Image and Block
    int imgWidth;
    int imgHeight;
    int blockWidth;
    int blockHeight;
    int numOfBins;
    Format format;
    Color color;

    // Channel Details
    int ySize;
    int uSize;
    int vSize;
    int imageSize;

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
    cl::Kernel singleChannelKernel;
    cl::Kernel singleChannelDetailKernel;

    // Ranges
    cl::NDRange globalRange;
    cl::NDRange localRange;
    cl::Event event;

    // Input Buffers
    cl::Buffer imageBuffer;
    cl::Buffer numOfBinsBuffer;
    cl::Buffer formatBuffer;

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
    std::vector<varhist> yVarianceBins;
    std::vector<varhist> uVarianceBins;
    std::vector<varhist> vVarianceBins;

    // Timers
    double elapsedTime;
};