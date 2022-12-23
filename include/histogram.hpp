/**
 * @file histogram.hpp
 * @brief This header file contains the histogram library class.
 */
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
    /**
     * @brief Defines the type used for the variance histogram (float for NVIDIA GPU and int for Intel GPU).
     */
    typedef int varhist;
#endif

/**
 * @brief This class implements the histogram library.
 * This library uses OpenCL to calculate fast histograms for the average and variance of pixel blocks using parallel processing in the GPU.
 */
class Histogram {
    public:

    /**
     * @brief This enumeration is to define the image format.
     * 
     */
    enum class Format {
        YUV,
        NV12
    };

    /**
     * @brief This enumeration is to define the chromatic format.
     * 
     */
    enum class Color {
        Chromatic,
        Grayscale
    };

    /**
     * @brief This enumeration is used to select the desired channel.
     * 
     */
    enum class Channel {
        Y,
        U,
        V
    };

    /**
     * @brief This enumeration is used to include or exclude detailed calculation (Calculate Average and Variance Values).
     * 
     */
    enum class Detail {
        Exclude,
        Include
    };

    /**
     * @brief This enumeration is used to display errors or not.
     * 
     */
    enum class ErrorLevel {
        NoError,
        ShowError
    };

    /**
     * @brief Default constructor for the histogram class.
     * The default constructor uses a YUV format, chromatic option, with full hd image size (1920x1080), 8x8 pixel block, and 16 bins.
     */
    Histogram();

    /**
     * @brief Constructor for the histogram class.
     * 
     * @param format the image format of the raw data.
     * @param color the chromatic options for the image.
     * @param imgWidth the width of the image.
     * @param imgHeight the height of the image.
     * @param blockWidth the width of the pixel blocks.
     * @param blockHeight the height of the pixel blocks.
     * @param numOfBins the number of bins for the histograms.
     */
    Histogram(Format format, Color color, int imgWidth, int imgHeight, int blockWidth, int blockHeight, int numOfBins);

    /**
     * @brief Copy constructor.
     * 
     * @param o 
     */
    Histogram(const Histogram &o);

    /**
     * @brief Destructor.
     * 
     */
    ~Histogram();

    /**
     * @brief Sets up the initial environment for the GPU.
     * Perform initialization of OpenCL device and allocates memory buffers.
     * Needs to be executed before any other calculation method.
     */
    void setupEnvironment();

    /**
     * @brief Prints the enviroment information.
     * Information includes the GPU device being used.
     */
    void printEnvironment();

    /**
     * @brief Write the input memory buffer with the raw image data.
     * 
     * @param imageVector vector that contains the raw image data in YUV or NV12 formats.
     */
    void writeInputBuffers(std::vector<int> imageVector);

    /**
     * @brief Write the input memory buffer with the raw image data.
     * 
     * @param ptr pointer to memory that contains the raw image data in YUV or NV12 formats.
     */
    void writeInputBuffers(const void *ptr);

    /**
     * @brief Sets the Image Size for the enviroment.
     * Used if the image size needs to be changed dynamically.
     * 
     * @param imgWidth the width of the image.
     * @param imgHeight the height of the image.
     */
    void setImageSize(int imgWidth, int imgHeight);

    /**
     * @brief Sets the Block Size for the enviroment.
     * Used if the block size needs to be changed dynamically.
     * 
     * @param blockWidth the width of the pixel blocks.
     * @param blockHeight the height of the pixel blocks.
     */
    void setBlockSize(int blockWidth, int blockHeight);

    /**
     * @brief Sets the Number of Bins for the enviroment.
     * Used if the Number of Bins needs to be changed dynamically.
     * 
     * @param numOfBins the number of bins for the histograms.
     */
    void setNumofBins(int numOfBins);

    /**
     * @brief Set the Error Level.
     * If set to display, errors will be displayed as they occur.
     * 
     * @param errorLevel the error options desired.
     */
    void setErrorLevel(ErrorLevel errorLevel);
    
    /**
     * @brief Calculates the histograms of the image in the buffers.
     * The calculation returns the average and variance histograms based on the average and variance of each group (block of pixels).
     * The resolution of the calculations takes into account the block size of the environemnt (block width and height).
     * The histogram is generated using the number of bins of the environment.
     */
    void calculateHistograms();

    /**
     * @brief Calculates the histograms of the image in the buffers.
     * The calculation returns the average and variance histograms based on the average and variance of each group (block of pixels).
     * The resolution of the calculations takes into account the block size of the environemnt (block width and height).
     * The histogram is generated using the number of bins of the environment.
     * If the option of detail is passed, this method also returns the average and variance that was calculate for each group (block of pixels).
     * 
     * @param detail the option to perform calculations with our without returning the details.
     */
    void calculateHistograms(Detail detail);

    /**
     * @brief Gets the average data for the given channel.
     * The average data is a vector that represents each group (block of pixels).
     * 
     * @param channel selects the channel to return the data from.
     * @return std::vector<float> with the average data.
     */
    std::vector<float> getAverage(Channel channel);

    /**
     * @brief Gets the variance data for the given channel.
     * 
     * The variance data is a vector that represents each group (block of pixels).
     * @param channel selects the channel to return the data from.
     * @return std::vector<float> with the variance data.
     */
    std::vector<float> getVariance(Channel channel);

    /**
     * @brief Gets the average histogram data for the given channel.
     * 
     * @param channel selects the channel to return the data from.
     * @return std::vector<int> with the average histogram data.
     */
    std::vector<int> getAverageHistogram(Channel channel);

    /**
     * @brief Gets the variance histogram data for the given channel.
     * 
     * @param channel selects the channel to return the data from.
     * @return std::vector<varhist> with the variance histogram data.
     */
    std::vector<varhist> getVarianceHistogram(Channel channel);

    /**
     * @brief Gets the elapsed time for the previous calculations.
     * 
     * @return double with the elapsed time.
     */
    double getElapsedTime();

    private:
    /**
     * @brief Calculates the sizes of buffers and vectors needed for the environment.
     * 
     */
    void calculateSizes();

    /**
     * @brief Creates the input memory buffers.
     * 
     */
    void createInputBuffers();

    /**
     * @brief Creates the output memory buffers.
     * 
     */
    void createOutputBuffers();

    /**
     * @brief Create a output vectors.
     * 
     */
    void createOutputVectors();

    /**
     * @brief Helper function used to adjust the dimension to fit the block size evenly. 
     *
     * @param dimension image dimension to be used.
     * @param blockDimension block dimension to be compared to
     * @return int with adjusted dimension.
     */
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