#pragma once

#include "histogram.hpp"

Histogram::Histogram() {
    imgWidth = 1920;
    imgHeight = 1080;
    blockWidth = 8;
    blockHeight = 8;
    numOfBins = 16;
    format = Format::YUV;
    color = Color::Chromatic;
    showErrors = false;
    elapsedTime = 0;
    environmentSetUp = false;
}

Histogram::Histogram(Format format, Color color, int imgWidth, int imgHeight, int blockWidth, int blockHeight, int numOfBins) {
    this->imgWidth = imgWidth;
    this->imgHeight = imgHeight;
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;
    this->numOfBins = numOfBins;
    this->format = format;
    this->color = color;
    elapsedTime = 0;
    showErrors = false;
    environmentSetUp = false;
}

Histogram::Histogram(const Histogram &o) {
    imgWidth = o.imgWidth;
    imgHeight = o.imgHeight;
    blockWidth = o.blockWidth;
    blockHeight = o.blockHeight;
    numOfBins = o.numOfBins;
    format = o.format;
    color = o.color;
    elapsedTime = 0;
    showErrors = o.showErrors;
    environmentSetUp = false;
}

Histogram::~Histogram() {}

void Histogram::setupEnvironment() {
    // Get platform and device information
    platform = cl::Platform::getDefault();
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    defaultDevice = devices[0];

    // Create context
    context = cl::Context(defaultDevice, NULL, NULL, NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Context ERROR: " << clError << std::endl;
    }

    // Create CommandQueue
    commandQueue = cl::CommandQueue(context, defaultDevice, cl::QueueProperties::Profiling, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Queue ERROR: " << clError << std::endl;
    }

    // Read Program Source
    std::ifstream sourceFile("histogram_kernel.cl");
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));

    // Load Program
    program = cl::Program(context, sourceCode, clError);
    if (showErrors && clError < 0) {
        std::cout << "Program ERROR: " << clError << std::endl;
    }

    // Build Program
    clError = program.build(defaultDevice, "-cl-std=CL3.0");
    if (showErrors && clError < 0) {
        std::cout << "Build Program ERROR: " << clError << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << std::endl;
    }

    // Load Kernels
    histogramsKernel = cl::Kernel(program, "calculateHistograms");
    histogramsDetailKernel = cl::Kernel(program, "calculateHistogramsWithDetail");
    singleChannelKernel = cl::Kernel(program, "calculateHistogramsSingleChannel");
    singleChannelDetailKernel = cl::Kernel(program, "calculateHistogramsSingleChannelWithDetail");

    calculateSizes();
    createInputBuffers();
    createOutputVectors();
    createOutputBuffers();

    environmentSetUp = true;
}

void Histogram::createOutputVectors() {
    // Initialize Output Vectors
    yAverage = std::vector<float>(yNumOfBlocks);
    uAverage = std::vector<float>(uNumOfBlocks);
    vAverage = std::vector<float>(vNumOfBlocks);
    yVariance = std::vector<float>(yNumOfBlocks);
    uVariance = std::vector<float>(uNumOfBlocks);
    vVariance = std::vector<float>(vNumOfBlocks);
    yAverageBins = std::vector<int>(numOfBins);
    uAverageBins = std::vector<int>(numOfBins);
    vAverageBins = std::vector<int>(numOfBins);
    yVarianceBins = std::vector<varhist>(numOfBins);
    uVarianceBins = std::vector<varhist>(numOfBins);
    vVarianceBins = std::vector<varhist>(numOfBins);
}

void Histogram::createInputBuffers() {
    // Initialize Input Buffers
    imageBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, imageSize * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create imageBuffer ERROR: " << clError << std::endl;
    }
    numOfBinsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create numOfBinsBuffer ERROR: " << clError << std::endl;
    }
    formatBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create formatBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::writeInputBuffers(std::vector<int> imageVector) {
    clError = commandQueue.enqueueWriteBuffer(imageBuffer, CL_TRUE, 0, imageSize * sizeof(int), &imageVector[0], NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write imageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(numOfBinsBuffer, CL_TRUE, 0, 1 * sizeof(int), &numOfBins, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write numOfBinsBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(formatBuffer, CL_TRUE, 0, 1 * sizeof(int), &format, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write formatBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::writeInputBuffers(const void *ptr) {
    clError = commandQueue.enqueueWriteBuffer(imageBuffer, CL_TRUE, 0, imageSize * sizeof(int), ptr, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write imageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(numOfBinsBuffer, CL_TRUE, 0, 1 * sizeof(int), &numOfBins, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write numOfBinsBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(formatBuffer, CL_TRUE, 0, 1 * sizeof(int), &format, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write formatBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::createOutputBuffers() {
    yAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, yNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yAverageBuffer ERROR: " << clError << std::endl;
    }
    uAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, uNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uAverageBuffer ERROR: " << clError << std::endl;
    }
    vAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, vNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vAverageBuffer ERROR: " << clError << std::endl;
    }

    yVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, yNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yAverageBuffer ERROR: " << clError << std::endl;
    }
    uVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, uNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uVarianceBuffer ERROR: " << clError << std::endl;
    }
    vVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, vNumOfBlocks * sizeof(float), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vVarianceBuffer ERROR: " << clError << std::endl;
    }

    yAverageHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yAverageHistBuffer ERROR: " << clError << std::endl;
    }
    uAverageHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uAverageHistBuffer ERROR: " << clError << std::endl;
    }
    vAverageHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vAverageHistBuffer ERROR: " << clError << std::endl;
    }

    yVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(varhist), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    uVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(varhist), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    vVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(varhist), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vVarianceHistBuffer ERROR: " << clError << std::endl;
    }

    // Initialize Hist Buffer
    clError = commandQueue.enqueueWriteBuffer(yAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &yAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading y_AverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(uAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &uAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading u_AverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(vAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &vAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading v_AverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(yVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &yVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading y_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(uVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &uVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading u_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(vVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &vVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading v_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::calculateSizes() {
    ySize = imgWidth * imgHeight;
    uSize = (imgWidth/2) * (imgHeight/2);
    vSize = (imgWidth/2) * (imgHeight/2);
    imageSize = ySize + uSize + vSize;

    yBlockWidth = blockWidth;
    yBlockHeight = blockHeight;
    yBlockSize = yBlockWidth * yBlockHeight;
    yNumOfBlocks = (imgWidth/yBlockWidth) * (imgHeight/yBlockHeight);

    uBlockWidth = blockWidth/2;
    uBlockHeight = blockHeight/2;
    uBlockSize = uBlockWidth * uBlockHeight;
    uNumOfBlocks = ((imgWidth/2)/uBlockWidth) * ((imgHeight/2)/uBlockHeight);

    vBlockWidth = blockWidth/2;
    vBlockHeight = blockHeight/2;
    vBlockSize = vBlockWidth * vBlockHeight;
    vNumOfBlocks = ((imgWidth/2)/vBlockWidth) * ((imgHeight/2)/vBlockHeight);

    globalRange = cl::NDRange(adjustDimension(imgWidth/2, yBlockWidth/2), adjustDimension(imgHeight/2, yBlockHeight/2));
    localRange = cl::NDRange(yBlockWidth/2, yBlockHeight/2);
}

void Histogram::calculateHistograms() {
    if (!environmentSetUp) {
        std::cout << "Environment not set up" << std::endl;
        return;
    }
    calculateHistograms(Detail::Exclude);
}

void Histogram::calculateHistograms(Detail detail) {
    if (!environmentSetUp) {
        std::cout << "Environment not set up" << std::endl;
        return;
    }

    // Reset Timers
    elapsedTime = 0;

    cl::Event event;
    cl::Kernel kernel;
    
    // Set Kernel Args
    if (color == Color::Chromatic) {
        if (detail == Detail::Exclude) {
            kernel = histogramsKernel;
            kernel.setArg(0, imageBuffer);
            kernel.setArg(1, numOfBinsBuffer);
            kernel.setArg(2, formatBuffer);
            kernel.setArg(3, yAverageHistBuffer);
            kernel.setArg(4, yVarianceHistBuffer);
            kernel.setArg(5, uAverageHistBuffer);
            kernel.setArg(6, uVarianceHistBuffer);
            kernel.setArg(7, vAverageHistBuffer);
            kernel.setArg(8, vVarianceHistBuffer);
            kernel.setArg(9, yBlockSize * sizeof(int), NULL);
            kernel.setArg(10, yBlockSize * sizeof(int), NULL);
            kernel.setArg(11, uBlockSize * sizeof(int), NULL);
            kernel.setArg(12, uBlockSize * sizeof(int), NULL);
            kernel.setArg(13, vBlockSize * sizeof(int), NULL);
            kernel.setArg(14, vBlockSize * sizeof(int), NULL);
        }
        else {
            kernel = histogramsDetailKernel;
            kernel.setArg(0, imageBuffer);
            kernel.setArg(1, numOfBinsBuffer);
            kernel.setArg(2, formatBuffer);
            kernel.setArg(3, yAverageBuffer);
            kernel.setArg(4, yVarianceBuffer);
            kernel.setArg(5, yAverageHistBuffer);
            kernel.setArg(6, yVarianceHistBuffer);
            kernel.setArg(7, uAverageBuffer);
            kernel.setArg(8, uVarianceBuffer);
            kernel.setArg(9, uAverageHistBuffer);
            kernel.setArg(10, uVarianceHistBuffer);
            kernel.setArg(11, vAverageBuffer);
            kernel.setArg(12, vVarianceBuffer);
            kernel.setArg(13, vAverageHistBuffer);
            kernel.setArg(14, vVarianceHistBuffer);
            kernel.setArg(15, yBlockSize * sizeof(int), NULL);
            kernel.setArg(16, yBlockSize * sizeof(int), NULL);
            kernel.setArg(17, uBlockSize * sizeof(int), NULL);
            kernel.setArg(18, uBlockSize * sizeof(int), NULL);
            kernel.setArg(19, vBlockSize * sizeof(int), NULL);
            kernel.setArg(20, vBlockSize * sizeof(int), NULL);
        }
    }
    else {
        if (detail == Detail::Exclude) {
            kernel = singleChannelKernel;
            kernel.setArg(0, imageBuffer);
            kernel.setArg(1, numOfBinsBuffer);
            kernel.setArg(2, yAverageHistBuffer);
            kernel.setArg(3, yVarianceHistBuffer);
            kernel.setArg(4, yBlockSize * sizeof(int), NULL);
            kernel.setArg(5, yBlockSize * sizeof(int), NULL);
        }
        else {
            kernel = singleChannelDetailKernel;
            kernel.setArg(0, imageBuffer);
            kernel.setArg(1, numOfBinsBuffer);
            kernel.setArg(2, yAverageBuffer);
            kernel.setArg(3, yVarianceBuffer);
            kernel.setArg(4, yAverageHistBuffer);
            kernel.setArg(5, yVarianceHistBuffer);
            kernel.setArg(6, yBlockSize * sizeof(int), NULL);
            kernel.setArg(7, yBlockSize * sizeof(int), NULL);
        }
    }
    clError = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange, NULL, &event);
    event.wait();
    if (showErrors && clError < 0) {
        std::cout << "Execution ERROR: " << clError << std::endl;
    }
    elapsedTime = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    

    // Read responsess
    if (detail == Detail::Include) 
    {
        clError = commandQueue.enqueueReadBuffer(yAverageBuffer, CL_TRUE, 0, yNumOfBlocks * sizeof(float), &yAverage[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading yAverageBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(yVarianceBuffer, CL_TRUE, 0, yNumOfBlocks * sizeof(float), &yVariance[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading yVarianceBuffer ERROR: " << clError << std::endl;
        }
        if (color == Color::Chromatic) {
            clError = commandQueue.enqueueReadBuffer(uAverageBuffer, CL_TRUE, 0, uNumOfBlocks * sizeof(float), &uAverage[0], NULL, NULL);
            if (showErrors && clError < 0) {
                std::cout << "Reading uAverageBuffer ERROR: " << clError << std::endl;
            }
            clError = commandQueue.enqueueReadBuffer(uVarianceBuffer, CL_TRUE, 0, uNumOfBlocks * sizeof(float), &uVariance[0], NULL, NULL);
            if (showErrors && clError < 0) {
                std::cout << "Reading uVarianceBuffer ERROR: " << clError << std::endl;
            }

            clError = commandQueue.enqueueReadBuffer(vAverageBuffer, CL_TRUE, 0, vNumOfBlocks * sizeof(float), &vAverage[0], NULL, NULL);
            if (showErrors && clError < 0) {
                std::cout << "Reading vAverageBuffer ERROR: " << clError << std::endl;
            }
            clError = commandQueue.enqueueReadBuffer(vVarianceBuffer, CL_TRUE, 0, vNumOfBlocks * sizeof(float), &vVariance[0], NULL, NULL);
            if (showErrors && clError < 0) {
                std::cout << "Reading vVarianceBuffer ERROR: " << clError << std::endl;
            }  
        }
    }

    clError = commandQueue.enqueueReadBuffer(yAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &yAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading yAverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueReadBuffer(yVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &yVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading yVarianceHistBuffer ERROR: " << clError << std::endl;
    }

    if (color == Color::Chromatic) {
        clError = commandQueue.enqueueReadBuffer(uAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &uAverageBins[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading uAverageHistBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(vAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &vAverageBins[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading vAverageHistBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(uVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &uVarianceBins[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading uVarianceHistBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(vVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(varhist), &vVarianceBins[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading vVarianceHistBuffer ERROR: " << clError << std::endl;
        }
    }
}

std::vector<float> Histogram::getAverage(Channel channel) {
    if (channel == Channel::Y) {
        return yAverage;
    }
    else if (channel == Channel::U) {
        return uAverage;
    }
    else if (channel == Channel::V) {
        return vAverage;
    }
    return yAverage;
}

std::vector<float> Histogram::getVariance(Channel channel) {
    if (channel == Channel::Y) {
        return yVariance;
    }
    else if (channel == Channel::U) {
        return uVariance;
    }
    else if (channel == Channel::V) {
        return vVariance;
    }
    return yVariance;
}

std::vector<int> Histogram::getAverageHistogram(Channel channel) {
    if (channel == Channel::Y) {
        return yAverageBins;
    }
    else if (channel == Channel::U) {
        return uAverageBins;
    }
    else if (channel == Channel::V) {
        return vAverageBins;
    }
    return yAverageBins;
}

std::vector<varhist> Histogram::getVarianceHistogram(Channel channel) {
    if (channel == Channel::Y) {
        return yVarianceBins;
    }
    else if (channel == Channel::U) {
        return uVarianceBins;
    }
    else if (channel == Channel::V) {
        return vVarianceBins;
    }
    return yVarianceBins;
}

double Histogram::getElapsedTime() {
    return elapsedTime;
}

void Histogram::printEnvironment() {
    if (!environmentSetUp) {
        std::cout << "Environment not set up" << std::endl;
        return;
    }
    std::cout << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Device name: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Device OpenCL Version: " << devices[0].getInfo<CL_DEVICE_VERSION>() << std::endl;
    std::cout << "Device OpenCL C Version: " << devices[0].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
}

int Histogram::adjustDimension(int dimension, int blockDimension) {
    if (blockDimension == 0) {
        return dimension;
    }
    return dimension - (dimension % blockDimension);
}

void Histogram::setImageSize(int imgWidth, int imgHeight) {
    // Change settings
    this->imgWidth = imgWidth;
    this->imgHeight = imgHeight;
    
    // Recalculate sizes and reset buffers
    calculateSizes();
    createInputBuffers();
    createOutputVectors();
    createOutputBuffers();

}

void Histogram::setBlockSize(int blockWidth, int blockHeight) {
    // Change settings
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;

    // Recalculate sizes and reset buffers
    calculateSizes();
    createOutputVectors();
    createOutputBuffers();
}

void Histogram::setNumofBins(int numOfBins) {
    this->numOfBins = numOfBins;
}

void Histogram::setErrorLevel(ErrorLevel errorLevel) {
    if (errorLevel == ErrorLevel::NoError) {
        this->showErrors = false;
    }
    else {
        this->showErrors = true;
    }
}