#pragma once

#include "histogram.hpp"

Histogram::Histogram() {
    imgWidth = 1920;
    imgHeight = 1080;
    blockWidth = 8;
    blockHeight = 8;
    numOfBins = 16;
    showErrors = true;
    environmentSetUp = false;
}

Histogram::Histogram(int imgWidth, int imgHeight, int blockWidth, int blockHeight, int numOfBins, bool showErrors) {
    this->imgWidth = imgWidth;
    this->imgHeight = imgHeight;
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;
    this->numOfBins = numOfBins;
    this->showErrors = showErrors;
    environmentSetUp = false;
}

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

    calculateSizes();

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
    yVarianceBins = std::vector<float>(numOfBins);
    uVarianceBins = std::vector<float>(numOfBins);
    vVarianceBins = std::vector<float>(numOfBins);

    createInputBuffers();
    createOutputBuffers();

    environmentSetUp = true;
}

void Histogram::createInputBuffers() {
    // Initialize Input Buffers
    yImageBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, ySize * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yImageBuffer ERROR: " << clError << std::endl;
    }
    uImageBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, uSize * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uImageBuffer ERROR: " << clError << std::endl;
    }
    vImageBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, vSize * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vImageBuffer ERROR: " << clError << std::endl;
    }
    numOfBinsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create numOfBinsBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::writeInputBuffers(std::vector<int> imageVector) {
    clError = commandQueue.enqueueWriteBuffer(yImageBuffer, CL_TRUE, 0, ySize * sizeof(int), &imageVector[0], NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write yImageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(uImageBuffer, CL_TRUE, 0, uSize * sizeof(int), &imageVector[ySize], NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write Error ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(vImageBuffer, CL_TRUE, 0, vSize * sizeof(int), &imageVector[ySize + uSize], NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write vImageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(numOfBinsBuffer, CL_TRUE, 0, 1 * sizeof(int), &numOfBins, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write numOfBinsBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::writeInputBuffers(const void *ptr) {
    clError = commandQueue.enqueueWriteBuffer(yImageBuffer, CL_TRUE, 0, ySize * sizeof(uint8_t), ptr, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write yImageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(uImageBuffer, CL_TRUE, 0, uSize * sizeof(uint8_t), (uint8_t*)ptr + ySize, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write Error ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(vImageBuffer, CL_TRUE, 0, vSize * sizeof(uint8_t), (uint8_t*)ptr + ySize + uSize, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write vImageBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(numOfBinsBuffer, CL_TRUE, 0, 1 * sizeof(int), &numOfBins, NULL, NULL);            
    if (showErrors && clError < 0) {
        std::cout << "Write numOfBinsBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::createOutputBuffers() {
    yAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, yNumOfBlocks * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yAverageBuffer ERROR: " << clError << std::endl;
    }
    uAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, uNumOfBlocks * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uAverageBuffer ERROR: " << clError << std::endl;
    }
    vAverageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, vNumOfBlocks * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create vAverageBuffer ERROR: " << clError << std::endl;
    }

    yVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, yNumOfBlocks * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yAverageBuffer ERROR: " << clError << std::endl;
    }
    uVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, uNumOfBlocks * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uVarianceBuffer ERROR: " << clError << std::endl;
    }
    vVarianceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, vNumOfBlocks * sizeof(int), NULL, &clError);
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

    yVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create yVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    uVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
    if (showErrors && clError < 0) {
        std::cout << "Create uVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    vVarianceHistBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numOfBins * sizeof(int), NULL, &clError);
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
    clError = commandQueue.enqueueWriteBuffer(yVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &yVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading y_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(uVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &uVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading u_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueWriteBuffer(vVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &vVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading v_VarianceHistBuffer ERROR: " << clError << std::endl;
    }
}

void Histogram::calculateSizes() {
    ySize = imgWidth * imgHeight;
    uSize = (imgWidth/2) * (imgHeight/2);
    vSize = (imgWidth/2) * (imgHeight/2);

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

    yGlobalRange = cl::NDRange(adjustDimension(imgWidth/2, yBlockWidth/2), adjustDimension(imgHeight/2, yBlockHeight/2));
    yLocalRange = cl::NDRange(yBlockWidth/2, yBlockHeight/2);
    uGlobalRange = cl::NDRange(adjustDimension(imgWidth/4, uBlockWidth/2), adjustDimension(imgHeight/4, uBlockHeight/2));
    uLocalRange = cl::NDRange(uBlockWidth/2, uBlockHeight/2);
    vGlobalRange = cl::NDRange(adjustDimension(imgWidth/4, vBlockWidth/2), adjustDimension(imgHeight/4, vBlockHeight/2));
    vLocalRange = cl::NDRange(vBlockWidth/2, vBlockHeight/2);
}

void Histogram::calculateHistograms(bool detailed) {
    if (!environmentSetUp) {
        std::cout << "Environment not set up" << std::endl;
        return;
    }

    // Reset Timers
    yElapsedTime = 0;
    uElapsedTime = 0;
    vElapsedTime = 0;

    cl::Event event;
    cl::Kernel kernel;

    // Select Kernel
    if (detailed) {
        kernel = histogramsDetailKernel;
    }
    else {
        kernel = histogramsKernel;
    }
    
    // Execute Y Channel
    if (detailed) {
        kernel.setArg(0, yImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, yAverageBuffer);
        kernel.setArg(3, yVarianceBuffer);
        kernel.setArg(4, yAverageHistBuffer);
        kernel.setArg(5, yVarianceHistBuffer);
        kernel.setArg(6, yBlockSize * sizeof(int), NULL);
        kernel.setArg(7, yBlockSize * sizeof(int), NULL);
    }
    else {
        kernel.setArg(0, yImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, yAverageHistBuffer);
        kernel.setArg(3, yVarianceHistBuffer);
        kernel.setArg(4, yBlockSize * sizeof(int), NULL);
        kernel.setArg(5, yBlockSize * sizeof(int), NULL);
    }
    clError = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, yGlobalRange, yLocalRange, NULL, &event);
    event.wait();
    if (showErrors && clError < 0) {
        std::cout << "Execution Y ERROR: " << clError << std::endl;
    }
    yElapsedTime = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    
    // Execute U Channel
    if (detailed) {
        kernel.setArg(0, uImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, uAverageBuffer);
        kernel.setArg(3, uVarianceBuffer);
        kernel.setArg(4, uAverageHistBuffer);
        kernel.setArg(5, uVarianceHistBuffer);
        kernel.setArg(6, uBlockSize * sizeof(int), NULL);
        kernel.setArg(7, uBlockSize * sizeof(int), NULL);
    }
    else {
        kernel.setArg(0, uImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, uAverageHistBuffer);
        kernel.setArg(3, uVarianceHistBuffer);
        kernel.setArg(4, uBlockSize * sizeof(int), NULL);
        kernel.setArg(5, uBlockSize * sizeof(int), NULL);
    }
    clError = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, uGlobalRange, uLocalRange, NULL, &event);
    event.wait();
    if (showErrors && clError < 0) {
        std::cout << "Execution U ERROR: " << clError << std::endl;
    }
    uElapsedTime = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

    // Execute V Channel
    if (detailed) {
        kernel.setArg(0, vImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, vAverageBuffer);
        kernel.setArg(3, vVarianceBuffer);
        kernel.setArg(4, vAverageHistBuffer);
        kernel.setArg(5, vVarianceHistBuffer);
        kernel.setArg(6, vBlockSize * sizeof(int), NULL);
        kernel.setArg(7, vBlockSize * sizeof(int), NULL);
    }
    else {
        kernel.setArg(0, vImageBuffer);
        kernel.setArg(1, numOfBinsBuffer);
        kernel.setArg(2, vAverageHistBuffer);
        kernel.setArg(3, vVarianceHistBuffer);
        kernel.setArg(4, vBlockSize * sizeof(int), NULL);
        kernel.setArg(5, vBlockSize * sizeof(int), NULL);
    }
    clError = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, vGlobalRange, vLocalRange, NULL, &event);
    event.wait();
    if (showErrors && clError < 0) {
        std::cout << "Execution V ERROR: " << clError << std::endl;
    }
    vElapsedTime = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    
    // Read responsess
    if (detailed) {
        clError = commandQueue.enqueueReadBuffer(yAverageBuffer, CL_TRUE, 0, yNumOfBlocks * sizeof(int), &yAverage[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading yAverageBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(uAverageBuffer, CL_TRUE, 0, uNumOfBlocks * sizeof(int), &uAverage[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading uAverageBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(vAverageBuffer, CL_TRUE, 0, vNumOfBlocks * sizeof(int), &vAverage[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading vAverageBuffer ERROR: " << clError << std::endl;
        }

        clError = commandQueue.enqueueReadBuffer(yVarianceBuffer, CL_TRUE, 0, yNumOfBlocks * sizeof(int), &yVariance[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading yVarianceBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(uVarianceBuffer, CL_TRUE, 0, uNumOfBlocks * sizeof(int), &uVariance[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading uVarianceBuffer ERROR: " << clError << std::endl;
        }
        clError = commandQueue.enqueueReadBuffer(vVarianceBuffer, CL_TRUE, 0, vNumOfBlocks * sizeof(int), &vVariance[0], NULL, NULL);
        if (showErrors && clError < 0) {
            std::cout << "Reading vVarianceBuffer ERROR: " << clError << std::endl;
        }  
    }
    clError = commandQueue.enqueueReadBuffer(yAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &yAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading yAverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueReadBuffer(uAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &uAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading uAverageHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueReadBuffer(vAverageHistBuffer, CL_TRUE, 0, numOfBins * sizeof(int), &vAverageBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading vAverageHistBuffer ERROR: " << clError << std::endl;
    }

    clError = commandQueue.enqueueReadBuffer(yVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &yVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading yVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueReadBuffer(uVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &uVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading uVarianceHistBuffer ERROR: " << clError << std::endl;
    }
    clError = commandQueue.enqueueReadBuffer(vVarianceHistBuffer, CL_TRUE, 0, numOfBins * sizeof(float), &vVarianceBins[0], NULL, NULL);
    if (showErrors && clError < 0) {
        std::cout << "Reading vVarianceHistBuffer ERROR: " << clError << std::endl;
    }
}

std::vector<float> Histogram::getAverage(std::string channel) {
    if (channel == "Y") {
        return yAverage;
    }
    else if (channel == "U") {
        return uAverage;
    }
    else if (channel == "V") {
        return vAverage;
    }
    return yAverage;
}

std::vector<float> Histogram::getVariance(std::string channel) {
    if (channel == "Y") {
        return yVariance;
    }
    else if (channel == "U") {
        return uVariance;
    }
    else if (channel == "V") {
        return vVariance;
    }
    return yVariance;
}

std::vector<int> Histogram::getAverageHistogram(std::string channel) {
    if (channel == "Y") {
        return yAverageBins;
    }
    else if (channel == "U") {
        return uAverageBins;
    }
    else if (channel == "V") {
        return vAverageBins;
    }
    return yAverageBins;
}

std::vector<float> Histogram::getVarianceHistogram(std::string channel) {
    if (channel == "Y") {
        return yVarianceBins;
    }
    else if (channel == "U") {
        return uVarianceBins;
    }
    else if (channel == "V") {
        return vVarianceBins;
    }
    return yVarianceBins;
}

double Histogram::getElapsedTime(std::string channel) {
    if (channel == "Y") {
        return yElapsedTime;
    }
    else if (channel == "U") {
        return uElapsedTime;
    }
    else if (channel == "V") {
        return vElapsedTime;
    }
    return yElapsedTime;
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

