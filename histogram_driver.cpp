#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "histogram.hpp"
#include "histogram_driver_util.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

// Show/hide debug/test messages
const bool DEBUG_MODE_CPU = true;
const bool DEBUG_MODE_GPU = true;
const bool SHOW_CPU_TEST = false;

// Choose outputs to be generated
//const bool EXPORT_AVERAGE_HISTOGRAM = true;                 // NEEDS AVERAGE HISTOGRAM TEST 1 FOR GPU HISTOGRAM
//const bool EXPORT_VARIANCE_HISTOGRAM = true;                // NEEDS VARIANCE HISTOGRAM TEST 1 FOR GPU HISTOGRAM

// Set path for input image
const std::string FILEPATH = "input/DOTA2_I420_1920x1080.yuv"; 

// Set width height
const int IMG_WIDTH = 1920;
const int IMG_HEIGHT = 1080;

// Set block size
const int BLOCK_WIDTH = 8;
const int BLOCK_HEIGHT = 8;

// Set number of bins for the histograms
const int NUM_OF_BINS = 16;

int main(int argc, char const *argv[])
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Using image file: " << FILEPATH << std::endl << std::endl;

    // Calculate channel sizes for each frame
    int ySize = IMG_WIDTH * IMG_HEIGHT;
    int uSize = (IMG_WIDTH/2) * (IMG_HEIGHT/2);
    int vSize = (IMG_WIDTH/2) * (IMG_HEIGHT/2);
    int imageSize = ySize + uSize + vSize;
    if (DEBUG_MODE_CPU) {
        std::cout << "Y SIZE: " << ySize << std::endl;
        std::cout << "U SIZE: " << uSize << std::endl;
        std::cout << "V SIZE: " << vSize << std::endl;
        std::cout << "Image file size: " << imageSize << std::endl;
    }
    
    // Calculate blocksize and numbers of blocks for each channel;
    int yBlockWidth = BLOCK_WIDTH;
    int yBlockHeight = BLOCK_HEIGHT;
    int yBlockSize = yBlockWidth * yBlockHeight;
    int yNumOfBlocks = (IMG_WIDTH/yBlockWidth) * (IMG_HEIGHT/yBlockHeight);
    if (DEBUG_MODE_CPU) {
	    std::cout << "Y BLOCK SIZE: " << yBlockSize << std::endl;
        std::cout << "Y NUM OF BLOCKS: " << yNumOfBlocks << std::endl;
    }

    int uBlockWidth = BLOCK_WIDTH/2;
    int uBlockHeight = BLOCK_HEIGHT/2;
    int uBlockSize = uBlockWidth * uBlockHeight;
    int uNumOfBlocks = ((IMG_WIDTH/2)/uBlockWidth) * ((IMG_HEIGHT/2)/uBlockHeight);
    if (DEBUG_MODE_CPU) {
	    std::cout << "U BLOCK SIZE: " << uBlockSize << std::endl;
        std::cout << "U NUM OF BLOCKS: " << uNumOfBlocks << std::endl;
    }

    int vBlockWidth = BLOCK_WIDTH/2;
    int vBlockHeight = BLOCK_HEIGHT/2;
    int vBlockSize = vBlockWidth * vBlockHeight;
    int vNumOfBlocks = ((IMG_WIDTH/2)/vBlockWidth) * ((IMG_HEIGHT/2)/vBlockHeight);
    if (DEBUG_MODE_CPU) {
	    std::cout << "V BLOCK SIZE: " << vBlockSize << std::endl;
        std::cout << "V NUM OF BLOCKS: " << vNumOfBlocks << std::endl;
    }

    // Open file
    std::ifstream inputYUV (FILEPATH, std::ios::binary);
    if (!inputYUV.is_open()) {        
        std::cout << "Error opening file " << FILEPATH << std::endl;
        return(0);        
    }
    inputYUV.seekg(0, inputYUV.end);
    int fSize = inputYUV.tellg();
    if (DEBUG_MODE_CPU) {
        std::cout << "Read File Size: " << fSize << std::endl;
    }
    inputYUV.clear();
    inputYUV.seekg(inputYUV.beg);

    int actualSize = std::filesystem::file_size(FILEPATH);
    if (DEBUG_MODE_CPU) {
        std::cout << "Actual file size: " << actualSize << std::endl;
    }

    if (fSize != actualSize) {
        std::cout << "Size read different than actual file size"<< std::endl;
        inputYUV.close();
        return(0);   
    }

    if (fSize != imageSize) {
        std::cout << "Size read different than image file size" << std::endl;
        inputYUV.close();
        return(0);   
    }

    // Read file into vector
    std::vector<int> imageVector(imageSize);
    for (int i = 0; i < imageSize; i++) {
        imageVector.at(i) = inputYUV.get();
    }
    inputYUV.close();

    if (DEBUG_MODE_CPU) {
        std::cout << "\n================IMAGE AND BLOCK CONFIGURATION=================\n\n";

        std::cout << "Image dimensions: " << IMG_WIDTH << "x" << IMG_HEIGHT << std::endl;
        std::cout << "Block dimensions: " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << std::endl;
        std::cout << "Number of bins:" << NUM_OF_BINS << std::endl;
    }

    if (SHOW_CPU_TEST) {
        std::cout << "\n=============================CPU==============================\n\n";
    }

    // Create Output vectors for Average, Average Bins (histogram),
    // Variance, Variance Bins(histogram) for each channel
    std::vector<double> yAverageCPU(yNumOfBlocks);
    std::vector<double> uAverageCPU(uNumOfBlocks);
    std::vector<double> vAverageCPU(vNumOfBlocks);
    std::vector<double> yVarianceCPU(yNumOfBlocks);
    std::vector<double> uVarianceCPU(uNumOfBlocks);
    std::vector<double> vVarianceCPU(vNumOfBlocks);
    std::vector<int> yAverageBinsCPU(NUM_OF_BINS);
    std::vector<int> uAverageBinsCPU(NUM_OF_BINS);
    std::vector<int> vAverageBinsCPU(NUM_OF_BINS);
    std::vector<double> yVarianceBinsCPU(NUM_OF_BINS);
    std::vector<double> uVarianceBinsCPU(NUM_OF_BINS);
    std::vector<double> vVarianceBinsCPU(NUM_OF_BINS);

    // Create Timer Variables
    double yElapsedTimeAverageCPU, uElapsedTimeAverageCPU, vElapsedTimeAverageCPU;
    double yElapsedTimeVarianceCPU, uElapsedTimeVarianceCPU, vElapsedTimeVarianceCPU;
    double yElapsedTimeAverageHistCPU, uElapsedTimeAverageHistCPU, vElapsedTimeAverageHistCPU;
    double yElapsedTimeVarianceHistCPU, uElapsedTimeVarianceHistCPU, vElapsedTimeVarianceHistCPU;

    if (SHOW_CPU_TEST) {
        std::cout << "\n--------------------------AVERAGES----------------------------\n\n";
    }
    // Average of Channel Y
    TimeInterval timer("milli");
    calculateAverage(imageVector, 0, IMG_WIDTH, yNumOfBlocks, yBlockSize, yBlockWidth, yBlockHeight, yAverageCPU);
    yElapsedTimeAverageCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time Y Channel Average (ms) = " << yElapsedTimeAverageCPU << std::endl;
    }

    
    // Average of Channel U
    timer = TimeInterval("milli");
    calculateAverage(imageVector, ySize, IMG_WIDTH/2, uNumOfBlocks, uBlockSize, uBlockWidth, uBlockHeight, uAverageCPU);
    uElapsedTimeAverageCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time U Channel Average (ms) = " << uElapsedTimeAverageCPU << std::endl;
    }

    // Average of Channel V
    timer = TimeInterval("milli");
    calculateAverage(imageVector, ySize + uSize, IMG_WIDTH/2, vNumOfBlocks, vBlockSize, vBlockWidth, vBlockHeight, vAverageCPU);
    vElapsedTimeAverageCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time V Channel Average (ms) = " << vElapsedTimeAverageCPU << std::endl;
    }
    

    if (SHOW_CPU_TEST) {
        std::cout << "\n--------------------------VARIANCES---------------------------\n\n";
    }
    // Variance of Channel Y
    timer = TimeInterval("milli");
    calculateVariance(imageVector, 0, IMG_WIDTH, yNumOfBlocks, yBlockSize, yBlockWidth, yBlockHeight, yAverageCPU, yVarianceCPU);
    yElapsedTimeVarianceCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time Y Channel Variance (ms) = " << yElapsedTimeVarianceCPU << std::endl;
    }

    
    // Average of Channel U
    timer = TimeInterval("milli");
    calculateVariance(imageVector, ySize, IMG_WIDTH/2, uNumOfBlocks, uBlockSize, uBlockWidth, uBlockHeight, uAverageCPU, uVarianceCPU);
    uElapsedTimeVarianceCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time U Channel Variance (ms) = " << uElapsedTimeVarianceCPU << std::endl;
    }

    // Average of Channel V
    timer = TimeInterval("milli");
    calculateVariance(imageVector, ySize + uSize, IMG_WIDTH/2, vNumOfBlocks, vBlockSize, vBlockWidth, vBlockHeight, vAverageCPU, vVarianceCPU);
    vElapsedTimeVarianceCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time V Channel Variance (ms) = " << vElapsedTimeVarianceCPU << std::endl;
    }
    

    if (SHOW_CPU_TEST) {
        std::cout << "\n-------------------------HISTOGRAMS---------------------------\n\n";
    }
    // Average Histogram of Channel Y
    timer = TimeInterval("milli");
    calculateHistogram(yAverageCPU, NUM_OF_BINS, yAverageBinsCPU);
    yElapsedTimeAverageHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time Y Channel Average Hist (ms) = " << yElapsedTimeAverageHistCPU << std::endl;
    }

    
    // Average Histogram of Channel U
    timer = TimeInterval("milli");
    calculateHistogram(uAverageCPU, NUM_OF_BINS, uAverageBinsCPU);
    uElapsedTimeAverageHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time U Channel Average Hist (ms) = " << uElapsedTimeAverageHistCPU << std::endl;
    }

    // Average Histogram of Channel V
    timer = TimeInterval("milli");
    calculateHistogram(vAverageCPU, NUM_OF_BINS, vAverageBinsCPU);
    vElapsedTimeAverageHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time V Channel Average Hist (ms) = " << vElapsedTimeAverageHistCPU << std::endl;
    }
    

    // Variance Histogram of Channel Y
    timer = TimeInterval("milli");
    calculateHistogram(yAverageCPU, NUM_OF_BINS, yVarianceBinsCPU, yVarianceCPU);
    yElapsedTimeVarianceHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time Y Channel Variance Hist (ms) = " << yElapsedTimeVarianceHistCPU << std::endl;
    }
    
    
    // Variance Histogram of Channel U
    timer = TimeInterval("milli");
    calculateHistogram(uAverageCPU, NUM_OF_BINS, uVarianceBinsCPU, uVarianceCPU);
    uElapsedTimeVarianceHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time U Channel Variance Hist (ms) = " << uElapsedTimeVarianceHistCPU << std::endl;
    }

    // Variance Histogram of Channel V
    timer = TimeInterval("milli");
    calculateHistogram(vAverageCPU, NUM_OF_BINS, vVarianceBinsCPU, vVarianceCPU);
    vElapsedTimeVarianceHistCPU = timer.Elapsed();
    if (SHOW_CPU_TEST) {
        std::cout << "Elapsed time V Channel Variance Hist (ms) = " << vElapsedTimeVarianceHistCPU << std::endl;
    }
    

    if (SHOW_CPU_TEST) {
        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Average (Y + U + V) (ms) = " << yElapsedTimeAverageCPU + uElapsedTimeAverageCPU + vElapsedTimeAverageCPU << std::endl;
        std::cout << "Elapsed time Variance (Y + U + V) (ms) = " << yElapsedTimeVarianceCPU + uElapsedTimeVarianceCPU + vElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Average Hist (Y + U + V) (ms) = " << yElapsedTimeAverageHistCPU + uElapsedTimeAverageHistCPU + vElapsedTimeAverageHistCPU << std::endl;
        std::cout << "Elapsed time Variance Hist (Y + U + V) (ms) = " << yElapsedTimeVarianceHistCPU + uElapsedTimeVarianceHistCPU + vElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel Y (Avg + Var) (ms) = " << yElapsedTimeAverageCPU + yElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel U (Avg + Var) (ms) = " << uElapsedTimeAverageCPU + uElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel V (Avg + Var) (ms) = " << vElapsedTimeAverageCPU + vElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel Y (Avg + Var + Hist) (ms) = " << yElapsedTimeAverageCPU + yElapsedTimeVarianceCPU + yElapsedTimeAverageHistCPU + yElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel U (Avg + Var + Hist) (ms) = " << uElapsedTimeAverageCPU + uElapsedTimeVarianceCPU + uElapsedTimeAverageHistCPU + uElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel V (Avg + Var + Hist) (ms) = " << vElapsedTimeAverageCPU + vElapsedTimeVarianceCPU + vElapsedTimeAverageHistCPU + vElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Total Elapsed time (ms) = " << yElapsedTimeAverageCPU + yElapsedTimeVarianceCPU + yElapsedTimeAverageHistCPU + yElapsedTimeVarianceHistCPU + uElapsedTimeAverageCPU + uElapsedTimeVarianceCPU + uElapsedTimeAverageHistCPU + uElapsedTimeVarianceHistCPU + vElapsedTimeAverageCPU + vElapsedTimeVarianceCPU + vElapsedTimeAverageHistCPU + vElapsedTimeVarianceHistCPU << std::endl;
    
    }

    std::cout << "\n=============================GPU==============================\n\n";

    // Create Output Vectors
    std::vector<float> yAverageGPU(yNumOfBlocks);
    std::vector<float> uAverageGPU(uNumOfBlocks);
    std::vector<float> vAverageGPU(vNumOfBlocks);
    std::vector<float> yVarianceGPU(yNumOfBlocks);
    std::vector<float> uVarianceGPU(uNumOfBlocks);
    std::vector<float> vVarianceGPU(vNumOfBlocks);
    std::vector<int> yAverageHistGPU(NUM_OF_BINS);
    std::vector<int> uAverageHistGPU(NUM_OF_BINS);
    std::vector<int> vAverageHistGPU(NUM_OF_BINS);
    std::vector<float> yVarianceHistGPU(NUM_OF_BINS);
    std::vector<float> uVarianceHistGPU(NUM_OF_BINS);
    std::vector<float> vVarianceHistGPU(NUM_OF_BINS);

    // Create instance of Histogram Library
    Histogram histogram(IMG_WIDTH, IMG_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, NUM_OF_BINS, true);

    // Setup Environment
    histogram.setupEnvironment();
    histogram.printEnvironment();

    // Write Image Buffer
    histogram.writeInputBuffers(imageVector);

    // Calculate Histograms
    histogram.calculateHistograms(true);

    // Get output values
    yAverageGPU = histogram.getAverage("Y");
    uAverageGPU = histogram.getAverage("U");
    vAverageGPU = histogram.getAverage("V");

    yVarianceGPU = histogram.getVariance("Y");
    uVarianceGPU = histogram.getVariance("U");
    vVarianceGPU = histogram.getVariance("V");

    yAverageHistGPU = histogram.getAverageHistogram("Y");
    uAverageHistGPU = histogram.getAverageHistogram("U");
    vAverageHistGPU = histogram.getAverageHistogram("V");

    yVarianceHistGPU = histogram.getVarianceHistogram("Y");
    uVarianceHistGPU = histogram.getVarianceHistogram("U");
    vVarianceHistGPU = histogram.getVarianceHistogram("V");

    // Create Timer Variables
    double yElapsedTimeAllHistGPU = histogram.getElapsedTime("Y");
    double uElapsedTimeAllHistGPU = histogram.getElapsedTime("U");
    double vElapsedTimeAllHistGPU = histogram.getElapsedTime("V");

    // Validate Average Histogram Vectors
    std::cout << "\n---------------------------VALIDATING----------------------------\n\n";
    std::cout << "Validating Y Average GPU: ";
    validateVectorError(yAverageGPU, yAverageCPU);

    std::cout << "Validating U Average GPU: ";
    validateVectorError(uAverageGPU, uAverageCPU);

    std::cout << "Validating V Average GPU: ";
    validateVectorError(vAverageGPU, vAverageCPU);

    // Validate Variance Histogram Vectors
    std::cout << "Validating Y Variance GPU: ";
    validateVectorError(yVarianceGPU, yVarianceCPU);

    std::cout << "Validating U Variance GPU: ";
    validateVectorError(uVarianceGPU, uVarianceCPU);

    std::cout << "Validating V Variance GPU: ";
    validateVectorError(vVarianceGPU, vVarianceCPU);

    // Validate Average Histogram Vectors
    std::cout << "Validating Y Average Hist GPU: ";
    validateVectorError(yAverageHistGPU, yAverageBinsCPU);

    std::cout << "Validating U Average Hist GPU: ";
    validateVectorError(uAverageHistGPU, uAverageBinsCPU);

    std::cout << "Validating V Average Hist GPU: ";
    validateVectorError(vAverageHistGPU, vAverageBinsCPU);

    // Validate Variance Histogram Vectors
    std::cout << "Validating Y Variance Hist GPU: ";
    validateVectorError(yVarianceHistGPU, yVarianceBinsCPU);

    std::cout << "Validating U Variance Hist GPU: ";
    validateVectorError(uVarianceHistGPU, uVarianceBinsCPU);

    std::cout << "Validating V Variance Hist GPU: ";
    validateVectorError(vVarianceHistGPU, vVarianceBinsCPU);
    

    std::cout << "\n---------------------------PERFORMANCE----------------------------\n\n";
    std::cout << "Elapsed time Channel Y (ms) = " << yElapsedTimeAllHistGPU << std::endl;
    std::cout << "Elapsed time Channel U (ms) = " << uElapsedTimeAllHistGPU << std::endl;
    std::cout << "Elapsed time Channel V (ms) = " << vElapsedTimeAllHistGPU << std::endl;
    std::cout << "Elapsed time (Y + U + V) (ms) = " << yElapsedTimeAllHistGPU + uElapsedTimeAllHistGPU + vElapsedTimeAllHistGPU << std::endl;
    
 
    imageVector.clear();

    return 0;
}
