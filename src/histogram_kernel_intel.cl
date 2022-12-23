/**
 * @file histogram_kernel_intel.cl
 * @brief Kernel file for Intel GPU
 */
#pragma CL_VERSION_3_0

/**
 * @brief Kernel function that calculates the histograms for a single channel.
 * The kernel calculates the average and variance histograms based on the average and variance of each group (block of pixels).
 * It accepts YUV/NV12 format for the Luma channel or YUV format for the Chroma channel.
 * @param pixels pointer to raw image data.
 * @param numOfBins the number of bins on the calculated histograms.
 * @param averageBins the histogram data for the average.
 * @param varianceBins the histogram data for the variance.
 * @param blockSumAverage local memory for the accumulative sum (reduction) for the the average of the group.
 * @param blockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group.
 */
kernel void calculateHistogramsSingleChannel(global const int *pixels, global const int *numOfBins, global int *averageBins, global int *varianceBins, local int *blockSumAverage, local int *blockSumVariance) {    
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;

    // Get block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id relative positions
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    int gid = (gidY * globalWidth) + gidX;

    // Luma Upsampling Offsets
    int gidOffsetX = gid + blockWidth;
    int gidOffsetY = gid + (blockHeight * globalWidth);
    int gidOffsetXY = gidOffsetY + blockWidth;

    // Copy memory from global to local and add offset positions
    blockSumAverage[lid] = pixels[gid];
    blockSumAverage[lid] += pixels[gidOffsetX];
    blockSumAverage[lid] += pixels[gidOffsetY];
    blockSumAverage[lid] += pixels[gidOffsetXY];

    blockSumVariance[lid] = pixels[gid] * pixels[gid];
    blockSumVariance[lid] += pixels[gidOffsetX] * pixels[gidOffsetX];
    blockSumVariance[lid] += pixels[gidOffsetY] * pixels[gidOffsetY];
    blockSumVariance[lid] += pixels[gidOffsetXY] * pixels[gidOffsetXY];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction
    if (blockSize == 1024 && lid < 512) {
        blockSumAverage[lid] += blockSumAverage[lid + 512];
        blockSumVariance[lid] += blockSumVariance[lid + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && lid < 256) {
        blockSumAverage[lid] += blockSumAverage[lid + 256];
        blockSumVariance[lid] += blockSumVariance[lid + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && lid < 128) {
        blockSumAverage[lid] += blockSumAverage[lid + 128];
        blockSumVariance[lid] += blockSumVariance[lid + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && lid < 64) {
        blockSumAverage[lid] += blockSumAverage[lid + 64];
        blockSumVariance[lid] += blockSumVariance[lid + 64];
    }

    if (lid < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            blockSumAverage[lid] += blockSumAverage[lid + 32];
            blockSumVariance[lid] += blockSumVariance[lid + 32];
        }

        if (blockSize >= 32) {
            blockSumAverage[lid] += blockSumAverage[lid + 16];
            blockSumVariance[lid] += blockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[lid] += blockSumAverage[lid + 8];
            blockSumVariance[lid] += blockSumVariance[lid + 8];
        }

        if (blockSize >= 8) {
            blockSumAverage[lid] += blockSumAverage[lid + 4];
            blockSumVariance[lid] += blockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[lid] += blockSumAverage[lid + 2];
            blockSumVariance[lid] += blockSumVariance[lid + 2];
        }

        if (blockSize >= 2) {
            blockSumAverage[lid] += blockSumAverage[lid + 1];
            blockSumVariance[lid] += blockSumVariance[lid + 1];
        } 
    }
    
    // Update average array
    if (lid == 0) {
        // Calculate average        
        float average = (float)blockSumAverage[0]/(blockSize*4);

        // Calculate variance
        float variance = (float)blockSumVariance[0]/(blockSize*4) - average * average;

        // Calculate bin
        int interval = ((int)average*numOfBins[0])>>8;

        // Atomic increment
        atomic_inc(&averageBins[interval]);
        atomic_add(&varianceBins[interval], (int)variance);
    }
}

/**
 * @brief Kernel function that calculates the histograms for a single channel with details.
 * The kernel calculates the average and variance histograms based on the average and variance of each group (block of pixels).
 * The kernel also returns details for the values for the average and variance of group (block of pixels) that were calculated.
 * It accepts YUV/NV12 format for the Luma channel or YUV format for the Chroma channel.
 * @param pixels pointer to raw image data.
 * @param numOfBins the number of bins on the calculated histograms.
 * @param average the average data for each group.
 * @param variance the variance data for each group.
 * @param averageBins the histogram data for the average.
 * @param varianceBins the histogram data for the variance.
 * @param blockSumAverage local memory for the accumulative sum (reduction) for the the average of the group.
 * @param blockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group.
 */
kernel void calculateHistogramsSingleChannelWithDetail(global const int *pixels, global const int *numOfBins, global float *average, global float *variance, global int *averageBins, global int *varianceBins, local int *blockSumAverage, local int *blockSumVariance) {
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;

    // Get block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id relative positions
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    int gid = (gidY * globalWidth) + gidX;

    // Luma Upsampling Offsets
    int gidOffsetX = gid + blockWidth;
    int gidOffsetY = gid + (blockHeight * globalWidth);
    int gidOffsetXY = gidOffsetY + blockWidth;
    
    // Copy memory from global to local and add offset positions
    blockSumAverage[lid] = pixels[gid];
    blockSumAverage[lid] += pixels[gidOffsetX];
    blockSumAverage[lid] += pixels[gidOffsetY];
    blockSumAverage[lid] += pixels[gidOffsetXY];

    blockSumVariance[lid] = pixels[gid] * pixels[gid];
    blockSumVariance[lid] += pixels[gidOffsetX] * pixels[gidOffsetX];
    blockSumVariance[lid] += pixels[gidOffsetY] * pixels[gidOffsetY];
    blockSumVariance[lid] += pixels[gidOffsetXY] * pixels[gidOffsetXY];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction
    if (blockSize == 1024 && lid < 512) {
        blockSumAverage[lid] += blockSumAverage[lid + 512];
        blockSumVariance[lid] += blockSumVariance[lid + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && lid < 256) {
        blockSumAverage[lid] += blockSumAverage[lid + 256];
        blockSumVariance[lid] += blockSumVariance[lid + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && lid < 128) {
        blockSumAverage[lid] += blockSumAverage[lid + 128];
        blockSumVariance[lid] += blockSumVariance[lid + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && lid < 64) {
        blockSumAverage[lid] += blockSumAverage[lid + 64];
        blockSumVariance[lid] += blockSumVariance[lid + 64];
    }

    if (lid < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            blockSumAverage[lid] += blockSumAverage[lid + 32];
            blockSumVariance[lid] += blockSumVariance[lid + 32];
        }

        if (blockSize >= 32) {
            blockSumAverage[lid] += blockSumAverage[lid + 16];
            blockSumVariance[lid] += blockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[lid] += blockSumAverage[lid + 8];
            blockSumVariance[lid] += blockSumVariance[lid + 8];
        }

        if (blockSize >= 8) {
            blockSumAverage[lid] += blockSumAverage[lid + 4];
            blockSumVariance[lid] += blockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[lid] += blockSumAverage[lid + 2];
            blockSumVariance[lid] += blockSumVariance[lid + 2];
        }

        if (blockSize >= 2) {
            blockSumAverage[lid] += blockSumAverage[lid + 1];
            blockSumVariance[lid] += blockSumVariance[lid + 1];
        } 
    }
    
    // Update average array
    if (lid == 0) {
        // Calculate block linear id
        int bid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

        // Calculate average        
        average[bid] = (float)blockSumAverage[0]/(blockSize*4);

        // Calculate variance
        variance[bid] = (float)blockSumVariance[0]/(blockSize*4) - (average[bid] * average[bid]);

        // Calculate bin
        int interval = ((int)average[bid]*numOfBins[0])>>8;

        // Atomic increment
        atomic_inc(&averageBins[interval]);
        atomic_add(&varianceBins[interval], (int)variance[bid]);
    }
}

/**
 * @brief Kernel function that calculates the histograms for all channels.
 * The kernel calculates the average and variance histograms based on the average and variance of each group (block of pixels).
 * It accepts YUV/NV12 format and performs the calculation for all channels (YUV).
 * @param pixels pointer to raw image data.
 * @param numOfBins the number of bins on the calculated histograms.
 * @param format the format of the raw image data. 0 = YUV, 1 = NV12.
 * @param yAverageBins the histogram data for the average for channel Y.
 * @param yVarianceBins the histogram data for the variance for channel Y.
 * @param uAverageBins the histogram data for the average for channel U.
 * @param uVarianceBins the histogram data for the variance for channel U.
 * @param vAverageBins the histogram data for the average for channel V.
 * @param vVarianceBins the histogram data for the variance for channel V.
 * @param yBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel Y.
 * @param yBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel Y.
 * @param uBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel U.
 * @param uBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel U.
 * @param vBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel V.
 * @param vBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel V.
 */
kernel void calculateHistograms(global const int *pixels, global const int *numOfBins, global const int *format, global int *yAverageBins, global int *yVarianceBins, global int *uAverageBins, global int *uVarianceBins, global int *vAverageBins, global int *vVarianceBins, local int *yBlockSumAverage, local int *yBlockSumVariance, local int *uBlockSumAverage, local int *uBlockSumVariance, local int *vBlockSumAverage, local int *vBlockSumVariance) {
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;
    int globalHeight = get_global_size(1) * 2;
    int globalSize = globalWidth * globalHeight;

    // Get Block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id relative positions
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    int gid = (gidY * globalWidth) + gidX;

    // Luma Upsampling Offsets
    int gidOffsetX = gid + blockWidth;
    int gidOffsetY = gid + (blockHeight * globalWidth);
    int gidOffsetXY = gidOffsetY + blockWidth;

    // Chroma Offsets
    int gidOffsetU, gidOffsetV;

    // Format 0 = YUV, Format 1 = NV12
    if (format[0] == 0) {
        gidOffsetU = get_global_linear_id() + globalSize;
        gidOffsetV = gidOffsetU + (get_global_size(0) * get_global_size(1));
    }
    else {
        gidOffsetU = (2 * get_global_linear_id()) + globalSize;
        gidOffsetV = gidOffsetU + 1;
    }
    
    // Copy memory from global to local and add offset positions for each channel
    yBlockSumAverage[lid] = pixels[gid];
    yBlockSumAverage[lid] += pixels[gidOffsetX];
    yBlockSumAverage[lid] += pixels[gidOffsetY];
    yBlockSumAverage[lid] += pixels[gidOffsetXY];

    yBlockSumVariance[lid] = pixels[gid] * pixels[gid];
    yBlockSumVariance[lid] += pixels[gidOffsetX] * pixels[gidOffsetX];
    yBlockSumVariance[lid] += pixels[gidOffsetY] * pixels[gidOffsetY];
    yBlockSumVariance[lid] += pixels[gidOffsetXY] * pixels[gidOffsetXY];

    uBlockSumAverage[lid] = pixels[gidOffsetU];
    uBlockSumVariance[lid] = pixels[gidOffsetU] * pixels[gidOffsetU];

    vBlockSumAverage[lid] = pixels[gidOffsetV];
    vBlockSumVariance[lid] = pixels[gidOffsetV] * pixels[gidOffsetV];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction
    if (blockSize == 1024 && lid < 512) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 512];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 512];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 512];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 512];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 512];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && lid < 256) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 256];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 256];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 256];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 256];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 256];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && lid < 128) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 128];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 128];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 128];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 128];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 128];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && lid < 64) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 64];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 64];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 64];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 64];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 64];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 64];
    }

    if (lid < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 32];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 32];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 32];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 32];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 32];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 32];
        }

        if (blockSize >= 32) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 16];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 16];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 16];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 16];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 16];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 8];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 8];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 8];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 8];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 8];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 8];
        }

        if (blockSize >= 8) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 4];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 4];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 4];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 4];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 4];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 2];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 2];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 2];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 2];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 2];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 2];
        }

        if (blockSize >= 2) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 1];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 1];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 1];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 1];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 1];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 1];
        } 
    }
    
    // Update average array
    if (lid == 0) {
        // Calculate average        
        float yAverage = (float)yBlockSumAverage[0]/(blockWidth*2*blockHeight*2);
        float uAverage = (float)uBlockSumAverage[0]/(blockSize);
        float vAverage = (float)vBlockSumAverage[0]/(blockSize);

        // Calculate variance
        float yVariance = (float)yBlockSumVariance[0]/(blockWidth*2*blockHeight*2) - yAverage * yAverage;
        float uVariance = (float)uBlockSumVariance[0]/(blockSize) - uAverage * uAverage;
        float vVariance = (float)vBlockSumVariance[0]/(blockSize) - vAverage * vAverage;

        // Calculate bin
        int yInterval = ((int)yAverage*numOfBins[0])>>8;
        int uInterval = ((int)uAverage*numOfBins[0])>>8;
        int vInterval = ((int)vAverage*numOfBins[0])>>8;

        // Atomic increment
        atomic_inc(&yAverageBins[yInterval]);
        atomic_inc(&uAverageBins[uInterval]);
        atomic_inc(&vAverageBins[vInterval]);
        atomic_add(&yVarianceBins[yInterval], (int)yVariance);
        atomic_add(&uVarianceBins[uInterval], (int)uVariance);
        atomic_add(&vVarianceBins[vInterval], (int)vVariance);
    }
}

/**
 * @brief Kernel function that calculates the histograms for all channels with details.
 * The kernel calculates the average and variance histograms based on the average and variance of each group (block of pixels).
 * The kernel also returns details for the values for the average and variance of group (block of pixels) that were calculated.
 * It accepts YUV/NV12 format and performs the calculation for all channels (YUV).
 * @param pixels pointer to raw image data.
 * @param numOfBins the number of bins on the calculated histograms.
 * @param format the format of the raw image data. 0 = YUV, 1 = NV12.
 * @param yAverage the average data for each group for channel Y.
 * @param yVariance the variance data for each group for channel Y.
 * @param yAverageBins the histogram data for the average for channel Y.
 * @param yVarianceBins the histogram data for the variance for channel Y.
 * @param uAverage the average data for each group for channel U.
 * @param uVariance the variance data for each group for channel U.
 * @param uAverageBins the histogram data for the average for channel U.
 * @param uVarianceBins the histogram data for the variance for channel U.
 * @param vAverage the average data for each group for channel V.
 * @param vVariance the variance data for each group for channel V.
 * @param vAverageBins the histogram data for the average for channel V.
 * @param vVarianceBins the histogram data for the variance for channel V.
 * @param yBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel Y.
 * @param yBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel Y.
 * @param uBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel U.
 * @param uBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel U.
 * @param vBlockSumAverage local memory for the accumulative sum (reduction) for the the average of the group for channel V.
 * @param vBlockSumVariance local memory for the accumulative sum (reduction) for the the variance of the group for channel V.
 */
kernel void calculateHistogramsWithDetail(global const int *pixels, global const int *numOfBins, global const int *format, global float *yAverage, global float *yVariance, global int *yAverageBins, global int *yVarianceBins, global float *uAverage, global float *uVariance, global int *uAverageBins, global int *uVarianceBins, global float *vAverage, global float *vVariance, global int *vAverageBins, global int *vVarianceBins, local int *yBlockSumAverage, local int *yBlockSumVariance, local int *uBlockSumAverage, local int *uBlockSumVariance, local int *vBlockSumAverage, local int *vBlockSumVariance) {
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;
    int globalHeight = get_global_size(1) * 2;
    int globalSize = globalWidth * globalHeight;

    // Get Block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id relative positions
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    int gid = (gidY * globalWidth) + gidX;

    // Luma Upsampling Offsets
    int gidOffsetX = gid + blockWidth;
    int gidOffsetY = gid + (blockHeight * globalWidth);
    int gidOffsetXY = gidOffsetY + blockWidth;

    // Chroma Offsets
    int gidOffsetU, gidOffsetV;

    // Format 0 = YUV, Format 1 = NV12
    if (format[0] == 0) {
        gidOffsetU = get_global_linear_id() + globalSize;
        gidOffsetV = gidOffsetU + (get_global_size(0) * get_global_size(1));
    }
    else {
        gidOffsetU = (2 * get_global_linear_id()) + globalSize;
        gidOffsetV = gidOffsetU + 1;
    }
    
    // Copy memory from global to local and add offset positions for each channel
    yBlockSumAverage[lid] = pixels[gid];
    yBlockSumAverage[lid] += pixels[gidOffsetX];
    yBlockSumAverage[lid] += pixels[gidOffsetY];
    yBlockSumAverage[lid] += pixels[gidOffsetXY];

    yBlockSumVariance[lid] = pixels[gid] * pixels[gid];
    yBlockSumVariance[lid] += pixels[gidOffsetX] * pixels[gidOffsetX];
    yBlockSumVariance[lid] += pixels[gidOffsetY] * pixels[gidOffsetY];
    yBlockSumVariance[lid] += pixels[gidOffsetXY] * pixels[gidOffsetXY];

    uBlockSumAverage[lid] = pixels[gidOffsetU];
    uBlockSumVariance[lid] = pixels[gidOffsetU] * pixels[gidOffsetU];

    vBlockSumAverage[lid] = pixels[gidOffsetV];
    vBlockSumVariance[lid] = pixels[gidOffsetV] * pixels[gidOffsetV];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction
    if (blockSize == 1024 && lid < 512) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 512];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 512];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 512];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 512];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 512];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && lid < 256) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 256];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 256];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 256];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 256];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 256];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && lid < 128) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 128];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 128];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 128];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 128];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 128];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && lid < 64) {
        yBlockSumAverage[lid] += yBlockSumAverage[lid + 64];
        yBlockSumVariance[lid] += yBlockSumVariance[lid + 64];
        uBlockSumAverage[lid] += uBlockSumAverage[lid + 64];
        uBlockSumVariance[lid] += uBlockSumVariance[lid + 64];
        vBlockSumAverage[lid] += vBlockSumAverage[lid + 64];
        vBlockSumVariance[lid] += vBlockSumVariance[lid + 64];
    }

    if (lid < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 32];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 32];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 32];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 32];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 32];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 32];
        }

        if (blockSize >= 32) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 16];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 16];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 16];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 16];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 16];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 8];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 8];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 8];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 8];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 8];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 8];
        }

        if (blockSize >= 8) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 4];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 4];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 4];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 4];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 4];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 2];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 2];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 2];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 2];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 2];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 2];
        }

        if (blockSize >= 2) {
            yBlockSumAverage[lid] += yBlockSumAverage[lid + 1];
            yBlockSumVariance[lid] += yBlockSumVariance[lid + 1];
            uBlockSumAverage[lid] += uBlockSumAverage[lid + 1];
            uBlockSumVariance[lid] += uBlockSumVariance[lid + 1];
            vBlockSumAverage[lid] += vBlockSumAverage[lid + 1];
            vBlockSumVariance[lid] += vBlockSumVariance[lid + 1];
        } 
    }
    
    // Update average array
    if (lid == 0) {
        // Calculate block linear id
        int bid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

        // Calculate average        
        yAverage[bid] = (float)yBlockSumAverage[0]/(2*blockWidth*2*blockHeight);
        uAverage[bid] = (float)uBlockSumAverage[0]/(blockSize);
        vAverage[bid] = (float)vBlockSumAverage[0]/(blockSize);

        // Calculate variance
        yVariance[bid] = (float)yBlockSumVariance[0]/(2*blockWidth*2*blockHeight) - (yAverage[bid] * yAverage[bid]);
        uVariance[bid] = (float)uBlockSumVariance[0]/(blockSize) - (uAverage[bid] * uAverage[bid]);
        vVariance[bid] = (float)vBlockSumVariance[0]/(blockSize) - (vAverage[bid] * vAverage[bid]);

        // Calculate bin
        int yInterval = ((int)yAverage[bid]*numOfBins[0])>>8;
        int uInterval = ((int)uAverage[bid]*numOfBins[0])>>8;
        int vInterval = ((int)vAverage[bid]*numOfBins[0])>>8;

        // Atomic increment
        atomic_inc(&yAverageBins[yInterval]);
        atomic_inc(&uAverageBins[uInterval]);
        atomic_inc(&vAverageBins[vInterval]);
        atomic_add(&yVarianceBins[yInterval], (int)yVariance[bid]);
        atomic_add(&uVarianceBins[uInterval], (int)uVariance[bid]);
        atomic_add(&vVarianceBins[vInterval], (int)vVariance[bid]);
    }
}