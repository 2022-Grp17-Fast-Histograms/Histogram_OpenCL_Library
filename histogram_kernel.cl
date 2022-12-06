#pragma CL_VERSION_3_0

//inline PTX atomic_add using float (Only works for nvidia)
void atomic_add_float(global float *p, float val)
{
    float prev;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;" 
        : "=f"(prev) 
        : "l"(p) , "f"(val) 
        : "memory" 
    );
}

kernel void calculateHistograms(global const int *pixels, global const int *numOfBins, global int *averageBins, global float *varianceBins, local int *blockSumAverage, local int *blockSumVariance) {
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;

    // Get block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    
    int gid = (gidY * globalWidth) + gidX;
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
            blockSumAverage[lid] += blockSumAverage[lid + 16];
            blockSumVariance[lid] += blockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[lid] += blockSumAverage[lid + 8];
            blockSumVariance[lid] += blockSumVariance[lid + 8];
            blockSumAverage[lid] += blockSumAverage[lid + 4];
            blockSumVariance[lid] += blockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[lid] += blockSumAverage[lid + 2];
            blockSumVariance[lid] += blockSumVariance[lid + 2];
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

        atomic_inc(&averageBins[interval]);
        atomic_add_float(&varianceBins[interval], variance);
    }
}

kernel void calculateHistogramsWithDetail(global const int *pixels, global const int *numOfBins, global float *average, global float *variance, global int *averageBins, global float *varianceBins, local int *blockSumAverage, local int *blockSumVariance) {
    // Get local id
    int lid = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;

    // Get block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id
    int gidX = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int gidY = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    
    int gid = (gidY * globalWidth) + gidX;
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
            blockSumAverage[lid] += blockSumAverage[lid + 16];
            blockSumVariance[lid] += blockSumVariance[lid + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[lid] += blockSumAverage[lid + 8];
            blockSumVariance[lid] += blockSumVariance[lid + 8];
            blockSumAverage[lid] += blockSumAverage[lid + 4];
            blockSumVariance[lid] += blockSumVariance[lid + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[lid] += blockSumAverage[lid + 2];
            blockSumVariance[lid] += blockSumVariance[lid + 2];
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

        atomic_inc(&averageBins[interval]);
        atomic_add_float(&varianceBins[interval], variance[bid]);
    }
}

kernel void calculateHistogram(global const float *input, global const int *numOfBins, global int *bins) {
    // Get Blocal Id
    int i = get_global_id(0);

    // Calculate bin
    int interval = ((int)input[i]*numOfBins[0])>>8;
    atomic_inc(&bins[interval]);
}