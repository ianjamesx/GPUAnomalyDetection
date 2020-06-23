#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;


//device error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//host auxillary operations
#include "host.cuh"

//data read utilities (some host methods needed)
#include "dataread.cuh"

//device (GPU) auxillary operations
#include "device/device_aux.cuh"

//device (GPU) primary operations
#include "device/device.cuh"

__global__
void hello(){
    printf("entering...\n");

    int i, j, l;

    int index, stride;
    getIndex(blockIdx.x, blockDim.x, threadIdx.x, index);
    getStride(blockDim.x, gridDim.x, stride);

    printf("Hello! From index %d\n", index);
}

int main(){

    /*
    init dataset
    */

    float *records;
    int *record_types;
    int record_count, record_size;

    init_dataset(record_count, record_size, records, record_types);

    int i, j;

    for(i = 0; i < record_count; i++){
        printRecord_full(records, record_count, record_size, i);
    }

    /*
    generate pairs
    */

    int pair_size = 2;

    int pair_count = nk_count(record_size, pair_size);

    //allocate managed memory for pair list
    float *pairs;
    int pair_list_size = pairlist_size(record_count, record_size, pair_size);
    gpuErrchk( cudaMallocManaged(&pairs, pair_list_size * sizeof(float)) );

    //generate pairs on device
    generatePairs<<<1, 2>>>(records, pairs, record_count, record_size, pair_count, pair_size);
    cudaDeviceSynchronize();

    //print all pairings
    for(i = 0; i < record_count; i++){
       // printAllPairs_host(pairs, pair_count, i, pair_size);
    }

    /*
    allocate parent list and output list
    */

    //list used for all batches, keeps track of all substructures found
    float *parent_list;
    gpuErrchk( cudaMallocManaged(&parent_list, pair_list_size * sizeof(float)) );
    
    //buffer used for holding found substructures before comparing to parentlist
    float *pair_buffer;
    gpuErrchk( cudaMallocManaged(&pair_buffer, pair_list_size * sizeof(float)) );

    //list of occurances of each substructure, indices align to parentlist
    int *occurance_list;
    int occurance_list_size = occurancelist_size(record_count, record_size, pair_size);
    gpuErrchk( cudaMallocManaged(&occurance_list, occurance_list_size * sizeof(int)) );
    occurancelist_init(occurance_list, occurance_list_size);

    //also maintain a buffer for each iteration of occurances before copying to parentlist
    int *occurance_buffer;
    gpuErrchk( cudaMallocManaged(&occurance_buffer, occurance_list_size * sizeof(int)) );
    occurancelist_init(occurance_buffer, occurance_list_size);

    /*
    locate patterns in the pairings
    */

    //copy all pairs to buffer
    initOutputBuffer(pair_buffer, record_count, pair_count, pair_size);
    copyPairsToBuffer_host(pairs, pair_buffer, record_count, pair_count, pair_size);

    for(i = 0; i < record_count; i++){
        //printAllPairs_host(pair_buffer, pair_count, i, pair_size);
    }

    int initBlocks = 64, initThreads = 512;
    int allthreads = initBlocks * initThreads;

    while(allthreads){
        //launch thread with allthreads
        allthreads /= 2;
    }

    //locatePatterns(pair_buffer, occurance_buffer, record_count, pair_count, pair_size);

    //printOccurances(occurance_buffer, record_count, pair_count);

    //printAllPairsAllIndices_full(pair_buffer, record_count, pair_count, pair_size);

    //printOccurances(occurance_buffer, record_count, pair_count);

    return 0;
}
