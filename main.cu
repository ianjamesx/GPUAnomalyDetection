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

    int index=1, stride;
    //getIndex(blockIdx.x, blockDim.x, threadIdx.x, index);
    //getStride(blockDim.x, gridDim.x, stride);

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
        //printRecord_full(records, record_count, record_size, i);
    }

    /*
    generate pairs
    */

    int pair_size = 2;
    int pair_count = nk_count(record_size, pair_size);

    //size of each pair compression batch
    cout << pair_count << " " << pair_size << endl;
    int batchsize = getPairBatchSize(pair_count, pair_size, .60);
    int rounds = ceil(record_count / batchsize);

    //bound round/batchsize
    if(rounds < 1) rounds = 1;
    if(batchsize > record_count) batchsize = record_count;

    //rounds = 2;
    //batchsize = 3;

    cout << "rounds, batchsize: " << rounds << ", " << batchsize << endl;

    int parentlist_size = pairlist_size(record_count, record_size, pair_size);
    int OClist_size = occurancelist_size(record_count, record_size, pair_size);

    //for host to maintain (parent list, occurance list)
    float *parent_list = new float[parentlist_size];
    int *OClist = new int[OClist_size];
    initParentList(parent_list, record_count, record_size, pair_size);
    initOCList(OClist, record_count, record_size, pair_size);

    //for device to compress (pair buffer, occurance buffer)
    float *pair_buffer;
    int *OCbuffer;

    int pair_buffer_size = pairlist_size(batchsize, record_size, pair_size);
    int occurance_list_size = occurancelist_size(batchsize, record_size, pair_size);
    
    gpuErrchk(cudaMallocManaged(&pair_buffer, pair_buffer_size * sizeof(float)));
    gpuErrchk(cudaMallocManaged(&OCbuffer, occurance_list_size * sizeof(int)));

    for(i = 0; i < rounds; i++){

        //get starting/stopping record index for this batch
        int record_start, record_stop;
        getBatchStart(i, rounds, record_count, batchsize, record_start, record_stop);

        int blocks = 1, threads = 2;

        //generate pairing for this round
        generatePairs<<<blocks, threads>>>(records, pair_buffer, record_start, record_stop, record_size, pair_count, pair_size);
        cudaDeviceSynchronize();

        printFullParentList(pair_buffer, batchsize, pair_count, pair_size);
        printf("--------------\n");

        int record_pair_count = batchsize > record_count ? record_count : batchsize;

        while(threads > 1){
            locatePatterns<<<pair_count, threads>>>(pair_buffer, OCbuffer, record_pair_count, pair_count, pair_size, threads);
            cudaDeviceSynchronize();
            //printOccurances(OCbuffer, record_pair_count, pair_count);
            //printf("-----------\n");
            threads /= 2;
        }

        //run on single thread for final iteration
        threads = 1;
        locatePatterns<<<pair_count, 1>>>(pair_buffer, OCbuffer, record_pair_count, pair_count, pair_size, threads);
        cudaDeviceSynchronize();

        //save all patterns/occurances from buffer to parent list
        saveAllToParentList(parent_list, pair_buffer, OClist, OCbuffer, batchsize, pair_count, pair_size);

        //printFullParentList(pair_buffer, record_count, pair_count, pair_size);
        //printFullParentList(parent_list, record_count, pair_count, pair_size);

        //printf("-----------------\n");

        //printOccurances(OCbuffer, record_pair_count, pair_count);
        //printOccurances(OClist, record_pair_count, pair_count);

    }

    printf("--------------\n");

    //printFullParentList(parent_list, record_count, pair_count, pair_size);
    //printOccurances(OClist, record_count, pair_count);

    return 0;
}
