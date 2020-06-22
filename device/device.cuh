__device__
void getIndex(int blockIdx, int blockDim, int threadIdx, int &index){
    index = blockIdx * blockDim + threadIdx;
}

__device__
void getStride(int blockDimx, int gridDimx, int &stride){
    stride = blockDimx * gridDimx;
}

__global__
void generatePairs(float *recordlist, float *pairlist, int record_count, int record_size, int pair_count, int pair_size){

    int i, j, l;

    int index, stride;
    getIndex(blockIdx.x, blockDim.x, threadIdx.x, index);
    getStride(blockDim.x, gridDim.x, stride);

    printf("Hello! From index %d, my stride: %d (%d, %d)\n", index, stride, record_size, record_count);

    for(i = index; i < record_count; i += stride){ //<-- stride here by index

        int pair_index = 0;

        for(j = 0; j < record_size; j++){
            for(l = j+1; l < record_size; l++){

                int record_indices[2] = {j, l};

                //printf("thread %d iter %d --> %d, %d\n", index, i, record_indices[0], record_indices[1]);

                savepair_device(pairlist, recordlist, record_size, i, record_indices, pair_count, pair_index, pair_size);

                pair_index++;

            }
        }
    }
}

/*
copy all pairs from the pairlist to the output buffer
so when we start replacing with -1s we dont ruin our original pairs
*/

/*
__global__
void copyPairsToBuffer(float *pair_list, float *pair_buffer, int record_count, int pair_count, int pair_size){

    int i, j;

    for(j = 0; j < pair_count; j++){ //<-- stride here (by blocks)

        for(i = 0; i < record_count; i++){
            copypair_device(pair_list, pair_buffer, pair_count, i, j, pair_size);
        }

    }

}
*/

/*
distribute a unique pairing to each thread-block
thread block will find all patterns in pairings and move to the output buffer
then threads will search output buffer for duplicates, and merge into list of unique patterns
keeping track of number of occurances for each

total ops: Ck(R) * NK
where NK = number of unique pairings per record (820), R = record count
*/
/*
__global__
void locatePatterns(float *pair_buffer, int *occurance_list, int record_count, int pair_count, int pair_size){

    int i, j, l;

    for(l = 0; l < pair_count; l++){ //<-- stride here (by blocks)

        int curr_pair = l;

        //generate range for this thread to cover
        for(i = 0; i < record_count; i++){

            //index of the record to compare all others to
            int curr_record1 = i;

            //first check if current pair has been removed, if it has, skip this pair
            int removed;
            pairRemoved(pair_buffer, pair_count, curr_record1, curr_pair, pair_size, removed);
            if(removed) continue;

            //number of occurances of this pattern in other records
            //default is one as pattern appearing in this record counts as an occurance
            int occurances = 1;

            for(j = i+1; j < record_count; j++){

                int curr_record2 = j;
                int isequal;
                comparePairs(pair_buffer, pair_count, curr_record1, curr_record2, curr_pair, pair_size, isequal);

                //if we found matching pair, remove the pair from the buffer so we dont count it multiple times in compression
                if(isequal){
                    //cout << "match on patterm " << curr_pair << " for record " << curr_record1 << ", " << curr_record2 << endl;
                    removeBufferPair(pair_buffer, pair_count, curr_record2, curr_pair, pair_size);
                    occurances++;
                }
            }

            //add number of occurances of this pattern to list
            addOccurances(occurance_list, pair_count, curr_record1, curr_pair, occurances);

        }

    }

}
*
__global__
void reducePatterns(float *pairlist, float *output_buffer, int *occurance_list, int record_count, int pair_count, int pair_size){

    int i, j, l;

    for(l = 0; l < pair_count; l++){ //<-- stride here (by blocks)

        int curr_pair = l;

        for(i = 0; i < record_count; i++){

            //get number of occurances for this record
            int occuranceindex_real;
            getOccuranceIndex_real(pair_count, j, i, occuranceindex_real);

            if(occurance_list[i]){

            }
        }
    }
}

void addToParentList(float *pairlist, float *output_buffer, float *parent_list, int *occurance_list, int record_count, int pair_count, int pair_size){

  int i, j, l;
  for(l = 0; l < pair_count; l++){
    int curr_pair = l;

    for(i = 0; i < record_count; i++){
      int curr_record = i;

      //before copying a pair over, first make sure it has not been removed
      int removed;
      pairRemoved(output_buffer, pair_count, curr_record, curr_pair, pair_size, removed);

      if(!removed){
        copyPair(output_buffer, parent_list, pair_count, curr_record, curr_pair, pair_size);
      }
    }
  }
}
*/