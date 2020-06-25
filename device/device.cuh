
__global__
void generatePairs(float *recordlist, float *pairlist, int record_start, int record_stop, int record_size, int pair_count, int pair_size){

    int i, j, l;

    int index, stride;
    getIndex(blockIdx.x, blockDim.x, threadIdx.x, index);
    getStride(blockDim.x, gridDim.x, stride);

    for(i = (index + record_start); i < record_stop; i += stride){ //<-- stride here by index

        int pair_index = 0;

        //printf("Thread %d pairing record %d\n", index, i);

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
distribute a unique pairing to each block
have each thread in block cover a range of record pairings
compress all patterns, by end, all repeated patterns will be replaced with (-1...-1), OC replaced with 0
*/

__global__
void locatePatterns(float *pair_buffer, int *occurance_list, int record_count, int pair_count, int pair_size, int threadmax){

    //iterators
    int curr_pair, curr_record, comparing_record;

    //get range for this thread to cover
    int index, start, stop;
    getRange(threadIdx.x, threadmax, record_count, start, stop);
    //printf("Thread %d getting records %d ---> %d\n", threadIdx.x, start, stop);

    //pair covered is blockID
    curr_pair = blockIdx.x;

    //for(curr_pair = 0; curr_pair < pair_count; curr_pair++){ 
        for(curr_record = start; curr_record < stop; curr_record++){

            //first check if current pair has been removed, if it has, skip this pair
            int removed;
            pairRemoved(pair_buffer, pair_count, curr_record, curr_pair, pair_size, removed);
            if(removed) continue;

            //number of occurances of this pattern in other records
            int occurances;
            occuranceCountInit(occurance_list, pair_count, curr_record, curr_pair, occurances);

            for(comparing_record = curr_record+1; comparing_record < stop; comparing_record++){

                int isequal;
                comparePairs(pair_buffer, pair_count, curr_record, comparing_record, curr_pair, pair_size, isequal);

                if(isequal){

                    //increment occurances to number of occurances in other index
                    int otherOccurances;
                    getOccuranceCount(occurance_list, pair_count, comparing_record, curr_pair, otherOccurances);
                    occurances += otherOccurances;

                    removeBufferPair(pair_buffer, pair_count, comparing_record, curr_pair, pair_size);
                    removeBufferOC(occurance_list, pair_count, comparing_record, curr_pair);
                }
            }

            //add number of occurances of this pattern to list
            addOccurances(occurance_list, pair_count, curr_record, curr_pair, occurances);
        }
    //}
}

/*
similar to pattern compression, take all compressed patterns (occurance counts, patterns)
add occurances to same pattern in parent list if already exists, if not, add pattern to parent list
have a block cover each unique pairing
*/
/*
__global__
void reduceToParentList(float *pair_buffer, float *parent_list, int *occurance_list, int *occurance_buffer, int record_count, int pair_count, int pair_size, int threadmax){

    int curr_pair, curr_record, comparing_record;

    int index, start, stop;
    getRange(threadIdx.x, threadmax, record_count, start, stop);
    printf("Thread %d getting records %d ---> %d\n", threadIdx.x, start, stop);
    curr_pair = blockIdx.x;

    for(curr_pair = 0; curr_pair < pair_count; curr_pair++){ 
        for(curr_record = start; curr_record < stop; curr_record++){

            //first check if current pair has been removed, if it has, skip this pair
            int removed;
            pairRemoved(pair_buffer, pair_count, curr_record, curr_pair, pair_size, removed);
            if(removed) continue;

            int occurances = 1;
            
            for(comparing_record = curr_record+1; comparing_record < stop; comparing_record++){

                int isequal;
                comparePairs(pair_buffer, pair_count, curr_record, comparing_record, curr_pair, pair_size, isequal);

                if(isequal){
                    removeBufferPair(pair_buffer, pair_count, comparing_record, curr_pair, pair_size);
                    occurances++;
                }
            }

            //add number of occurances of this pattern to list
            addOccurances(occurance_list, pair_count, curr_record, curr_pair, occurances);
        }
    }

}
*/
/*
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