
/*
locate most common pattern out of all pairings
put pattern data in pd
return 1 if patterns of frequency over 1 still occur, 0 if not
*/

/*
get the pairing indices to search for, given block index, and record size
essentially, map each block to a unique i, j value to ensure all blocks have
a different pairing to work on, if block > CkN (in this case, 820), return invalid i, j
*/
__device__
void getPairingIndices(int block, int rsize, int &first, int &second){

    int i, j, total = 0;
    for(i = 0; i < rsize; i++){
        for(j = i+1; j < rsize; j++){
            if(total == block){
                first = i;
                second = j;
                return;
            }
            total++;
        }
    }

    first = -1;
    second = -1;
}

//get index of attribute in 1d representation of 2d array
__device__
void getAttributeIndex(int record_size, int i, int j, int &index){
    index = (i * record_size) + j;
}

__global__
void mostFreqPattern(float *records, int record_size, int record_count, patternData *pd){

    //get pairing indices for this block
    int index1, index2;
    getPairingIndices(blockIdx.x, record_size, index1, index2);

    //will need threads * 3 for occurance array, formatted as occurances, element1, element2 ...
    extern __shared__ int occurances[];

    int max_oc = 0; //largest number of occurances found
    float max_p1, max_p2; //most common pairing

    int iters = 0;

    int i, j;
    for(i = threadIdx.x; i < record_count; i += blockDim.x){ //stride here

        //get current indicies for assigned pair for current record
        int r1_attr1_ind, r1_attr2_ind;
        getAttributeIndex(record_size, i, index1, r1_attr1_ind);
        getAttributeIndex(record_size, i, index2, r1_attr2_ind);

        int curr_oc = 0;

        if(blockIdx.x == 0){
            //printf("comparing to record %d (attrs %d %d) on thread %d\n", i, index1, index2, threadIdx.x);
        }

        //ensure pairing is not negative duplicate (e.g. elements of same compression, -4,-4, -5,-5 etc)
        if(records[r1_attr1_ind] == records[r1_attr2_ind] && records[r1_attr1_ind] < 0) continue;

        for(j = i+1; j < record_count; j++){

            iters++;

            //get record pairing to compare to
            int r2_attr1_ind, r2_attr2_ind;
            getAttributeIndex(record_size, j, index1, r2_attr1_ind);
            getAttributeIndex(record_size, j, index2, r2_attr2_ind);
            
            //debug code
            /*
            int equal = 0;
            if(records[r1_attr1_ind] == records[r2_attr1_ind] && records[r1_attr2_ind] == records[r2_attr2_ind]) equal = 1;
            if(blockIdx.x == 808 && threadIdx.x == 0){
                printf("Comparing attr %.2f %.2f & %.2f %.2f : %d\n", records[r1_attr1_ind], records[r2_attr1_ind], records[r1_attr2_ind], records[r2_attr2_ind], equal);
            }
            */

            //see if record attributes are equal
            if(records[r1_attr1_ind] == records[r2_attr1_ind] && records[r1_attr2_ind] == records[r2_attr2_ind]){
                curr_oc++;
            }
        }

        //if we get a new max, save it as most common pattern
        if(curr_oc > max_oc){
            max_oc = curr_oc;
            max_p1 = records[r1_attr1_ind];
            max_p2 = records[r1_attr2_ind];
        }

        //debug code
        /*
        if(blockIdx.x == 808 && threadIdx.x == 0){
            printf("Found %d matches on pair %.2f %.2f\n", max_oc, records[r1_attr1_ind], records[r1_attr2_ind], records[r1_attr2_ind]);
        }
        */
    }

    //put most common pattern in shared array at this threads index
    occurances[threadIdx.x] = max_oc;
    __syncthreads();

    //on root thread (0) find most common pattern
    if(threadIdx.x == 0){
        int allmax_oc = max_oc, allmax_index = 0;
        for(i = 0; i < blockDim.x; i++){
            if(occurances[i] > allmax_oc){
                allmax_oc = occurances[i];
                allmax_index = i;
            }
        }
        //printf("Block %d Index %d has %d OCs\n", blockIdx.x, allmax_index, allmax_oc);
        occurances[0] = allmax_index;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        //printf("block %d done\n", blockIdx.x);
    }

    //if this thread is most frequent pattern, assign data to output array at unique pattern spot
    if(occurances[0] == threadIdx.x){
        pd[blockIdx.x].index1 = index1;
        pd[blockIdx.x].index2 = index2;
        pd[blockIdx.x].pattern.first = max_p1;
        pd[blockIdx.x].pattern.second = max_p2;
        pd[blockIdx.x].occurances = max_oc;
        //printf("Block %d Index %d has %d occurances on pair %.2f %.2f\n", blockIdx.x, threadIdx.x, max_oc, max_p1, max_p2);
    }

}