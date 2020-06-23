
//get unique index for a thread
__device__
void getIndex(int blockIdx, int blockDim, int threadIdx, int &index){
    index = blockIdx * blockDim + threadIdx;
}

//get stride for a thread
__device__
void getStride(int blockDimx, int gridDimx, int &stride){
    stride = blockDimx * gridDimx;
}

//get range of workload for a thread to work on
__device__
void getRange(int index, int totalThreads, int workload, int &start, int &stop){

    //get amount of stuff for each thread to cover (round up)
    int workPerNode = ceil(workload / totalThreads);
    if(workPerNode < 1) workPerNode = 1;

    //get starting and stopping index for this thread
    start = index * workPerThread;
    stop = start + workPerThread;

    //if this is the last thread, go until the end of the workload
    if(index >= totalThreads-1){
        stop = workload;
    }

}

/*
get starting index of a pair based off
    nk  - all pairs (820)
    i   - record index
    j   - pair index
    k   - pair size
*/
__device__
void getPairIndex_real(int nk, int i, int j, int k, int &index){
    index = (j * k) + (nk * k * i);
}

__device__
void getRecordAttributeIndex_real(int c, int i, int j, int &index){
    index = (i * c) + j;
}

__device__
void getOccuranceIndex_real(int c, int i, int j, int &index){
    index = (i * c) + j;
}

__device__
void printPair_full(float *all_pairs, int pair_count, int record_index, int pair_index, int pair_size){

    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, pair_size, pairindex_start);

    int i;
    for(i = 0; i < pair_size; i++){
        int pair_index_real = (pairindex_start + i);
        printf("%.2f ", all_pairs[pair_index_real]);
    }

    printf("\n");

}

__device__
void printAllPairs_full(float *all_pairs, int pair_count, int record_index, int pair_size){

    int i, j;

    for(i = 0; i < pair_count; i++){

        int pairindex_start;
        getPairIndex_real(pair_count, record_index, i, pair_size, pairindex_start);

        //print pair, as well as index for pair
        printf("Pair %d (%d) : ", i, pairindex_start);

        for(j = 0; j < pair_size; j++){
            int pair_index_real = (pairindex_start + j);
            printf(" (%d) %.2f ", pair_index_real, all_pairs[pair_index_real]);
        }

         printf("\n");

    }

}
/*
void printAllPairs_full_nobreaks(float *all_pairs, int pair_count, int record_index, int pair_size){

    int i, j;

    for(i = 0; i < pair_count; i++){
        int pairindex_start;
        getPairIndex_real(pair_count, 1, i, pair_size, pairindex_start);

        printf("( ");
        for(j = 0; j < pair_size; j++){
            int pair_index_real = (pairindex_start + j);
            cout << all_pairs[pair_index_real] << " ";
        }

        printf("), ");

    }
    printf("\n");

}
*/
/*
save pair of attributes from a record index to its corresponding pair index
*/
__device__
void savepair_device(float *all_pairs, float *recordlist, int record_size, int record_index, int record_attribute_indices[], int pair_count, int pair_index, int k){

    //get starting index of pair for all_pairs array
    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_start);

    //printf("Saving %d Elements %d - %d ... %d\n", record_index, record_attribute_indices[0], record_attribute_indices[1], pairindex_start);
    int i;
    for(i = 0; i < k; i++){

        //get indices for list of pairs, and list of record indices to copy over
        int all_pair_index = (pairindex_start + i);

        int record_attribute_index;
        getRecordAttributeIndex_real(record_size, record_index, record_attribute_indices[i], record_attribute_index);

        all_pairs[all_pair_index] = recordlist[record_attribute_index];
    }

}

/*
copy a pair from a src array (e.g. initial pair list) to a destination (e.g. parent list)
*/
__device__
void copypair_device(float *src, float *dest, int pair_count, int record_index, int pair_index, int k){

    int pairindex_real;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_real);

    int i;
    for(i = 0; i < k; i++){
        int curr_index = (pairindex_real + i);
        dest[curr_index] = src[curr_index];
    }
}

/*
increment occurances in occurance array
*/
__device__
void addOccurances(int *occurance_list, int pair_count, int record_index, int pair_index, int occurances){

    int occuranceindex_real;
    getOccuranceIndex_real(pair_count, record_index, pair_index, occuranceindex_real);
    occurance_list[occuranceindex_real] += occurances;

}

__device__
void printOccurances(int *occurance_list, int record_count, int pair_count){

    int i, j;
    for(i = 0; i < pair_count; i++){

        printf("Pair (%d): ", i);

        for(j = 0; j < record_count; j++){

            int occuranceindex_real;
            getOccuranceIndex_real(pair_count, j, i, occuranceindex_real);

            printf("%d ", occurance_list[occuranceindex_real]);

        }

        printf("\n");

    }

}

/*
retrieve a pair of attributes from list of all pairs
*/
__device__
void getPair(float *all_pairs, int pair_count, int record_index, int pair_index, int k, float *pair){

    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_start);

    int i;
    for(i = 0; i < k; i++){
        int all_pair_index = (pairindex_start + i);
        pair[i] = all_pairs[all_pair_index];
    }

}

/*
sets equal to 1 if pairs are equal
sets equal to 0 if pairs are not equal
*/
__device__
void comparePairs(float *all_pairs, int pair_count, int record_index_1, int record_index_2, int pair_index, int k, int &equal){

    equal = 1;

    //get starting index of both pairs
    int pair_index_1, pair_index_2;
    getPairIndex_real(pair_count, record_index_1, pair_index, k, pair_index_1);
    getPairIndex_real(pair_count, record_index_2, pair_index, k, pair_index_2);

    int i;
    for(i = 0; i < k; i++){

        int pair_1_curr_index = (pair_index_1 + i);
        int pair_2_curr_index = (pair_index_2 + i);

        //if we find two pairs have an element that does not match
        //set equal to 0, break early
        if(all_pairs[pair_1_curr_index] != all_pairs[pair_2_curr_index]){
            equal = 0;
            //break;
        }

    }

}

/*
when we want to remove a pair from the buffer, we replace all parts of the pair with -1s
purpose of removing pairs: to ensure that we dont double count pairs during location phase
*/
__device__
void removeBufferPair(float *all_pairs, int pair_count, int record_index, int pair_index, int k){

    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_start);

    int i;
    for(i = 0; i < k; i++){
        int all_pair_index = (pairindex_start + i);
        all_pairs[all_pair_index] = -1.0;
    }

}

/*
check if a pair has been removed (any part of the pair is -1.0)
*/
__device__
void pairRemoved(float *all_pairs, int pair_count, int record_index, int pair_index, int k, int &removed){

    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_start);

    removed = 0;

    int i;
    for(i = 0; i < k; i++){
        int all_pair_index = (pairindex_start + i);

        if(all_pairs[all_pair_index] == -1.0){
            removed = 1;
        }
    }
}

/*
see if a pair is in parent list by iterating over each pair in parent list and comparing
if so, save parent index, if not parent index will be -1
*/
__device__
void pairInParentList(float *all_pairs, float *parent_list, int record_count, int record_index, int pair_count, int pair_index, int k, int &parent_index){

    parent_index = -1;

    //get starting index of pair to search for
    int pair_index_real;
    getPairIndex_real(pair_count, record_index, pair_index, k, pair_index);

    int i, j;

    for(i = 0; i < record_count; i++){

        int parent_index_curr;
        getPairIndex_real(pair_count, i, pair_index, k, parent_index_curr);

        int match = 1;

        //compare each element of pair to current pair in parent list
        for(j = 0; j < k; j++){

            int pair_curr = (pair_index_real + i);
            int parent_curr = (parent_index_curr + i);

            //if we find two pairs have an element that does not match
            //set equal to 0, break early
            if(all_pairs[pair_curr] != parent_list[parent_curr]){
                match = 0;
            }

        }

        //if we have a match, set to index found in parent list
        //return early so we dont have any false negatives
        if(match){
            parent_index = i;
            return;
        }

    }

}

__device__
void getNextParentIndex(float *parent_list, int record_count, int record_index, int pair_count, int pair_index, int k, int &next_index){

  int i;

  for(i = 0; i < record_count; i++){

      int parent_index_curr;
      getPairIndex_real(pair_count, i, pair_index, k, parent_index_curr);

      //index containing -1 signifies open space
      //keep this as next space and return
      if(parent_list[parent_index_curr] == -1.0){
        next_index = i;
        return;
      }

  }

}

/*
save pair to parent list
if pair is in parent list, do not save to parent list
if not, put at end
save index of pair in parent list regardless
*/
__device__
void savePairToParentList(float *pair_buffer, float *parent_list, int record_count, int record_index, int pair_count, int pair_index, int k, int &parent_index){

  pairInParentList(pair_buffer, parent_list, record_count, record_index, pair_count, pair_index, k, parent_index);

  //pair not in parent list, get next available space to save it in
  if(parent_index == -1){
    getNextParentIndex(parent_list, record_count, record_index, pair_count, pair_index, k, parent_index);

    //copy pair to next spot (cannot used copyPair(...) as indices for pair indexing will not be same in both buffer and parent list)
    int startindex_buffer, startindex_parent;
    getPairIndex_real(pair_count, record_index, pair_index, k, startindex_buffer);
    getPairIndex_real(pair_count, parent_index, pair_index, k, startindex_parent);
    int i;
    for(i = 0; i < k; i++){
        int curr_index_buffer = (startindex_buffer + i);
        int curr_index_parent = (startindex_parent + i);
        parent_list[curr_index_parent] = pair_buffer[curr_index_buffer];
    }

  }

}
