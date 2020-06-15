
/*
get starting index of a pair based off
    row - number of rows (corresponds to number of records)
    k   - number of elements per pairing
    i   - index of row
    j   - index of pair

    CURRENT STATUS: MIGHT WORK
*/
void getPairIndex_real(int nk, int i, int j, int k, int &index){
    index = (j * k) + (nk * k * i);
}

void getRecordAttributeIndex_real(int r, int i, int j, int &index){
    index = (i * r) + j;
}

void getOccuranceIndex_real(int r, int i, int j, int &index){
    index = (i * r) + j;
}

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
save pair of attributes from a record index to its corresponding pair index
*/
void savePair(float *all_pairs, float *recordlist, int record_count, int record_index, int record_attribute_indices[], int pair_count, int pair_index, int k){
    
    //get starting index of pair for all_pairs array
    int pairindex_start;
    getPairIndex_real(pair_count, record_index, pair_index, k, pairindex_start);

    //copy elements corresponding to record attributes from record_indices to pair
    int i;

    //cout << "Saving " << record_index << " Elements " << record_attribute_indices[0] << " - " << record_attribute_indices[1] << "... " << pairindex_start<< endl;
    for(i = 0; i < k; i++){

        //get indices for list of pairs, and list of record indices to copy over
        int all_pair_index = (pairindex_start + i);

        int record_attribute_index;// = (record_count * record_index) + record_indices[i];
        getRecordAttributeIndex_real(record_count, record_index, record_attribute_indices[i], record_attribute_index);

        //cout << recordlist[record_attribute_index] << " ";

        //printf("%d --> %.2f ", all_pair_index, recordlist[record_attribute_index]);

        all_pairs[all_pair_index] = recordlist[record_attribute_index];
    }

    //cout << "\n";

    //printf("\n");

}

/*
copy a pair from a src array (e.g. initial pair list) to a destination (e.g. parent list)
*/
void copyPair(float *src, float *dest, int pair_count, int record_index, int pair_index, int k){

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
void addOccurances(int *occurance_list, int record_count, int pattern_index, int pair_index, int occurances){

    int occuranceindex_real;
    getOccuranceIndex_real(record_count, pattern_index, pair_index, occuranceindex_real);

    occurance_list[occuranceindex_real] += occurances;
    //printf("added %d to %d %d: %d\n", occurances, record_count, pair_index,  occurance_list[occuranceindex_real]);

}

void printOccurances(int *occurance_list, int record_count, int pair_count){
/*
    int occlen = pair_count * record_count;
    int i;

    for(i = 0; i < occlen; i++){
        if(occurance_list[i] > 0){
            printf("Pair %d\n", occurance_list[i]);
        }
    }
*/

    int i, j;
    for(i = 0; i < pair_count; i++){

        if(i != 711) continue;

        printf("Pair (%d): ", i);

        for(j = 0; j < record_count; j++){

            int occuranceindex_real;
            getOccuranceIndex_real(record_count, j, i, occuranceindex_real);

            printf("%d ", occurance_list[occuranceindex_real]);

        }

        printf("\n");

    }

}

/*
retrieve a pair of attributes from list of all pairs
*/
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

void pairInParentList(float *all_pairs, double *parent_list, int record_count, int record_index, int pair_index, int k, int &parent_index){

    parent_index = -1;

    //get starting index of pair to search for
    int pair_index_real;
    getPairIndex_real(record_count, record_index, pair_index, k, pair_index);

    int i, j;

    for(i = 0; i < record_count; i++){

        int parent_index_curr;
        getPairIndex_real(record_count, i, pair_index_real, k, parent_index_curr);

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

        //if we have a match, set return index to index found in parent list
        if(match){
            parent_index = i;
        }

    }

}

//void savePairToParentList()