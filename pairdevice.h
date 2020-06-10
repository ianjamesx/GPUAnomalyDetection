
/*
get starting index of a pair based off
    row - number of rows (corresponds to number of records)
    k   - number of elements per pairing
    i   - index of row
    j   - index of pair
*/
void getPairIndex_real(int row, int i, int j, int k, int *index){
    *index = (i*row + j) * k;
}

/*
save pair of attributes from a record index to its corresponding pair index
*/
void savePair(double *all_pairs, double *record, int record_count, int record_index, int record_indices[], int pair_index, int k){
    
    //get starting index of pair for all_pairs array
    int pairindex_start;
    getPairIndex_real(record_count, record_index, pair_index, k, &pairindex_start);

    //copy elements corresponding to record attributes from record_indices to pair
    int i;
    for(i = 0; i < k; i++){

        //get indices for list of pairs, and list of record indices to copy over
        int all_pair_index = (pairindex_start + i);
        int record_attribute_index = record_indices[i];

        all_pairs[all_pair_index] = record[record_attribute_index];
    }

}

/*
retrieve a pair of attributes from list of all pairs
*/
void getPair(double *all_pairs, double pair[], int pair_i, int pair_j, int k){

    int pairindex_start;
    getPairIndex_real(record_count, record_index, pair_index, k, &pairindex_start);

    int i;
    for(i = 0; i < k; i++){
        int all_pair_index = (pairindex_start + i);
        pair[i] = all_pairs[all_pair_index]
    }

}

/*
sets equal to 1 if pairs are equal
sets equal to 0 if pairs are not equal
*/
void comparePairs(double pair_1[], double pair_2[], int k, int *equal){

    *equal = 1;

    int i;
    for(i = 0; i < k; i++){
        //if one attribute of pair is not equal, we can break early
        if(pair_1[i] != pair_2[i]){
            *equal = 0;
            break;
        }
    }

}