int nk_count(int n, int k){

    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;

}

void pairList_init(float *&pair_list, int record_count, int record_attribute_count, int pair_size){

    //number of unique pairings per record (CkN, or N choose K)
    int nk = nk_count(record_attribute_count, pair_size);

    //for allocating each records pairings, we need k (pair_size) spaces for each pair
    int record_pairspace = nk * pair_size;

    int size = (record_pairspace * record_count);
    pair_list = new float[size];
}

void occuranceList_init(int *&occurance_list, int record_count, int record_attribute_count, int pair_size){

    //number of unique pairings per record (CkN, or N choose K)
    int nk = nk_count(record_attribute_count, pair_size);

    int size = (nk * record_count);
    occurance_list = new int[size];
}


