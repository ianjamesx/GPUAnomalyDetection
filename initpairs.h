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

void pairList_init(double **all_pairs, int record_count, int record_attribute_count, int pair_length){

    //number of unique pairings per record (CkN, or N choose K)
    int nk = nk_count(record_attribute_count, pair_length);

    //for allocating each records pairings, we need k (pair_length) spaces for each pair
    int record_pairspace = nk * pair_length;

    int size = (record_pairspace * record_count) * sizeof(double);
    *all_pairs = malloc(size);
}

