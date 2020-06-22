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

/*

indexing operations

*/

void getPairIndex_host(int nk, int i, int j, int k, int &index){
    index = (j * k) + (nk * k * i);
}

/*

shared memory array size getters

*/
int pairlist_size(int record_count, int record_size, int pair_size){

    //number of unique pairings per record (CkN, or N choose K)
    int nk = nk_count(record_size, pair_size);

    //for allocating each records pairings, we need k (pair_size) spaces for each pair
    int record_pairspace = nk * pair_size;

    int size = (record_pairspace * record_count);

    return size;
}

int occurancelist_size(int record_count, int record_size, int pair_size){

    //number of unique pairings per record (CkN, or N choose K)
    int nk = nk_count(record_size, pair_size);

    int size = (nk * record_count);
    return size;
}

void occurancelist_init(int *occurance_list, int size){

    //initialize all occurances to 0
    int i;
    for(i = 0; i < size; i++){
        occurance_list[i] = 0;
    }
}

/*
print every pattern found and stored in an output buffer
*/
void printAllPairsFromBuffer_full(float *output_buffer, int record_count, int pair_count, int pair_index, int pair_size){

    int i, j;
    for(i = 0; i < record_count; i++){

        int pairindex_start;
        getPairIndex_host(pair_count, i, pair_index, pair_size, pairindex_start);

        for(j= 0; j < pair_size; j++){

            int pair_index_real = (pairindex_start + j);

            //if value of pattern is -1, then no pattern has been stored yet
            //do not print if under 0
            if(output_buffer[pair_index_real] >= 0.0){
                printf("%.2f ", output_buffer[pair_index_real]);
            }
        }

        printf("\n");

    }

}

void printAllPairsAllIndices_full(float *output_buffer, int record_count, int pair_count, int pair_size){


    int i, j, l;

    for(l = 0; l < pair_count; l++){

        int pair_index = l;

        cout << "Pair " << pair_index << ": ";
        int any = 0;

        for(i = 0; i < record_count; i++){

            int pairindex_start;
            getPairIndex_host(pair_count, i, pair_index, pair_size, pairindex_start);

            cout << "( ";

            for(j= 0; j < pair_size; j++){

                int pair_index_real = (pairindex_start + j);

                //if value of pattern is -1, then no pattern has been stored yet
                //do not print if under 0
                //if(output_buffer[pair_index_real] >= 0.0){
                    any = 1;
                    cout << output_buffer[pair_index_real] << " ";
                //}
            }

            cout << ")";

        }

        cout << endl;

    }

}


/*
initialize output buffer to all -1, signaling no pattern stored as of yet
*/
void initOutputBuffer(float *output_buffer, int record_count, int pair_count, int pair_size){

    int bufferlen = (pair_size * pair_count) * record_count;
    int i;
    for(i = 0; i < bufferlen; i++){
        output_buffer[i] = -1.0;
    }

}

void copyPairsToBuffer_host(float *pair_list, float *pair_buffer, int record_count, int pair_count, int pair_size){
    
    int bufferlen = (pair_size * pair_count) * record_count;
    int i;
    for(i = 0; i < bufferlen; i++){
        pair_buffer[i] = pair_list[i];
    }

}

void printAllPairs_host(float *all_pairs, int pair_count, int record_index, int pair_size){

    int i, j;

    for(i = 0; i < pair_count; i++){

        int pairindex_start;
        getPairIndex_host(pair_count, record_index, i, pair_size, pairindex_start);

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

record operations 

*/

void attributeindex_real(int c, int i, int j, int *index){
    *index = (i * c) + j;
}

void getRecord(float *all_records, int record_count, int record_size, int record_index, float **record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        *record[i] = all_records[attributeindex];
    }

}

void saveRecord(float *all_records, int record_count, int record_size, int record_index, float *record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_size, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        all_records[attributeindex] = record[i];
    }

}

void printRecord_full(float *all_records, int record_count, int record_size, int record_index){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_size, record_index, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        cout << all_records[attributeindex] << " ";
    }
    cout << endl;

}