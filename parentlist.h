/*
print every pattern found and stored in an output buffer
*/
void printAllPairsFromBuffer_full(float *output_buffer, int record_count, int pair_count, int pair_index, int pair_size){

    int i, j;
    for(i = 0; i < record_count; i++){

        int pairindex_start;
        getPairIndex_real(pair_count, i, pair_index, pair_size, pairindex_start);

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
