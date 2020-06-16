
//__GLOBAL__
void generatePairs(float *recordlist, float *pairlist, int record_count, int record_size, int pair_count, int pair_size){

    int i, j, l;
    for(i = 0; i < record_count; i++){ //<-- stride here by unique id

        int pair_index = 0;

        for(j = 0; j < record_size; j++){
            for(l = j+1; l < record_size; l++){

                int record_indices[2] = {j, l};

                savePair(pairlist, recordlist, record_size, i, record_indices, pair_count, pair_index, pair_size);
                
                pair_index++;
            
            }
        }
    }
}

/*
distribute a unique pairing to each thread-block
thread block will find all patterns in pairings and move to the output buffer
then threads will search output buffer for duplicates, and merge into list of unique patterns
keeping track of number of occurances for each

also, replace the pattern in the pairlist with the index in parent list
will have to use negative index

total ops: Ck(R) * NK
where NK = number of unique pairings per record (820), R = record count 
*/

void locatePatterns(float *pairlist, float *output_buffer, int *occurance_list, int record_count, int pair_count, int pair_size){

    int i, j, l;

    for(l = 0; l < pair_count; l++){ //<-- stride here (by blocks)

        int curr_pair = l;

        //generate range for this thread to cover
        for(i = 0; i < record_count; i++){

            //index of the record to compare all others to
            int curr_record1 = i;
            //number of occurances of this pattern in other records
            int occurances = 0;

            for(j = i+1; j < record_count; j++){

                int curr_record2 = j;
                int isequal;
                comparePairs(pairlist, pair_count, curr_record1, curr_record2, curr_pair, pair_size, isequal);

                //if we found matching pair, copy this pair to the output buffer at corresponding index for current record
                if(isequal){
                    //cout << "match on patterm " << curr_pair << " for record " << curr_record1 << ", " << curr_record2 << endl;
                    copyPair(pairlist, output_buffer, record_count, curr_record1, curr_pair, pair_size);
                    occurances++;
                }
            }
            //add number of occurances of this pattern to list
            addOccurances(occurance_list, pair_count, curr_record1, curr_pair, occurances);
        }

    }

}

/*
void reducePatterns(float *parent_list, double *output_buffer){


}
*/