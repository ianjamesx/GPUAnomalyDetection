#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.h"
#include "recorddevice.h"
#include "pairdevice.h"
#include "initpairs.h"
#include "parentlist.h"
#include "hostread.h"

#define pair_size 2

//__GLOBAL__
void generatePairs(double *recordlist, double *pairlist, int record_count, int record_size, int pair_count){

    int i, j, l;
    for(i = 0; i < record_count; i++){ //<-- stride here by unique id

        int pair_index = 0;

        for(j = 0; j < record_size; j++){
            for(l = j+1; l < record_size; l++){

                int record_indices[2] = {j, l};

                savePair(pairlist, recordlist, record_count, i, record_indices, pair_count, pair_index, pair_size);
                
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

void locatePatterns(double *pairlist, double *output_buffer, int *occurance_list, int record_count, int pair_count){

    int i, j, l;

    int comps = 0;
    int matches = 0;

    //comparePairs(double *all_pairs, int record_count, int record_index_1, int record_index_2, int pair_index, int k, int *equal)

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

                comparePairs(pairlist, pair_count, curr_record1, curr_record2, curr_pair, pair_size, &isequal);

                //if we found matching pair, copy this pair to the output buffer at corresponding index for current record
                if(isequal){
 
                    copyPair(pairlist, output_buffer, record_count, curr_record1, curr_pair, pair_size);

                    occurances++;

                    matches++;

                }
                
                comps++;

            }

            if(curr_pair == 711){
                printf("~~\n");
                printPair_full(pairlist, pair_count, curr_record1, curr_pair, pair_size);
            }
            

            addOccurances(occurance_list, record_count, curr_record1, curr_pair, occurances);

        }

    }

    printf("comps: %d, matches: %d\n", comps, matches);

}
/*
void reducePatterns(double *parent_list, double *output_buffer){


}*/

int main(int argc, char **argv){

    /*
    read in dataset
    */
    Matrix M;
    int *recordflags;
    char **attacks;

    hostread(&M, recordflags, attacks);

    /*
    convert matrix read in format to device friendly array
    */

    double *recordlist;
    int record_count = M.rows;
    int record_size = M.cols;

    recordlist_init(&recordlist, record_count, record_size);
    int i;
    for(i = 0; i < (record_count * record_size); i++){
        recordlist[i] = M.data[i];
    }

    /*
    generate pairs
    */

    int pair_count = nk_count(record_size, pair_size);

    double *pairlist;
    pairList_init(&pairlist, record_count, record_size, pair_size);
    generatePairs(recordlist, pairlist, record_count, record_size, pair_count);

    /*
    allocate parent list and output list
    */

    //list used for all batches, keeps track of all substructures found
    double *parentlist;
    pairList_init(&parentlist, record_count, record_size, pair_size);

    //buffer used for holding found substructures before comparing to parentlist
    double *outputlist_buffer;
    pairList_init(&outputlist_buffer, record_count, record_size, pair_size);

    //list of occurances of each substructure, indices align to parentlist
    int *occurancelist;
    occuranceList_init(&occurancelist, record_count, record_size, pair_size);

    /*
    run compression phase 1
    */

    printf("%d %d %d\n", record_count, pair_count, pair_size);;

    //printAllPairs_full(pairlist, pair_count, 0, pair_size);
    //printAllPairs_full(pairlist, pair_count, 1, pair_size);

/*
    int equal = 5;
    printf("COMPARING record 1, record 2, pair 2\n");
    comparePairs(pairlist, record_count, 1, 12, 31, pair_size, &equal);
    printf("\n equal is %d\n", equal);

*/

    printf("Record Count: %d, nk: %d\n", record_count, pair_count);

    initOutputBuffer(outputlist_buffer, record_count, pair_count, pair_size);

    locatePatterns(pairlist, outputlist_buffer, occurancelist, 2, pair_count);

    printf("-----\n");

    printAllPairsFromBuffer_full(outputlist_buffer, 2, pair_count, 711, pair_size);

    printf("-----\n");

    printOccurances(occurancelist, record_count, pair_count);

    //printAllPairs_full(outputlist_buffer, pair_count, 0, pair_size);

    return 0;
}
