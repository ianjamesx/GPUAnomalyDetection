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
#include "hostread.h"

#define pair_size 2

//__GLOBAL__
void generatePairs(double *recordlist, double *pairlist, int record_count, int record_size){

    int i, j, l;
    for(i = 0; i < record_count; i++){ //<-- stride here by unique id

        int pair_index = 0;

        for(j = 0; j < record_size; j++){
            for(l = j+1; l < record_size; l++){

                int record_indices[2] = {j, l};

                printf("Pair %.2f %.2f\n", recordlist[record_indices[0]], recordlist[record_indices[1]]);

                savePair(pairlist, recordlist, record_count, i, record_indices, pair_index, pair_size);
                
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
*/
/*
void locatePatterns(double *pairlist, double *output_buffer, int *occurance_list, int record_count, int pair_count){

    int i, j, l;

    //comparePairs(double *all_pairs, int record_count, int record_index_1, int record_index_2, int pair_index, int k, int *equal)

    for(l = 0; l < pair_count; l++){ //<-- stride here (by blocks)

        int curr_pair = l;

        for(i = 0; i < record_count; i++){

            int curr_record1 = i;

            for(j = i+1; j < record_count; j++){

                int curr_record2 = j;

                int isequal;

                comparePairs(pairlist, record_count, curr_record1, curr_record2, curr_pair, pair_count, &isequal);

                printf("%d", isequal);

            }
        }

    }

}

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

    double *pairlist;
    pairList_init(&pairlist, record_count, record_size, pair_size);
    generatePairs(recordlist, pairlist, 1, record_size);

    /*
    allocate parent list and output list
    */

    //list used for all batches, keeps track of all substructures found
    double *parentlist;
    pairList_init(&parentlist, record_count, record_size, pair_size);

    //buffer used for holding found substructures before comparing to parentlist
    double *parentlist_buffer;
    pairList_init(&parentlist_buffer, record_count, record_size, pair_size);

    //list of occurances of each substructure, indices align to parentlist
    int *occurancelist;
    occuranceList_init(&occurancelist, record_count, record_size, 1);
    
    /*
    run compression phase 1
    */

    int nk = nk_count(record_size, pair_size);

    printRecord_full(recordlist, record_count, record_size, 0);
    printf("------\n");
    
    //printPair_full(pairlist, record_count, 1, 2, pair_size);
    //printPair_full(pairlist, record_count, 2, 2, pair_size);
    //printRecord_full(recordlist, record_count, record_size, 1);

    //printAllPairs_full(pairlist, record_count, 0, nk, pair_size);

    int equal = 5;
    printf("COMPARING record 1, record 2, pair 2\n");
    comparePairs(pairlist, record_count, 1, 2, 2, pair_size, &equal);

    printf("\n equal is %d\n", equal);

    return 0;
}
