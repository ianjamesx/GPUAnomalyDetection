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
    for(i = 0; i < record_count; i++){ //<-- stride here

        int pair_index = 0;

        for(j = 0; j < record_size; j++){
            for(l = j+1; l < record_size; l++){

                int record_indices[2] = {j, l};

                savePair(pairlist, recordlist, record_count, i, record_indices, pair_index, pair_size);
                
                pair_index++;
            
            }
        }
    }
}

void compressPairs(double *pairlist, double *parent_list, double *output_list, int record_count, int pair_count){

    int i, j;

    for(i = 0; i < record_count; i++){
        for(j = 0; j < pair_count; j++){

            
        }
    }

}

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

    double *parentlist;
    pairList_init(&parentlist, record_count, record_size, pair_size);

    double *outputlist;
    //for outputlist, pair size is 1, as it only holds index of pair in parentlist
    pairList_init(&outputlist, record_count, record_size, 1);

    /*
    run compression phase 1
    */

    int nk = int nk_count(record_size, pair_size);

    return 0;
}
