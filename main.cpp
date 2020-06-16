#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

#include "record.h"
#include "dataread.h"
#include "initpairs.h"

#include "device/pairdevice.h"
#include "device/device.h"

#include "parentlist.h"

int main(){

    /*
    init dataset
    */

    float *records;
    int *record_types;
    int record_count, record_size;

    init_dataset(record_count, record_size, records, record_types);

    /*
    generate pairs
    */

    int pair_size = 2;

    int i, j;

    int pair_count = nk_count(record_size, pair_size);

    float *pairs;
    pairList_init(pairs, record_count, record_size, pair_size);
/*
    for(i = 0; i < record_count * record_size; i++){
        cout << records[i] << endl;
    }*/

    generatePairs(records, pairs, record_count, record_size, pair_count, pair_size);

    //printAllPairs_full(pairs, pair_count, 0, pair_size);

    /*
    allocate parent list and output list
    */

    //list used for all batches, keeps track of all substructures found
    float *parentlist;
    pairList_init(parentlist, record_count, record_size, pair_size);

    //buffer used for holding found substructures before comparing to parentlist
    float *outputlist_buffer;
    pairList_init(outputlist_buffer, record_count, record_size, pair_size);

    //list of occurances of each substructure, indices align to parentlist
    int *occurancelist;
    occuranceList_init(occurancelist, record_count, record_size, pair_size);

    /*
    locate patterns in the pairings
    */

    initOutputBuffer(outputlist_buffer, record_count, pair_count, pair_size);

    locatePatterns(pairs, outputlist_buffer, occurancelist, record_count, pair_count, pair_size);

    printOccurances(occurancelist, record_count, pair_count);


    return 0;
}