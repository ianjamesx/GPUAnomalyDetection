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

#include "device/pairdevice.h"
#include "device/device.h"

#include "hostinit.h"

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

    generatePairs(records, pairs, record_count, record_size, pair_count, pair_size);

    /*
    allocate parent list and output list
    */

    //list used for all batches, keeps track of all substructures found
    float *parent_list;
    pairList_init(parent_list, record_count, record_size, pair_size);

    //buffer used for holding found substructures before comparing to parentlist
    float *pair_buffer;
    pairList_init(pair_buffer, record_count, record_size, pair_size);

    //list of occurances of each substructure, indices align to parentlist
    int *occurance_buffer;
    occuranceList_init(occurance_buffer, record_count, record_size, pair_size);

    /*
    locate patterns in the pairings
    */

    initOutputBuffer(pair_buffer, record_count, pair_count, pair_size);

    copyPairsToBuffer(pairs, pair_buffer, record_count, pair_count, pair_size);

    locatePatterns(pair_buffer, occurance_buffer, record_count, pair_count, pair_size);

    //printOccurances(occurance_buffer, record_count, pair_count);

    cout << "----\n";


    printAllPairsAllIndices_full(pair_buffer, record_count, pair_count, pair_size);

    printOccurances(occurance_buffer, record_count, pair_count);

    return 0;
}