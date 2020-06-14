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

#include "stringutils.h"
#include "recorddevice.h"
#include "dataset/dataread.h"

using namespace std;

int main(){

    float *records;
    int *record_types;

    int record_count, record_size;

    init_dataset(record_count, record_size, records, record_types);

    return 0;
}