#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <utility>
#include <set>
#include <chrono> 

//thrust utils (for GPU accel)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;
using namespace std::chrono; 

#include "dataread.h"

//device error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printRecords(vector<vector<float> > records){

    int i, j;
    for(i = 0; i < records.size(); i++){
        for(j = 0; j < records[i].size(); j++){
            cout << records[i][j] << ' ';
        }
        cout << endl;
    }

}

void vectorsubtraction(vector<float> v1, vector<float> v2, vector<float> &v3){

    int i;
    //vectors must be same size
    if(v1.size() != v2.size()) return;
    for(i = 0; i < v1.size(); i++){
        v3.push_back((v1[i] - v2[i]));
    }
}

float normalizevector(vector<float> v){

    int i;
    float total = 0.0;
    for(i = 0; i < v.size(); i++){
        total += v[i];
    }

    cout << "normalized: " << sqrt(total) << endl;

    return sqrt(total);
}

double edgeweight(vector<float> v1, vector<float> v2){

    vector<float> v3;
    vectorsubtraction(v1, v2, v3);

    int i;
    for(i = 0; i < v1.size(); i++){
        cout << v1[i] << " ";
    }
    cout << endl;
    for(i = 0; i < v2.size(); i++){
        cout << v2[i] << " ";
    }
    cout << endl;
    for(i = 0; i < v3.size(); i++){
        cout << v3[i] << " ";
    }
    cout << endl;

    float normalized = normalizevector(v3);

    double base = exp(normalized);

    cout << "e^x " << base << endl;

    return (1 / base);
}


int main(){

    /*
    init dataset
    */

    vector<vector<float> > records;
    vector<int> record_types;
    int record_count, record_size;
    init_dataset(record_count, record_size, records, record_types);

    cout <<  record_count << endl;

    float matrix[record_count][record_count];
/*
    double param, result;
    param = 5.0;
    result = exp (param);
    printf ("The exponential value of %f is %f.\n", param, result );
*/

    double edge = edgeweight(records[0], records[4]);

    cout << fixed << "0, 1: " << edge << endl;

    cout << "------------------\n";

    //double edge2 = edgeweight(records[0], records[2]);

/*
    int i, j;
    for(i = 0; i < records.size(); i++){
        for(j = 0; j < records.size(); j++){
            if(i == j) cout << i << ", " << j << ": " << 0 << endl;
            cout << i << ", " << j << ": " << edgeweight(records[i], records[j]) << endl;
        }
    }
*/
    return 0;
}