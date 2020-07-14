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
#include <iomanip>

//thrust utils (for GPU accel)

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;
using namespace std::chrono;

#include "dataread.h"

struct range {
  float min;
  float max;
};

#include "cluster_device.cuh"

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

//floating point comps
bool cmfl(float x, float y, float epsilon = 0.01f){
  if(fabs(x - y) < epsilon)
     return true; //they are same
     return false; //they are not same
}

//get index of edge from 1d representation of a matrix
int getMatrixIndex(int record_size, int i, int j){
    return (i * record_size) + j;
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

bool symbolic(int index){
  if(index == 1 || index == 2 || index == 3){
    return true;
  }
  return false;
}

/*
get range [min, max] of all continous attributes
*/
vector<range> attributeRanges(vector<vector<float> > records){

  vector<range> ranges;

  int i, j;
  int attr_count = records[0].size();
  for(i = 0; i < attr_count; i++){

    int attr = i;

    //init min, max
    int min = records[0][attr];
    int max = records[0][attr];

    for(j = 0; j < records.size(); j++){

      //update min/max
      float curr = records[j][attr];
      if(curr < min) min = curr;
      if(curr > max) max = curr;

    }

    //save min/max in ranges
    range curr_range;
    curr_range.min = min;
    curr_range.max = max;
    ranges.push_back(curr_range);
  }

  return ranges;

}

float scaleValue(float r_min, float r_max, float t_min, float t_max, float m){

  //if rmax == rmin (usualy for 0 values) return t_min (usually 0, to symbolize all values will be identical)
  if(r_max == r_min){
    return t_min;
  }

  float val = (((m - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min);

  return val;

}

void vectorsubtraction(vector<float> v1, vector<float> v2, vector<float> &v3){

    int i;
    //vectors must be same size
    if(v1.size() != v2.size()) return;
    for(i = 0; i < v1.size(); i++){
        //if symbolic attribute, push 1 to represent not equal, 0 to represent equal
        if(symbolic(i)){
          v3.push_back(v1[i] != v2[i]);
          //if continous, subtract values
        } else {
          v3.push_back((v1[i] - v2[i]));
        }
    }
}

float normalizevector(vector<float> v){

    int i;
    float total = 0.0;
    for(i = 0; i < v.size(); i++){
        total += (v[i] * v[i]);
    }

    return sqrt(total);
}

double edgeweight(vector<float> v1, vector<float> v2){

    vector<float> v3;
    vectorsubtraction(v1, v2, v3);

    float normalized = normalizevector(v3);
    double base = exp(normalized);
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


    //double edge = edgeweight(records[0], records[4]);

    vector<range> ranges = attributeRanges(records);

    int i, j;

    for(i = 0; i < ranges.size(); i++){
      cout << "(" << ranges[i].min << ", " << ranges[i].max << ") ";
    }

    cout << endl;

    /*
    scale values
    */

    for(i = 0; i < records.size(); i++){
      for(j = 0; j < records[i].size(); j++){
        if(symbolic(j)){ //unless symbolic attribute
          continue;
        }
        records[i][j] = scaleValue(ranges[j].min, ranges[j].max, 0, 1, records[i][j]);
      }
    }

    //edgeGeneration<<<2, 4>>>(NULL, 2, 2);

    /*
    populate edge matrix
    */


    //float *edgematrix = new float[records.size() * records.size()];

    //printRecords(records);

    float *edgematrix, *recordmatrix;
    int edge_msize = records.size() * records.size();
    int records_msize = records.size() * record_size;
    gpuErrchk(cudaMallocManaged(&edgematrix, edge_msize * sizeof(float)));
    gpuErrchk(cudaMallocManaged(&recordmatrix, records_msize * sizeof(float)));

    //copy records to a unified array from stl vector (so it can work on cuda kernel)
    int curr = 0;
    for(i = 0; i < records.size(); i++){
      for(j = 0; j < record_size; j++){
        recordmatrix[curr] = records[i][j];
        curr++;
      }
    }


    /*
    printRecords(records);
    cout << "-------------\n";
    for(i = 0; i < record_count; i++){
      for(j = 0; j < record_size; j++){
        int currindex = getMatrixIndex(record_size, i, j);
        cout << recordmatrix[currindex] << " ";
      }
      cout << endl;
    }
    */

    //start kernel
  
    edgeGeneration<<<16, 64>>>(edgematrix, recordmatrix, record_size, record_count);
    cudaDeviceSynchronize();

    cout << "--------------\n";

    //printRecords(records);

    
    /*

    for(i = 0; i < records.size(); i++){
        for(j = 0; j < records.size(); j++){
          int curredge = getMatrixIndex(records.size(), i, j);
          cout << setprecision(4) << edgematrix[curredge] << " ";
        }
      cout << endl;
    }

    */


    
    /*
    //COMPARING DEVICE TO HOST IMPLEMENTATION ON EDGE CALCS
    
    float *edgematrix_host = new float[edge_msize];
    for(i = 0; i < records.size(); i++){
        for(j = 0; j < records.size(); j++){
          int curredge = getMatrixIndex(records.size(), i, j);
          if(i == j){
            edgematrix_host[curredge] = 0;
            continue;
          }
          edgematrix_host[curredge] = edgeweight(records[i], records[j]);
        }
    }


    for(i = 0; i < records.size(); i++){
      for(j = 0; j < records.size(); j++){
        int curredge = getMatrixIndex(records.size(), i, j);
        if(!cmfl(edgematrix[curredge], edgematrix_host[curredge])){
          cout << edgematrix_host[curredge] << " : " << edgematrix[curredge] << endl;
        }
      }
    }
    printf("ALL GOOD!\n");
    */
    
    



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
