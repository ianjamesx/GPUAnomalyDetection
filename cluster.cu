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

struct edgedata {
  float weight;
  int vertex;
};

struct location {
  float x;
  float y;
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

bool edgeRemoved(edgedata edge){
  if(edge.vertex < 0){
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

    /*
    populate edge matrix
    */

    int k = 25;

    float *recordmatrix;
    edgedata *edgematrix;
    int edge_msize = records.size() * k;//records.size();
    int records_msize = records.size() * record_size;
    cout << "ems: " << edge_msize << ", " << records_msize << endl;
    gpuErrchk(cudaMallocManaged(&edgematrix, edge_msize * sizeof(edgedata)));
    gpuErrchk(cudaMallocManaged(&recordmatrix, records_msize * sizeof(float)));

    //copy records to a unified array from stl vector (so it can work on cuda kernel)
    int curr = 0;
    for(i = 0; i < records.size(); i++){
      for(j = 0; j < record_size; j++){
        recordmatrix[curr] = records[i][j];
        curr++;
      }
    }

    //init all elements in edge matrix to -1
    for(i = 0; i < edge_msize; i++){
      edgematrix[i].weight = -1.0;
    }


    //start kernel
    edgeGeneration<<<16, 64>>>(edgematrix, recordmatrix, record_size, record_count, k);
    cudaDeviceSynchronize();

    cout << "--------------\n";

    //printRecords(records);

    
    //print k nearest neighbors
    for(i = 0; i < 10; i++){
      for(j = 0; j < 5; j++){
        int curredge = getMatrixIndex(k, i, j);
        //cout << setprecision(2) << edgematrix[curredge].weight << " ";
        cout << setprecision(4) << "( " << edgematrix[curredge].weight << " " << edgematrix[curredge].vertex << ") ";
      }
      cout << endl;
    }

    cout << "........\n";

    for(i = record_count-10; i < record_count; i++){
      for(j = 0; j < 5; j++){
        int curredge = getMatrixIndex(k, i, j);
        //cout << setprecision(2) << edgematrix[curredge].weight << " ";
        cout << setprecision(4) << "( " << edgematrix[curredge].weight << " " << edgematrix[curredge].vertex << ") ";
      }
      cout << endl;
    }
    

    /*
    begin clustering approach
    */

    /*
    init matrix used for keeping edge lengths/scoring
    at end of algorithm edgelengths will be total of all lengths recorded
    */

    float *edgelengths = new float[record_count * k];
    for(i = 0; i < (record_count * k); i++){
      edgelengths[i] = 0.0;
    }
  
    /*
    generate random positions based on indices of records
    */

    int *p = new int[record_count];
    float *locavgs = new float[record_count];

    int t;
    for(t = 0; t < 5; t++){

      for(i = 0; i < record_count; i++){
        //assign each vertex a random location
        p[i] = rand() % (record_count*25);
      }
  
      int l;
      for(l = 0; l < 5; l++){
  
        for(i = 0; i < record_count; i++){
  
          //first, get sum of all valence weights on this vertex (so we can get an average)
          float valenceweight = 0.0;
          for(j = 0; j < k; j++){
    
            //get vertex to compare to
            int valenceedge = getMatrixIndex(k, i, j);
            valenceweight += edgematrix[valenceedge].weight;
          }
    
          //move vertices based on force from neighbors
          float force = 0.0;
          for(j = 0; j < k; j++){
    
            //get vertex to compare to
            int compedge = getMatrixIndex(k, i, j);
            int compvertex = edgematrix[compedge].vertex;

            //add force from neighbor vertex
            force += ((p[i] - p[compvertex]) * edgematrix[compedge].weight);
          }
    
  
          //account for gain μ
          force *= (1/(valenceweight+1));
  
          //change position in accordance to force present on it
          p[i] -= force;
          
        }
  
      }
  
      //after moving vertices, record lengths of each edge
      for(i = 0; i < record_count; i++){
        for(j = 0; j < k; j++){
  
          //get vertex to compare to
          int compedge = getMatrixIndex(k, i, j);
          int compvertex = edgematrix[compedge].vertex;
 
          //get distance between two vertices
          float distance = (p[i] - p[compvertex]);
          if(distance < 0){
            distance *= -1;
          }
  
          edgelengths[compedge] += distance;
          //edgelengths[compedge] = distance;
        }
        
      }

    }

    //now we start cutting edges with higher lengths than others, as such, k value with vary vertex to vertex

    //take average of all edge lengths (as of now, it is total of lengths after all trials)
    for(i = 0; i < record_count; i++){
      for(j = 0; j < k; j++){
        int compedge = getMatrixIndex(k, i, j);
        edgelengths[compedge] /= 5;
      }
    }
    

    //get average of all edges connected to vertex i
    for(i = 0; i < record_count; i++){
      float localtotal = 0.0;
      for(j = 0; j < k; j++){
        int compedge = getMatrixIndex(k, i, j);
        localtotal += edgelengths[compedge];
      }
      float localavg = localtotal / k;
      locavgs[i] = localavg;
      
      //cut edges (set weight, vertex to -1) with a length longer than average
      for(j = 0; j < k; j++){
        int compedge = getMatrixIndex(k, i, j);
        if(edgelengths[compedge] > localavg){
          edgematrix[compedge].vertex = -1;
          edgematrix[compedge].weight = -1;
          edgelengths[compedge] = -1;
        };
      }
    }

    for(i = 0; i < 10; i++){
      cout << locavgs[i] << ": ";
      for(j = 0; j < 15; j++){
        int curredge = getMatrixIndex(k, i, j);
        cout << setprecision(4) << "( " << edgelengths[curredge] << " " << edgematrix[curredge].vertex << ") ";
      }
      cout << endl;
    }

    cout << "-----------------------\n";

    for(i = record_count-10; i < record_count; i++){
      cout << locavgs[i] << ": ";
      for(j = 0; j < 15; j++){
        int curredge = getMatrixIndex(k, i, j);
        cout << setprecision(4) << "( " << edgelengths[curredge] << " " << edgematrix[curredge].vertex << ") ";
      }
      cout << endl;
    }

    return 0;
}
