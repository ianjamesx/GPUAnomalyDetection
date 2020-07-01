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
#include <utility>
#include <set>
#include <chrono> 

//thrust utils (for GPU accel)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>

using namespace std;

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

//typedef/structs
typedef pair<float, float> Pairing;

typedef struct patternData {

    int index1;
    int index2;
    Pairing pattern;
    int occurances;

} patternData;


#include "frequent.cuh"

void printPD(patternData pd){

    cout << "Pattern: (" << pd.pattern.first << ", " << pd.pattern.second << "), ";
    cout << "Indices: (" << pd.index1 << ", " << pd.index2 << "), ";
    cout << "Occurances: " << pd.occurances << endl;

}

void generatePairing(vector<vector<float> > records, vector<Pairing> &pairings, int p1, int p2){

    int i;
    for(i = 0; i < records.size(); i++){

        //exclude pair IF, content is equal and less than zero, as pairings of this format (-1, -1) are considered one element
        if(records[i][p1] == records[i][p2] && records[i][p1] < 0){
            continue;
        }
        if(p1 == 0 && p2 == 2){
            //cout << records[i][p1] << " ~ " << records[i][p2] << endl;
        }
        pairings.push_back(Pairing(records[i][p1], records[i][p2]));
    }

}

/*
get most common pairing for a given list of of pairings
will return count of pairings, mostcommon will be overwritten to be most common pairing
*/
int mostCommmonPairing(vector<Pairing> pairings, Pairing &mostcommon){

    int i, max = -1;
    set<Pairing> patterns;
    for(i = 0; i < pairings.size(); i++){

        //very pattern isnt composed of compressed pattern in itself
        if(pairings[i].first == pairings[i].second && pairings[i].first < 0) continue;

        //very pattern has not been counted already (if it is in set of patterns)
        if(patterns.find(pairings[i]) != patterns.end()) continue;
        

        //get number of occurances for this pair
        //cout << "counting " << pairings.size()-i << " elements\n";
        //thrust::device_vector<Pairing> d_vec(pairings+(i+1));
        int occurances = count(pairings.begin()+(i+1), pairings.end(), pairings[i]);

        //put into set of already-tested patterns
        patterns.insert(pairings[i]);

        //see if this is greatest count so far
        if(occurances > max){
            max = occurances;
            mostcommon = pairings[i];
        }
    }

    return max;
}

/*
locate most common pattern out of all pairings
put pattern data in pd
return 1 if patterns of frequency over 1 still occur, 0 if not
*/
int mostFreqPattern(vector<vector<float> > records, patternData &pd){

    int i, j;

    int allmode = -1;
    int p1, p2;
    Pairing pattern;

    int rsize = records[0].size();

    for(i = 0; i < rsize; i++){

        //cout << "at i " << i << " of " << rsize << endl;

        for(j = (i+1); j < rsize; j++){

            //cout << "at j " << j << " of " << rsize << endl;

            Pairing mostcommon;
            vector<Pairing> pairings;

            generatePairing(records, pairings, i, j);

            int currmode = mostCommmonPairing(pairings, mostcommon);

            if(currmode > allmode){
                //save occurances counted, pair, and indices
                allmode = currmode;
                pattern = mostcommon;
                p1 = i;
                p2 = j;
            }
        }
    }

    pd.occurances = allmode;
    pd.pattern = pattern;
    pd.index1 = p1;
    pd.index2 = p2;

    //return 1 if still >2, if not, return 0
    if(allmode > 0){
        return 1;
    }

    return 0;

}

int mostFreqPatternDevice(vector<vector<float> > records, patternData &pd){

    //convert 2d vector to 1d array to use on device

    float *record_dev;
    int record_count = records.size(), record_size = records[0].size();
    int record_dev_size = record_size * record_count;
    gpuErrchk(cudaMallocManaged(&record_dev, record_dev_size * sizeof(float)));

    //copy records over
    int i, j, curr_index = 0;
    for(i = 0; i < record_count; i++){
        for(j = 0; j < record_size; j++){
            record_dev[curr_index] = records[i][j];
            curr_index++;
        }
    }

    int blocks = 820;
    int threads = 512;

    //allocate for patterns
    patternData *patterns;
    gpuErrchk(cudaMallocManaged(&patterns, blocks * sizeof(patternData)));

    //find most frequent pattern (on gpu)
    mostFreqPattern<<<blocks, threads, (threads*sizeof(int))>>>(record_dev, record_size, record_count, patterns);
    cudaDeviceSynchronize();

    //see which of the 820 pairings in patterns list is most common
    int maxindex = 0;
    int maxoc = 0;
    for(i = 0; i < blocks; i++){
        if(patterns[i].occurances > maxoc){
            maxoc = patterns[i].occurances;
            maxindex = i;
        }
    }

    pd = patterns[maxindex];

    //free all cuda data
    cudaFree(record_dev);
    cudaFree(patterns);

    if(maxoc > 0){
        return 1;
    }

    return 0;

}

void compressPatterns(vector<vector<float> > &records, vector<Pairing> &parentlist, patternData pd){

    int i, j;
    int rsize = records[0].size();

    //push pattern into parentlist, get negative index+1, and pairing to replace
    parentlist.push_back(pd.pattern);
    int plIndex = parentlist.size();
    plIndex *= -1;

    float p1 = pd.pattern.first, p2 = pd.pattern.second;

    //caching corresponding variables, will be overwritten on first search
    int compIndex1 = -1, compIndex2 = -1;

    //replace all instances of the pattern with negative index in parentlist
    for(i = 0; i < records.size(); i++){
        if(records[i][pd.index1] == p1 && records[i][pd.index2] == p2){

            //also replace corresponding compressed indices if compressions are compressed
            //can tell if they are compressed if a pattern has a negative value
            if(p1 < 0){
                for(j = 0; j < records[i].size(); j++){
                    if(records[i][j] == p1){
                        records[i][j] = plIndex;
                    }
                }
            }

            if(p2 < 0){
                for(j = 0; j < records[i].size(); j++){
                    if(records[i][j] == p2){
                        records[i][j] = plIndex;
                    }
                }
            }

            records[i][pd.index1] = plIndex;
            records[i][pd.index2] = plIndex;
        }
    }
}

/*
get the size, in elements (compressed or uncompressed) of a record
*/
int getRecordSize(vector<float> record){

    int i, j;
    int size = 0;
    set<float> compressions;

    for(i = 0; i < record.size(); i++){
        //put negative elements (compressed pairs) in set
        if(record[i] < 0){
            compressions.insert(record[i]);
        //if attribute not compressed, add
        } else {
            size++;
        }
    }

    //return count of all compressed pairs + uncompressed pairs
    return (compressions.size() + size);
}

void initRecordSizes(vector<vector<int> > &recordsizes, int size){
    int i;
    for(i = 0; i < size; i++){
        recordsizes.push_back(vector<int>());
    }
}

void updateRecordSizes(vector<vector<float> > records, vector<vector<int> > &recordsizes){
    int i;
    for(i = 0; i < records.size(); i++){
        int currsize = getRecordSize(records[i]);
        recordsizes[i].push_back(currsize);
    }
}

float rankRecord(vector<float> record, vector<int> attributecount){

    int i;
    int n = attributecount.size();

    //init previous attribute count to initial size of record
    int prevAttrCount = attributecount[0];
    int constattr = prevAttrCount;

    float summation = 1;
    for(i = 1; i < n; i++){

      //get weight for this iteration
      int iter = n - i + 1;

      //determine if compression occured on ith iteration
      //if it was, Ci = 0, else 1/x
      double compression;
      if(attributecount[i] == prevAttrCount){
          compression = 0.0;
      } else {
          compression = 1.0 / constattr;
      }

      //update previous attribute count for next iteration
      prevAttrCount = attributecount[i];
      double iterationcomp = (iter * compression);
      summation += iterationcomp;
    }

    double limiter = 1.0 / n;
    summation *= limiter;

    float rank = 1.0 - summation;
    return rank;

}

void printall(vector<vector<float> > records){

    int i, j;
    for(i = 0; i < records.size(); i++){
        for(j = 0; j < records[i].size(); j++){
            cout << records[i][j] << ' ';
        }
        cout << " | " << getRecordSize(records[i]) << endl;
        //cout << ": " << record_types[i] << endl;
    }

}

int main(){

    /*
    init dataset
    */

    vector<vector<float> > records;
    vector<int> record_types;
    int record_count, record_size;

    init_dataset(record_count, record_size, records, record_types);

    cout << "Finished read in\n";

    /*
    generate pairs
    */

    //printRecord_full(records, record_count, record_size, 0);

    int i, j;
    int common = 0, uncommon = 0;

    //printall(records);

    vector<Pairing> parentlist;
    vector<int> occurances;
    vector<int> iterations;
    vector<vector<int> > recordsizes;

    //init recordsizes
    initRecordSizes(recordsizes, record_count);

    //printall(records);

    vector<float> ranks;

    int found = 1;

    int curr_iter = 0;

    while(found){

        //find most frequent pattern

        cout << "iteration " << curr_iter << endl;

        patternData pd;
        found = mostFreqPattern(records, pd);
        //found = mostFreqPatternDevice(records, pd);

        occurances.push_back(pd.occurances);

        //compress pattern found
        //cout << "done... compressing pattern..." << endl;
        if(found){
            compressPatterns(records, parentlist, pd);
        }

        //count sizes of each record after compression
        updateRecordSizes(records, recordsizes);

        //update iteration
        iterations.push_back(curr_iter);
        curr_iter++;

    }

    for(i = 0; i < parentlist.size(); i++){
        //cout << parentlist[i].first << ", " << parentlist[i].second << ": " << occurances[i] << ", " << iterations[i] << endl;
    }
    cout << "---------\n";
    //printall(records);
    cout << "total iters: " << curr_iter << endl;

    //score all records
    for(i = 0; i < record_count; i++){
      float score = rankRecord(records[i], recordsizes[i]);
      cout << "record " << i << ": " << score << endl;
    }

    return 0;

}
