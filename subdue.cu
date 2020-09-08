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

//typedef/structs
typedef pair<float, float> Pairing;

typedef struct patternData {

    int index1;
    int index2;
    Pairing pattern;
    int occurances;

} patternData;

#include "subdue_device.cuh"

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
    int threads = 32;

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

    int i;
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
          compression = 1.0 / attributecount[i];
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

bool isAttack(int type){
    if(type == 0){
        return false;
    }
    return true;
}

void rankstats(int *record_types, float *record_ranks, int record_count, float runtime, int iterations){

    float threshold = .5;

    int TP = 0, FP = 0, TN = 0, FN = 0;
    
    //for writing anomalies to file
    ofstream anoms;
    ofstream normal;
    ofstream report;
    anoms.open("./logs/anoms.log");
    normal.open("./logs/normal.log");
    report.open("./logs/report.log");

    int allattacks = 0;
    int normalrec = 0;

    int i;
    for(i = 0; i < record_count; i++){

        //true negative case
        if((!isAttack(record_types[i])) && record_ranks[i] < threshold){
            TN++;
        }
        //false negative case
        if(isAttack(record_types[i]) && record_ranks[i] < threshold){
            FN++;
        }
        //true positive case
        if(isAttack(record_types[i]) && record_ranks[i] >= threshold){
            TP++;
        }
        //false positive case
        if((!isAttack(record_types[i])) && record_ranks[i] >= threshold){
            FP++;
            normal << "FALSE POS " << record_types[i] << ": " << record_ranks[i] << endl;
        }

        if(isAttack(record_types[i])){
            anoms << "ANOMALY " << record_types[i] << ": " << record_ranks[i] << endl;
            allattacks++;
        } else {
            normalrec++;
        }

        if(record_ranks[i] >= threshold){
            anoms << "Normal " << record_types[i] << ": " << record_ranks[i] << endl;
        }

    }

    anoms.close();
    normal.close();

    double TPR = (TP * 1.0) / static_cast<double>((TP + FN));
    double accuracy = (TP + TN) / static_cast<double>((TP + TN + FP + FN));

    report << "Iterations: " << iterations << endl;
    report << "Runtime: " << runtime << endl;

    report << "TN: " << TN << endl;
    report << "FN: " << FN << endl;
    report << "TP: " << TP << endl;
    report << "FP: " << FP << endl;
    report << "sensitivity (TPR): " << TPR << endl;
    report << "accuracy: " << accuracy << endl;

    report << "Attacks: " << allattacks << ", normal: " << normalrec << endl;

    report.close();

}

void printranks(int *record_types, float *record_ranks, int record_count, int max){

    int i;
    for(i = record_count-1; i > record_count-max; i--){
        cout << "Type: " << record_types[i] << ", percent: " << record_ranks[i] << endl;
    }
}

void getAccuracy(int *record_types, float *record_ranks, int record_count, float threshold){

    

    int i;
    for(i = 0; i < record_count; i++){
        
    }

}

void printTypeOccurances(vector<int> record_types){

    int ocs[23];
    int i;

    for(i = 0; i < 23; i++){
        ocs[i] = 0;
    }

    for(i = 0; i < record_types.size(); i++){
        ocs[record_types[i]]++;
    }
/*
    for(i = 0; i < 23; i++){
        cout << "TYPE " << i << ": " << ocs[i] << endl;
    }
*/
}

int main(){

    /*
    init dataset
    */

    vector<vector<float> > records;
    vector<int> record_types;
    int record_count, record_size;

    init_dataset(record_count, record_size, records, record_types);

    /*
    generate pairs
    */

    int i;

    vector<Pairing> parentlist;
    vector<int> occurances;
    vector<int> iterations;
    vector<vector<int> > recordsizes;

    //init recordsizes
    initRecordSizes(recordsizes, record_count);

    printTypeOccurances(record_types);

    cout << "Records: " << record_count << endl;

    float termination_ratio = .25;
    int found = 1;
    int curr_iter = 0;

    auto fullstart = high_resolution_clock::now(); 

    while(found){

        //find most frequent pattern
        patternData pd;

        auto start = high_resolution_clock::now(); 

        //found = mostFreqPattern(records, pd);
        found = mostFreqPatternDevice(records, pd);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start); 
        //cout << "pattern found in: " << (duration.count() * .000001) << endl; 

        occurances.push_back(pd.occurances); 

        //if a pattern was found, compress it
        if(found){
            compressPatterns(records, parentlist, pd);
        } 

        //count sizes of each record after compression
        updateRecordSizes(records, recordsizes);

        //update iteration
        iterations.push_back(curr_iter);
        curr_iter++;

        if(curr_iter % 5 == 0){
            cout << "ratio: " << (pd.occurances / (record_count * 1.0)) << endl;
        }

        if((pd.occurances / (record_count * 1.0)) < termination_ratio){
            break;
        }
    }

    auto fullstop = high_resolution_clock::now();
    auto fullduration = duration_cast<microseconds>(fullstop - fullstart);

    cout << "Iterations: " << curr_iter << endl;
    cout << "Time for algo: " << (fullduration.count() * .000001) << endl;

    //score all records
    vector<float> record_ranks;
    for(i = 0; i < record_count; i++){
        record_ranks.push_back(rankRecord(records[i], recordsizes[i]));
    }

    float *rank_arr = new float[record_count];
    int *type_arr = new int[record_count];
    copy(record_ranks.begin(), record_ranks.end(), rank_arr);
    copy(record_types.begin(), record_types.end(), type_arr);
    thrust::sort_by_key(thrust::host, rank_arr, rank_arr + record_count, type_arr);

    rankstats(type_arr, rank_arr, record_count, (fullduration.count() * .000001), curr_iter);
    printranks(type_arr, rank_arr, record_count, 60);

    return 0;

}
