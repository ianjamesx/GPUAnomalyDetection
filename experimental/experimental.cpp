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

using namespace std;

#include "dataread.h"

//typedef/structs

typedef pair<float, float> Pairing;

typedef struct patternData {

    int index1;
    int index2;
    Pairing pattern;
    int occurances;

} patternData;

void printPD(patternData pd){

    cout << "Pattern: (" << pd.pattern.first << ", " << pd.pattern.second << "), ";
    cout << "Indices: (" << pd.index1 << ", " << pd.index2 << "), ";
    cout << "Occurances: " << pd.occurances << endl;

}

/*
void generatePairs(vector<vector<float> > records, pair<float, float> pair_arr[21], int curr){

    int i, j, currpair = 0;
    for(i = 0; i < records[curr].size(); i++){
        for(j = i+1; j < records[curr].size(); j++){
            pair_arr[currpair].first = records[curr][i];
            pair_arr[currpair].second = records[curr][j];
            currpair++;
        }
    }
}
*/
/*
generate all pairings for a specific pair

void replaceWithPLIndex(vector<Pairing> &pairings, Pairing p, int index){
    replace(pairings.begin(), pairings.end(), Pairing(p.first, p.second), Pairing(index, index));
}

int pairInParentList(vector<Pairing> plist, Pairing p){

    vector<Pairing>::iterator iter;
    iter = find(plist.begin(), plist.end(), p);
    if(iter != plist.end()){
        return distance(plist.begin(), iter);
    } else {
        return -1;
    }

}

bool isParentListIndex(vector<Pairing> plist, int negativeIndex){

    int realIndex = (negativeIndex * -1) - 1;

    if(plist.size()-1 < realIndex){
        return false;
    }
    return true;

}

*/

int isCompressed(vector<Pairing> pairings, int index){


return 1;
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
    for(i = 0; i < pairings.size(); i++){

        //cout << pairings[i].first << " <> " << pairings[i].second << " | " << i << ", " << pairings.size() << "...\n";

        if(pairings[i].first == pairings[i].second && pairings[i].first < 0){
            continue;
        }

        //get number of occurances for this pair
        //cout << "counting " << pairings.size()-i << " elements\n";
        int occurances = count(pairings.begin()+(i+1), pairings.end(), pairings[i]);

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

/*
given the negative (compresseed) index of an element in records, find corresponding one
this ensures that all elements of a compression are further compression when paired with
another already compressed pair
*/
/*
int getCorrespondingCompression(vector<float> record, int negindex, int index){

    //get corresponding index of pair in parent list
    //e.g., if given value -2, index 5, find corresponding negative value -2 (not of index 5)
    int i, j, corresponding;
    for(i = 0; i < record.size(); i++){

        if(record[i] == negindex && i != index){
            cout << "corresponding index of " << negindex << " (" << index << ") found at " << i << endl;
            return i;
        }

    }

    cout << "returning -1, failed to find " << negindex << " not at " << index << endl;
    for(i = 0; i < record.size(); i++){
        cout << record[i] << " ";
    }
    cout << endl;
    return -1;

}
*/

/*
when a compression itself is being compressed, we need to replace
all parts of the compressed pair with the new value
*/
void replaceFullCompression(vector<float> &record, float negindex, int plIndex){

    int i;
    for(i = 0; i < record.size(); i++){
        if(record[i] == negindex){
            record[i] = plIndex;
        }
    }

}

void compressPatterns(vector<vector<float> > &records, vector<Pairing> &parentlist, patternData pd){

    int i, j;
    int rsize = records[0].size();

    //push pattern into parentlist, get negative index+1, and pairing to replace
    parentlist.push_back(pd.pattern);
    int plIndex = (parentlist.size()) * -1;
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

float rankRecord(vector<float> record, vector<int> recordsizes){

    int i;
    int n = recordsizes.size();
    float rank = 0;

    float summation = 0;
    for(i = 0; i < n; i++){
      int iter = n - i + 1;
      summation += (iter * (1.0 / recordsizes[i]));
    }

    cout << "sum: " << summation << endl;

    summation *= (1 / n);

    cout << "curr sum: " << summation << endl;
    return (1 - summation);

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

    vector<float> ranks;

    int found = 1;

    int curr_iter = 0;

    while(found){

        //find most frequent pattern

        //cout << "finding pattern..." << endl;
        patternData pd;
        found = mostFreqPattern(records, pd);

        //printPD(pd);

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

        //cout << "done...." << endl;

    }

    for(i = 0; i < parentlist.size(); i++){
        //cout << parentlist[i].first << ", " << parentlist[i].second << ": " << occurances[i] << ", " << iterations[i] << endl;
    }
    cout << "---------\n";
    //printall(records);

    for(i = 0; i < recordsizes.size(); i++){
        //cout << recordsizes[i][recordsizes[i].size()-1] << endl;
    }

    cout << "total iters: " << curr_iter << endl;

    float score = rankRecord(records[0], recordsizes[0]);

    for(i = 0; i < record_count; i++){
      //float score = rankRecord(records[i], recordsizes[i]);
      //cout << i << ": " << score << endl;
    }

    //rankRecord(records[0], parentlist, occurances, iterations);

    return 0;

}
