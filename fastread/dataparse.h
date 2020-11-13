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
using namespace std;

/*
remove unwanted chars from string
*/
void cleanString(string str){

    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    str.erase(remove(str.begin(), str.end(), '.'), str.end());
    str.erase(remove(str.begin(), str.end(), ' '), str.end());

}

/*
given one line from text file, split elements of that line into vector of record attributes
*/
void parseRecord(string str, vector<string> &result){

    //create string stream from the string
    stringstream s_stream(str); 

    //get string delimited by comma
    while(s_stream.good()) {
        string substr;
        getline(s_stream, substr, ',');
        result.push_back(substr);
    }
/*
    //clean up all splitted strings
    for(int i = 0; i < result.size(); i++) {
        cleanString(result.at(i));
    }
*/
}

void parseFileContent(string file, vector<vector<string> > &records){

    //create string stream from the string
    stringstream s_stream(file); 

    int counter = 0;

    //get record string deliminated by line break
    //parse that record, put into vector
    while(s_stream.good()) {
        string record;
        getline(s_stream, record, '\n');

/*
        counter++;
        if(counter % 100000 == 0){
            cout << "Record " << counter << endl;
        }
*/

        vector<string> curr;
        parseRecord(record, curr);

        records.push_back(curr);
    }

}

/*
get index of a property from vector
used for record attributes with string based properties
*/
float getPropertyIndex(string property, vector<string> &v){

    int i;
    for(i = 0; i < v.size(); i++){
        if(property == v[i]){
            return static_cast<float>(i);
        }
    }

    v.push_back(property);
    return static_cast<float>(v.size()-1);

}

/*
take a vector of records parsed into strings
convert all attributes into floats
hard code in any cases of string based symoblic properties in the dataset (e.g. protocols)
*/
void generateNumericRecords(vector<vector<string> > rvect, vector<vector<float> > &records, vector<int> &types){

    int i, j;

    //vectors to keep track of language-based properties;
    vector<string> protocals;
    vector<string> services;
    vector<string> flags;

    vector<string> record_types;

    int record_count = rvect.size();
    int record_size = rvect[0].size()-1;

    for(i = 0; i < rvect.size(); i++){

        //cout << "Converted " << i << endl;

      records.push_back(vector<float>());

        //array of floats containing all members of a record
        //float *curr = new float[record_size];

        for(j = 0; j < record_size; j++){

            switch(j){
                //if we have a language based property, put property index into array
                case 1:
                    records[i].push_back(getPropertyIndex(rvect[i][j], protocals));
                break;
                case 2:
                    records[i].push_back(getPropertyIndex(rvect[i][j], services));
                break;
                case 3:
                    records[i].push_back(getPropertyIndex(rvect[i][j], flags));
                break;
                default:
                    //convert c-str to float, put in array
                    float x = static_cast<float>(atof(rvect[i][j].c_str()));
                    records[i].push_back(x);
            }
        }

        //save type of record
        types.push_back(getPropertyIndex(rvect[i][record_size], record_types));
    }
}