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

#include "fastread.h"
#include "dataparse.h"

int main(){

    //read data from file
    string filename = "./kdd.data";
    const char *filec = filename.c_str();
    char *filecontent = fastfileread(filec);

    //get records from file
    vector<vector<string> > records;
    parseFileContent(string(filecontent), records);

    cout << "total records: " << records.size() << endl;

    vector<vector<float> > records_numeric;
    vector<int> types;
    generateNumericRecords(records, records_numeric, types);

    int i, j;
    for(i = 0; i < 12; i++){
        cout << i << ": ";
        for(j = 0; j < records_numeric[i].size(); j++){
            cout << records_numeric[i][j] << " ";
        }
        cout << " | " << types[i];
        cout << "\n";
    }

    return 0;
}