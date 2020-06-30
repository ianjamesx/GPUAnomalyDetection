
/*
error checking
*/
void err(string message){
  cout << message << endl;
  exit(1);
}

/*
some string utils
*/

void cleanString(string str){

  str.erase(remove(str.begin(), str.end(), '\n'), str.end());
  str.erase(remove(str.begin(), str.end(), '.'), str.end());
  str.erase(remove(str.begin(), str.end(), ' '), str.end());

}

void splitString(string str, char delim, vector<string> &result){

  stringstream s_stream(str); //create string stream from the string

  while(s_stream.good()) {
    string substr;
    getline(s_stream, substr, delim); //get first string delimited by comma
    result.push_back(substr);
  }

  for(int i = 0; i < result.size(); i++) {    //print all splitted strings
    cleanString(result.at(i));
  }

}

/*
display percentage based on amount done and total
*/
void percentread(int curr, int total, string message){

  int best = total/20;
  if(curr % best == 0){
    int percent = ((float) curr / total) * 100;
    cout << percent << "% " << message << '\n';
  }
}

/*
read in full dataset (or file of features), path passed by user
*/
vector<vector<string> > readfile(string path){

    string line;
    ifstream infile;
    infile.open(path);

    vector<vector<string> > records;
    int curr = 0;

    while (getline(infile, line)){
        istringstream iss(line);

        //load attributes of current record into vector
        records.push_back(vector<string>());
        splitString(line, ',', records[curr]);
        curr++;
    }

    return records;

}

/*
get index of a property from vector
used for record attributes with linguistic properties
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
conver 2d vector to 1d array of floats
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

        //save record to array
        //saveRecord(records, record_count, record_size, i, curr);

        //save type of record
        types.push_back(getPropertyIndex(rvect[i][record_size], record_types));
    }

}

void init_dataset(int &record_count, int &record_size, vector<vector<float> > &records, vector<int> &types){

    vector<vector<string> > record_tokens = readfile("./dataset/sample.data");
    //vector<vector<string> > record_tokens = readfile("./dataset/simple.data");
    //vector<vector<string> > record_tokens = readfile("./dataset/1k.data");
    //vector<vector<string> > record_tokens = readfile("./dataset/kddcup_10percent.data");

    //allocate space for number of records * number of attributes per record (e.g 500k * 42)
    record_count = record_tokens.size();
    record_size = record_tokens[0].size()-1;

    generateNumericRecords(record_tokens, records, types);

}
