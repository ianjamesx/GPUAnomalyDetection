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