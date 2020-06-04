
#define RECORDS  4905000  //number of records in dataset (rough estimate)
#define FEATURES 42       //features of a record, e.g. protocal, byte size
#define ATTACKS  23       //number of attacks
#define CONTINOUS 0
#define SYMBOLIC 1

typedef struct Feature {
  char *name;
  int type; //continous = 0 / symbolic = 1
} Feature;

/*
error checking
*/
void err(char *message){
  printf("ERR: %s\n", message);
  exit(1);
}

/*
string cleaning
*/
void removeCharacter(char* str, char c) {
    char *pr = str, *pw = str;
    while (*pr) {
        *pw = *pr++;
        pw += (*pw != c);
    }
    *pw = '\0';
}

void cleanString(char *str){
  removeCharacter(str, '.');
  removeCharacter(str, '\n');
  removeCharacter(str, ' ');
}

void cleanManyStrings(char **stringArr, int count){
  int i;
  for(i = 0; i < count; i++){
    cleanString(stringArr[i]);
  }
}

/*
display percentage based on amount done and total
*/
int percentread(int curr, int total, char *message){

  int best = total/20;
  if(curr % best == 0){
    int percent = ((float) curr / total) * 100;
    printf("%d%% %s\n", percent, message);
  }
}

/*
read in full dataset (or file of features), path passed by user
*/
char **readfile(char *path, int lines, int *exact){

  FILE *fp;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  //allocate data for all records after file open
  fp = fopen(path, "r");
  if(fp == NULL) err("dataset file could not be opened");
  char **data = (char**) malloc(lines * sizeof(char *));

  int i = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
    //allocate space for this record, copy over to dataset array
    data[i] = (char*) malloc(sizeof(char) * (read+1));
    strcpy(data[i], line);
    //display percent of file read in
    i++;
    percentread(i, RECORDS, "read...");
  }

  printf("Total lines: %d\n", i);
  *exact = i;

  fclose(fp);
  if(line) free(line);

  return data;

}

char **splitRecord(char *record){

   char **formatted = (char**) malloc(FEATURES * sizeof(char *));

   // Extract the first feature
   char *feature = strtok(record, ",");
   int i = 0;

   // loop through the string to extract all other features
   while(feature != NULL){
    int featurelen = strlen(feature);
    formatted[i] = (char*) malloc(sizeof(char) * (featurelen+1));
    strcpy(formatted[i], feature);

    feature = strtok(NULL, ",");
    i++;
   }

   return formatted;
}

Feature* createFeature(char *name, char *type){

  Feature *f = (Feature*) malloc(sizeof(Feature));

  //copy name
  int namelen = strlen(name);
  f->name = (char*) malloc(sizeof(char) * (namelen+1));
  strcpy(f->name, name);

  //determine type
  if(strcmp(type, " continuous") == 0) //if matches continous
    f->type = CONTINOUS;
   else
    f->type = SYMBOLIC;

  return f;
}

Feature **readFeatures(char *path, char **attacks){

  //list of lines read in from dataset
  int exact;
  char **features = readfile(path, FEATURES, &exact);
  //list of Features to return
  //one less than FEATURES as the last feature is the attack type (which will be kept in an array)
  Feature **flist = (Feature**) malloc(sizeof(Feature*) * FEATURES-1);

  //copy first string of features to attacks list
  *attacks = (char*) malloc(sizeof(char) * strlen(features[0]));
  strcpy(*attacks, features[0]);

  int i;
  for(i = 1; i < FEATURES; i++){
    char *name = strtok(features[i], ":");
    char *type = strtok(NULL, ".");
    flist[i-1] = createFeature(name, type);
  }

  return flist;

}

double hash(unsigned char *str){
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c;

    return (double) hash;
}

double *numberfyRecord(char **recordtokens){

  //all record attributes except attack type
  double *record = malloc(sizeof(double) * FEATURES-1);

  //convert all strings in array to numbers
  int i;
  for(i = 0; i < FEATURES-1; i++){
    if(i == 1 || i == 2 || i == 3){ //features using plaintext data, need to be hashed
      record[i] = hash(recordtokens[i]);
      continue;
    }
    record[i] = atof(recordtokens[i]);
  }

  //free all strings after we have converted
 /* for(i = 0; i < FEATURES; i++){
    free(recordtokens[i]);
  }
  free(recordtokens);
*/
  return record;

}

int getTypeIndex(char **alltypes, char *currtype){

  cleanString(currtype);

  int i;
  for(i = 0; i < ATTACKS; i++){
    if(strcmp(alltypes[i], currtype) == 0){
      return i;
    }
  }

  return -1;

}

char **getAttackTypes(char *attacks){

  char **allattacks = (char**) malloc(ATTACKS * sizeof(char *));
  int i = 0;
  
  //split str at ,
  char *attack = strtok(attacks, ",");
  while(attack != NULL){
      //allocate space for attack name, copy, then split next token
      allattacks[i] = (char*) malloc(sizeof(char) * strlen(attack));
      strcpy(allattacks[i], attack);
      attack = strtok(NULL, ",");
      i++;
   }

   return allattacks;

}

void printRecord(double *record){
  int i;
  for(i = 0; i < FEATURES-1; i++){
    printf("%.2f\n", record[i]);
  }
}

void generateRecordMatrix(Matrix *M, char **records, int recordcount, int *recordtypes, char **alltypes){

  //all of our data is held in a matrix
  initMatrix(M, recordcount, FEATURES-1);

  int i;
  for(i = 0; i < recordcount; i++){

    //get array of record attributes
    char **rec = splitRecord(records[i]);

    //transfer array of numberfied record attributes to matrix
    double *numberfied = numberfyRecord(rec);
    transferArray(M, numberfied, i);

    //get type of 
    recordtypes[i] = getTypeIndex(alltypes, rec[FEATURES-1]);
    percentread(i, recordcount, "converted...");

  }

}

/*

HOST READ
read in all data, attacks types, on host (CPU bound operation)

generate for user:
  M           -Matrix of all network records, with attributes stored as doubles so it is compatible with device
  records     -list (char **) of all records, so when a certain record is read as an anomaly, it can be printed
  attacks     -list (char **) of all attacktypes
  recordtypes -list (int **) of all records types (attack/normal), value corresponds to index of a certain attack type in attacks

*/

void hostread(Matrix *M, int *recordtypes, char **attacks){

  char *datasetPath = "./dataset/sample.data";

  //char *datasetPath = "./dataset/kddcup_10percent.data";
  char *featuresetPath = "./dataset/kddcup.names";

  //get all records from file
  int recordTotal;
  char **records = readfile(datasetPath, RECORDS, &recordTotal);

  //also get all features of a record for reference, as well as attack types
  char *attacktypes;
  Feature **features = readFeatures(featuresetPath, &attacktypes);
  
  //split attacks into array of types to match up indices of values in recordtypes
  attacks = getAttackTypes(attacktypes);
  cleanManyStrings(attacks, ATTACKS);

  //type of record (normal or attack type), value corresponds to name of attack in **attacks, index corresponds to index in records list 
  recordtypes = malloc(sizeof(int) * recordTotal);

  //convert all records to numeric (double) records, put into a matrix
  generateRecordMatrix(M, records, recordTotal, recordtypes, attacks);

}