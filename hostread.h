
#define RECORDS  4905000  //number of records in dataset (rough estimate)
#define FEATURES 42       //features of a record, e.g. protocal, byte size
#define ATTACKS  23       //number of attacks
#define CONTINOUS 0
#define SYMBOLIC 1


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
  char **attacks;
  char **features = readFeatures(featuresetPath, &attacks);


  //type of record (normal or attack type), value corresponds to name of attack in **attacks, index corresponds to index in records list
  recordtypes = malloc(sizeof(int) * recordTotal);

  //convert all records to numeric (double) records, put into a matrix
  generateRecordMatrix(M, records, recordTotal, recordtypes, attacks);

}
