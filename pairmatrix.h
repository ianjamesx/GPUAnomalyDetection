/*
matrix operations
for 2d arrays, to save space, use 1d array
keep track of rows and columns
*/

typedef struct PairMatrix {
  int rows;
  int cols;
  int n;
  int k;
  double **data;
} PairMatrix;

int nk_count(int n, int k){

  if (k > n) return 0;
  if (k * 2 > n) k = n-k;
  if (k == 0) return 1;

  int result = n;
  for( int i = 2; i <= k; ++i ) {
      result *= (n-i+1);
      result /= i;
  }
  return result;

}

/*
allocate for matrix
*/
void initPairMatrix(PairMatrix *M, int r, int n, int k){

  M->rows = r;

  //columns will be Ck(n), where k = number of elements per pair, n = elements
  int nk = nk_count(n, k);
  M->cols = nk;
  M->n = n;
  M->k = k;

  //malloc space for each pair to hold a pointer to array (vertex)
  int size = (nk * r) * sizeof(double*);
  M->data = malloc(size);
}

/*
write single element to matrix
*/
void savePair(PairMatrix *M, double *pair, int i, int j){
  int index = INDEX(M->rows, M->cols, i, j);

  M->data[index] = malloc(sizeof(double) * M->k);

  int i;
  for(i = 0; i < M->k; i++){
    M->data[index][i] = pair[i];
  }
}

/*
get pair from matrix
base case: pair does not exist, return NULL
pair has a negative value, symbolizes the pair has been compressed
*/
double *getPair(PairMatrix *M, int i, int j){
  int index = INDEX(M->rows, M->cols, i, j);

  if(M->data[index] == NULL)
    return NULL;

  if(M->data[index][0] < 0)
    return M->data[index][0];

  int i;
  double *pair = malloc(sizeof(double) * M->k);
  for(i = 0; i < M->k; i++){
    pair[i] = M->data[index][i];
  }

  return pair;
}

/*
replace pairs first element with index of pair in parentlist
negative index will symbolize Compression
*/
void compressPair(PairMatrix *M, int i, int j, int parentindex){
  int index = INDEX(M->rows, M->cols, i, j);
  M->data[index][0] = ((parentindex+1) * -1);
}
