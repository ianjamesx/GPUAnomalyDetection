/*
matrix operations
for 2d arrays, to save space, use 1d array
keep track of rows and columns
*/

#define PAIRINDEX(r,c, i, j, p) i*r + j

typedef struct Pairs {
  
  int k;    //length of a pair (usually just 2 or 3)
  int n;    //number of pairings per record
  int rows; //number of records
  double *data;

} Pairs;

/*
allocate for matrix
*/
void initPairs(Pairs *P, int k, int n, int r){
  P->k = k;
  P->n = n;
  P->rows = r;

  int size = (k*n*r) * sizeof(double);
  P->data = malloc(size);
}

/*
write single element to matrix
*/
void writePairs(Matrix *M, double val, int i, int j){
    int index = INDEX(M->rows, M->cols, i, j);
    M->data[index] = val;
}

/*
get element from matrix
*/
double getPair(Matrix *M, int record, int pair){
  int index = INDEX(M->rows, M->cols, i, j);
  return M->data[index];
}

/*
copy array to a matrix
then free array
*/
void transferArray(Matrix *M, double *array, int row_index){
  int i;
  for(i = 0; i < M->cols; i++){
    writeMatrix(M, array[i], row_index, i);
  }
  free(array);
}

void freeMatrix(Matrix *M){
  free(M->data);
}

void printMatrix(Matrix *M){
  int i, j;
  for(i = 0; i < M->rows; i++){
    for(j = 0; j < M->cols; j++){
      int index = INDEX(M->rows, M->cols, i, j);
      printf("%.2f ", M->data[index]);
    }
    printf("\n");
  }
}

void printRow(Matrix *M, int row){
  int i;
  for(i = 0; i < M->cols; i++){
    int index = INDEX(M->rows, M->cols, row, i);
    printf("%d: %.2f ", i, M->data[index]);
    printf("\n");
  }
}