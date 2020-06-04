/*
matrix operations
for 2d arrays, to save space, use 1d array
keep track of rows and columns
*/

#define INDEX(r,c,i,j) i*r + j

typedef struct Matrix {
  int rows;
  int cols;
  double *data;
} Matrix;

/*
allocate for matrix
*/
void initMatrix(Matrix *M, int r, int c){
  M->rows = r;
  M->cols = c;

  int size = (r*c) * sizeof(double);
  M->data = malloc(size);
}

/*
write single element to matrix
*/
void writeMatrix(Matrix *M, double val, int i, int j){
    int index = INDEX(M->rows, M->cols, i, j);
    M->data[index] = val;
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