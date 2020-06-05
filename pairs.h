/*
pair operations
*/

#define PAIRINDEX(r,c, i, j) i*r + j

typedef struct Pairs {

  int k;    //length of a pair (usually just 2 or 3)
  int n;    //number of pairings per record
  int rows; //number of records
  double *data;

} Pairs;

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
allocate for pairs
*/
void initPairs(Pairs *P, int k, int n, int r){
  P->k = k;
  P->n = n;
  P->rows = r;

  int nk = nk_count(n, k);

  int size = (nk * r) * sizeof(double);
  //printf("Allo %d\n", size);
  P->data = malloc(size);
}

/*
write single pairing to matrix
*/
void savePair(Pairs *P, double pair[], int i, int j){
  int index = PAIRINDEX(P->rows, P->n, i, j);
  int l, k = P->k;

  for(l = 0; l < k; l++){
    P->data[index + l] = pair[l];
  }
}

/*
return an array of pairs from matrix
*/
double *getPair(Pairs *P, int i, int j){

  int index = PAIRINDEX(P->rows, P->n, i, j);
  int l, k = P->k;

  //populate array with all elements of this pair
  double *pairs = malloc(sizeof(double) * k);
  for(l = 0; l < k; l++){
    pairs[l] = P->data[index + l];
  }

  return pairs;
}

void freePairs(Pairs *P){
  free(P->data);
}
