/*
pair operations
*/

#define PAIRINDEX(r,c, k, i, j) (i*r + j) * k

typedef struct Pairs {

  int k;    //length of a pair (usually just 2 or 3)
  int n;    //number of pairings per record (usually 41)
  int rows; //number of records (a lot, >500k)
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
note that there will be Ck(N) [n choose k] * r pairings
*/
void initPairs(Pairs *P, int k, int n, int r){
  P->k = k;
  P->n = n;
  P->rows = r;

  //have to multiply number of pairings * elements per pair when allocating
  int nk = nk_count(n, k) * k;

  int size = (nk * r) * sizeof(double);
  P->data = malloc(size);
}

/*
write single pairing to matrix
*/
void savePair(Pairs *P, double *pair, int i, int j){

  int index = PAIRINDEX(P->rows, P->n, P->k, i, j);
  
  //index used to keep track of pairing index is l
  //to get index of pair, need to mulitply pair index by k, (pindex * k) + k is ending index

  int l;
  //printf("%d: ", index/P->k);
  for(l = 0; l < P->k; l++){
    //printf("%.2f to %d, ", pair[l], (index+l));
    P->data[index + l] = pair[l];
  }
  //printf("\n");
}

/*
return an array of pairs from matrix
*/
double *getPair(Pairs *P, int i, int j){

  int index = PAIRINDEX(P->rows, P->n, P->k, i, j);
  int l;

  //populate array with all elements of this pair
  double *pairs = malloc(sizeof(double) * P->k);

  for(l = 0; l < P->k; l++){
    pairs[l] = P->data[index + l];
  }

  return pairs;
}

void freePairs(Pairs *P){
  free(P->data);
}

void fullprint(Pairs *P){
  int i, j;
  int nk = nk_count(P->n, P->k);
  for(i = 0; i < P->n; i++){
    for(j = 0; j < nk; j++){
      int index = PAIRINDEX(P->rows, P->n, P->k, i, j);
      printf("%d %d | %.2f - %.2f\n", i, j, P->data[index], P->data[index + 1]);
    }
  }
}
