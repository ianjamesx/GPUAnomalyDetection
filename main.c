#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.h"
#include "pairs.h"
#include "hostread.h"

void saveRecordPairs(Pairs *P, Matrix M, int index){

  int i, j;

  int total = 0;

  for(i = 0; i < M.cols; i++){
    for(j = i+1; j < M.cols; j++){
      int pairIndex = (i * M.cols) + j;
      printf("%d| %d %d | %.2f %.2f\n", pairIndex, i, j, getValue(&M, index, i), getValue(&M, index, j));
      total++;
      double pair[2] = {getValue(&M, index, i), getValue(&M, index, j)};
      //savePair(&P, pair, index, pairIndex);
    }
    printf("----------\n");
  }

  printf("%d\n", total);

}

void generatePairs(Matrix M){

  int pairsize = 2;

  Pairs P;
  initPairs(&P, pairsize, M.cols, M.rows);

  int i;
  for(i = 0; i <1; i++){
    saveRecordPairs(&P, M, i);
  }

}

int main(int argc, char **argv){

    Matrix M;
    int *recordflags;
    char **attacks;

    hostread(&M, recordflags, attacks);

    printRow(&M, 0);

    generatePairs(M);

    printf("%d\n", nk_count(40, 2));

    /*int test[] = {1,2,3,4};

    int i, j, l;
    for(i = 0; i < 4; i++){
      for(j = i+1; j < 4; j++){
        printf("%d, %d\n", test[i], test[j]);
      }
    }*/

    return 0;
}

/*



*/

/*

parent-linked list approach

original records:
[4,1,3,5]
[4,1,2,5]
[0,1,2,5]

...

parentlist  := [(4,1), (2,5)]

[
    0 -> 0
    1 -> 0 -> 1
    2 -> 1
]

*/
