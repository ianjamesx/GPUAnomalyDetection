#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.h"
#include "pairs.h"
#include "hostread.h"

/*
index - current record index, used for storing record pairing
*/
void saveRecordPairs(Pairs *P, Matrix M, int index){

  int i, j;

  int pairindex = 0;

  for(i = 0; i < M.cols; i++){
    for(j = i+1; j < M.cols; j++){
      
      //generate pair
      double *pair = malloc(sizeof(double) * 2);
      pair[0] = getValue(&M, index, i);
      pair[1] = getValue(&M, index, j);
      
      //save
      savePair(P, pair, index, pairindex);
      pairindex++;
    
    }

  }

}

void generatePairs(Matrix M){

  int pairsize = 2;

  Pairs P;
  initPairs(&P, pairsize, M.cols, M.rows);

  int i;
  for(i = 0; i < M.rows; i++){
    saveRecordPairs(&P, M, i);
  }

  //fullprint(&P);

}

int main(int argc, char **argv){

    Matrix M;
    int *recordflags;
    char **attacks;

    hostread(&M, recordflags, attacks);

    printRow(&M, 0);

    generatePairs(M);

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
