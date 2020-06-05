#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.h"
#include "hostread.h"

void generatePairs(Matrix M){



}

int main(int argc, char **argv){

    Matrix M;
    int *recordflags;
    char **attacks;

    hostread(&M, recordflags, attacks);

    //printRow(&M, 23);

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