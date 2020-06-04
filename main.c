#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.h"
#include "hostread.h"

int main(int argc, char **argv){

    Matrix M;
    int *recordflags;
    char **attacks;

    hostread(&M, recordflags, attacks);

    //printRow(&M, 23);
}

[4,2,41,3,5,4,12,4,15,34,53,1,4,1,2,41,2]
[4,2,41,3,5,4,12,4,15,34,53,1,4,1,2,41,2]
[4,2,41,3,5,4,12,4,15,34,53,1,4,1,2,41,2]

