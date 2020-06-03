#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


int main(){

    int ex[5] = {0, 1, 2, 3, 4};
    
    int n = 5, k = 2;
    int i, j, total = 0;


    for(i = 0; i < n; i++){ 
        for(j = i+1; j < n; j++){
            printf("%d, %d\n", i, j);
            total++;
        }
    }

    printf("%d\n", total);

    return 0;

}