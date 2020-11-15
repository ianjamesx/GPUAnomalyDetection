#include <sys/types.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

/*
Given a file, memory map into array of chars
need to get file size, then cast to array of characters
*/
char *fastfileread(const char* filepath){

    int fd = open(filepath, O_RDONLY, S_IRUSR | S_IWUSR);
    struct stat sb;

    if(fstat(fd, &sb) == -1) perror("Could not get file size\n");

    printf("File size: %ld\n", sb.st_size);

    char *filecontent = (char*) mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    return filecontent;

}