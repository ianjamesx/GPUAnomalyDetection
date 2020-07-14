

//get index of attribute in 1d representation of 2d array
__device__
int getMatrixIndex_device(int record_size, int i, int j){
    return (i * record_size) + j;
}

__device__
int uniqueIndex(){
    return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__
int symbolic_device(int index){
    if(index == 1 || index == 2 || index == 3){
        return 1;
    }
    return 0;
}

__device__
void vectorsubtraction(float *records, float *vbuffer, int r1, int r2, int vsize){

    int i;
    //vectors must be same size
    for(i = 0; i < vsize; i++){

        //first, get indices of attributes from records
        int r1index = getMatrixIndex_device(vsize, r1, i);
        int r2index = getMatrixIndex_device(vsize, r2, i);

        //if symbolic attribute, push 1 to represent not equal, 0 to represent equal
        if(symbolic_device(i)){
            vbuffer[i] = (records[r1index] != records[r2index]);
          //if continous, subtract values
        } else {
            vbuffer[i] = (records[r1index] - records[r2index]);
        }
    }
}



__device__
float normalizevector(float *v, int vsize){

    int i;
    float total = 0.0;
    for(i = 0; i < vsize; i++){
        total += (v[i] * v[i]);
    }

    return sqrt(total);
}
/*
__device__
double edgeweight(float *v1, float *v2){

    vector<float> v3;
    vectorsubtraction(v1, v2, v3);

    float normalized = normalizevector(v3);
    double base = exp(normalized);
    return (1 / base);
}
*/


__global__
void edgeGeneration(float *edges, float *records, int record_size, int record_count){

    int i, j;
    int index = uniqueIndex();
    int stride = blockDim.x * gridDim.x;

    //vector buffer used for subtracting v[i] - v[j] on each iteration
    //set up here as kernel is slow to allocate
    float *vbuffer = new float[record_size];

    for(i = index; i < record_count; i += stride){
        for(j = 0; j < record_count; j++){

            //if compare to this record, simply set to 0 as it will be identical
            if(i == j){
                int edgeindex = getMatrixIndex_device(record_count, i, j);
                edges[edgeindex] = 0;
                continue;
            }

            vectorsubtraction(records, vbuffer, i, j, record_size);

            //normalize vector, set edge weight
            float normalized = normalizevector(vbuffer, record_size);
            float base = exp(normalized);

            int edgeindex = getMatrixIndex_device(record_count, i, j);
            edges[edgeindex] = (1 / base);

        }

    }

}
