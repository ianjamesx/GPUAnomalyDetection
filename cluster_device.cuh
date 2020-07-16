

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

__device__
void knn(edgedata *edges, int k, int src, int dest, float weight){

    //int edgeindex = getMatrixIndex_device(record_count, i, j);

    //see if this weight is larger than any others in the list
    int i;
    for(i = 0; i < k; i++){
        int edgeindex = getMatrixIndex_device(k, src, i);
        if(edges[edgeindex].weight < weight){

            //save current weight/vertex to buffers
            float wbuffer = edges[edgeindex].weight;
            int vbuffer= edges[edgeindex].vertex;

            //replace current index in src vertex knn with dest
            edges[edgeindex].weight = weight;
            edges[edgeindex].vertex = dest;

            //recurse through with edge being replaced if there are any other candidates
            knn(edges, k, src, vbuffer, wbuffer);

            //break so we dont replace any other elements
            break;
        }
    }

}

__global__
void edgeGeneration(edgedata *edges, float *records, int record_size, int record_count, int k){

    int i, j;
    int index = uniqueIndex();
    int stride = blockDim.x * gridDim.x;

    //vector buffer used for subtracting v[i] - v[j] on each iteration
    //set up here as kernel is slow to allocate
    float *vbuffer = new float[record_size];

    for(i = index; i < record_count; i += stride){

        //if(index == 0)
                //printf("Thread %d on %d %d\n", index, i);

        for(j = 0; j < record_count; j++){

            //if we are comparing this record to itself, skip
            if(i != j){

                vectorsubtraction(records, vbuffer, i, j, record_size);

                //normalize vector, set edge weight
                float normalized = normalizevector(vbuffer, record_size);
                float base = exp(normalized);
                float weight = (1 / base);

                //update knn graph with the new weight
                knn(edges, k, i, j, weight);

            }
            
        }

    }

    delete [] vbuffer;

}
