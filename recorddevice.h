void attributeindex_real(int r, int i, int j, int *index){
    *index = (i * r) + j;
}

void getRecord(float *all_records, int record_count, int record_size, int record_index, float **record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        *record[i] = all_records[attributeindex];
    }

}

void saveRecord(float *all_records, int record_count, int record_size, int record_index, float *record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_size, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        all_records[attributeindex] = record[i];
    }

}

void printRecord_full(float *all_records, int record_count, int record_size, int record_index){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        cout << all_records[attributeindex] << endl;
    }

}