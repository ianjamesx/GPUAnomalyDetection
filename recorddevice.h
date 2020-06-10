void attributeindex_real(int r, int i, int j, int *index){
    *index = (i * r) + j;
}

void recordlist_init(double **all_records, int record_count, int record_size){
    
    //record_count  - number of records (rows)
    //record_size   - number of attributes in a record (columns)
    int size = (record_count * record_size) * sizeof(double);
    *all_records = malloc(size);
}

void getRecord(double *all_records, int record_count, int record_size, int record_index, double **record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        *record[i] = all_records[attributeindex];
    }

}

void saveRecord(double *all_records, int record_count, int record_size, int record_index, double *record){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        all_records[attributeindex] = record[i];
    }

}

void printRecord_full(double *all_records, int record_count, int record_size, int record_index){

    //get start and end indices of record in list of all records
    int recordindex_start;
    attributeindex_real(record_index, record_count, 0, &recordindex_start);

    int i;
    for(i = 0; i < record_size; i++){
        int attributeindex = recordindex_start + i;
        printf("%.2f\n", all_records[attributeindex]);
    }

}