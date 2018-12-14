#include"read_npy.h"
#include<string.h>
#include<stdio.h>
#include<stdlib.h>
void parse_npy_header(FILE* fp,struct data_box *npy) {
    char buffer[256];
    char *loc1, *loc2;
    size_t res = fread(buffer,sizeof(char),11,fp);
    if(res != 11){
        printf("parse_npy_header: failed fread");
        exit(0);
    }
    fgets(buffer,256,fp);
    //printf("buffer=%s",buffer);
    //fortran order
    //loc1 = strstr(buffer, "fortran_order")+16;
    //npy->fortran_order = strstr(buffer, "True")==NULL?0:1;
    //shape
    loc1 = strstr(buffer, "(")+1;//header.find("(");
    loc2 = strstr(buffer, ")")-1;//header.find(")");
    if(*loc2 == ','){
        npy->ndims = 1;
    }else{
        npy->ndims=0;
        //printf("loc2-loc1=%d\n",loc2-loc1);
        for(int i=0;i<=(loc2-loc1);i++){
            if(*(loc1+i)==','){
                npy->ndims++;
            }
        }
        npy->ndims += 1;
    }
    npy->shape = (int *)malloc(npy->ndims*sizeof(int));
    for(int i = 0;i < npy->ndims;i++) {
        npy->shape[i] = atoi(loc1);
        loc1 = strchr(loc1,',')+1;
    }
    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = strstr(buffer, "descr")+9;
    int littleEndian = (*loc1 == '<' || *loc1 == '|' ? 1 : 0);
    if(littleEndian==0){
        printf("endian wrong\n");
        exit(0);
    }
    //npy->is_int=*(loc1+1)=='i'?1:0;
    //npy->word_size = atoi(loc1+2);双精度浮点型的word_size是8
}

struct data_box *load_the_npy_file(FILE* fp) {
    struct data_box *npy=(struct data_box *)malloc(sizeof(struct data_box));
    parse_npy_header(fp,npy);
    unsigned long long size = 1; //long long so no overflow when multiplying by word_size
    for(unsigned int i = 0;i < (npy->ndims);i++) size *= npy->shape[i];
    npy->data = (double *)malloc(size*8);//new char[size*word_size];
    size_t nread = fread(npy->data,8,size,fp);

    //printf("ndims=%d,word_size=%d\n",npy->ndims,8);

    if(nread != size){
        //throw std::runtime_error("load_the_npy_file: failed fread");
        printf("load_the_npy_file: failed fread");
        exit(0);
    }
    return npy;
}

struct data_box *npy_load(char *fname) {
    FILE* fp = fopen(fname, "rb");
    if(!fp) {
        printf("npy_load: Error! Unable to open file %s!\n",fname);
        exit(0);
    }
    struct data_box *npy = load_the_npy_file(fp);
    fclose(fp);
    return npy;
}
