#ifndef __READ_NPY_H
#define __READ_NPY_H

#include<stdio.h>
#include"cnn.h"
//struct data_box{
//    double *data;
//    int *shape;
//    int ndims;
//};
//void parse_npy_header(FILE* ,struct data_box *);
struct data_box *npy_load(char *fname,int id,int nb_procs);
#endif // __READ_NPY_H
