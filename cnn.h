#ifndef __CNN_H
#define __CNN_H
#include"read_npy.h"

#define TRAIN 21
#define TEST 22
struct feed_data{
    double *data;
    double *label;
    int sample_num;
};

struct solver{
    double lr;
    double beta1;
    double beta2;
    double eps;
    double t;
};

struct CNN{
	int layerNum;
	int state;
	int weight_size;
	struct solver adam_para;
	struct layer *layer_box[19];
	struct softmax *softmax_obj;
    struct data_box *(*loss_func)();
};


struct layer{
    struct data_box *x;
    struct data_box *dx;
    struct data_box *out;
    struct data_box *dout;
    void (*forward_pass)(struct layer *,int);
    void (*backward_pass)(struct layer *);
    void (*update)(struct CNN *,struct layer *);
    void (*load_weight)(struct layer *l,double *p);
    void (*pack_dweight)(struct layer *l,double *p);
    int weight_size;
};

struct data_box{
        double *data;
        int *shape;
        int ndims;
};

struct CNN *cnn_init(struct data_box *);
double* go(struct CNN *,int);
void adam(struct CNN *cnn,double *x,double *dx,double *m,double *v,int size);
void feed(struct CNN *cnn,struct feed_data *data);
void load_weight(struct CNN *cnn,double *buf);
void pack_dweight(struct CNN *cnn,double *buf);

#endif // __CNN_H
