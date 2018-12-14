#ifndef __DENSE_LAYER_H
#define __DENSE_LAYER_H

#include"cnn.h"

struct dense_layer{
//与上一层相连
    struct data_box *x;
    struct data_box *dx;
    //与下一层相连
    struct data_box *out;
    struct data_box *dout;

    void (*forward_pass)(struct layer *,int);
    void (*backward_pass)(struct layer *);
    void (*update)(struct CNN *,struct layer *);
    void (*load_weight)(struct layer *l,double *p);
    void (*pack_dweight)(struct layer *l,double *p);
    void (*pack_weight)(struct layer *l,double *p);

    int weight_size;

    int sample_size;
    int neuron_num;

//该层内部参数
    double *w;
    double *b;
    double *dw;
    double *db;
//adam优化器参数
    double *wm;
    double *wv;
    double *bm;
    double *bv;
};

void dense_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,int neuron_num);
void dense_layer_forward_pass(struct layer *,int);
void dense_layer_backward_pass(struct layer *);
void dense_layer_update(struct CNN *,struct layer *);
void dense_layer_load_weight(struct layer *l,double *p);
void dense_layer_pack_dweight(struct layer *l,double *p);
void dense_layer_pack_weight(struct layer *l,double *p);

#endif // __DENSE_LAYER_H
