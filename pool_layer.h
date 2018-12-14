#ifndef __POOL_LAYER_H
#define __POOL_LAYER_H

#include"cnn.h"

struct pool_layer{
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

    int pool_h,pool_w;
    int *index;
};

void pool_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,int pool_h,int pool_w);
void pool_layer_forward_pass(struct layer *,int);
void pool_layer_backward_pass(struct layer *);
void pool_layer_update(struct CNN *,struct layer *);
void pool_layer_load_weight(struct layer *l,double *p);
void pool_layer_pack_dweight(struct layer *l,double *p);
void pool_layer_pack_weight(struct layer *l,double *p);

#endif // __POOL_LAYER_H
