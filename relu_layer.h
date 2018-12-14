#ifndef __RELU_LAYER_H
#define __RELU_LAYER_H
#include"cnn.h"
struct relu_layer{
//与上下层相连
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

void relu_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p);
void relu_layer_forward_pass(struct layer *,int);
void relu_layer_backward_pass(struct layer *);
void relu_update(struct CNN *,struct layer *);
void relu_layer_load_weight(struct layer *l,double *p);
void relu_layer_pack_dweight(struct layer *l,double *p);

#endif // RELU_LAYER_H
