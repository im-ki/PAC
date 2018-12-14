#ifndef __BN_LAYER_H
#define __BN_LAYER_H

#include"cnn.h"

struct BN_layer{
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

    int weight_size;

    int sample_size;
    double eps;
    double momentum;

//该层内部参数
    double *gamma;
    double *beta;
    double *dgamma;
    double *dbeta;
//adam优化器参数
    double *gammam;
    double *gammav;
    double *betam;
    double *betav;

    double *running_mean;
    double *running_var;
    double *mean;
    double *var;
    double *istd;
    double *x_hat;
};

void BN_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p);
void BN_layer_forward_pass(struct layer *,int);
void BN_layer_backward_pass(struct layer *);
void BN_layer_update(struct CNN *,struct layer *);
void BN_layer_load_weight(struct layer *l,double *p);
void BN_layer_pack_dweight(struct layer *l,double *p);

#endif // __BN_LAYER_H
