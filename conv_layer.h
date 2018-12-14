#ifndef __CONV_LAYER_H
#define __CONV_LAYER_H

#include"cnn.h"

struct conv_layer{
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
    int filter_n,filter_h,filter_w,stride_h,stride_w;

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

void conv_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,\
                     int filter_n,int filter_h,int filter_w,int stride_h,int stride_w);
void conv_layer_forward_pass(struct layer *,int);
void conv_layer_backward_pass(struct layer *);
void conv_layer_update(struct CNN *,struct layer *);
void conv_layer_load_weight(struct layer *l,double *p);
void conv_layer_pack_dweight(struct layer *l,double *p);


#endif // __CONV_LAYER_H

