#ifndef __SOFTMAX_H
#define __SOFTMAX_H
#include"cnn.h"
struct softmax{
    int sample_size;

//与上一层相连
    struct data_box *in;
    struct data_box *out;
    double *label;

    double *(*forward_pass)(struct softmax *,int);
};
struct softmax *softmax_init(struct data_box *common_p,struct data_box *dcommon_p);
double *softmax_forward_pass(struct softmax *,int);
void max_exp_div(double *box,int num);
#endif // __SOFTMAX_H
