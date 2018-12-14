#include"pool_layer.h"
#include"cnn.h"
#include<stdlib.h>
#include<time.h>
#include<string.h>
void pool_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,int pool_h,int pool_w){
    struct pool_layer *new_layer=(struct pool_layer *)malloc(sizeof(struct pool_layer));
//神经网络的属性初始化
    new_layer->forward_pass=pool_layer_forward_pass;
    new_layer->backward_pass=pool_layer_backward_pass;
    new_layer->update=pool_layer_update;
    new_layer->load_weight=pool_layer_load_weight;
    new_layer->pack_dweight=pool_layer_pack_dweight;

    new_layer->weight_size=0;

    struct data_box *common_p=*com_p;
    struct data_box *dcommon_p=*dcom_p;

    int sample_num=common_p->shape[0];
    int input_h=common_p->shape[1];
    int input_w=common_p->shape[2];
    int channel=common_p->shape[3];

    new_layer->pool_h=pool_h;
    new_layer->pool_w=pool_w;

    new_layer->out=(struct data_box *)malloc(sizeof(struct data_box));
    new_layer->dout=(struct data_box *)malloc(sizeof(struct data_box));

    int out_h=input_h/pool_h;
    int out_w=input_w/pool_w;
//    if(out_h==0) out_h=1;
//    if(out_w==0) out_w=1;

//神经网络的参数初始化
    new_layer->index=(int *)malloc(sample_num*out_h*out_w*channel*sizeof(int));
//用于训练的内存空间初始化
    new_layer->out->data=(double *)malloc(sample_num*out_h*out_w*channel*sizeof(double));
    new_layer->out->shape=(int *)malloc(4*sizeof(int));
    new_layer->out->shape[0]=sample_num;
    new_layer->out->shape[1]=out_h;
    new_layer->out->shape[2]=out_w;
    new_layer->out->shape[3]=channel;
    new_layer->out->ndims=4;

    new_layer->dout->data=(double *)malloc(sample_num*out_h*out_w*channel*sizeof(double));
    new_layer->dout->shape=(int *)malloc(4*sizeof(int));
    new_layer->dout->shape[0]=sample_num;
    new_layer->dout->shape[1]=out_h;
    new_layer->dout->shape[2]=out_w;
    new_layer->dout->shape[3]=channel;
    new_layer->dout->ndims=4;
//用于上下连接的内存空间赋值
    new_layer->x=common_p;
    new_layer->dx=dcommon_p;
    *com_p=new_layer->out;
    *dcom_p=new_layer->dout;

    *init_layer=(struct layer*)new_layer;
}

void pool_layer_forward_pass(struct layer *l,int state){
    struct pool_layer *layer=(struct pool_layer *)l;
    int sample_num=layer->x->shape[0];
    int input_h=layer->x->shape[1];
    int input_w=layer->x->shape[2];
    int channel=layer->x->shape[3];

    int pool_h=layer->pool_h;
    int pool_w=layer->pool_w;
    int out_h=layer->out->shape[1];
    int out_w=layer->out->shape[2];

    int *index=(int *)malloc(sample_num*out_h*out_w*channel*sizeof(int));
    for(int i=0;i<sample_num;i++){
        for(int j=0;j<channel;j++){
            for(int k=0;k<out_h;k++){
                for(int l=0;l<out_w;l++){
                    int root_index=i*input_h*input_w*channel+k*pool_h*input_w*channel+l*pool_w*channel+j;
                    for(int m=0;m<pool_h;m++){
                        for(int n=0;n<pool_w;n++){
                                index[m*pool_w+n]=root_index+m*input_w*channel+n*channel;
                        }
                    }
                    int max_index=0;
                    for(int n=1;n<pool_h*pool_w;n++){
                        if(layer->x->data[index[n]] > layer->x->data[index[max_index]]){
                            max_index=n;
                        }
                    }
                    layer->out->data[i*out_h*out_w*channel+k*out_w*channel+l*channel+j]=layer->x->data[index[max_index]];
                    layer->index[i*out_h*out_w*channel+k*out_w*channel+l*channel+j]=index[max_index];
                }
            }
        }
    }
    free(index);
    layer->out->shape[0]=sample_num;
}

void pool_layer_backward_pass(struct layer *l){
    struct pool_layer *layer=(struct pool_layer *)l;
    int sample_num=layer->x->shape[0];
    int input_h=layer->x->shape[1];
    int input_w=layer->x->shape[2];
    int channel=layer->x->shape[3];

    int out_h=layer->out->shape[1];
    int out_w=layer->out->shape[2];

    int *index=layer->index;

    memset(layer->dx->data,0,sample_num*input_h*input_w*channel*sizeof(double));
    for(int i=0;i<sample_num*out_h*out_w*channel;i++){
        layer->dx->data[index[i]]=layer->dout->data[i];
    }
}

void pool_layer_update(struct CNN *cnn,struct layer *l){
    return;
}

void pool_layer_load_weight(struct layer *l,double *p){
    return;
}

void pool_layer_pack_dweight(struct layer *l,double *p){
    return;
}
