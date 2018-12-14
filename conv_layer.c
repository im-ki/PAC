#include"conv_layer.h"
#include"cnn.h"
#include<stdlib.h>
#include<time.h>
#include<string.h>
void conv_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,\
                     int filter_n,int filter_h,int filter_w,int stride_h,int stride_w){
    struct conv_layer *new_layer=(struct conv_layer *)malloc(sizeof(struct conv_layer));
//神经网络的属性初始化
    new_layer->forward_pass=conv_layer_forward_pass;
    new_layer->backward_pass=conv_layer_backward_pass;
    new_layer->update=conv_layer_update;
    new_layer->load_weight=conv_layer_load_weight;
    new_layer->pack_dweight=conv_layer_pack_dweight;
    new_layer->pack_weight=conv_layer_pack_weight;

    struct data_box *common_p=*com_p;
    struct data_box *dcommon_p=*dcom_p;

    int sample_num=common_p->shape[0];
    int input_h=common_p->shape[1];
    int input_w=common_p->shape[2];
    int channel=common_p->shape[3];

    new_layer->filter_n=filter_n;
    new_layer->filter_h=filter_h;
    new_layer->filter_w=filter_w;
    new_layer->stride_h=stride_h;
    new_layer->stride_w=stride_w;

    new_layer->out=(struct data_box *)malloc(sizeof(struct data_box));
    new_layer->dout=(struct data_box *)malloc(sizeof(struct data_box));

    int out_h=(input_h-filter_h)/stride_h+1;
    int out_w=(input_w-filter_w)/stride_w+1;

//神经网络的参数初始化
    new_layer->w=(double *)malloc(filter_n*filter_h*filter_w*channel*sizeof(double));
    srand(time(NULL));
    for(int i=0;i<filter_n*filter_h*filter_w*channel;i++){
        new_layer->w[i]=rand()/(double)(RAND_MAX)*2-1;
    }
    new_layer->dw=(double *)malloc(filter_n*filter_h*filter_w*channel*sizeof(double));
//    new_layer->wm=(double *)calloc(filter_n*filter_h*filter_w*channel , sizeof(double));
//    new_layer->wv=(double *)calloc(filter_n*filter_h*filter_w*channel , sizeof(double));

    new_layer->b=(double *)calloc(filter_n , sizeof(double));
    new_layer->db=(double *)malloc(filter_n*sizeof(double));
//    new_layer->bm=(double *)calloc(filter_n , sizeof(double));
//    new_layer->bv=(double *)calloc(filter_n , sizeof(double));

    new_layer->weight_size=filter_n*filter_h*filter_w*channel+filter_n;
//用于训练的内存空间初始化
    new_layer->out->data=(double *)malloc(sample_num*out_h*out_w*filter_n*sizeof(double));
    new_layer->out->shape=(int *)malloc(4*sizeof(int));
    new_layer->out->shape[0]=common_p->shape[0];
    new_layer->out->shape[1]=out_h;
    new_layer->out->shape[2]=out_w;
    new_layer->out->shape[3]=filter_n;
    new_layer->out->ndims=4;

    new_layer->dout->data=(double *)malloc(sample_num*out_h*out_w*filter_n*sizeof(double));
    new_layer->dout->shape=(int *)malloc(4*sizeof(int));
    new_layer->dout->shape[0]=common_p->shape[0];
    new_layer->dout->shape[1]=out_h;
    new_layer->dout->shape[2]=out_w;
    new_layer->dout->shape[3]=filter_n;
    new_layer->dout->ndims=4;
//用于上下连接的内存空间赋值
    new_layer->x=common_p;
    new_layer->dx=dcommon_p;
    *com_p=new_layer->out;
    *dcom_p=new_layer->dout;

    *init_layer=(struct layer*)new_layer;
}

void conv_layer_forward_pass(struct layer *l,int state){
    struct conv_layer *layer=(struct conv_layer *)l;
    int sample_num=layer->x->shape[0];
    int input_h=layer->x->shape[1];
    int input_w=layer->x->shape[2];
    int channel=layer->x->shape[3];

    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;
    int stride_h=layer->stride_h;
    int stride_w=layer->stride_w;
    int out_h=layer->out->shape[1];
    int out_w=layer->out->shape[2];

    int single_filter_size=filter_h*filter_w*channel;
    int *index=(int *)malloc(single_filter_size*sizeof(int));

    for(int i=0;i<sample_num;i++){
        for(int j=0;j<out_h;j++){
            for(int k=0;k<out_w;k++){
                int root_index=i*input_h*input_w*channel+j*stride_h*input_w*channel+k*stride_w*channel;
                for(int l=0;l<filter_h;l++){
                    for(int m=0;m<filter_w;m++){
                        for(int n=0;n<channel;n++){
                            index[l*filter_w*channel+m*channel+n]=root_index+l*input_w*channel+m*channel+n;
                        }
                    }
                }
                double sum=0;
                for(int l=0;l<filter_n;l++,sum=0){
                    for(int m=0;m<single_filter_size;m++){
                        sum += layer->w[l*single_filter_size+m] * layer->x->data[index[m]];
                    }
                    sum+=layer->b[l];
                    layer->out->data[i*out_h*out_w*filter_n+j*out_w*filter_n+k*filter_n+l]=sum;
                    
                    //printf("%f ",sum);
                }
            }
        }
    }
    free(index);
    layer->out->shape[0]=sample_num;
}

void conv_layer_backward_pass(struct layer *l){
    struct conv_layer *layer=(struct conv_layer *)l;
    int sample_num=layer->x->shape[0];
    int input_h=layer->x->shape[1];
    int input_w=layer->x->shape[2];
    int channel=layer->x->shape[3];

    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;
    int stride_h=layer->stride_h;
    int stride_w=layer->stride_w;
    int out_h=layer->out->shape[1];
    int out_w=layer->out->shape[2];

    int single_filter_size=filter_h*filter_w*channel;
    int *index=(int *)malloc(single_filter_size*sizeof(int));
//dw and dx
    memset(layer->dw,0,filter_n*single_filter_size*sizeof(double));
    memset(layer->dx->data,0,sample_num*input_h*input_w*channel*sizeof(double));
    for(int i=0;i<sample_num;i++){
        for(int j=0;j<out_h;j++){
            for(int k=0;k<out_w;k++){
                int root_index=i*input_h*input_w*channel+j*stride_h*input_w*channel+k*stride_w*channel;
                for(int l=0;l<filter_h;l++){
                    for(int m=0;m<filter_w;m++){
                        for(int n=0;n<channel;n++){
                            index[l*filter_w*channel+m*channel+n]=root_index+l*input_w*channel+m*channel+n;
                        }
                    }
                }
                double filter_dout=0;
                for(int l=0;l<filter_n;l++){
                    filter_dout=layer->dout->data[i*out_h*out_w*filter_n+j*out_w*filter_n+k*filter_n+l];
                    for(int m=0;m<single_filter_size;m++){
                        layer->dw[l*single_filter_size+m] += filter_dout * layer->x->data[index[m]];
                        layer->dx->data[index[m]] += filter_dout * layer->w[l*single_filter_size+m];
                    }
                }
            }
        }
    }
//db
    double sum=0;
    for(int i=0;i<filter_n;i++,sum=0){
        for(int j=0;j<sample_num;j++){
            for(int k=0;k<out_h;k++){
                for(int l=0;l<out_w;l++){
                    sum+=layer->dout->data[j*out_h*out_w*filter_n+k*out_w*filter_n+l*filter_n+i];
                }
            }
        }
        layer->db[i]=sum;
    }
    free(index);
}

void conv_layer_update(struct CNN *cnn,struct layer *l){
    struct conv_layer *layer=(struct conv_layer *)l;

    int channel=layer->x->shape[3];
    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;

    adam(cnn,layer->w,layer->dw,layer->wm,layer->wv,filter_n*filter_h*filter_w*channel);
    adam(cnn,layer->b,layer->db,layer->bm,layer->bv,filter_n);
}

void conv_layer_load_weight(struct layer *l,double *p){
    struct conv_layer *layer=(struct conv_layer *)l;

    int channel=layer->x->shape[3];
    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;

    memcpy(layer->w,p,filter_n*filter_h*filter_w*channel*sizeof(double));
    memcpy(layer->b,p+filter_n*filter_h*filter_w*channel,filter_n*sizeof(double));
}

void conv_layer_pack_dweight(struct layer *l,double *p){
    struct conv_layer *layer=(struct conv_layer *)l;

    int channel=layer->x->shape[3];
    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;

    memadd(p,layer->dw,filter_n*filter_h*filter_w*channel);
    memadd(p+filter_n*filter_h*filter_w*channel,layer->db,filter_n);
}

void conv_layer_pack_weight(struct layer *l,double *p){
    struct conv_layer *layer=(struct conv_layer *)l;

    int channel=layer->x->shape[3];
    int filter_n=layer->filter_n;
    int filter_h=layer->filter_h;
    int filter_w=layer->filter_w;

    memcpy(p,layer->w,filter_n*filter_h*filter_w*channel*sizeof(double));
    memcpy(p+filter_n*filter_h*filter_w*channel,layer->b,filter_n*sizeof(double));
}
