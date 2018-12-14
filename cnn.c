#include"cnn.h"
#include"conv_layer.h"
#include"relu_layer.h"
#include"pool_layer.h"
#include"BN_layer.h"
#include"dense_layer.h"
#include"softmax.h"
#include<stdlib.h>
#include<math.h>
struct CNN *cnn_init(struct data_box *input_data){
    struct CNN *cnn = (struct CNN *)malloc(sizeof(struct CNN));
    int *shape=input_data->shape;
    int ndims=input_data->ndims;
    cnn->state=TRAIN;

    cnn->adam_para.beta1=0.9;
    cnn->adam_para.beta2=0.99;
    cnn->adam_para.eps=1e-8;
    cnn->adam_para.lr=1e-3;
    cnn->adam_para.t=0;

    struct data_box *common_p;
    struct data_box *dcommon_p;

    common_p=(struct data_box *)malloc(sizeof(struct data_box));
    //common_p->data=(double *)malloc(shape[0]*neuron_num*sizeof(double));
    common_p->shape=shape;
    common_p->ndims=ndims;

    int size=1;
    for(int i=0;i<ndims;i++) size*=shape[i];

    dcommon_p=(struct data_box *)malloc(sizeof(struct data_box));
    dcommon_p->data=(double *)malloc(size*sizeof(double));
    dcommon_p->shape=shape;
    dcommon_p->ndims=ndims;

    //定义网络的结构

//    conv_layer_init(&cnn->layer_box[0],common_p,dcommon_p);
//    BN_layer_init(&cnn->layer_box[1],common_p,dcommon_p);
//    relu_layer_init(&cnn->layer_box[2],common_p,dcommon_p);
//    conv_layer_init(&cnn->layer_box[3],common_p,dcommon_p);
//    BN_layer_init(&cnn->layer_box[4],common_p,dcommon_p);
//    relu_layer_init(&cnn->layer_box[5],common_p,dcommon_p);
//    pool_layer_init(&cnn->layer_box[6],common_p,dcommon_p);
//    conv_layer_init(&cnn->layer_box[7],common_p,dcommon_p);
//    BN_layer_init(&cnn->layer_box[8],common_p,dcommon_p);
//    relu_layer_init(&cnn->layer_box[9],common_p,dcommon_p);
//    pool_layer_init(&cnn->layer_box[10],common_p,dcommon_p);
//    dense_layer_init(&cnn->layer_box[0],&common_p,&dcommon_p,2048);
//    relu_layer_init(&cnn->layer_box[1],&common_p,&dcommon_p);
	conv_layer_init(&cnn->layer_box[0],&common_p,&dcommon_p,32,5,6,1,1);
    BN_layer_init(&cnn->layer_box[1],&common_p,&dcommon_p);
    relu_layer_init(&cnn->layer_box[2],&common_p,&dcommon_p);
    pool_layer_init(&cnn->layer_box[3],&common_p,&dcommon_p,2,2);
    conv_layer_init(&cnn->layer_box[4],&common_p,&dcommon_p,64,5,4,1,2);
    BN_layer_init(&cnn->layer_box[5],&common_p,&dcommon_p);
    relu_layer_init(&cnn->layer_box[6],&common_p,&dcommon_p);
    pool_layer_init(&cnn->layer_box[7],&common_p,&dcommon_p,1,2);
    conv_layer_init(&cnn->layer_box[8],&common_p,&dcommon_p,64,1,6,1,1);
    BN_layer_init(&cnn->layer_box[9],&common_p,&dcommon_p);
    relu_layer_init(&cnn->layer_box[10],&common_p,&dcommon_p);
    pool_layer_init(&cnn->layer_box[11],&common_p,&dcommon_p,1,2);
    dense_layer_init(&cnn->layer_box[12],&common_p,&dcommon_p,1024);
    BN_layer_init(&cnn->layer_box[13],&common_p,&dcommon_p);
    relu_layer_init(&cnn->layer_box[14],&common_p,&dcommon_p);
    dense_layer_init(&cnn->layer_box[15],&common_p,&dcommon_p,512);
    BN_layer_init(&cnn->layer_box[16],&common_p,&dcommon_p);
    relu_layer_init(&cnn->layer_box[17],&common_p,&dcommon_p);
    dense_layer_init(&cnn->layer_box[18],&common_p,&dcommon_p,3);

    cnn->softmax_obj=softmax_init(common_p,dcommon_p);

    cnn->weight_size=0;
    for(int i=0;i<sizeof(cnn->layer_box)/sizeof(struct layer *);i++){
        cnn->weight_size+=cnn->layer_box[i]->weight_size;
    }

    return cnn;
}

//神经网络运行流程控制
double *go(struct CNN *cnn,int state){
    double *result_box;//返回loss和acc
    cnn->state=state;
//前向传播
    for(int i=0;i<sizeof(cnn->layer_box)/sizeof(struct layer *);i++){
        struct layer *layer=cnn->layer_box[i];
        layer->forward_pass(layer,TRAIN);
    }
    result_box=cnn->softmax_obj->forward_pass(cnn->softmax_obj,state);
    if(state==TRAIN){
    //反向传播
        for(int i=sizeof(cnn->layer_box)/sizeof(struct layer *)-1;i>=0;i--){
            struct layer *layer=cnn->layer_box[i];
            layer->backward_pass(layer);
        }
        cnn->adam_para.t++;
    //更新梯度
        for(int i=0;i<sizeof(cnn->layer_box)/sizeof(struct layer *);i++){
            struct layer *layer=cnn->layer_box[i];
            layer->update(cnn,layer);
        }
    }else{//state==test

    }
    return result_box;//记得释放内存
}

//adam优化器
void adam(struct CNN *cnn,double *x,double *dx,double *m,double *v,int size){
    double b1,b2,lr,eps,t,coef;
    b1=cnn->adam_para.beta1;
    b2=cnn->adam_para.beta2;
    eps=cnn->adam_para.eps;
    lr=cnn->adam_para.lr;
    t=cnn->adam_para.t;

    coef=-lr*sqrt(1-pow(b2,t))/(1-pow(b1,t));

    for(int i=0;i<size;i++){
        m[i]=b1*m[i]+(1-b1)*dx[i];
        v[i]=b2*v[i]+(1-b2)*dx[i]*dx[i];
        x[i]+=coef*m[i]/(sqrt(v[i])+eps);
    }
}

//喂数据
void feed(struct CNN *cnn,struct feed_data *data){
    struct layer *layer=cnn->layer_box[0];
    layer->x->data=data->data;
    layer->x->shape[0]=data->sample_num;
    cnn->softmax_obj->label=data->label;
}

void load_weight(struct CNN *cnn,double *buf){
    int current_size=0;
    struct layer *layer;
    for(int i=0;i<sizeof(cnn->layer_box)/sizeof(struct layer *);i++){
        layer=cnn->layer_box[i];
        layer->load_weight(layer,buf+current_size);
        current_size+=layer->weight_size;
    }
}

void pack_dweight(struct CNN *cnn,double *buf){
    int current_size=0;
    struct layer *layer;
    for(int i=0;i<sizeof(cnn->layer_box)/sizeof(struct layer *);i++){
        layer=cnn->layer_box[i];
        layer->pack_dweight(layer,buf+current_size);
        current_size+=layer->weight_size;
    }
}
