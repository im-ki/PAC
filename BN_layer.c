#include"BN_layer.h"
#include"cnn.h"
#include<stdlib.h>
#include<math.h>
#include<string.h>
void BN_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p){
    struct BN_layer *new_layer=(struct BN_layer *)malloc(sizeof(struct BN_layer));
//神经网络的属性初始化
    struct data_box *common_p=*com_p;
    struct data_box *dcommon_p=*dcom_p;

    new_layer->forward_pass=BN_layer_forward_pass;
    new_layer->backward_pass=BN_layer_backward_pass;
    new_layer->update=BN_layer_update;
    new_layer->load_weight=BN_layer_load_weight;
    new_layer->pack_dweight=BN_layer_pack_dweight;

    new_layer->eps=1e-5;
    new_layer->momentum=0.9;

    new_layer->out=(struct data_box *)malloc(sizeof(struct data_box));
    new_layer->dout=(struct data_box *)malloc(sizeof(struct data_box));

//神经网络的参数初始化
    int num=1;
    for(int i=1;i<common_p->ndims;i++){
        num*=common_p->shape[i];
    }
    new_layer->sample_size=num;

    new_layer->gamma=(double *)malloc(num*sizeof(double));
    for(int i=0;i<num;i++){
        new_layer->gamma[i]=1;
    }
    new_layer->beta=(double *)calloc(num,sizeof(double));

    new_layer->weight_size=num+num;

    new_layer->dgamma=(double *)malloc(num*sizeof(double));
    new_layer->dbeta=(double *)malloc(num*sizeof(double));
    //优化器参数
    new_layer->gammam=(double *)calloc(num,sizeof(double));
    new_layer->gammav=(double *)calloc(num,sizeof(double));
    new_layer->betam=(double *)calloc(num,sizeof(double));
    new_layer->betav=(double *)calloc(num,sizeof(double));

    new_layer->running_mean=(double *)calloc(num,sizeof(double));
    new_layer->running_var=(double *)calloc(num,sizeof(double));
    new_layer->mean=(double *)malloc(num*sizeof(double));
    new_layer->var=(double *)malloc(num*sizeof(double));
    new_layer->istd=new_layer->var;

    new_layer->x_hat=(double *)malloc(common_p->shape[0]*num*sizeof(double));

//用于训练的内存空间初始化
    new_layer->out->data=(double *)malloc(common_p->shape[0]*num*sizeof(double));
    new_layer->out->shape=common_p->shape;
    new_layer->out->ndims=common_p->ndims;
    new_layer->dout->data=(double *)malloc(common_p->shape[0]*num*sizeof(double));
    new_layer->dout->shape=common_p->shape;
    new_layer->dout->ndims=common_p->ndims;
//用于上下连接的内存空间赋值
    new_layer->x=common_p;
    new_layer->dx=dcommon_p;
    *com_p=new_layer->out;
    *dcom_p=new_layer->dout;

    *init_layer=(struct layer*)new_layer;
}

void BN_layer_forward_pass(struct layer *l,int state){
    struct BN_layer *layer=(struct BN_layer *)l;

    int sample_num=layer->x->shape[0];
    int sample_size=layer->sample_size;
    double eps=layer->eps;
    double m=layer->momentum;
    double gamma,beta,mean,sqrt_var,istd;

    if(state==TRAIN){
        double col_sum=0;
        for(int i=0;i<sample_size;i++,col_sum=0){
            for(int j=0;j<sample_num;j++){
                col_sum+=layer->x->data[j*sample_size+i];
            }
            layer->mean[i]=col_sum/sample_num;
        }
        double var_sum=0;
        for(int i=0;i<sample_size;i++,var_sum=0){
            mean=layer->mean[i];
            for(int j=0;j<sample_num;j++){
                var_sum+=pow(layer->x->data[j*sample_size+i]- mean , 2);
            }
            layer->var[i]=var_sum/sample_num;
        }
        for(int i=0;i<sample_size;i++){
            layer->running_mean[i]=m*layer->running_mean[i]+(1-m)*layer->mean[i];
            layer->running_var[i]=m*layer->running_var[i]+(1-m)*layer->var[i];
        }
        for(int i=0;i<sample_size;i++){
            layer->istd[i]=1.0/sqrt(layer->var[i]+eps);
        }
        for(int i=0;i<sample_size;i++){
            mean=layer->mean[i];
            istd=layer->istd[i];
            for(int j=0;j<sample_num;j++){
                layer->x_hat[j*sample_size+i] = (layer->x->data[j*sample_size+i] - mean) * istd;
            }
        }
        for(int i=0;i<sample_size;i++){
            gamma=layer->gamma[i];
            beta=layer->beta[i];
            for(int j=0;j<sample_num;j++){
                layer->out->data[j*sample_size+i] = gamma * layer->x_hat[j*sample_size+i] + beta;
    //            printf("%f ",layer->out->data[j*sample_size+i]);
            }
        }
    }else{
        for(int i=0;i<sample_size;i++){
            gamma=layer->gamma[i];
            beta=layer->beta[i];
            mean=layer->running_mean[i];
            sqrt_var=sqrt(layer->running_var[i]+eps);
            for(int j=0;j<sample_num;j++){
                layer->out->data[j*sample_size+i] = gamma * (layer->x->data[j*sample_size+i] - mean) /sqrt_var + beta;
            }
        }
    }
    layer->out->shape[0]=sample_num;
//    //
//    for(int i=0;i<sample_size;i++){
//        for(int j=0;j<sample_num;j++){
//            printf("%f ",layer->out->data[j*sample_size+i]);
//        }
//    }
////    //
}

void BN_layer_backward_pass(struct layer *l){
    struct BN_layer *layer=(struct BN_layer *)l;
    int sample_num=layer->x->shape[0];
    int sample_size=layer->sample_size;
//dbeta
    double col_sum=0;
    for(int i=0;i<sample_size;i++,col_sum=0){
        for(int j=0;j<sample_num;j++){
            col_sum += layer->dout->data[j*sample_size+i];
        }
        layer->dbeta[i]=col_sum;
    }
//dgamma
    for(int i=0;i<sample_size;i++,col_sum=0){
        for(int j=0;j<sample_num;j++){
            col_sum += layer->x_hat[j*sample_size+i] * layer->dout->data[j*sample_size+i];
        }
        layer->dgamma[i]=col_sum;
    }
//dx
    double col_coef;
    double dgamma;
    double dbeta;
    for(int i=0;i<sample_size;i++){
        col_coef = layer->gamma[i] * layer->istd[i] / sample_num;
        dgamma=layer->dgamma[i];
        dbeta=layer->dbeta[i];
        for(int j=0;j<sample_num;j++){
            layer->dx->data[j*sample_size+i] = col_coef * (sample_num * layer->dout->data[j*sample_size+i] - layer->x_hat[j*sample_size+i] * dgamma - dbeta);
        }
    }
}

void BN_layer_update(struct CNN *cnn,struct layer *l){
    struct BN_layer *layer=(struct BN_layer *)l;
    adam(cnn,layer->gamma,layer->dgamma,layer->gammam,layer->gammav,layer->sample_size);
    adam(cnn,layer->beta,layer->dbeta,layer->betam,layer->betav,layer->sample_size);
}

void BN_layer_load_weight(struct layer *l,double *p){
    struct BN_layer *layer=(struct BN_layer *)l;
    int num=layer->sample_size;
    memcpy(layer->gamma,p,num*sizeof(double));
    memcpy(layer->beta,p+num,num*sizeof(double));
}

void BN_layer_pack_dweight(struct layer *l,double *p){
    struct BN_layer *layer=(struct BN_layer *)l;
    int num=layer->sample_size;
    memcpy(p,layer->dgamma,num*sizeof(double));
    memcpy(p+num,layer->dbeta,num*sizeof(double));
}
