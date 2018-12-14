#include"dense_layer.h"
#include"cnn.h"
#include<stdlib.h>
#include<time.h>
#include<string.h>
void dense_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p,int neuron_num){
    struct dense_layer *new_layer=(struct dense_layer *)malloc(sizeof(struct dense_layer));
//神经网络的属性初始化
    struct data_box *common_p=*com_p;
    struct data_box *dcommon_p=*dcom_p;

    new_layer->forward_pass=dense_layer_forward_pass;
    new_layer->backward_pass=dense_layer_backward_pass;
    new_layer->update=dense_layer_update;
    new_layer->load_weight=dense_layer_load_weight;
    new_layer->pack_dweight=dense_layer_pack_dweight;

    new_layer->neuron_num=neuron_num;

    new_layer->out=(struct data_box *)malloc(sizeof(struct data_box));
    new_layer->dout=(struct data_box *)malloc(sizeof(struct data_box));

//神经网络的参数初始化
    int num=1;
    for(int i=1;i<common_p->ndims;i++){
        num*=common_p->shape[i];
    }
    new_layer->sample_size=num;
    new_layer->w=(double *)malloc(num*neuron_num*sizeof(double));
    srand(time(NULL));
    for(int i=0;i<num*neuron_num;i++){
        new_layer->w[i]=rand()/(double)(RAND_MAX)*2-1;
    }
    new_layer->b=(double *)calloc(neuron_num,sizeof(double));
    new_layer->dw=(double *)malloc(num*neuron_num*sizeof(double));
    new_layer->db=(double *)malloc(neuron_num*sizeof(double));

    new_layer->weight_size=num*neuron_num+neuron_num;
    //优化器参数
    new_layer->wm=(double *)calloc(num*neuron_num,sizeof(double));
    new_layer->wv=(double *)calloc(num*neuron_num,sizeof(double));
    new_layer->bm=(double *)calloc(neuron_num,sizeof(double));
    new_layer->bv=(double *)calloc(neuron_num,sizeof(double));
//用于训练的内存空间初始化
    new_layer->out->data=(double *)malloc(common_p->shape[0]*neuron_num*sizeof(double));
    new_layer->out->shape=(int *)malloc(2*sizeof(int));
    new_layer->out->shape[0]=common_p->shape[0];
    new_layer->out->shape[1]=neuron_num;
    new_layer->out->ndims=2;
    new_layer->dout->data=(double *)malloc(common_p->shape[0]*neuron_num*sizeof(double));
    new_layer->dout->shape=(int *)malloc(2*sizeof(int));
    new_layer->dout->shape[0]=common_p->shape[0];
    new_layer->dout->shape[1]=neuron_num;
    new_layer->dout->ndims=2;
//用于上下连接的内存空间赋值
    new_layer->x=common_p;
    new_layer->dx=dcommon_p;
    *com_p=new_layer->out;
    *dcom_p=new_layer->dout;

    *init_layer=(struct layer*)new_layer;
}

void dense_layer_forward_pass(struct layer *l,int state){
    struct dense_layer *layer=(struct dense_layer *)l;
    int sample_num=layer->x->shape[0];
    int sample_size=layer->sample_size;
    int neuron_num=layer->neuron_num;
    double mul_sum=0;

    //x.dot(w)+b
    for(int i=0;i<sample_num;i++){
        for(int k=0;k<neuron_num;k++){
            for(int j=0;j<sample_size;j++){
                mul_sum+=layer->x->data[i*sample_size+j]*layer->w[j*neuron_num+k];
            }
            mul_sum+=layer->b[k];
            layer->out->data[i*neuron_num+k]=mul_sum;
            mul_sum=0;
        }
    }
    layer->out->shape[0]=sample_num;
    layer->out->shape[1]=neuron_num;
}

void dense_layer_backward_pass(struct layer *l){
    struct dense_layer *layer=(struct dense_layer *)l;
    int sample_num=layer->x->shape[0];
    int sample_size=layer->sample_size;
    int neuron_num=layer->neuron_num;
//dx=dout.dot(w.T)
    double mul_sum=0;
    for(int i=0;i<sample_num;i++){
        for(int j=0;j<sample_size;j++){
            for(int k=0;k<neuron_num;k++){
                mul_sum+=layer->dout->data[i*neuron_num+k]*layer->w[j*neuron_num+k];
            }
            layer->dx->data[i*sample_size+j]=mul_sum;
            mul_sum=0;
        }
    }
//dw=x.T.dot(dout)
    for(int j=0;j<sample_size;j++){
        for(int k=0;k<neuron_num;k++){
            for(int i=0;i<sample_num;i++){
                mul_sum+=layer->x->data[i*sample_size+j]*layer->dout->data[i*neuron_num+k];
            }
            layer->dw[j*neuron_num+k]=mul_sum;
            mul_sum=0;
        }
    }
//db=np.ones(dout.shape[0]).dot(dout)
    for(int m=0;m<neuron_num;m++){
        for(int n=0;n<sample_num;n++){
            mul_sum+=layer->dout->data[n*neuron_num+m];
        }
        layer->db[m]=mul_sum;
        mul_sum=0;
    }
}

void dense_layer_update(struct CNN *cnn,struct layer *l){
    struct dense_layer *layer=(struct dense_layer *)l;
    adam(cnn,layer->w,layer->dw,layer->wm,layer->wv,layer->neuron_num*layer->sample_size);
    adam(cnn,layer->b,layer->db,layer->bm,layer->bv,layer->neuron_num);
}

void dense_layer_load_weight(struct layer *l,double *p){
    struct dense_layer *layer=(struct dense_layer *)l;

    int num=layer->sample_size;
    int neuron_num=layer->neuron_num;

    memcpy(layer->w,p,num*neuron_num*sizeof(double));
    memcpy(layer->b,p+num*neuron_num,neuron_num*sizeof(double));
}

void dense_layer_pack_dweight(struct layer *l,double *p){
    struct dense_layer *layer=(struct dense_layer *)l;

    int num=layer->sample_size;
    int neuron_num=layer->neuron_num;

    memcpy(p,layer->dw,num*neuron_num*sizeof(double));
    memcpy(p+num*neuron_num,layer->db,neuron_num*sizeof(double));
}
