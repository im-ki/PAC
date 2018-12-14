#include<stdlib.h>
#include"cnn.h"
#include"relu_layer.h"
void relu_layer_init(struct layer **init_layer,struct data_box **com_p,struct data_box **dcom_p){
    struct relu_layer *new_layer=(struct relu_layer *)malloc(sizeof(struct relu_layer));

    struct data_box *common_p=*com_p;
    struct data_box *dcommon_p=*dcom_p;
//神经网络的属性初始化
    new_layer->forward_pass=relu_layer_forward_pass;
    new_layer->backward_pass=relu_layer_backward_pass;
    new_layer->update=relu_update;
    new_layer->load_weight=relu_layer_load_weight;
    new_layer->pack_dweight=relu_layer_pack_dweight;

    new_layer->weight_size=0;

//神经网络的参数初始化
    new_layer->x=common_p;
    new_layer->dx=dcommon_p;
    new_layer->out=common_p;
    new_layer->dout=dcommon_p;

    *init_layer=(struct layer*)new_layer;
}

void relu_layer_forward_pass(struct layer *l,int state){
    struct relu_layer *layer=(struct relu_layer *)l;
    int *shape=layer->x->shape;
    int ndims=layer->x->ndims;
    int size=1;
    for(int i=0;i<ndims;i++){
        size*=shape[i];
    }
    for(int i=0;i<size;i++){
        if(layer->x->data[i]<0) layer->x->data[i]=0;
    }
}

void relu_layer_backward_pass(struct layer *l){
    struct relu_layer *layer=(struct relu_layer *)l;
    int *shape=layer->x->shape;
    int ndims=layer->x->ndims;
    int size=1;
    for(int i=0;i<ndims;i++){
        size*=shape[i];
    }
    for(int i=0;i<size;i++){
        if(layer->x->data[i]==0) layer->dx->data[i]=0;
    }
}

void relu_update(struct CNN *cnn,struct layer *l){
    return;
}

void relu_layer_load_weight(struct layer *l,double *p){
    return;
}

void relu_layer_pack_dweight(struct layer *l,double *p){
    return;
}
