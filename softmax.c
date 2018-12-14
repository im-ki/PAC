#include"softmax.h"
#include<math.h>
#include<stdlib.h>
#include<string.h>
struct softmax *softmax_init(struct data_box *common_p,struct data_box *dcommon_p){
    struct softmax *softmax_obj=(struct softmax *)malloc(sizeof(struct softmax));

    softmax_obj->in=common_p;
    softmax_obj->out=dcommon_p;
    softmax_obj->forward_pass=softmax_forward_pass;
    return softmax_obj;
}

double *softmax_forward_pass(struct softmax *softmax_obj,int state){
    double *in,*out;
    int sample_num=softmax_obj->in->shape[0];
    int sample_size=softmax_obj->in->shape[1];
    //printf("sample_num=%d sample_size=%d\n",sample_num,sample_size);
    double loss=0;
    double acc=0;
    double *result_box;
    in=softmax_obj->in->data;
    out=softmax_obj->out->data;

    for(int i=0;i<sample_num;i++){
        max_exp_div(in+i*sample_size,sample_size);
    }

    memcpy(out,in,sample_num*sample_size*sizeof(double));
    //算loss
    for(int i=0;i<sample_num;i++){
        for(int j=0;j<sample_size;j++){
            if(softmax_obj->label[i*sample_size+j]==1){
                loss-=log(in[i*sample_size+j]);
                break;
            }
        }
    }
    loss/=sample_num;
    //算梯度
    if(state==TRAIN){
        for(int i=0;i<sample_num;i++){
            for(int j=0;j<sample_size;j++){
                if(softmax_obj->label[i*sample_size+j]==1){
                    out[i*sample_size+j]-=1;
                    break;
                }
            }
        }
    }
    //算acc
    int max_index;
    for(int i=0;i<sample_num;i++){
        for(int j=0;j<sample_size;j++){
            if(softmax_obj->label[i*sample_size+j]==1){
                max_index=0;
                for(int k=1;k<sample_size;k++){
                    if(in[i*sample_size+k]>in[i*sample_size+max_index]){
                        max_index=k;
                    }
                }
                //printf("max index=%d,j=%d\n",max_index,j);
                if(max_index==j) acc+=1;
                break;
            }
        }
    }
    acc/=sample_num;
    result_box=(double *)malloc(2*sizeof(double));
    result_box[0]=loss;
    result_box[1]=acc;
    return result_box;//记得释放内存
}

void max_exp_div(double *box,int num){
    int max_index=0;
    double max_num;
    double sum=0;
//minus the max number and exp and sum
    for(int i=1;i<num;i++){
        if(box[i]>box[max_index]) max_index=i;
    }
    max_num=box[max_index];
    for(int i=0;i<num;i++){
        box[i]=exp(box[i]-max_num);
        sum+=box[i];
    }
//divide
    for(int i=0;i<num;i++){
        box[i]/=sum;
    }
}
