#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "read_npy.h"
#include "string.h"
#include <time.h>

void shuffle(struct data_box *data_box,struct data_box *label_box);
int main(void){
    //读取数据
    struct data_box *train_data=npy_load("X_train.npy");
    struct data_box *train_label=npy_load("y_train.npy");
    struct data_box *test_data=npy_load("X_test.npy");
    struct data_box *test_label=npy_load("y_test.npy");
    //输出数据的形状
    printf("X_train:%d,%d,%d,%d\n",train_data->shape[0],train_data->shape[1],train_data->shape[2],train_data->shape[3]);
    printf("y_train:%d,%d\n",train_label->shape[0],train_label->shape[1]);
    printf("X_test:%d,%d,%d,%d\n",test_data->shape[0],test_data->shape[1],test_data->shape[2],test_data->shape[3]);
    printf("y_test:%d,%d\n",test_label->shape[0],test_label->shape[1]);
    //定义一个用于喂数据的结构体
    struct feed_data feed_box;
    //打乱输入的数据
    shuffle(train_data,train_label);
    shuffle(test_data,test_label);
    //定义迭代的次数和每一个batch包含的数据个数
    int nb_epoch=200;
    int batch_size=1000;
    //定义一个用于存放条件的结构体
    struct data_box con;
    int sample_size=1,classes;
    con.ndims=train_data->ndims;
    con.shape=(int*)malloc(con.ndims*sizeof(int));
    con.shape[0]=batch_size;
    for(int i=1;i<con.ndims;i++){
        con.shape[i]=train_data->shape[i];
        sample_size*=con.shape[i];
    }
    classes=train_label->shape[1];
    //使用上面定义的结构体初始化神经网络
    struct CNN *cnn=cnn_init(&con);

    int loop_time,the_last_time;
    double loss,acc;
    double *result;
    for(int i=0;i<nb_epoch;i++){
        //train
        loop_time=train_data->shape[0]/batch_size;
        the_last_time=train_data->shape[0]%batch_size;
        if(the_last_time>0) loop_time++;
        loss=acc=0;
        for(int j=0;j<loop_time;j++){
            feed_box.data=train_data->data+j*batch_size*sample_size;
            feed_box.label=train_label->data+j*batch_size*classes;
            if(j==loop_time-1&&the_last_time!=0){
                feed_box.sample_num=the_last_time;
            }else{
                feed_box.sample_num=batch_size;
            }
            //喂数据
            feed(cnn,&feed_box);
            //运行神经网络
            result=go(cnn,TRAIN);
            //计算平均loss和acc并输出
            loss=result[0];
            acc=result[1];
            printf("train : epoch:%d,loop:%d,loss:%f,acc:%f\n",i,j,loss,acc);

            free(result);
        }
        //test
        loop_time=test_data->shape[0]/batch_size;
        the_last_time=test_data->shape[0]%batch_size;
        if(the_last_time>0) loop_time++;
        loss=acc=0;
        for(int j=0;j<loop_time;j++){
            feed_box.data=test_data->data+j*sample_size;
            feed_box.label=test_label->data+j*classes;
            if(j==loop_time-1&&the_last_time!=0){
                feed_box.sample_num=the_last_time;
            }else{
                feed_box.sample_num=batch_size;
            }
            feed(cnn,&feed_box);
            result=go(cnn,TEST);
//            loss=result[0];
//            acc=result[1];
            loss=loss*j/(j+1)+result[0]/(j+1);
            acc=acc*j/(j+1)+result[1]/(j+1);
            free(result);
            printf("test : loss:%f,acc:%f\n",loss,acc);
        }
    }
    return 0;
}

//洗牌函数
void shuffle(struct data_box *data_box,struct data_box *label_box){
    int sample_num=data_box->shape[0];
    int sample_size=1;
    int class_num=label_box->shape[1];
    double *temp;
    int r;//store random number
    for(int i=1;i<data_box->ndims;i++){
        sample_size*=data_box->shape[i];
    }
    srand(time(NULL));
    temp=(double *)malloc(sample_size*sizeof(double));
    for(int i=0;i<sample_num;i++){
        r=rand()%sample_num;
        //交换data
        memcpy(temp,data_box->data+i*sample_size,sample_size*sizeof(double));
        memcpy(data_box->data+i*sample_size,data_box->data+r*sample_size,sample_size*sizeof(double));
        memcpy(data_box->data+r*sample_size,temp,sample_size*sizeof(double));
        //交换label
        memcpy(temp,label_box->data+i*class_num,class_num*sizeof(double));
        memcpy(label_box->data+i*class_num,label_box->data+r*class_num,class_num*sizeof(double));
        memcpy(label_box->data+r*class_num,temp,class_num*sizeof(double));
    }
    free(temp);
}
