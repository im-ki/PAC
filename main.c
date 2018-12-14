#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "read_npy.h"
#include "string.h"
#include <time.h>
#include "mpi.h"
int main(void){
    //读取数据
    int id,nb_procs;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&nb_procs);

    struct data_box *train_data=npy_load("X_train.npy",id,nb_procs);
    struct data_box *train_label=npy_load("y_train.npy",id,nb_procs);
    struct data_box *test_data=npy_load("X_test.npy",id,nb_procs);
    struct data_box *test_label=npy_load("y_test.npy",id,nb_procs);
   //输出数据的形状
//    printf("X_train:%d,%d,%d,%d\n",train_data->shape[0],train_data->shape[1],train_data->shape[2],train_data->shape[3]);
//    printf("y_train:%d,%d\n",train_label->shape[0],train_label->shape[1]);
//    printf("X_test:%d,%d,%d,%d\n",test_data->shape[0],test_data->shape[1],test_data->shape[2],test_data->shape[3]);
//    printf("y_test:%d,%d\n",test_label->shape[0],test_label->shape[1]);
    //定义一个用于喂数据的结构体
    struct feed_data feed_box;
    //定义迭代的次数和每一个batch包含的数据个数
    int nb_epoch=50;
    int batch_size=200;
    batch_size/=nb_procs;
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
    int weight_size=cnn->weight_size;
    double *w,*dw,*m,*v,*buf,*loop_result;
    w=(double *)malloc(weight_size*sizeof(double));
    buf=(double *)calloc(weight_size,sizeof(double));
    if(id==0){
        dw=(double *)malloc(weight_size*sizeof(double));
        m=(double *)calloc(weight_size,sizeof(double));
        v=(double *)calloc(weight_size,sizeof(double));
        loop_result=(double *)malloc(2*sizeof(double));
    }

    int loop_time,the_last_time;
    double loss,acc;
    double *result;

    if(id==0){
        pack_weight(cnn,w);
    }
    MPI_Bcast(w,weight_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
    load_weight(cnn,w);

    for(int i=0;i<nb_epoch;i++){
        //train
        loop_time=train_data->shape[0]/batch_size;
        the_last_time=train_data->shape[0]%batch_size;
        if(the_last_time>0) loop_time++;
        loss=acc=0;

		double begin,end;
		if(id==0){
			begin=MPI_Wtime();
		}
        for(int j=0;j<loop_time;j++){
//            printf("%d %d %d %d\n",i,nb_epoch,j,loop_time);
            memset(buf,0,weight_size*sizeof(double));

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
            MPI_Reduce(result,loop_result,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

            if(id==0){
                printf("train : epoch:%d,loop:%d,loss:%f,acc:%f\n",i,j,loop_result[0]/nb_procs,loop_result[1]/nb_procs);
            }
            free(result);

            pack_dweight(cnn,buf);
				//begin=MPI_Wtime();
            MPI_Reduce(buf,dw,weight_size,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
				//end=MPI_Wtime();
            if(id==0){
                adam(cnn,w,dw,m,v,weight_size);
                cnn->adam_para.t++;
				//printf("%fs\n",end-begin);
            }
            MPI_Bcast(w,weight_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
            load_weight(cnn,w);
			if(j==9){
				if(id==0){
					end=MPI_Wtime();
					printf("10 loop:%fs\n",end-begin);
				}
			}
        }
//        MPI_Reduce(buf,dw,weight_size,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
//        if(id==0){
//            adam(cnn,w,dw,m,v,weight_size);
//        }
//        MPI_Bcast(w,weight_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
//        load_weight(cnn,w);


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
    MPI_Finalize();
    return 0;
}
//
////洗牌函数
//void shuffle(struct data_box *data_box,struct data_box *label_box){
//    int sample_num=data_box->shape[0];
//    int sample_size=1;
//    int class_num=label_box->shape[1];
//    double *temp;
//    int r;//store random number
//    for(int i=1;i<data_box->ndims;i++){
//        sample_size*=data_box->shape[i];
//    }
//    srand(time(NULL));
//    temp=(double *)malloc(sample_size*sizeof(double));
//    for(int i=0;i<sample_num;i++){
//        r=rand()%sample_num;
//        //交换data
//        memcpy(temp,data_box->data+i*sample_size,sample_size*sizeof(double));
//        memcpy(data_box->data+i*sample_size,data_box->data+r*sample_size,sample_size*sizeof(double));
//        memcpy(data_box->data+r*sample_size,temp,sample_size*sizeof(double));
//        //交换label
//        memcpy(temp,label_box->data+i*class_num,class_num*sizeof(double));
//        memcpy(label_box->data+i*class_num,label_box->data+r*class_num,class_num*sizeof(double));
//        memcpy(label_box->data+r*class_num,temp,class_num*sizeof(double));
//    }
//    free(temp);
//}
