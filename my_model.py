import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.convolutional import Conv2D, MaxPooling2D  
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import gc
import multiprocessing
import time
import sys
import StringIO
from keras import regularizers
import pickle
import pandas as pd

# There are 40 different classes  
nb_classes = 3
nb_epoch = 20
batch_size = 200
    
def Net_model(config):
    img_rows=config['img_rows']
    img_cols=config['img_cols']
    lr=config['lr']
    decay=config['decay']
    momentum=config['momentum']
    
    nb_filters1=config['nb_filters1']
    nb_conv1_rows=config['nb_conv1_rows']
    nb_conv1_cols=config['nb_conv1_cols']
    stride1_rows=config['stride1_rows']
    stride1_cols=config['stride1_cols']
    nb_pool1_rows=config['nb_pool1_rows']
    nb_pool1_cols=config['nb_pool1_cols']

    nb_filters2=config['nb_filters2']
    nb_conv2_rows=config['nb_conv2_rows']
    nb_conv2_cols=config['nb_conv2_cols']
    stride2_rows=config['stride2_rows']
    stride2_cols=config['stride2_cols']
    nb_pool2_rows=config['nb_pool2_rows']
    nb_pool2_cols=config['nb_pool2_cols']

    nb_filters3=config['nb_filters3']
    nb_conv3_rows=config['nb_conv3_rows']
    nb_conv3_cols=config['nb_conv3_cols']
    stride3_rows=config['stride3_rows']
    stride3_cols=config['stride3_cols']
    nb_pool3_rows=config['nb_pool3_rows']
    nb_pool3_cols=config['nb_pool3_cols']

#    nb_filters4=config['nb_filters4']
#    nb_conv4_rows=config['nb_conv4_rows']
#    nb_conv4_cols=config['nb_conv4_cols']
#    stride4_rows=config['stride4_rows']
#    stride4_cols=config['stride4_cols']
#    nb_pool4_rows=config['nb_pool4_rows']
#    nb_pool4_cols=config['nb_pool4_cols']

    reg1=config['reg1']
    reg2=config['reg2']
    reg3=config['reg3']
    reg4=config['reg4']
    reg5=config['reg5']

    fc1 = config['fc1']
    fc2 = config['fc2']
    
    model = Sequential()  
    model.add(Conv2D(nb_filters1, (nb_conv1_rows, nb_conv1_cols), strides=(stride1_rows, stride1_cols), 
                            border_mode='valid',input_shape=(img_rows, img_cols, 1),W_regularizer=regularizers.l2(reg1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(nb_pool1_rows, nb_pool1_cols)))  

    model.add(Conv2D(nb_filters2, (nb_conv2_rows, nb_conv2_cols),strides=(stride2_rows, stride2_cols),W_regularizer=regularizers.l2(reg2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(nb_pool2_rows, nb_pool2_cols)))
    
    model.add(Conv2D(nb_filters3, (nb_conv3_rows, nb_conv3_cols),strides=(stride3_rows, stride3_cols),W_regularizer=regularizers.l2(reg3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(nb_pool3_rows, nb_pool3_cols)))

#    model.add(Conv2D(nb_filters4, (nb_conv4_rows, nb_conv4_cols),strides=(stride4_rows, stride4_cols)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))  
#    model.add(MaxPooling2D(pool_size=(nb_pool4_rows, nb_pool4_cols)))

#   model.add(Conv2D(nb_filters4, (1, nb_conv4),strides=(1, stride4)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))  
#    model.add(MaxPooling2D(pool_size=(1, nb_pool4)))

##    model.add(Conv2D(nb_filters5, (1, nb_conv5),strides=(1, stride5)))
##    model.add(BatchNormalization())
##    model.add(Activation('relu'))  
##    model.add(MaxPooling2D(pool_size=(1, nb_pool5)))
    
    model.add(Flatten())
    model.add(Dense(fc1,kernel_regularizer=regularizers.l2(reg4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(fc2,kernel_regularizer=regularizers.l2(reg5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))  
   # model.add(Activation('relu'))
    model.add(Activation('softmax'))  
  
    adam = Adam(lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  
      
    return model  
  
def train_model(model,X_train,Y_train,X_test,Y_test,nb_epoch):  
    hist=model.fit(X_train, Y_train, shuffle = True, batch_size=batch_size, nb_epoch=nb_epoch,  
          verbose=1, validation_data=(X_test, Y_test))
#    model.save_weights('model_weights.h5',overwrite=True)
    hist=hist.history
    max_arg=hist['val_acc'].index(max(hist['val_acc']))
    score=[]
    score.append(hist['val_loss'][max_arg])
    score.append(hist['val_acc'][max_arg])
    return score

def record(config,score):
    config_list=config.keys()
    mark=''
    for i in config_list:
        if i!='img_cols' and i!='img_rows':
            if i[0:3]=='nb_':
                mark+=i[3:]+'='+str(config[i])+','
            else:
                mark+=i+'='+str(config[i])+','
    mark=mark+"score="+str(score[0])+",acc="+str(score[1])
    #print mark
    with open('result', 'a') as result:
        result.write(mark+'\n')

def train_and_record(config,X_train,y_train,X_test,y_test):
    model=Net_model(config)
    
    orig_stdout = sys.stdout
    output = StringIO.StringIO()
    sys.stdout = output
    model.summary()
    sys.stdout = orig_stdout
    summary = output.getvalue()
    index=summary.find('Total params: ')+len('Total params: ')
    new_seq=summary[index:]
    print '\n'
    print new_seq
    index=new_seq.find('.')
    new_seq=new_seq[:index]
    new_seq=new_seq.split(',')
    a=''
    for i in new_seq:
        a+=i
    new_seq=int(a)
    print new_seq
    print '\n'
    #if new_seq>20000000:
    #    return 0

    #test_process=multiprocessing.Process(target=train_model,args=(model,X_train,y_train,X_test,y_test,1))
    #test_process.start()
    #test_process.join(60)
    #if test_process.is_alive():
    #    test_process.terminate()
    #    return 0
    
    score=train_model(model,X_train,y_train,X_test,y_test,nb_epoch)
    record(config,score)
#    weights = model.get_weights()
#    with open('weights.pkl','wb') as f:
#        pickle.dump(weights,f)

    classes=model.predict_classes(X_test,verbose=0)
    y_test_classes = np.zeros(len(y_test))
    for i in range(len(y_test)):
        y_test_classes[i] = y_test[i].index(max(y_test[i]))
    result=np.zeros((len(y_test),2))
    for i in range(len(y_test)):
        result[i,0]=classes[i]
        result[i,1]=y_test_classes[i]
    result=pd.DataFrame(result)
    result.to_csv("test_result.csv", index=False)
    model.save('my_model.h5')

def max_divide(a,b):
    return a/b+int(bool(a%b)) #a/b+1,ceiling

#def data_mix(X_train,y_train):
#    y_=[i.index(max(i)) for i in y_train]
#    y_=np.array(y_)
#    
#    first_class=X_train[y_==0]
#    second_class=X_train[y_==1]
#    third_class=X_train[y_==2]
#
#    for j in range(30):#range(third_class.shape[0]):
#        to_add=third_class[j]
#        to_add=to_add.reshape((1,to_add.shape[0],to_add.shape[1],to_add.shape[2]))
#        fisrt_new=first_class+to_add
#        second_new=second_class+to_add
#        third_new=third_class+to_add
#        X_train=np.vstack((X_train,fisrt_new,second_new,third_new))
#        y_train = y_train + first_class.shape[0]*[[1,0,0],] + second_class.shape[0]*[[0,1,0],] + third_class.shape[0]*[[0,0,1],]
#
#    return X_train,y_train

def data_mix(X_train,y_train):
    y_=[i.index(max(i)) for i in y_train]
    y_=np.array(y_)
    
    first_class=X_train[y_==0]
    second_class=X_train[y_==1]
    third_class=X_train[y_==2]
    print first_class.shape[0],second_class.shape[0],third_class.shape[0]

#    for j in range(30):#range(third_class.shape[0]):
#        to_add=third_class[j]
#        to_add=to_add.reshape((1,to_add.shape[0],to_add.shape[1],to_add.shape[2]))
#        fisrt_new=first_class+to_add
#        second_new=second_class+to_add
#        third_new=third_class+to_add
#        X_train=np.vstack((X_train,fisrt_new,second_new,third_new))
#        y_train = y_train + first_class.shape[0]*[[1,0,0],] + second_class.shape[0]*[[0,1,0],] + third_class.shape[0]*[[0,0,1],]

    return X_train,y_train


def train_it(X_train=None, X_test=None, y_train=None, y_test=None):
    # the data, shuffled and split between tran and test sets
#    X_train=X_train[:,:32]
#    X_test=X_test[:,:32]
    X_train, y_train = data_mix(X_train, y_train)

    config={}
    config['img_rows']=X_train.shape[1]
    config['img_cols']=X_train.shape[2]
    config['momentum']=0.9
    config['decay']=1e-6
    config['lr']=1e-4
    config['nb_filters1']=nb_filters1=32
    config['nb_filters2']=nb_filters2=64
    config['nb_filters3']=nb_filters3=64
#    config['nb_filters4']=nb_filters4=512

    layer0_rows = X_train.shape[1]
    layer0_cols = X_train.shape[2]
    layer0 = (layer0_rows, layer0_cols)
##    for nb_conv1_rows in [X_train.shape[1],]:
    config['nb_conv1_rows']=3
    config['nb_conv1_cols']=4
    config['stride1_rows']=1
    config['stride1_cols']=1
    config['nb_pool1_rows']=1
    config['nb_pool1_cols']=2

    config['nb_conv2_rows']=3
    config['nb_conv2_cols']=5
    config['stride2_rows']=1
    config['stride2_cols']=1
    config['nb_pool2_rows']=1
    config['nb_pool2_cols']=2
    
    config['nb_conv3_rows']=3
    config['nb_conv3_cols']=6
    config['stride3_rows']=1
    config['stride3_cols']=1
    config['nb_pool3_rows']=1
    config['nb_pool3_cols']=2

#    config['nb_conv4_rows']=1
#    config['nb_conv4_cols']=2
#    config['stride4_rows']=1
#    config['stride4_cols']=1
#    config['nb_pool4_rows']=1
#    config['nb_pool4_cols']=3
    
    config['reg1']=0 
    config['reg2']=0
    config['reg3']=0
    config['reg4']=0
    config['reg5']=0

    config['fc1'] = 1024
    config['fc2'] = 512

#    for i in [j/100.0 for j in range(1,11)]:
#        config['reg1'] = i
    try:
        sub_process=multiprocessing.Process(target=train_and_record,args=(config,X_train,y_train,X_test,y_test))
        sub_process.start()
        sub_process.join()
    except:
        pass
    gc.collect()




