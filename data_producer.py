import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from scipy.io import wavfile

def produce_data_set(window=6144,interval=6144,samplep=128,produce_file=1,del_old=1,data=None):
##    window=6144
##    interval=6144
##    #fft sample size
##    samplep=128

###################delete old file##########################
    if del_old==1:
        import os
        files=os.listdir('./')
        for i in ['X_train.pkl','X_test.pkl','y_train.pkl','y_test.pkl']:
            if i in files:
                os.remove(i)
    if data==None:
        with open('one_hot_label.pkl','rb') as man_file:
            label=pickle.load(man_file)
        with open('sound_without_unlabelled.pkl','rb') as man_file:
            sound=pickle.load(man_file)
        data=(sound,label)
    else:
        sound = data[0]
        label = data[1]

##    ################change label#############################
##    one_hot_label=[]
##    sound_without_unlabelled=[]
##    for i in range(len(label)):
##        if label[i]=='extrastole' or label[i]=='extrahls':
##            a=[1,0,0]
##            one_hot_label.append(a)
##            sound_without_unlabelled.append(sound[i])
##        elif label[i]=='artifact':
##            a=[0,0,1]
##            one_hot_label.append(a)
##            sound_without_unlabelled.append(sound[i])
##        elif label[i]=='normal':
##            a=[1,0,0]
##            one_hot_label.append(a)
##            sound_without_unlabelled.append(sound[i])
##        elif label[i]=='murmur':
##            a=[0,1,0]
##            one_hot_label.append(a)
##            sound_without_unlabelled.append(sound[i])
##        else:
##            pass
##    ##################split train test set######################
##    sound=sound_without_unlabelled
##    label=one_hot_label
    new_label=[]
    new_sound=[]

    for i in range(len(label)-1,-1,-1):
        if len(sound[i])>window:
            new_sound.append(sound[i])
            new_label.append(label[i])

    sound=new_sound
    label=new_label

    X_train, X_test, y_train, y_test = train_test_split(sound, label, test_size=0.30, random_state=42)
    test_sound_X = X_test
    test_sound_y = y_test
    ####################cut sound sample into pieces################


    X_train, y_train = sample_sound(X_train, y_train)
    X_test, y_test = sample_sound(X_test, y_test)

    ##with open('X_train_cut.pkl','wb') as man_file:  
    ##    pickle.dump(X_train,man_file)
    ##    
    ##with open('X_test_cut.pkl','wb') as man_file:  
    ##    pickle.dump(X_test,man_file)
    ##    
    ##with open('y_train_cut.pkl','wb') as man_file:  
    ##    pickle.dump(y_train,man_file)
    ##    
    ##with open('y_test_cut.pkl','wb') as man_file:  
    ##    pickle.dump(y_test,man_file)

    #####################convert to graph#######################
    X_train_result=to_graph(X_train)
    X_test_result=to_graph(X_test)
    
    ##plt.imshow(result)
    ##plt.show()

#    print len(X_train)
#    print len(y_train)
#    print len(X_test)
#    print len(y_test)

##    if produce_file==1:
##        with open('X_train.pkl','wb') as man_file:  
##            pickle.dump(X_train_result,man_file)
##            
##        with open('X_test.pkl','wb') as man_file:  
##            pickle.dump(X_test_result,man_file)
##            
##        with open('y_train.pkl','wb') as man_file:  
##            pickle.dump(y_train,man_file)
##            
##        with open('y_test.pkl','wb') as man_file:  
##            pickle.dump(y_test,man_file)

##    if __name__!='__main__':
    print len(test_sound_X)
    print len(test_sound_y)
    print X_train_result.dtype
    return X_train_result,X_test_result,y_train,y_test,data,test_sound_X,test_sound_y

def to_graph(X,window=5632,samplep=128):
    x=samplep/2
    y=window/samplep*2-1
    ham=np.hamming(samplep)
    X_=np.zeros((len(X),x,y,1))
    for i in range(len(X)):
        a=X[i]
        a=np.array(a)
        result=np.zeros((y,x))
        for j in range(y):
            b=a[j*samplep/2:j*samplep/2+samplep]
            b=b*ham
            b=np.fft.fft(b)
            b=b[:x]
            result[j]=np.abs(b)
        result=result.transpose()
        result=result.reshape((x,y,1))
        var=np.sqrt(np.var(result))
        mean=np.mean(result)
        X_[i]=(result-mean)/var
    X_ = X_[:,:10,:,:]
    return X_
        
def sample_sound(X,y,window=5632,interval=640):
    class0=640
    class1=265
    class2=90
    new_sound=[]
    new_label=[]
    
    y_=[i.index(max(i)) for i in y]
    for i in range(len(X)):
        if y_[i]==0:
            interval=class0
        elif y_[i]==1:
            interval=class1
        else:
            interval=class2
        j=int(len(X[i])>window)*(1+(len(X[i])-window)/interval)
        for k in range(j):
            new_sound.append(X[i][k*interval:k*interval+window])
            new_label.append(y[i])
    return new_sound,new_label

def test_sound(model,test_sound_X,test_sound_y):
    right=0.0
    classify_result = np.zeros((3,3))
    for i in range(len(test_sound_X)):
        X_test = [test_sound_X[i],]
        y_test = [test_sound_y[i],]
        X,_ = sample_sound(X_test,y_test)
        X = to_graph(X)
        result=model.predict(X,verbose=0)
        result=result.sum(axis=0)
        result=result.argmax()
        y = test_sound_y[i]
        y = y.index(max(y))
        classify_result[y,result] += 1
        if result==y:
            print i,'/',len(test_sound_X)," right"
            right+=1
        else:
            print i,'/',len(test_sound_X)," wrong"
            wavfile.write(str(i)+'-predict-'+str(result)+'-label-'+str(y)+'.wav',4000,np.array(test_sound_X[i]))
    print "acc=",right/len(test_sound_X) 
    #calculate precision of each class
    print "precision of normal:",classify_result[0,0]/classify_result.sum(axis = 1)[0]
    print "precision of heart problem:",classify_result[1,1]/classify_result.sum(axis = 1)[1]
    print "precision of artifact:",classify_result[2,2]/classify_result.sum(axis = 1)[2]
    #calculate youden index of artifact
    tp = classify_result[2,2]
    fn = classify_result.sum(axis = 1)[2] - tp
    fp = classify_result.sum(axis = 0)[2] - tp
    tn = np.sum(classify_result) - tp - fp - fn
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    sensitivity = tp/(tp+fn)#ill
    specificity = tn/(fp+tn)
    youden = sensitivity + specificity - 1
    fscore = 2*p*r/(p+r)
    print "artifact sensitivity:",sensitivity
    print "artifact specificity:",specificity
    print "Youden Index of artifact:",youden
    print "F1-score of artifact:", fscore
    #calculate youden index of heart problem
    tp = classify_result[1,1]
    fp = classify_result.sum(axis = 0)[1] - tp
    fn = classify_result.sum(axis = 1)[1] - tp
    tn = np.sum(classify_result) - tp - fp - fn
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    youden = sensitivity + specificity - 1
    print "heart problem sensitivity:",sensitivity
    print "heart problem specificity:",specificity
    print "Youden Index of heart problem:",youden
    #calculate youden index of normal
    tp = classify_result[0,0]
    fp = classify_result.sum(axis = 1)[0] - tp
    fn = classify_result.sum(axis = 0)[0] - tp
    tn = np.sum(classify_result) - tp - fp - fn
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    youden = sensitivity + specificity - 1
    print "normal sensitivity:",sensitivity
    print "normal specificity:",specificity
    print "Youden Index of normal:",youden
    print classify_result                                    

if __name__=='__main__':
    produce_data_set(window=5632,interval=640,samplep=128,produce_file=1,del_old=1)

