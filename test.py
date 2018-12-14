import data_producer
import my_model

data=None

def test(window,interval,samplep,data):
    X_train,X_test,y_train,y_test,data,sound_test_X,sound_test_y = data_producer.produce_data_set(window=window,
        interval=interval,samplep=samplep,produce_file=0,del_old=0,data=data)
    del data
    from keras.models import load_model
    model = load_model("my_model.h5")
    data_producer.test_sound(model, sound_test_X, sound_test_y)

if __name__=='__main__':
    test(5632,640,128,data)
