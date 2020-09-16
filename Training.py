# create by topkoka , mr.aditep campira  19-07-2020
# import warnings filter
# ปิด warnings
from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
filterwarnings('ignore')
filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import os
import requests
import io
import uuid
import sys
np.set_printoptions(threshold=sys.maxsize)

def readDataURL():

    url = 'https://raw.githubusercontent.com/topkoka/transform_101/master/datas.csv'
    r = requests.get(url)
    if r.ok:
        data = r.content.decode('utf8')
        df = pd.read_csv(io.StringIO(data))
        df.info()
        z = pd.DataFrame(df, columns=['produce']).astype('int')
        x = pd.DataFrame(df,
                         columns=['water', 'disaster', 'suitability', 'plant_maintenance',
                                  'plant_sale_price']).to_numpy()
    return x, z


def readDataCSV():
    # ที่อยู่ไฟล์ข้อมูล
    script_dir = os.path.dirname(__file__)
    READFILE = script_dir + "/datas.csv"
    # โหลด
    data = pd.read_csv(READFILE)
    data.info()
    # แบ่งข้อมูล
    z = pd.DataFrame(data, columns=['produce']).astype('int')
    x = pd.DataFrame(data,
                     columns=['water', 'disaster', 'suitability', 'plant_maintenance', 'plant_sale_price']).to_numpy()
    return x, z



# model Dl
def nn_model(X, kernelInitializer, biasinitializer, activations):
    NN_model = Sequential()
    NN_model.add(
        Dense(128, kernel_initializer=kernelInitializer, bias_initializer=biasinitializer, input_dim=X.shape[1],
              activation=activations))
    for hidden in range(X.shape[1]):
        NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    NN_model.compile(loss='mean_absolute_error', optimizer='adam',
                     metrics=['RootMeanSquaredError','mean_absolute_error'])
    NN_model.summary()
    return NN_model


def create_model(X):
    # ปรับ น้ำหนัก
    # kernelInitializers = ['RandomNormal', 'RandomUniform', 'TruncatedNormal',
    #                       'VarianceScaling', 'lecun_uniform', 'glorot_normal',
    #                       'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform']

    kernelInitializers = ['RandomNormal']
                          
    # ปรับ Bias
    biasinitializers = ['Zeros']
    # ปรับ activations เริ่มต้น
    activations = ['relu']

    # add model
    model_list = []
    for kernelInitializer in kernelInitializers:
        for biasinitializer in biasinitializers:
            for activation in activations:
                print(kernelInitializer, activation)
                model = nn_model(X, kernelInitializer, biasinitializer, activation)
                model_list.append([str(uuid.uuid4()), model, 'DL :( weights ' + kernelInitializer + ' : bias ' + biasinitializer + ' : activation ' + activation + ')'])

    # เตรียม dict เก็บข้อมูล Model
    accuracy: dict = {}
    model: dict = {}
    Namemodel: dict = {}
    for i in range(len(model_list)):
        accuracy[str(model_list[i][0])] = 'NULL'
        model[model_list[i][0]] = None
        Namemodel[model_list[i][0]] = model_list[i][2]
    return model_list, accuracy, model, Namemodel




def TrainingModel_tf(X_train,z_train,X_test,z_test,model):
    # กำหนด callback ถ้า loss ไม่ลดลง 15 ครั้งให้หยุด
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15),
    ]
    # สอน
    hist = model.fit(X_train, z_train.ravel(), epochs=30,initial_epoch=0 ,batch_size=128, validation_split=0.2, verbose= 1  , workers = 0,
                     validation_data=(X_test, z_test), callbacks=my_callbacks)
    print('\nhistory dict:', hist.history)
    results = model.evaluate(X_test, z_test, batch_size=128)
    MAE_Loss = np.array(hist.history['loss'][50:]) - np.array(hist.history['val_loss'][50:])
    
    # วัดผล mape
    answer = model.predict(X_test, verbose=0, use_multiprocessing=True, workers=0)
    # answer_test = np.array(answer)*100/ np.array(z_test)
    answer_test = pd.DataFrame([answer.reshape(1,-1)[0],z_test, np.abs(answer.reshape(1, -1)[0] - z_test)/z_test]).T
    answer_test.rename(columns={0: 'answer', 1: 'predict', 2: 'mape'},
            inplace=True)
    mape = answer_test['mape'].mean()*100
    answer_test.to_csv("values.csv")
    print("Mean absolute percentage error" ,mape)
    
    return [results[1],float(np.abs(sum(MAE_Loss)))],mape,hist

def dump_model_tf(fileName, model,PathFiles):
    # save model
    try:
        model.save(PathFiles + fileName + ".hdf5")
        print(fileName, 'Dump_file OK...')
    except IOError as e:
        print(e)

def plotGLoss(hist):
    # ตรวจสอบ underfit goodfit overfit
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Overfit and underfit')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig('CheckModel_.png')
    plt.show()

if __name__ == '__main__':

    # โหลดข้อมูล และ ตรวจสอบข้อมูล
    x, z = readDataURL()
    # x, z = readDataCSV()

    # โหลด model
    model_list, accuracy, model, Namemodel = create_model(x)
    classifiers = model_list
    #print(model)
    # เรียนรู้

    # แบ่งข้อมูล
    X_train, X_test, z_train, z_test = train_test_split(x, z.values.ravel(), test_size=0.2)
    
    # สอน
    for name, model,_ in classifiers:
      results,accuracy_mean,hist = TrainingModel_tf(X_train,z_train,X_test,z_test,model)
      # ตรวจสอบ model
      plotGLoss(hist) 

    # # save model
    dump_model_tf("produceModel", model,"")
