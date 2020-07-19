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
from sklearn.model_selection import train_test_split
import os

def readData():
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
    return  x,z

def nn_model(X):
    # สร้าง model
    NN_model = Sequential()
    NN_model.add(
        Dense(128, kernel_initializer='lecun_normal', bias_initializer='Zeros', input_dim=X.shape[1],
              activation='relu'))
    for hidden in range(X.shape[1]):
        NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    NN_model.compile(loss='mean_absolute_error', optimizer='adam',
                     metrics=['RootMeanSquaredError'])
    NN_model.summary()
    return NN_model



def TrainingModel_tf(X,z,model):
    # แบ่งข้อมูล
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # สอน
    hist = model.fit(X_train, z_train.ravel(), epochs=300,initial_epoch=0 ,batch_size=32, validation_split=0.2, verbose= 1  , workers = 0,
                     validation_data=(X_test, z_test))

    return hist

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
    x,z = readData()

    # โหลด model
    model = nn_model(x)

    # เรียนรู้
    hist = TrainingModel_tf(x,z.values.ravel(),model)

    # ตรวจสอบ model
    plotGLoss(hist)

    # save model
    dump_model_tf("produceModel", model,"")
