# create by topkoka , mr.aditep campira  19-07-2020
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model



def predict(water,disaster,suitability,plant_maintenance,plant_sale_price):
    model = load_model("produceModel.hdf5")
    data = np.array([[water,disaster,suitability,plant_maintenance,plant_sale_price]])
    answer = model.predict(data, verbose=0, use_multiprocessing=True, workers=0)
    print('{:.10f}'.format(answer[0][0]))



if __name__ == '__main__':
    # water	disaster	suitability	        plant_maintenance	plant_breed 	plant_sale_price	produce
    # 0.0000	100.0000	50.0000	        850.00	                0101	        9.00            	500.00
    water = 0
    disaster = 100.0000
    suitability = 50
    plant_maintenance =  850.00
    plant_sale_price =   9.00

    predict(water,disaster,suitability,plant_maintenance,plant_sale_price)