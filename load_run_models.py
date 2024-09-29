import numpy as np
import pandas as pd

import datetime
from dateutil.relativedelta import relativedelta
import os

from keras.models import load_model
from keras.utils import pad_sequences
import andi

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from create_datasets import log

def read_reshape_txt(file,max_length):
    with open(file, "r") as f:    
        task = [np.array(x.split(";"), dtype="float32") for x in f.readlines()]
        task = np.array([x[1:] for x in task])
        
        """import andi
        AD = andi.andi_datasets()
        dataset = AD.andi_dataset(N = 1_000, tasks = 2, dimensions = 1)
        task = np.array([x[1:] for x in [np.array(dataset[2][0][0]),np.array(dataset[2][0][1])]])
        """
        task = task[:,]
    
    task_HomLen = pad_sequences(task, maxlen=max_length, dtype="float32")
    return task_HomLen.reshape((-1, max_length, 1)) # reshape all to the same size of max_lenght

def predict_task1(model,df,start_date, end_date,task,len_min,len_max,now):
    """
    Parameters:
    model:
    df: df with the tickers 
    """
    returns_str = f"{start_date.date()}-{end_date.date()}-{len_min}-{len_max}"
    new_prediction = f'task{task}_len_{len_min}_{len_max}_{start_date.date()}_{end_date.date()}'
    if new_prediction not in df.columns:
        df[new_prediction] = ""
        error_  = 0
        for i,RIC in enumerate(df['Identifier']):
            RIC_dates = RIC + '-' + returns_str + '.txt'
            name_file_txt = os.path.join('data' , RIC_dates)
            
            try:
                task_HomLenRes = read_reshape_txt(name_file_txt,len_max)
            
                # Clasified the data
                with tf.device('/CPU:0'):
                    exps = model.predict(task_HomLenRes, verbose=0)
                # Averaged all the exponents in the trayectorias
                df.loc[i , f'task{task}_len_{len_min}_{len_max}_{start_date.date()}_{end_date.date()}'] = np.mean(exps)
                #print('Predicted',i,RIC)
            except:
                #print('Error',RIC)
                error_ += 1
            log(RIC+' predicted and save',now)
        print('Errors:',error_,'/',len(df['Identifier']))
        return df
    else:
        return df

def predict_task2(model,df,start_date, end_date,task,len_min,len_max,now):
    returns_str = f"{start_date.date()}-{end_date.date()}-{len_min}-{len_max}"

    df[f'task{task}_len_{len_min}_{len_max}_{start_date.date()}_{end_date.date()}'] = ""

    for i,RIC in enumerate(df['Identifier']):
        RIC_dates = RIC + '-' + returns_str + '.txt'
        name_file_txt = os.path.join('data' , RIC_dates)
        
        if  RIC_dates in os.listdir('data'):
            log(RIC+' read',now)
            
            task_HomLenRes = read_reshape_txt(name_file_txt,len_max)
            
            # Clasified the data
            model_predicted = model.predict(task_HomLenRes)
            
            df.iloc[i,-1] = model_predicted
            log(RIC+' predicted and save',now)
    return df

if __name__ == '__main__':

    # Chosse the interval
    start_data = datetime.datetime(year=2012,month=4,day=1)
    end_data = datetime.datetime(year=2024,month=4,day=1)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    start_window = datetime.datetime(year=2012,month=4,day=1)
    end_window = datetime.datetime(year=2016,month=4,day=1)
    
    month_jumps = 1
    num_of_jumps = 8*12
    lenghts_min = [10,100,400,800]
    lenghts_max = [20,200,500,1_000]
    
    task = 1
    path = 'ANDI_Challenge/models/task'+str(task)
    data = 'analisis/S&P_500_exp.xlsx'
    df = pd.read_excel(data)

    # Loop through all the model for the intervals
    if task == 1:
        for len_min,len_max in zip(lenghts_min,lenghts_max):
            name_model = f'task{task}_len_{len_min}_{len_max}.h5'
            model = load_model(os.path.join(path,name_model))
            
            for jump in range(0,num_of_jumps+1,month_jumps): # Loop through the date to make a rolling window 
                start_date = start_window + relativedelta(months=jump)
                end_date = end_window + relativedelta(months=jump)
        
                # Make predictions
                df = predict_task1(model,df,start_date, end_date,task,len_min,len_max,now)
            df.to_excel(data,index=False)
            print(len_min,'-',len_max)
    
    elif task == 2:
        len_min,len_max = 800,1_000
        name_model = f'task{task}_dim{1}.h5'
        df = pd.read_excel(data,index_col=0)
            
        # Load the data and prepare
        model = load_model(os.path.join(path,name_model))

        # Make predictions
        df = predict_task2(model,df,start_date, end_date,task,len_min,len_max,now)
        df.to_excel(data)
        print(len_min,'-',len_max)


