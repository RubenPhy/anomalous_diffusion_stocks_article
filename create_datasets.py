from unicodedata import name
import pandas as pd
import numpy as np

#from download_data import download_history

import datetime
import os
from dateutil.relativedelta import relativedelta

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def download_hist(df,ticker, start_date, end_date, path_file, df_of_file, now, interval='daily'):
    """
    If the data is not already downloaded, it downloads the data and save it as csv
    Otherwise, it reads the data and filter the data for the interval dates 
    """

    file = os.path.split(path_file)[-1]
    # Check if there is a file with the interval that we want
    df_filter = df_of_file[(df_of_file['start_date'] <= start_date) & (df_of_file['end_date'] >= end_date) & (df_of_file['ticker'] == ticker)]
    
    # Check if the df_filter is not empty, i.e. there is a file with the interval that we want
    if df_filter is not None and not df_filter.empty:
        if df_filter.shape[0] > 1:

            # Make a line with an expection error
            print("Error, more than one file", file)
            raise Exception(f"More than one file could be use to filter the data:{ticker} {start_date} {end_date}")

        else:
            csv_name = df_filter.values[0]
            log(f"Datos históricos ya existentes:{csv_name[0]}-{str(csv_name[1])[:10]}-{str(csv_name[2])[:10]}.csv",now)
            
            # Read the historical data and filter for your interval.
            
            return df[(start_date <= pd.to_datetime(df['Date'])) & (pd.to_datetime(df['Date']) <= end_date)]
    else:
        start_date = file.split('-',1)[1][0:10]+' 00:00:00'
        end_date = file.split('-',1)[1][11:21]+' 23:59:59'
        try:
            df = download_history(ticker=ticker,
                                  fields='*',
                                  start_date=start_date,
                                  end_date=end_date,
                                  interval=interval) 
            df.to_csv(path_file)
            print('Download:',ticker)
            log(f"Downloaded and save:{file}\nSize:{df.shape[0]}",now)
            return df
        except Exception as e:
            print(e)
            log(str(e),now)
            return None

def save_hist(df,name_file, min_T,max_T,now):
    """
    If the file already exist do nothing
    If the file doesn't exist clean the df from NA and compute the log returns
    """
    # Check if the file already exist
    if any([name_file.split('\\')[1] in x for x in os.listdir('data')]):
        return
    
    # Drop the NA
    size_df = len(df)
    df= df.dropna(subset=['CLOSE'])
    if size_df != len(df):
        log(str(size_df-len(df))+' rows with NA',now)

    size_df = len(df)
    if df.shape[0] < min_T:
        log("Error in the size of the df:"+name_file, now)
        return
    
    if df.shape[0] < max_T and df.shape[0] > min_T:
        max_T = df.shape[0]

    # Split the df['CLOSE'] in intervals of max_T
    p = np.floor(size_df / max_T)
    trajectories = np.array([np.array(df['CLOSE'][i*max_T:(i+1)*max_T]) for i in range(int(p))])

    # Compute the returns compare with the x_0
    norm_trajectories = []
    for x in trajectories:
        norm_trajectories.append([np.log(_/x[0])*100 for _ in x])
    
    if not all(isinstance(_,float) for _ in sum(norm_trajectories,[])):
        log("Error in the returns of:"+name_file, now)
    
    # Save the retuns in a txt
    with open(name_file,'w') as f:
        for trajectory in norm_trajectories:
            f.write('1.0;'+';'.join([str(x) for x in trajectory])+'\n')
    
    log("Returns compute",now)

def log(txt,now):
    f = open(__file__[:-3]+'_'+now+'.log', 'a')
    f.write(txt + '\r')
    f.close()

def compute_returns(i,tickers,start_data,end_data, start_window, end_window, month_jumps, num_of_jumps, lenghts_min, lenghts_max, df_files,current_files, now):
    ticker = tickers[i]
    csv_historical_file = os.path.join('data' , f"{ticker}-{start_data.date()}-{end_data.date()}.csv")
    df_csv = pd.read_csv(csv_historical_file)
    time_end = datetime.datetime.now()
    
    for jump in range(0,num_of_jumps+1,month_jumps): # Loop through the dates to make a rolling window 
        start_date = start_window + relativedelta(months=jump)
        end_date = end_window + relativedelta(months=jump)
        
        for len_min,len_max in zip(lenghts_min,lenghts_max): # Loop through the lenghts of the returns
        
            log(ticker,now)
            returns_txt = f"{ticker}-{start_date.date()}-{end_date.date()}-{len_min}-{len_max}.txt" 
            name_file_txt = os.path.join('data' , returns_txt)
            
            if returns_txt in current_files: # Check if the file already exist
                log('Done',now)
            else:
                
                df_window = download_hist(df_csv,ticker, start_date, end_date, csv_historical_file, df_files, now)
            
                name_file_txt = os.path.join('data' , f"{ticker}-{start_date.date()}-{end_date.date()}-{len_min}-{len_max}.txt")
                if df_window is not None and df_window.shape[0] > len_min:    
                    save_hist(df_window,name_file_txt, len_min, len_max,now)
                    log('Done',now)
                    
                else:
                    log("EMPTY"+name_file_txt,now)
    print(i,ticker,datetime.datetime.now()-time_end)

def create_df_file(path):
    list_rics = []
    list_start_date = []
    list_end_dates = []

    for file in os.listdir(path):
        if file.endswith('.csv'):
            list_rics.append(file.split('-')[0])
            list_start_date.append(file.split('-',1)[1][0:10])
            list_end_dates.append(file.split('-',1)[1][11:21])

    df = pd.DataFrame({'ticker':list_rics, 'start_date':list_start_date, 'end_date':list_end_dates})
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    return df

def historical_vol(tickers, start_data, len_max):
    list_of_files = os.listdir('data')
    volatilty = []

    df_max_day = pd.read_csv(os.path.join('data','AFL.N-2012-03-29-2024-03-01.csv'))
    days = [pd.Timestamp(x) for x in df_max_day['Date']]
    
    if os.path.exists('analisis/S&P_500_HV_'+str(len_max)+'.csv'):
        return

    for ticker in tickers:
    
        csv_historical_file = [x for x in list_of_files if ticker in x and '.csv' in x][0]
        df_hist = pd.read_csv(os.path.join('data',csv_historical_file))

        if df_hist['CLOSE'].isna().sum() > 10:
            print("Problems in ",ticker)
       
        df_hist['Retuns'] = df_hist['CLOSE'].pct_change()
        hist_vol = []
        
        # The average of the std of the returns from day to len_max days previous
        min_date = pd.to_datetime(df_hist['Date']).min()+relativedelta(days=len_max)
        total_min_data = start_data + relativedelta(days=len_max)
        for i,d in enumerate(days):
            if i > len_max and min_date<=d and d>=total_min_data:
                avg_std = np.std(df_hist['Retuns'][i-len_max:i]) * np.sqrt(252) * 100
                hist_vol.append(avg_std)
            else:
                hist_vol.append(np.nan)
        print(ticker,len(hist_vol),len(days))
        volatilty.append(hist_vol)


    hist_vol_matrix = np.array(volatilty)
    df = pd.DataFrame(hist_vol_matrix,columns=days, index=tickers)        
    df.to_csv('analisis/S&P_500_HV_'+str(len_max)+'.csv')
        
def historical_imp_vol(tickers, start_data, len_max):
    list_of_files = os.listdir('data/IV/')
    volatilty = []

    df_max_day = pd.read_csv(os.path.join('data/IV','AFL.N__2012-04-01_2024-04-01.csv'))
    #convert the date to a timestamp without tz=UTC
    days = [pd.Timestamp(x[:10], tz=None) for x in df_max_day['Date']]
    
    for ticker in tickers:
        csv_historical_file = [x for x in list_of_files if ticker in x and '.csv' in x][0]
        df_hist = pd.read_csv(os.path.join('data/IV',csv_historical_file))

        df_hist['Date'] = [pd.Timestamp(x[:10], tz=None) for x in df_hist['Date']]
        hist_vol = []
        
        # The average of the std of the returns from day to len_max days previous
        min_date = pd.to_datetime(df_hist['Date']).min()+relativedelta(days=len_max)
        total_min_data = start_data + relativedelta(days=len_max)
        for i,d in enumerate(days):
            if i > len_max and min_date<=d and d>=total_min_data:
                avg = np.mean(df_hist['30 Day At-The-Money Implied Volatility Index for Put Options'][i-len_max:i])
                hist_vol.append(avg)
            else:
                hist_vol.append(np.nan)
        print(ticker,len(hist_vol),len(days))
        volatilty.append(hist_vol)
    hist_vol_matrix = np.array(volatilty)
    df = pd.DataFrame(hist_vol_matrix,columns=days, index=tickers)        
    df.to_csv('analisis/S&P_500_IV_'+str(len_max)+'.csv')


if __name__ == '__main__':

    start_data = datetime.datetime(year=2012,month=4,day=1)
    end_data = datetime.datetime(year=2024,month=3,day=1)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    difference = relativedelta(end_data, start_data)
    num_of_jumps = difference.years * 12 + difference.months

    start_window = datetime.datetime(year=2012,month=4,day=1)
    end_window = datetime.datetime(year=2016,month=4,day=1)
    
    path_sp = 'analisis/S&P_500.xlsx'
    tickers = pd.read_excel(path_sp)['Identifier'].to_list()

    
    lenghts_min = [10,100,400,800]
    lenghts_max = [20,200,500,1_000]
    month_jump_list = [1,12,24,48]

    df_files = create_df_file('data')
    current_files = os.listdir('data')

    for len_min,len_max,month_jump in zip(lenghts_min,lenghts_max,month_jump_list):
        #historical_vol(tickers, start_data, len_max)
        historical_imp_vol(tickers, start_data, len_max)

    month_jumps = 1
    num_of_jumps = 8*12-1

    #for i in range(len(tickers)):
    #    compute_returns(i,tickers,start_data,end_data, start_window, end_window, month_jumps, num_of_jumps, lenghts_min, lenghts_max, df_files,current_files, now)
    

    print("end")
