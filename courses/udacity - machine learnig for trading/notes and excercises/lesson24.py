 #-*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from time import mktime
import json
from bs4 import BeautifulSoup
import requests

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns, get_rolling_mean, get_rolling_std, get_bollinger_bands
from lesson22n23 import LinRegLearner, PolyRegLearner, KNNLearner, calculate_df_input_output_data, get_rms_error

#get historical data from cryptocompare
#from http://www.quantatrisk.com/2017/03/20/download-crypto-currency-time-series-portfolio-python/

def Datetime2TimeStamp(dt):
    epoch = pd.to_datetime('1970-01-01')
    timestamp = (dt - epoch).total_seconds()
    return timestamp

def Timestamp2Datetime(ts):
    return pd.to_datetime(ts, unit='s')

def FetchCryptoOHLC(fsym, tsym):
    # function fetches a crypto price-series for fsym/tsym and stores
    # it in pandas DataFrame
    cols = ['date', 'timestamp', 'open', 'high', 'low', 'close']
    lst = ['time', 'open', 'high', 'low', 'close']

    ts = Datetime2TimeStamp(pd.datetime.now())
    print(ts)
    
    for j in range(2):
        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(int(ts)) + "&limit=2000"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())
        for i in range(1, 2001):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if(x == 0):
                    tmp.append(str(Timestamp2Datetime(y)))
                tmp.append(y)
            if(np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df.date)
        df.drop('date', axis=1, inplace=True)
        curr_timestamp = int(df.ix[0][0])
        if(j == 0):
            df0 = df.copy()
        else:
            data = pd.concat([df, df0], axis=0)
 
    return data

def do_main():

    #get the data
    fsym = 'ARK'
    tsym = 'BTC'
    data = FetchCryptoOHLC(fsym, tsym)
  
    plt.figure(figsize=(10,4))
    plt.plot(data.close)
    plt.legend(loc=2)
    plt.title(fsym, fontsize=12)
    plt.ylabel(tsym, fontsize=12)

    rm = get_rolling_mean(data['close'], 20) #20 days
    rstd = get_rolling_std(data['close'], 20) #20 days
    b_upper_band, b_lower_band = get_bollinger_bands(rm, rstd)

    rm.plot(label='Rolling mean') #plot on the same plot "access object"
    b_upper_band.plot(label='Upper band')
    b_lower_band.plot(label='Lower band')

    plt.show()

if __name__ == "__main__":
    do_main()


