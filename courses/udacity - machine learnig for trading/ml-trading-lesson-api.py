import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) #change dir to this script location
import sys
sys.path.insert(0, os.path.abspath("..")) #add the parent folder to the sys path

import time
import datetime
from time import mktime
import math
import json
import requests

from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode

import pandas as pd
import pandas_datareader.data as pd_web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from bs4 import BeautifulSoup

import scipy.optimize as spo
from sklearn import cluster, covariance, manifold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split, ShuffleSplit
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

import gdax
from bittrex.bittrex import Bittrex

#----------------------------------------------------------
def GetDateRangeLastDays(num_days, end_dt_offset=None):
    #e.g offset: pd.DateOffset(days=30)
    end_dt = pd.datetime.now()
    if end_dt_offset is not None:
        end_dt = pd.datetime.now() - end_dt_offset
    return pd.date_range(end=end_dt, periods=num_days, freq='D', normalize=True)

#----------------------------------------------------------
def GetDateRangeLastHours(num_hours, end_dt_offset=None):
    end_dt = pd.datetime.now()
    if end_dt_offset is not None:
        end_dt = pd.datetime.now() - end_dt_offset
    return pd.date_range(end=end_dt, periods=num_hours, freq='H', normalize=False)

#----------------------------------------------------------
def GetDateRangeLastMinutes(num_minutes, end_dt_offset=None):
    end_dt = pd.datetime.now()
    if end_dt_offset is not None:
        end_dt = pd.datetime.now() - end_dt_offset
    return pd.date_range(end=end_dt, periods=num_minutes, freq='min', normalize=False)

#----------------------------------------------------------
def Datetime2TimeStamp(dt):
    epoch = pd.to_datetime('1970-01-01')
    timestamp = (dt - epoch).total_seconds()
    return timestamp

#----------------------------------------------------------
def Timestamp2Datetime(ts):
    return pd.to_datetime(ts, unit='s')

#----------------------------------------------------------
def GetGdaxData(fsym, tsym, dates, data_granularity=60*60*24):
    #get price history from gdax
    df = pd.DataFrame(index=dates)

    gdax_client = gdax.PublicClient()

    start_date_iso = dates[0].isoformat() #dates[0] is a pandas TimeStamp
    end_date_iso = (dates[-1] + pd.DateOffset(days=1)).isoformat() #add on more day because get_product_historic_rates end date is exclusive

    history = gdax_client.get_product_historic_rates(fsym + '-' + tsym,
                        granularity=data_granularity, start=start_date_iso, end=end_date_iso)

    #convert to pandas dataframe
    history_df = pd.DataFrame(history)
    #rename columns
    history_df.rename(columns={0:'date', 1:'low', 2:'high', 3:'open', 4:'close', 5:'volume'}, inplace=True)
   
    #convert date
    history_df['date'] = Timestamp2Datetime(history_df['date'])
    #set date as index
    history_df.set_index('date', inplace=True)

    df = df.join(history_df, how='left')
    df = df.dropna()

    return df

#----------------------------------------------------------
#from http://www.quantatrisk.com/2017/03/20/download-crypto-currency-time-series-portfolio-python/
def GetCryptocompareHisto(func_name, fsym, tsym, dates):
    # function fetches a crypto price-series for fsym/tsym and stores
    # it in pandas DataFrame
    cols = ['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'VolumeFrom', 'VolumeTo']
    lst = ['time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']

    final_ts = dates[-1].value / 10 ** 9
    curr_ts = final_ts

    num_samples = dates.size
    samples_limit = 2000
    num_requests = int(math.ceil(num_samples/float(samples_limit)))
    
    for j in range(num_requests):
        max_samples = samples_limit
        if (j == num_requests - 1):
            max_samples = num_samples % samples_limit #avoid overshot on last request

        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/" + func_name + "?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(int(curr_ts)) + "&limit=" + str(max_samples)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())
        for i in range(1, max_samples + 1):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if(x == 0):
                    tmp.append(Timestamp2Datetime(y))
                tmp.append(y)
            if(np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df['Date'])
        df.drop('Date', axis=1, inplace=True)
        curr_ts = int(df.ix[0][0])
        if(j == 0):
            data = df.copy()
        else:
            data = pd.concat([df, data], axis=0)
 
    data = data.dropna()
    return data

#----------------------------------------------------------
#http://pandas-datareader.readthedocs.io/en/latest/remote_data.html
def GetFinanceHistorical(symbol, dates, source='yahoo'):
    start_date = Timestamp2Datetime(dates[0])
    end_date = Timestamp2Datetime(dates[-1])
    return pd_web.DataReader(symbol, source, start_date, end_date)

#----------------------------------------------------------
def SetupBittrex(secrets_file_path = None):
    if (secrets_file_path is None):
        secrets_file_path = "bittrex-sec.json"

    with open(secrets_file_path) as secrets_file:
        secrets = json.load(secrets_file)
        secrets_file.close()
        return Bittrex(secrets['key'], secrets['secret'])
    
    return None

#----------------------------------------------------------
def GetBittrexOrderbook(bittrex_client, market, depth=20):
    order_book_json = bittrex_client.get_orderbook(market, 'both', depth)

    buy_df = pd.DataFrame.from_dict(order_book_json['result']['buy'])
    buy_df.index = buy_df['Rate']
    buy_df.drop('Rate', axis=1, inplace=True)
    buy_df['Sum'] = buy_df.cumsum(axis=0)
    buy_df['SumRel'] = buy_df['Sum']/buy_df['Quantity'].sum()

    sell_df = pd.DataFrame.from_dict(order_book_json['result']['sell'])
    sell_df.index = sell_df['Rate']
    sell_df.drop('Rate', axis=1, inplace=True)
    sell_df['Sum'] = sell_df.cumsum(axis=0)
    sell_df['SumRel'] = sell_df['Sum']/sell_df['Quantity'].sum()

    return buy_df, sell_df

#----------------------------------------------------------
def MergeSymbolsData(df_array, name_array, parameter):
    merged_df = None

    for i in range(len(df_array)):
        df_array[i].rename(columns={parameter:name_array[i]}, inplace=True)

        if (i == 0):
            merged_df = df_array[i][name_array[i]].to_frame()
        else:
            merged_df = merged_df.join(df_array[i][name_array[i]], how='left')

    return merged_df

#----------------------------------------------------------
def NormalizeDF(df):
    return df / df.ix[0,:]  #divide by the first row/initial value

#----------------------------------------------------------
def FillMissingData(df):
    #ffill first than bfill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

#----------------------------------------------------------
def StopWatchFunction(profiled_function, *args):
    t1 = time.time()
    profiled_function(*args)
    t2 = time.time()
    return t2-t1

#----------------------------------------------------------
def JoinSeriesAB(series_a, series_b):
    df_a = series_a.to_frame()
    df_a.rename(columns={df_a.columns[0]:'A'}, inplace=True)
    df_b = series_b.to_frame()
    df_b.rename(columns={df_b.columns[0]:'B'}, inplace=True)

    return df_a.join(df_b, how='left')

#----------------------------------------------------------
def GetRollingMean(series, window):
    return series.rolling(window=window, center=False, min_periods=2).mean()

#----------------------------------------------------------
def GetRollingSTD(series, window):
    return series.rolling(window=window, center=False, min_periods=2).std()

#----------------------------------------------------------
def GetBollingerBands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band

#----------------------------------------------------------
def GetExpandingMean(series):
    return series.expanding(min_periods=2).mean()

#----------------------------------------------------------
def GetExpandingSTD(series):
    return series.expanding(min_periods=2).std()

#----------------------------------------------------------
def GetBollingerBands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band

#----------------------------------------------------------
def GetKurtosis(series):
    return series.kurtosis()

#----------------------------------------------------------
def GetCorrelation(series_a, series_b):
    joined_df = JoinSeriesAB(series_a, series_b)
    return joined_df.corr(method='pearson') #pearson is the most commun method to calc correlation

#---------------------------------------------------------
def CalculateMomentum(df, samples = 1):
    mom = df.copy()
    mom[samples:] = df[samples:] - df[:-samples].values #pandas will try to match index on element wise operations. SO you need VALUES to reset the index
    
    return mom

#----------------------------------------------------------
def ComputeDailyReturns(df):
    dr = df.copy()
    dr[1:] = (df[1:] / df[:-1].values) - 1 #pandas will try to match index on element wise operations. SO you need VALUES to reset the index
    
    if (len(dr.shape) > 1):
        dr.ix[0, :] = 0 #set row 0 to 0
    else:
        dr[0] = 0
    
    return dr

#----------------------------------------------------------
def PlotBollingerBands(df, dont_plot = False):
    ax = df.plot(title="BollingerBands")

    rm = GetRollingMean(df, df.shape[0])
    rstd = GetRollingSTD(df, df.shape[0])
    b_upper_band, b_lower_band = GetBollingerBands(rm, rstd)

    rm.plot(label='Rolling mean', ax=ax) #plot on the same plot "access object"
    b_upper_band.plot(label='Upper band', ax=ax)
    b_lower_band.plot(label='Lower band', ax=ax)

    if not dont_plot:
        plt.show()
    return ax

#----------------------------------------------------------
def PlotHistogram(series, num_bins=20):
    series.hist(bins=num_bins)
    mean = series.mean()
    std = series.std()

    plt.axvline(mean, color='y', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

#----------------------------------------------------------
def PlotScatter(series_a, series_b):
    joined_df = JoinSeriesAB(series_a, series_b)

    joined_df.plot(kind='scatter', x='A', y='B')
    plt.show()

#----------------------------------------------------------
def GetPortfolioValue(df, initial_portfolio_alloc): 
    temp_df = df / df.ix[0, :] #normalize
    temp_df = temp_df * initial_portfolio_alloc.transpose() #multiply by the initial_portfolio_alloc to get the moneys
    return temp_df.sum(axis=1) #sum  all stocks

#----------------------------------------------------------
def GetPortfolioCumulativeReturn(porfolio_value):
    return (porfolio_value[-1] / porfolio_value[0]) - 1

#----------------------------------------------------------
def GetPortfolioSharpRatio(daily_return, mean_daily_return_rf, sample_per_year):
    return math.sqrt(sample_per_year) * (daily_return.mean() - mean_daily_return_rf) / daily_return.std()

#----------------------------------------------------------
def PorfolioOptimizeSharpeRationFunction(allocations, df):
    portfolio_value = GetPortfolioValue(df, allocations)
    daily_return = ComputeDailyReturns(portfolio_value)[1:] #remove the initial daily return which is 0
    mean_daily_return_rf = 0 #risk free return is 0% these days

    sharpe_ratio = GetPortfolioSharpRatio(daily_return, mean_daily_return_rf, 365) #252 = daily samples per year
    return sharpe_ratio - 1 #return the inverse, so we can minimize it

#----------------------------------------------------------
def GetOptimizedPortfolioAllocations(df):
    num_of_symbols = df.shape[1]
    print("num_of_symbols:", num_of_symbols)
    portfolio_allocations_0 = np.full((num_of_symbols, 1), 1/float(num_of_symbols), dtype=float)
    cons = ({'type':'eq', 'fun': lambda x: 1 -x.sum()})  #must return 0 to be true

    bnds = ((0, 1.0), (0, 1.0), (0, 1.0))

    result = spo.minimize(PorfolioOptimizeSharpeRationFunction, portfolio_allocations_0, 
                            args=(df,), 
                            method='SLSQP', 
                            options={'disp':True},
                            bounds=bnds, #values of X are in range [0, 1]
                            constraints=cons ) #allocations must sum to 1.0
    return result.x

#----------------------------------------------------------
class LinRegLearner:
    def __init__(self):
        self.model = LinearRegression()
        pass

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]  #convert y to series!
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return pd.DataFrame(index=X.index, data={self.GetName():prediction})

    def GetName(self):
        return "LinRegLearner"

    def Save(self, filename):
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
#http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
#transform polynomial featues and then use a linear regression model
class PolyRegLearner:
    def __init__(self, n):
        self.n = n
        self.model = Pipeline([('poly', PolynomialFeatures(degree=self.n)), ('linear', LinearRegression(fit_intercept=False))])
        pass

    def fit(self, X, Y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return pd.DataFrame(index=X.index, data={self.GetName():prediction})

    def GetName(self):
        return "PolyRegLearner " + "n=" + str(self.n)

    def Save(self, filename):
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
class KNNLearner:
    def __init__(self, k):
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=self.k)
        pass

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return pd.DataFrame(index=X.index, data={self.GetName():prediction})

    def GetName(self):
        return "KNNLearner " + "k=" + str(self.k)

    def Save(self, filename):
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
class BaggingLearner:
# see more here: http://scikit-learn.org/stable/modules/ensemble.html#bagging
    def __init__(self, estimator):
        self.estimator = estimator
        self.model = BaggingRegressor(self.estimator.model)
        pass

    def fit(self, X, y):
        if isinstance(Y, pd.DataFrame):
            y = y.ix[:,0]
        self.model.fit(X, y)

    def predict(self, X):
        prediction = (pd.DataFrame(index=X.index, data={self.GetName():self.model.predict(X)}))
        return prediction

    def GetName(self):
        return "BaggingLearner (" + self.estimator.GetName() + ")"

    def Save(self, filename):
        self.estimator.Save(filename + '-estimator.pkl')
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.estimator.Load(filename + '-estimator.pkl')
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')
       
#----------------------------------------------------------
class RandomForestLearner:
# see more here: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    def __init__(self):
        self.model = RandomForestRegressor()
        pass

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]
        self.model.fit(X, y)

    def predict(self, X):
        prediction = (pd.DataFrame(index=X.index, data={self.GetName():self.model.predict(X)}))
        return prediction

    def GetName(self):
        return "RandomForestLearner"

    def Save(self, filename):
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
class SVRLearner:
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    def __init__(self, kernel = 'rbf'):
        self.kernel = kernel
        pipe = Pipeline([('scaler', StandardScaler()), ('reduce_dim', PCA()), ('svr', SVR(kernel=self.kernel))])

        #try to chose the best C and gamma
        C_range = np.logspace(-2, 10, 3)
        gamma_range = np.logspace(-9, 3, 3)
        reduce_dim=[None, PCA(5), PCA(10)]
        param_grid = dict(reduce_dim=reduce_dim, svr__gamma=gamma_range, svr__C=C_range) 

        self.model = GridSearchCV(pipe, param_grid=param_grid)
        pass

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]  #convert Y to series!
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return pd.DataFrame(index=X.index, data={self.GetName():prediction})

    def GetName(self):
        return "SVRLearner " + "kr=" + str(self.kernel)

    def Save(self, filename):
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
class RFELearner():
#https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
    def __init__(self, estimator, num_features):
        self.estimator = estimator
        self.num_features = num_features
        self.model = RFE(self.estimator.model, self.num_features)
        pass

    def fit(self, X, Y):
        if isinstance(y, pd.DataFrame):
            y = y.ix[:,0]  #convert Y to series!
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return pd.DataFrame(index=X.index, data={self.GetName():prediction})

    def GetName(self):
        return "RFELearner(" + self.estimator.GetName() + "):" + str(self.num_features)

    def Save(self, filename):
        self.estimator.Save(filename + '-estimator.pkl')
        joblib.dump(self.model, filename + '-model.pkl')
        joblib.dump(self, filename + '.pkl')

    def Load(self, filename):
        self.estimator.Load(filename + '-estimator.pkl')
        self.model = joblib.load(filename + '-model.pkl')
        self = joblib.load(filename + '.pkl')

#----------------------------------------------------------
def GetRMSError(y_predicted, y_test):
    error = y_predicted.ix[:,0]-y_test.ix[:,0]
    return math.sqrt(np.sum(error**2) / y_test.shape[0])

#----------------------------------------------------------
def GetInputOutputDataForPrediction(df, lookback_samples, prediction_offset):
    #get close data as df
    df = df['Close'].to_frame()

    samples_limit = lookback_samples+prediction_offset-1
    inputs = df.iloc[samples_limit:]
    #look back data
    for i in range(prediction_offset, samples_limit+1):
        lower_limit = samples_limit - i
        if lower_limit > 0:
            inputs.insert(inputs.shape[1], 'lookback sample' + str(i), df[lower_limit:-i].values)
        else:
            inputs.insert(inputs.shape[1], 'lookback sample' + str(i), df[:-i].values) #when lower limit is 0

    #daily return
    #dr = ComputeDailyReturns(df)
    #dr.rename(columns={dr.columns[0]:'daily returns'}, inplace=True)
    #inputs = inputs.join(dr, how='left')

    #remove output column to its own matrix
    outputs = inputs[inputs.columns[0]].to_frame()
    inputs.drop(inputs.columns[0], axis=1, inplace=True) #remove first collumn

    return inputs, outputs

#----------------------------------------------------------
def SeparateDataForTraining(df, lookback_samples, prediction_offset):
    df_copy = df.copy()

    #remove featuers
    excluded_features = ['Date', 'Timestamp', 'Volume', 'VolumeFrom', 'VolumeTo']
    for feature_name in excluded_features:
        if (feature_name in df_copy.columns):
            df_copy.drop(feature_name, axis=1, inplace=True)

    inputs = df_copy.iloc[lookback_samples:-prediction_offset]

    for i in range(1, lookback_samples + 1):
        col_max_index = inputs.shape[1]
        row_min = lookback_samples - i
        row_max = prediction_offset + i
        for j, col_name in enumerate(df_copy.columns):
            inputs.insert(col_max_index + j, col_name + ' T-' + str(i), df[row_min:-row_max][col_name].values)
    
    num_cut_samples = lookback_samples + prediction_offset
    outputs = df_copy.iloc[num_cut_samples - 1:].shift(-1)
    outputs = outputs.dropna()

    return inputs, outputs

def AddTechinalAnalysisFeatures(df):
    new_df = df.copy()
    close = new_df['Close']

    #variation
    variation = new_df['Close'] - new_df['Open']
    new_df.insert(new_df.shape[1], 'Variation' , variation)

    #daily return
    dr = ComputeDailyReturns(close)
    new_df.insert(new_df.shape[1], 'Daily Returns' , dr)
    
    #rma and boullinger bands
    rm = GetRollingMean(close, close.shape[0])
    rstd = GetRollingSTD(close, close.shape[0])
    em = GetExpandingMean(close)
    rm_mom1 = CalculateMomentum(rm)
    rm_mom3 = CalculateMomentum(rm, 3)

    new_df.insert(new_df.shape[1], 'Rolling Mean', rm)
    new_df.insert(new_df.shape[1], 'Rolling STD', rstd)
    new_df.insert(new_df.shape[1], 'Expanding Mean', em)
    new_df.insert(new_df.shape[1], 'RM Momentum1', rm_mom1)
    new_df.insert(new_df.shape[1], 'RM Momentum3', rm_mom3)

    #clean up
    FillMissingData(new_df)

    return new_df

#----------------------------------------------------------
def AddPredictionAsFeature(df, predictor, pre_training_x, pre_training_y, test_x):
    new_df = df.copy()

    predictor.fit(pre_training_x, pre_training_y)

    new_df.insert(new_df.shape[1], 'Prediction:' + predictor.GetName(), predictor.predict(test_x))

    return new_df

#---------------------------------------------------------
def PlotLearningCurves(estimator, X, y, title=None, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), shuffle=False):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=shuffle)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#----------------------------------------------------------
def ClusterAnalyses(dfs, names):
    close_prices = np.vstack([q['Close'] for q in dfs])
    open_prices = np.vstack([q['Open'] for q in dfs])
    variation = close_prices - open_prices

    # ########################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    # #########################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    # #########################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    # #########################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.viridis,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                bbox=dict(facecolor='w',
                        edgecolor=plt.cm.spectral(label / float(n_labels)),
                        alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
            embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
            embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()

#----------------------------------------------------------
def do_main():

    #get the data
    fsym = 'ARK'
    tsym = 'BTC'
    days = 30

    #lesson3
    #df = GetGdaxData(fsym, tsym, GetDateRangeLastDays(days))
    '''
    df = FetchCryptocompareHistoday(fsym, tsym, GetDateRangeLastDays(days))
    print(df)

    normalized = NormalizeDF(df)
    print(normalized)

    print("StopWatchGetGdax:" + str(StopWatchFunction(GetGdaxData, 'BTC', 'EUR', GetDateRangeLastDays(30))))
    print("StopWatchFetchCryptocompareHistoday:" + str(StopWatchFunction(FetchCryptocompareHist, 'histoday', 'BTC', 'EUR', GetDateRangeLastDays(30))))

    #lesson5
    dr = ComputeDailyReturns(df)
    print("dr:", dr['Close'])

    PlotBollingerBands(df['Close'])

    #lesson7
    PlotHistogram(dr['Close'])

    btc_df = FetchCryptocompareHist('histoday', BTC', 'USD', GetDateRangeLastDays(days))
    btc_dr = ComputeDailyReturns(btc_df)

    PlotScatter(dr['Close'], btc_dr['Close'])

    correlation = GetCorrelation(dr['Close'], btc_dr['Close'])
    print('Correlation =', correlation)
 
    #lesson8-10
    ark_df = FetchCryptocompareHist('histoday', 'ARK', 'BTC', GetDateRangeLastDays(days))
    xby_df = FetchCryptocompareHist('histoday', 'XBY', 'BTC', GetDateRangeLastDays(days))
    bts_df = FetchCryptocompareHist('histoday', 'BTS', 'BTC', GetDateRangeLastDays(days))

    merged_df = MergeSymbolsData([ark_df, xby_df, bts_df], ["ARK", "XBY", "BTS"], 'Close')
    print(merged_df.tail())

    optimized_portfolio_allocs = GetOptimizedPortfolioAllocations(merged_df)

    optimized_portfolio_value = GetPortfolioValue(merged_df, optimized_portfolio_allocs)
    print("Max profit=", optimized_portfolio_value[-1])

    optimized_daily_return = ComputeDailyReturns(optimized_portfolio_value)[1:]

    sharpe_ratio = GetPortfolioSharpRatio(optimized_daily_return, 0, 252)
    print("Max Sharpe Ratio=", sharpe_ratio)

    optimized_daily_return.plot(title='Daily Returns')
    plt.show()
   
    
    #lesson22-24
    test_df = FetchCryptocompareHisto('histohour', 'ARK', 'BTC', GetDateRangeLastHours(24*7))
    training_df = FetchCryptocompareHisto('histohour', 'ARK', 'BTC', GetDateRangeLastHours(24*14, pd.DateOffset(days=7)))

    training_df['Close'].plot()
    plt.show()

    x_training, y_training = GetInputOutputDataForPrediction(training_df, 10)
    x_test, y_test = GetInputOutputDataForPrediction(test_df, 10)

    linear_reg_learner = LinRegLearner()
    linear_reg_learner.fit(x_training, y_training)
    y_predicted_lr = linear_reg_learner.predict(x_test)

    ax = y_test.plot(title="Predict Learner", label='Test')
    y_predicted_lr.plot(label='Predict', ax=ax)
    plt.show()

    #Root Mean Squared Error
    rms_error_lr = GetRMSError(y_predicted_lr, y_test)
    print("rms_error_lr: " + str(rms_error_lr))

    #Bittrex
    bittrex = SetupBittrex()

    btc_balance = bittrex.get_balance('BTC')
    print("btc_balance:", btc_balance)

    ark_buy_orderbook, ark_sell_orderbook = FetchBittrexOrderbook(bittrex, 'BTC-ARK')
    print(ark_buy_orderbook)
    print(ark_sell_orderbook)

    #merge many
    fsym_list = ['BTC', 'ETH', 'LTC', 'ARK', 'BTS', 'BAT', 'NAV', 'XBY', 'IOC', 'NEO']

    days = 30
    data_list = []
    for i, ele in enumerate(fsym_list):
        print(i, ele)
        data_list.append(FetchCryptocompareHisto('histoday', ele, 'USD', GetDateRangeLastDays(days)))
    merged_df = MergeSymbolsData(data_list, fsym_list, 'Close')
    print(merged_df.tail())
   
    '''
    #predict BTC price attempt
    fsym = 'BTC'
    #btc_df = FetchCryptocompareHisto('histohour', fsym, 'USD', GetDateRangeLastHours(24*30))
    #ax = PlotBollingerBands(btc_df['Close'], True)

    pre_training_df = GetCryptocompareHisto('histohour', fsym, 'USD', GetDateRangeLastHours(24*5, pd.DateOffset(days=35)))
    training_df = GetCryptocompareHisto('histohour', fsym, 'USD', GetDateRangeLastHours(24*30, pd.DateOffset(days=5)))
    test_df = GetCryptocompareHisto('histohour', fsym, 'USD', GetDateRangeLastHours(24*5))
    
    #pre_training_df = GetFinanceHistorical('GOOG', GetDateRangeLastDays(365,  pd.DateOffset(days=365+30)), 'google') #used to train a estimator used as a feature, so it needs to be out of the training samples
    #training_df = GetFinanceHistorical('GOOG', GetDateRangeLastDays(365,  pd.DateOffset(days=30)), 'google')
    #test_df = GetFinanceHistorical('GOOG', GetDateRangeLastDays(30),'google')

    x_pre_training, y_pre_training = SeparateDataForTraining(pre_training_df, 4, 1)
    x_training, y_training = SeparateDataForTraining(training_df, 4, 1)
    x_test, y_test = SeparateDataForTraining(test_df, 4, 1)

    #how to plot multiple graphs
    #plt.figure(1)
    #plt.subplot(211)
    #plt.subplot(212)

    ax = y_test['Close'].plot(label="Test")

    #add features
    x_pre_training = AddTechinalAnalysisFeatures(x_pre_training)

    x_training = AddTechinalAnalysisFeatures(x_training)
    #x_training = AddPredictionAsFeature(x_training, RFELearner(SVRLearner('linear'), 15), x_pre_training, y_pre_training, x_training)
    
    x_test = AddTechinalAnalysisFeatures(x_test)
    #x_test = AddPredictionAsFeature(x_test, RFELearner(SVRLearner('linear'), 15), x_pre_training, y_pre_training, x_test)

    learners = []
    #learners.append(KNNLearner(3))
    learners.append(LinRegLearner())
    #learners.append(BaggingLearner(KNNLearner(3)))
    #learners.append(BaggingLearner(LinRegLearner()))
    #learners.append(RFELearner(LinRegLearner(), 5))
    #learners.append(RFELearner(LinRegLearner(), 10))
    learners.append(SVRLearner('rbf'))
    #learners.append(BaggingLearner(SVRLearner('linear')))
    #learners.append(RFELearner(SVRLearner('linear'), 10))
    #learners.append(RandomForestLearner())

    for learner in learners:
        learner.fit(x_training, y_training['Close'].to_frame())
        #learner.Save("learner_" + learner.GetName())
        #learner.Load("learner_" + learner.GetName())
        y_predicted_lr = learner.predict(x_test)
        
        y_predicted_lr.plot(label=learner.GetName(), ax=ax)
        print(learner.GetName() + "score:" + str(learner.model.score(x_test, y_test['Close'])))
    
    ax.legend()
    plt.show()

    for learner in learners:
        print("cross val|" + learner.GetName() + "score:" + str(cross_val_score(learner.model, x_training.append(x_test), y_training.append(y_test)['Close'], cv=KFold(n_splits=3))))
     
    '''
    data = GetCryptocompareHisto('histohour', 'BTC', 'USD', GetDateRangeLastHours(24*90))
    x_data, y_data = SeparateDataForTraining(data, 2, 1)

    #X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, shuffle=False)

    # score curves, each time with 20% data randomly selected as a validation set.
    learner = LinRegLearner()
    PlotLearningCurves(learner.model, x_data, y_data)
    plt.show()
    '''

    '''
    symbol_dict = {
        'BTC': 'BTC',
        'ETH': 'ETH',
        'XRP': 'Ripple',
        'BCH': 'BitcoinC',
        'LTC': 'LTC',
        'DASH': 'Dash',
        'ETC': 'ETH Classic',
        'OMG': 'OMG',
        'ARK': 'ARK',
        'LSK': 'LISK',
        'BAT': 'BAT',
        'WTC': 'WTC',
        'BTS': 'Bitshares',
        'IOC': 'IOC',
        'XBY': 'XBY',
        'XMR': 'Monero',
        'NEO': 'NEO'}

    symbols, names = np.array(sorted(symbol_dict.items())).T

    dfs = []
    dates = GetDateRangeLastHours(24*30)
    
    for symbol in symbols:
        print('Fetching history for %r' % symbol, file=sys.stderr)
        dfs.append(GetCryptocompareHisto('histohour', symbol, 'USD', dates))

    ClusterAnalyses(dfs, names)
    '''
#----------------------------------------------------------
if __name__ == "__main__":
    do_main()