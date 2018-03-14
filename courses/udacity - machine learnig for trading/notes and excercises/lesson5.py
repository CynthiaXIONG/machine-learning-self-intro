import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lesson3 import get_gdax_data

def get_rolling_mean(series, window):
	return series.rolling(window=window, center=False).mean()

def get_rolling_std(series, window):
	return series.rolling(window=window, center=False).std()

def get_bollinger_bands(rm, rstd):
	upper_band = rm + 2 * rstd
	lower_band = rm - 2 * rstd
	return upper_band, lower_band

def compute_daily_returns(df):
	dr = df.copy()
	dr[1:] = (df[1:] / df[:-1].values) - 1 #pandas will try to match index on element wise operations. SO you need VALUES to reset the index
	
	if (len(dr.shape) > 1):
		dr.ix[0,:] = 0 #set row 0 to 0
	else:
		dr[0] = 0
	
	return dr

def compute_daily_returns_alt(df): #also works
	dr = (df / df.shift(1)) - 1 #using pandas shift =D
	dr.ix[0,:] = 0 #set row 0 to 0
	return dr

def do_main():
	crypto_codes = ['BTC']
	currency_code = 'EUR'
	dates = pd.date_range('2017-05-01', '2017-08-24')

	df = get_gdax_data(crypto_codes, currency_code, dates)
	ax = df.plot(title="BTC Close stats", label='BTC')
	ax.set_xlabel("Date")
	ax.set_ylabel("Close Price")
	ax.legend(loc='upper left')

	#dataframe computations like mean, etc
	#http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats
	print("--mean--")
	print(df.mean())
	print("--median--")
	print(df.median())
	print("--std--")
	print(df.std())
	#http://pandas.pydata.org/pandas-docs/stable/computation.html?highlight=rolling%20statistics#moving-rolling-statistics-moments
	#rolling statistics -> statistics over just a "window" of data, done for all points
		#3.g.: bollinger bands -> limiting "thersholds" at 2sigma (rolling standard dev).
								# if price is there, good opportunity to buy/sell
	rm_BTC = get_rolling_mean(df['BTC close'], 20) #20 days
	rstd_BTC = get_rolling_std(df['BTC close'], 20) #20 days
	b_upper_band, b_lower_band = get_bollinger_bands(rm_BTC, rstd_BTC)

	rm_BTC.plot(label='Rolling mean', ax=ax) #plot on the same plot "access object"
	b_upper_band.plot(label='Upper band', ax=ax)
	b_lower_band.plot(label='Lower band', ax=ax)

	plt.show()

	#daily returns -> how much did the price go up and down on a day
					# todays price relative to yesterday
	daily_returns = compute_daily_returns(df)
	daily_returns.plot(title="Daily Returns")
	plt.show()

	#cumulative returns -> like daily, but price relative to initial price [0] (begging of the year for ex)
						# corresponds to a NORMALIZATION
	

if __name__ == "__main__":
	do_main()