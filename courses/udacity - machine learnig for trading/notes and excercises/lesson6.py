import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lesson3 import get_gdax_data

def fill_missing_data(df)
	df.fillna(method='ffill', inplace=True)
	df.fillna(method='bfill', inplace=True)

def do_main():
	#---Imcomplete DATA---
	#for missing data, we need to fill those gaps with data, so the NaNs dont blow our calculations
		#interpolating is not good (predicting the future)
	#use the last known value to fill the gap ->fill forward
	#for missing data in the begging ->fill backwards
	
	#use pandas fillna to fill the data (bfill and ffill)
	#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html

	crypto_codes = ['BTC','ETH', 'LTC']
	currency_code = 'EUR'
	dates = pd.date_range('2017-04-01', '2017-08-01') #time where ETH started, so data is missing

	df = get_gdax_data(crypto_codes, currency_code, dates)

	ax = df.plot(title="LTH Close stats", label='LTH')
	ax.set_xlabel("Date")
	ax.set_ylabel("Close Price")
	ax.legend(loc='upper left')
	plt.show()

	#forward fill first
	df.fillna(method='ffill', inplace=True)
	ax = df.plot(title="LTH Close stats", label='LTH')
	plt.show()

	#backward fill second
	df.fillna(method='bfill', inplace=True)
	ax = df.plot(title="LTH Close stats", label='LTH')
	plt.show()

if __name__ == "__main__":
	do_main()