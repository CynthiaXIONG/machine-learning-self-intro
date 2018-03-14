 #-*- coding: utf-8 -*-

import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns


def do_main():
	#---Histograms and scatter plots--- for Daily Returs
		#Histograms -> integral of the ocurrances -> usually matches the normal/guassian distribution
			#we can measue mean, std and kurtosis
				#kurtosis: 	>0, "fat tails", there are more occurances at the "tails"/end, than the guassian distrinution
						#	<0, "skinny tais", there are less than
						# measures how the "tails" probability is different from gaussian distrib
		#Scatter Plots -> plot the difference between two values (two 'stocks')
			#commun to use linear regression to get the function of this relation
				#slope(m) = β, represents how reactive a stock is to the market (SPY)
					#slope does not mean correlation!!
						#correlation is if the values are less scatter (and more close together)...the band is thin!
				#intersection with 0 (b) = α, if positive, means that this stock has better performance than market(SPY)

	crypto_codes = ['BTC']
	currency_code = 'EUR'
	dates = pd.date_range('2017-01-01', '2017-08-01')
	df = get_gdax_data(crypto_codes, currency_code, dates)

	ax = df.plot(title="Close stats", label='Close')
	plt.show()

	daily_returns = compute_daily_returns(df)
	daily_returns.plot(title="Daily Returns", label='Daily Returns')
	plt.show()

	#--histogram--
	daily_returns.hist(bins=40) #use 20 bins

	#add mean and std (lines) to histogram
	mean = daily_returns['BTC close'].mean()
	print("mean=", mean)
	std = daily_returns['BTC close'].std()
	print("std=", std)

	plt.axvline(mean, color='w', linestyle='dashed', linewidth=1)
	plt.axvline(std, color='r', linestyle='dashed', linewidth=1)
	plt.axvline(-std, color='r', linestyle='dashed', linewidth=1)

	plt.show()

	#compute kurtosis
	print("kurtosis=", daily_returns.kurtosis())

	#compare two histograms
	crypto_codes = ['LTC', 'ETH']
	dates = pd.date_range('2017-06-01', '2017-08-25')
	df = get_gdax_data(crypto_codes, currency_code, dates)
	ax = df.plot(title="Close stats", label='Close')
	plt.show()

	daily_returns = compute_daily_returns(df)
	daily_returns.hist(bins=40) 
	plt.show()

	#to plot both histograms together
	daily_returns['LTC close'].hist(bins=20, label='LTC')
	daily_returns['ETH close'].hist(bins=20, label='ETH')
	plt.show()

	#--Scatterplots--
	crypto_codes = ['BTC', 'LTC', 'ETH']
	df = get_gdax_data(crypto_codes, currency_code, dates)
	ax = df.plot(title="Close stats", label='Close')
	plt.show()

	daily_returns = compute_daily_returns(df)

	#Scatterplot BTC vs LTC
	daily_returns.plot(kind='scatter', x='BTC close', y='LTC close')
		#fit a line using regression/numpy
	beta_LTC, alpha_LTC = np.polyfit(daily_returns['BTC close'], daily_returns['LTC close'], 1) #degree 1 poly (line)
	print("beta_LTC=", beta_LTC)
	print("alpha_LTC=", alpha_LTC)
	plt.plot(daily_returns['BTC close'], beta_LTC*daily_returns['BTC close'] + alpha_LTC, '-', color='r')
	plt.show()
	

	daily_returns.plot(kind='scatter', x='BTC close', y='ETH close')
	beta_ETH, alpha_ETH = np.polyfit(daily_returns['BTC close'], daily_returns['ETH close'], 1) #degree 1 poly (line)
	print("beta_ETH=", beta_ETH)
	print("alpha_ETH=", alpha_ETH)
	plt.plot(daily_returns['BTC close'], beta_ETH*daily_returns['BTC close'] + alpha_ETH, '-', color='r')
	plt.show()

	#Correlation
	print daily_returns.corr(method='pearson') #pearson is the most commun method to calc correlation


if __name__ == "__main__":
	do_main()