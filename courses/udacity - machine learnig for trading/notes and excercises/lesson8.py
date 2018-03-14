 #-*- coding: utf-8 -*-

import math
import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns

#returns the monetary value of a portfolio of stocks for each 'day'
def get_portfolio_value(df, initial_portfolio_alloc): 
	temp_df = df / df.ix[0,:] #normalize
	temp_df = temp_df * initial_portfolio_alloc.transpose() #multiply by the initial_portfolio_alloc to get the moneys
	return temp_df.sum(axis=1) #sum  all stocks

def get_portfolio_cumulative_return(porfolio_value):
	return (porfolio_value[-1] / porfolio_value[0]) - 1

def get_portfolio_sharp_ratio(daily_return, mean_daily_return_rf, sample_per_year):
	return math.sqrt(sample_per_year) * (daily_return.mean() - mean_daily_return_rf) / daily_return.std()


def do_main():
	#---Porfolio values/stats---
		#portefolio_value -> how much are your stocks worth in total, each day
		#cumulative return -> portefolio_value[-1]/portefolio_value[0] - 1
		#avg_daily_ret -> related to its performance (the more the better)
		#std_daily_ret -> related with RISK (deviation/volatility)
		#sharpe_ratio -> metric to adjust return for risk (access if high return but better risk is worth or not)
						#= (portefolio_return - risk_free_return) / portefolio_return_std
							#risk_free_return is the return you would get in a risk free asset (bank account), which these days = 0 (ZERO!!!!)
						# = (mean(daily_ret - daily_rf)) / std(daily_ret) * k
							# need to multiply by a adjusting factor - k
								# k = sqrt(#sampes_per_year)
						# the higher the sharpe_ratio the better!!

	crypto_codes = ['BTC', 'LTC', 'ETH']
	currency_code = 'EUR'
	dates = pd.date_range('2017-06-01', '2017-08-25')
	df = get_gdax_data(crypto_codes, currency_code, dates)

	#get portefolio value
	portfolio_money = 1000
	portfolio__rel_alloc = np.array([0.4, 0.3, 0.3])
	portfolio_value = get_portfolio_value(df, portfolio__rel_alloc * portfolio_money)
	print("portfolio_value")
	print(portfolio_value.tail())

	#daily return
	daily_return = compute_daily_returns(portfolio_value)[1:] #remove the initial daily return which is 0
	print("daily_return")
	print(daily_return.tail())

	#cumulative return
	cumulative_return = get_portfolio_cumulative_return(portfolio_value);
	print("cumulative_return=", cumulative_return)

	#avg_daily_ret
	avg_daily_ret = daily_return.mean();
	print("avg_daily_ret=", avg_daily_ret)

	#std_daily_ret
	std_daily_ret = daily_return.std();
	print("std_daily_ret=", std_daily_ret)

	#sharpe_ratio
	sharpe_ratio = get_portfolio_sharp_ratio(daily_return, 0, 252); #252 active days a year
	print("sharpe_ratio=", sharpe_ratio)


		

if __name__ == "__main__":
	do_main()