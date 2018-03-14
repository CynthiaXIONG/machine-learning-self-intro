 #-*- coding: utf-8 -*-

import math
import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns
from lesson8 import get_portfolio_value
from lesson8 import get_portfolio_sharp_ratio

def minimize_function(allocations, data): #inverse of sharpe ratio as we want the highest as possible
	portfolio_value = get_portfolio_value(data, allocations)
	daily_return = compute_daily_returns(portfolio_value)[1:] #remove the initial daily return which is 0
	mean_daily_return_rf = 0 #risk free return is 0% these days

	sharpe_ratio = get_portfolio_sharp_ratio(daily_return, mean_daily_return_rf, 252) #252 = daily samples per year
	return sharpe_ratio - 1 #return the inverse, so we can minimize it

def do_main():
	#---Optimize a portfolio for higher shape ratio--
		#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

	#get the data
	crypto_codes = ['BTC', 'LTC', 'ETH']
	currency_code = 'EUR'
	dates = pd.date_range('2017-08-01', '2017-09-01')
	df = get_gdax_data(crypto_codes, currency_code, dates)

	#Initial Guess
	portfolio_allocations_0 = np.array([0.4, 0.3, 0.3])

	#Solver constraints and bounds
	cons  = ({'type':'eq', 'fun': lambda x: 1 -x.sum()})  #must return 0 to be true
	bnds = ((0, 1.0), (0, 1.0), (0, 1.0))
	result = spo.minimize(minimize_function, portfolio_allocations_0, 
							args=(df,), 
							method='SLSQP', 
							options={'disp':True},
							bounds=bnds, #values of X are in range [0, 1]
							constraints=cons ) #allocations must sum to 1.0 
	print("Optimized Allocations:")
	print(result.x)

	optimized_portfolio_value = get_portfolio_value(df, result.x)
	print("Max profit=", optimized_portfolio_value[-1])

	optimized_daily_return = compute_daily_returns(optimized_portfolio_value)[1:] #remove the initial daily return which is 0
		
	sharpe_ratio = get_portfolio_sharp_ratio(optimized_daily_return, 0, 252)
	print("Max Sharpe Ratio=", sharpe_ratio)

	optimized_daily_return.plot(title='Daily Returns')
	plt.show()

if __name__ == "__main__":
	do_main()