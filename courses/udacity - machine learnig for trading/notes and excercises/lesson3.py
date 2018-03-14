from coinbase.wallet.client import Client
import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_gdax_data(crypto_codes, currency_code, dates):
	#get history from gdax 
	gdax_client = gdax.PublicClient()

	df = pd.DataFrame(index=dates)

	for crypto_coin in crypto_codes:
		#start_date_unix_ts = dates[0].value // 10 ** 9 #convert from nanoseconds to seconds
		start_date_iso = dates[0].isoformat() #dates[0] is a pandas TimeStamp
		end_date_iso = (dates[-1] + pd.DateOffset(days=1)).isoformat() #add on more day because get_product_historic_rates end date is exclusive

		history = gdax_client.get_product_historic_rates(crypto_coin + '-' + currency_code,
							granularity=60*60*24, start=start_date_iso, end=end_date_iso)
		#convert to pandas dataframe
		coin_df = pd.DataFrame(history)
		#rename columns
		close_col_name = crypto_coin + ' close'
		coin_df.rename(columns={0:'date', 4:close_col_name}, inplace=True)
		#convert date
		coin_df['date'] = pd.to_datetime(coin_df['date'],unit='s')
		#set date as index
		coin_df.set_index('date', inplace=True)
		#join_data
		df = df.join(coin_df[close_col_name], how='left')

	return df

def do_main():
	client = Client(
		'Iha7dJ4XCKlMbn9q',
		'jxiTukpl7fXbMoNDooMQLjt8EH2bIk0Y')

	#list the wallets and transactions in our account
	#accounts = client.get_accounts()
	#for account in accounts.data:
		#balance = account.balance
	
	main_account = client.get_primary_account()
	
	crypto_codes = ['BTC', 'ETH', 'LTC']
	currency_code = 'EUR'  # can also use EUR, CAD, etc.

	"""
	df = pd.DataFrame(history)
	#rename columns
	df.rename(columns={0:'date', 1:'low', 2:'high', 3:'open', 4:'close', 5:'volume'}, inplace=True)
	#convert date
	df['date'] = pd.to_datetime(df['date'],unit='s')
	"""

	#define data range
	dates = pd.date_range('2017-01-01', '2017-08-24')

	df = get_gdax_data(crypto_codes, currency_code, dates)

	#normalize
	df = df / df.ix[0,:]  #divide by the first row/initial value

	#plot
	df.plot()
	plt.show()

if __name__ == "__main__":
	do_main()