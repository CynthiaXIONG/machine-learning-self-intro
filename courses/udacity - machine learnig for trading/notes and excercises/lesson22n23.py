 #-*- coding: utf-8 -*-

import math
import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns

class LinRegLearner:
	def __init__(self):
		pass

	def train(self, X, Y):
		self.regr = LinearRegression()
		self.regr.fit(X, Y)

	def query(self, X):
		return self.regr.predict(X)

#http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
#transform polynomial featues and then use a linear regression model
class PolyRegLearner:
	def __init__(self, n):
		self.n = n
		pass

	def train(self, X, Y):
		self.model = Pipeline([('poly', PolynomialFeatures(degree=self.n)), ('linear', LinearRegression(fit_intercept=False))])
		self.model.fit(X, Y)

	def query(self, X):
		return self.model.predict(X)


class KNNLearner:
	def __init__(self, k):
		self.k = k
		pass

	def train(self, X, Y):
		self.neigh = KNeighborsRegressor(n_neighbors=self.k)
		self.neigh.fit(X, Y)

	def query(self, X):
		return self.neigh.predict(X)

class EnsembleLearner:
# see more here: http://scikit-learn.org/stable/modules/ensemble.html#bagging
    def __init__(self):
        self.regr_linear = LinRegLearner()
        self.regr_poly_2 = PolyRegLearner(2)
        self.knn_bagging = BaggingRegressor(KNeighborsRegressor(), max_samples=0.6, max_features=1.0)
        pass

    def train(self, X, Y):
        self.regr_linear.train(X, Y)
        self.regr_poly_2.train(X, Y)
        self.knn_bagging.fit(X, Y)

    def query(self, X):
        solution = self.regr_linear.query(X)
        solution = np.add(solution, self.regr_poly_2.query(X))
        solution = np.add(solution, self.knn_bagging.predict(X))
        return (solution / 3)

def calculate_df_input_output_data(df, lookback_days):
	inputs = df.iloc[lookback_days:]
	#look back data
	for i in range(1, lookback_days+1):
		inputs['prev day' + str(i)] = df[lookback_days-i:-i].values

	#daily return
	dr = compute_daily_returns(df)
	dr.rename(columns={dr.columns[0]:'daily returns'}, inplace=True)

	inputs = inputs.join(dr, how='left')

	#remove output column to its own matrix
	outputs = inputs[inputs.columns[0]]
	inputs.drop(inputs.columns[0], axis=1, inplace=True) #remove first collumn

	return inputs, outputs

def get_rms_error(y_predicted, y_test):
	return math.sqrt(np.sum((y_predicted-y_test)**2) / y_test.size)


def do_main():

	#get the data
	crypto_codes = ['BTC']
	currency_code = 'EUR'
	training_dates = pd.date_range('2017-07-01', '2017-08-01')
	df_training = get_gdax_data(crypto_codes, currency_code, training_dates)

	test_dates = pd.date_range('2017-08-01', '2017-09-01')
	df_test = get_gdax_data(crypto_codes, currency_code, test_dates)

	x_training, y_training = calculate_df_input_output_data(df_training, 5)
	x_test, y_test = calculate_df_input_output_data(df_test, 5)

	print("y_test:")
	print(y_test)

	linear_reg_learner = LinRegLearner()
	linear_reg_learner.train(x_training, y_training)
	y_predicted_lr = linear_reg_learner.query(x_test)

	print("y_predicted_lr:")
	print(y_predicted_lr)

	#Root Mean Squared Error
	rms_error_lr = get_rms_error(y_predicted_lr, y_test)
	print("rms_error_lr: " + str(rms_error_lr))

	#Correlation
	correlation_lr = np.corrcoef(y_predicted_lr, y_test)
	print("correlation_lr: " + str(correlation_lr))


	#KNN
	knn_learner = KNNLearner(3)
	knn_learner.train(x_training, y_training)
	y_predicted_knn = knn_learner.query(x_test)

	print("y_predicted_knn:")
	print(y_predicted_knn)

	#Root Mean Squared Error
	rms_error_knn = get_rms_error(y_predicted_knn, y_test)
	print("rms_error_knn: " + str(rms_error_knn))

	#Correlation
	correlation_knn = np.corrcoef(y_predicted_knn, y_test)
	print("correlation_knn: " + str(correlation_knn))

	#knn in this case is very bad bc almost all x_test are outside of x_training range (price shot up) and knn cant extrapolate

	#PolyRegLearner
	poly_learner = PolyRegLearner(2)
	poly_learner.train(x_training, y_training)
	y_predicted_poly = poly_learner.query(x_test)

	print("y_predicted_poly:")
	print(y_predicted_poly)

	#Root Mean Squared Error
	rms_error_poly = get_rms_error(y_predicted_poly, y_test)
	print("rms_error_poly: " + str(rms_error_poly))

	#Correlation
	correlation_poly = np.corrcoef(y_predicted_poly, y_test)
	print("correlation_poly: " + str(correlation_poly))

	#Ensemble (lesson 24)
	ensemble_learner = EnsembleLearner()
	ensemble_learner.train(x_training, y_training)
	y_predicted_ensemble = ensemble_learner.query(x_test)

	print("y_predicted_ensemble:")
	print(y_predicted_ensemble)

	#Root Mean Squared Error
	rms_error_ensemble = get_rms_error(y_predicted_ensemble, y_test)
	print("rms_error_ensemble: " + str(rms_error_ensemble))

	#Correlation
	correlation_ensemble = np.corrcoef(y_predicted_ensemble, y_test)
	print("correlation_ensemble: " + str(correlation_ensemble))

if __name__ == "__main__":
	do_main()


