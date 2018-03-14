 #-*- coding: utf-8 -*-

import math
import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from lesson3 import get_gdax_data
from lesson5 import compute_daily_returns

def f(x):
	y = (x - 1.5)**2 + 0.5
	print("X={}, Y={}".format(x,y))
	return y

def error(line, data): #line is two coeficients (c0, c1 or m and b or beta and alpha)
	return np.sum((data[:,1] - (line[0]*data[:,0]+line[1]))**2)



def fit_line(data, error_func):
	#initial guess
	l = np.float32([0, np.mean(data[:,1])]) #slope = 0, b = mean(y values)

	#plot initial guess
	x_ends = np.float32([-5, 5])
	plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

	#call optimizer to minimize error function
	result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp':True}) #because error_func has a secondary argument (data), pass it as with the 'args'
	return result.x

def error_poly(C, data): #very similar. C is numpy.poly1d object that represents the polynomail coefficients
	return np.sum((data[:,1] - np.polyval(C, data[:, 0]))**2)

def fit_poly(data, error_func, poly_degree):
	#initial guess (all coefs = 1)
	c_guess = np.poly1d(np.ones(poly_degree + 1, dtype=np.float32))

	#plot initial guess
	x = np.linspace(-5, 5, 21)
	plt.plot(x, np.polyval(c_guess, x), 'y--', linewidth=2.0, label="Initial guess")

	#call optimizer to min error func
	result = spo.minimize(error_func, c_guess, args=(data,), method='SLSQP', options={'disp':True})
	return np.poly1d(result.x) #convert optimal result into a poly1d object

def do_main():
	#---Optimizing - Minimizing function - Parameterized Model---
		#using SciPy

	#use the spo minimizer
	x_guess = 0
	min_result = spo.minimize(f, x_guess, method='SLSQP', options={'disp':True}) #disp = true -> verbose mode
	print("Minima found at: X={}, Y={}".format(min_result.x,min_result.fun))

	#plot function and mark minima
	x_plot = np.linspace(0.5, 2.5, 21)
	y_plot = f(x_plot)
	plt.plot(x_plot, y_plot)
	plt.plot(min_result.x, min_result.fun, 'ro')
	plt.show()

	#fit a line to a given data set -> calculate the estimation function by minimizing the squared error
		#original line
	l_orig = np.float32([4,2])
	print("Original line: C0={}, C1={}".format(l_orig[0], l_orig[1]))
	x_orig = np.linspace(0, 10, 21)
	y_orig = l_orig[0] * x_orig + l_orig[1]
	plt.plot(x_orig, y_orig, 'b--', linewidth=2.0, label="Original line")
		#generate noisy data points
	noise_sigma = 3.0
	noise = np.random.normal(0, noise_sigma, y_orig.shape)
	data = np.asarray([x_orig, y_orig + noise]).T
	plt.plot(data[:,0], data[:,1], 'go', label="Data points")
	
		#fit a line
	fitted_line = fit_line(data, error)
	plt.plot(x_orig, fitted_line[0] * x_orig + fitted_line[1], 'r--', linewidth=2.0, label="Fitted line")

	
	#polynomial fit
	fitted_poly = fit_poly(data, error, 3)
	plt.plot(x_orig, np.polyval(fitted_poly, x_orig), 'g--', linewidth=2.0, label="Fitted poly")


	plt.show()

if __name__ == "__main__":
	do_main()