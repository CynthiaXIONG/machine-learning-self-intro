import gdax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def do_main():

	#https://docs.scipy.org/doc/numpy/user/basics.creation.html
	print np.array([(2,3,4), (5,6,7)])
	print np.empty((5,4))
	print np.ones((5,4))

	#https://docs.scipy.org/doc/numpy/reference/routines.random.html
	print np.random.random((5,4))
	print np.random.rand(5,4)
	print np.random.normal(50, 10, size=(2,3)) #50 mean, and 10 s.d.
	print np.random.randint(0, 10, size=(5,4)) #low 0, high 10 (not inclusive)

	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
	a = np.random.random((5,4))
	print len(a.shape)
	print a.shape[0]
	print a.shape[1]
	print a.size
	print a.dtype
	#https://docs.scipy.org/doc/numpy/reference/routines.math.html
	print a.sum()
	print a.min(axis=0) #min of each columnm
	print a.max(axis=1) #max of each row
	print a.mean() #mean of all elements
	#https://docs.scipy.org/doc/numpy/reference/routines.sort.html
	print a.argmax()

	#https://docs.python.org/2/library/time.html#time.time
	t1 = time.time()
	print("printing")
	t2 = time.time()
	print("Time elapsed:%ss" % str(t2-t1))

	#slicing
	#https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
	#. ":" -> range, e.g: nd[0:3, 0:3]  -. selects first 3x3 elements (last is not included)
	#. nd[:,1] -> all row elements in the '1' column
	#. nd[-1, -2] -> last row, seconde last col
	print a[:, 0:3:2] #returns elements in columns 0-3, but in "steps" of 2 rows (for every two rows)
	b = np.random.rand(5)
	print b
	indices = ([1,1,2,3])
	print b[indices]
	#indexing using bitarrays/bitmasks
	c = np.random.randint(0,30, size=(2,9))
	print c
	mean = c.mean()
	print mean
	print c[c<mean] #select all values from that condition (less than mean)
	c[c<mean] = mean
	print c

	#https://docs.scipy.org/doc/numpy/reference/routines.math.html#arithmetic-operations
	#arithmetic operations are applied bitwise



if __name__ == "__main__":
	do_main()