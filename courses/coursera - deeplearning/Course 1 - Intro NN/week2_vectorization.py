import numpy as np
import time

if __name__ == "__main__":

    m = 1000000
    a = np.random.rand(m)
    b = np.random.rand(m)

    start_t = time.time()
    c = np.dot(a,b)
    end_t = time.time()
    print("Vectorized Version:" + str((end_t-start_t)))

    c = 0
    start_t = time.time()
    for i in range(m):
        c += a[i] * b[i]
    end_t = time.time()
    print("Foot-loop Version:" + str((end_t-start_t)))

    #------------------------------------------------------
    start_t = time.time()
    u = np.random.rand(m)
    np.log(u)
    np.abs(u)
    np.maximum(u, 0)
    u**2
    1/u
    end_t = time.time()
    print("SIMD operations:" + str((end_t-start_t)))

    #------------------------------------------------------
    #Broadcasting https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    #calc rel values by column
    A = np.array([[56.0, 0.0, 4.4, 68.0],
                  [1.2, 104.0, 52.0, 8.0],
                  [1.9, 135.0, 99.0, 0.9]])
    print(A)
    sum_col = A.sum(axis=0)
    rel_A = A / sum_col.reshape(1,4) #sum_col , use reshape just to be sure...very cheap  <- this mult is only possible because python expands (broacasts), creating a matrix from sum_col
    print(rel_A)

        #careful...always explicit both dimensions
    a = np.random.rand(5)
    print (a == a.T)
    print (np.dot(a, a.T))

    b = np.random.rand(5, 1) # <-column vector   (1, 5) -> row vector
    print (b == b.T)
    print (np.dot(b, b.T))
    assert(b.shape == (5, 1))

    A = np.random.randn(4, 3)
    B = np.sum(A, axis=1, keepdims = True)
    print(B.shape)




