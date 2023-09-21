"""Implementation of Strassen's algorithm built upon numpy's matmul."""

import numpy as np

def strassen(X, Y):
    """Return the product of two square matrices.

    This function takes advantage of the fact that Strassen is asymptotically
    fast but has a large leading coefficient making more standard methods
    faster for smaller matrices. Because Strassen is divide and conquer,
    a vanilla implementation of it will leave the function reverting
    to matrix multiplication which is suboptimal, even if the dimension
    is high enough for the empirical trade off.

    This function trades off Strassen for NumPy's matmul when the dimension
    of the matrices is under 64. This number was determined by running
    tests for optimal crossover times. For more information see the readme
    file.
    
    Parameters
    ----------
    X, Y : numpy array
        Matrices being multiplied.

    Returns
    -------
    result : numpy array
        The product of the two matrices.
    
    """
    n = X.shape[0]

    if n < 64:
        #Strassen benefits from swithiching to np.matmul
        #for matrices of dimension k by k where k < 64
        return np.matmul(X, Y)

    if n % 2 == 1:
        #If the dimension is odd make it even by appending
        #a row and column of zeros
        X = np.pad(X, (0, 1), mode= 'constant')
        Y = np.pad(Y, (0, 1), mode= 'constant')

    m = (n + 1) // 2 #Equivalent to ceiling(n/2)

    #Generate the block matrices
    A, B = X[:m, :m], X[:m, m:]
    C, D = X[m:, :m], X[m:, m:]

    E, F = Y[:m, :m], Y[:m, m:]
    G, H = Y[m:, :m], Y[m:, m:]

    #Compute the required block products
    p1 = strassen(A, F - H)
    p2 = strassen(A + B, H)
    p3 = strassen(C + D, E)
    p4 = strassen(D, G - E)
    p5 = strassen(A + D, E + H)
    p6 = strassen(B - D, G + H)
    p7 = strassen(A - C, E + F)

    #Initialize and fill in our computations
    result = np.zeros((2 * m, 2 * m), dtype = np.int32)
    
    result[:m, :m] = p5 + p4 - p2 + p6
    result[:m, m:] = p1 + p2
    result[m:, :m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    #Return only the contributions of the original matrices
    return result[:n, :n]


if __name__ == '__main__':
    #This will take a second or two
    A = np.random.randint(100, size = (2500, 2500))
    B = np.random.randint(100, size = (2500, 2500))
    print('Multiplying two 2500 x 2500 matrices')
    strassen(A, B)
    print('Multiplication completed.')
