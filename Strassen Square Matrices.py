"""Multiplying square matrices via Strassen's algorithm."""

import numpy as np

def strassen(X, Y):
    """Return the product of two square matrices.

    This function computes the product of two square matrices via
    Volker Strassen's 1969 sub O(n^3) algorithm. 

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

    if n == 1:
        return X * Y

    if n % 2 == 1: 
        #If the dimension is odd make it even by appending
        #a row and column of zeros
        X = np.pad(X, (0, 1), mode= 'constant')
        Y = np.pad(Y, (0, 1), mode= 'constant')

    m = (n + 1) // 2 #Equivalent to ceiling(n/2)

    #Generate block matrices to be used for block products
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

    #Initialize and fill in the computations
    result = np.zeros((2 * m, 2 * m), dtype = np.int32)
    
    result[:m, :m] = p5 + p4 - p2 + p6
    result[:m, m:] = p1 + p2
    result[m:, :m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    #Return only the contributions of the original matrices
    return result[:n, :n]

if __name__ == '__main__':
    A = np.random.randint(100, size = (5, 5))
    B = np.random.randint(100, size = (5, 5))
    print(f'A = \n {A}')
    print(f'B = \n {B}')
    print(f'A * B = \n {strassen(A, B)}')
