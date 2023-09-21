"""Compare np.matmul vs the hybrid Strassen/Matmul algorithm."""

from matplotlib import pyplot as plt
import numpy as np
import time

def strassen_hybrid(X, Y, k):
    """Multiply square matrices swapping Strassen with matmul.

    This function computes the product of two square matrices via
    Strassen and np.matmul, swapping Strassen with np.matmul when the
    dimension of the matrices is less than a given parameter, "k".

    Parameters
    ----------
    X, Y : numpy array
        Matrices being multiplied.
    k : int
        Dimension at which np.matmul takes over.
    
    Returns
    -------
    result : numpy array
        The product of the two matrices.

    """
    n = X.shape[0]

    if n < 64:
        #Switch to np.matmul for matrices of dimension
        #k by k where k < 64.
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
    p1 = strassen_hybrid(A, F - H, k)
    p2 = strassen_hybrid(A + B, H, k)
    p3 = strassen_hybrid(C + D, E, k)
    p4 = strassen_hybrid(D, G - E, k)
    p5 = strassen_hybrid(A + D, E + H, k)
    p6 = strassen_hybrid(B - D, G + H, k)
    p7 = strassen_hybrid(A - C, E + F, k)

    #Initialize and fill in our computations
    result = np.zeros((2 * m, 2 * m), dtype = np.int32)
    
    result[:m, :m] = p5 + p4 - p2 + p6
    result[:m, m:] = p1 + p2
    result[m:, :m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    #Return only the contributions of the original matrices
    return result[:n, :n]

def time_hybrid(A, B, k):
    """Time taken by Strassen hybrid to multiply two matrices."""

    start = time.time()
    strassen_hybrid(A, B, k)
    end = time.time()
    
    return end - start

def time_matmul(A, B):
    """Time taken by np.matmul to multiply two matrices."""
    
    start = time.time()
    np.matmul(A, B)
    end = time.time()
    
    return end - start

def get_matrices(n):
    """Generate two matrices of dimension n by n each.

    Parameters
    ----------
    n : int
        Desired matrix dimension.

    Returns
    -------
    A, B : numpy array
        Matrices of desired dimension.
    
    """
    A = np.random.randint(100, size = (n, n))
    B = np.random.randint(100, size = (n, n))
    
    return A, B

def graphing_function(data, data_titles, axis_names, x_axis, graph_title):
    """Produce side by side bar-graphs.

    Parameters
    ----------
    data: list (of lists)
        List of data to be graphed, each entry is one data-set.
    data_titles : str, list of str
        Labels corresponding to each dataset in 'data'.
    axis_names : list of str
        Labels of the x and y axis (resp.)
    x_axis : list
        The labelled points of the x-axis
    graph_title : str
        Title of graph
    
    Returns
    -------
    Prints graph
    
    """
    barWidth = len(x_axis)    
    #Show bar graphs side by side
    for i in range(len(data)):
        cl_i = 'blue' if (i % 2 == 0) else 'red'
        plt.bar([x + i*barWidth for x in x_axis],
                data[i],
                width = barWidth,
                color = cl_i,
                edgecolor = 'black',
                label = data_titles[i])
 
    #Label Graph
    plt.xlabel(f'{axis_names[0]}')
    plt.ylabel(f'{axis_names[1]}')
    plt.legend()
    plt.title(graph_title)
 
    #Show graphic
    plt.show()

def strassen_vs_matmul(max_dim, trials, number_comparisons):
    """Graph Strassen and np.matmul times.

    Parameters
    ----------
    max_dim : int
        The maximum matrix dimension.
    trials : int
        Number of matrix multiplication to perform in any given dimension.
    number_comparisons : int
        Number of dimensions Strassen and matmul are compared.
    
    Returns
    -------
    Prints graph

    """
    matmul_avgs = []
    strassen_avgs = []
    step_size = (max_dim + 1) // number_comparisons
    
    for dim in range(0, max_dim + 1, step_size):
        
        total_mat, total_stra = 0, 0
        for _ in range(trials):
            #Generate matrices, multiply (both ways) and sum time taken
            A, B = get_matrices(dim)
            total_mat += time_matmul(A, B)
            total_stra += time_hybrid(A, B, 1)
            
        #Compute the average runtime for each method in a given dimension
        avg_mat = total_mat / trials
        avg_stra = total_stra / trials
        
        #Save both averages
        matmul_avgs.append(avg_mat)
        strassen_avgs.append(avg_stra)

    #Graph results
    graphing_function([matmul_avgs, strassen_avgs],
                      ['Matmul', 'Strassen'],
                      ['Dimension', 'Time (sec)'],
                      list(range(0, max_dim + 1, step_size)),
                      'Matmul vs Strassen Runtimes')

def hybrid_vs_matmul(max_dim, trials, number_comparisons):
    """Graph Strassen hybrid (switching at 64) and np.matmul times.

    Parameters
    ----------
    max_dim : int
        The maximum matrix dimension.
    trials : int
        Number of matrix multiplication to perform in any given dimension.
    number_comparisons : int
        Number of dimensions Strassen-hybrid and matmul are compared.
    
    Returns
    -------
    Prints graph

    """
    matmul_avgs = []
    hybrid_avgs = []
    step_size = (max_dim + 1) // number_comparisons
    
    for dim in range(0, max_dim + 1, step_size):
        
        total_mat, total_stra = 0, 0
        for _ in range(trials):
            #Generate matrices, multiply (both ways) and sum time taken
            A, B = get_matrices(dim)
            total_mat += time_matmul(A, B)
            total_stra += time_hybrid(A, B, 64) #64 is point of crossover
            
        #Compute the average runtime for each method in a given dimension
        avg_mat = total_mat / trials
        avg_stra = total_stra / trials
        
        #Save both averages
        matmul_avgs.append(avg_mat)
        hybrid_avgs.append(avg_stra)

    #Graph results
    graphing_function([matmul_avgs, hybrid_avgs],
                      ['Matmul', 'Hybrid'],
                      ['Dimension', 'Time (sec)'],
                      list(range(0, max_dim + 1, step_size)),
                      'Matmul vs Hybrid Runtimes')

def compare_crossovers(cross_overs, max_dim, trials, number_comparisons):
    """Graph different time to multiply given different crossovers.

    At each dimension this function will compare time taken to multiply
    matrices at dimension k = 64 and also all cross_over values. At any given
    dimension the graph compares k = 64 vs the min cross_over value with respect
    to average time taken.

    Parameters
    ----------
    cross_overs : list
        Crossover values to compare.
    max_dim : int
        Maximum dimension of comparisons, should be over 750.
    trials : int
        Number of matrix multiplication to perform in any given dimension.
    number_comparisons : int
        Number of dimensions Strassen-hybrid and matmul are compared.
    
    Returns
    -------
    Prints graph

    """
    best_alt_avgs = []
    avgs_64 = []
    step_size = (max_dim + 1) // number_comparisons

    lower_bound = 750 #For 'small' matrices every method is 'fast enough'.
    for dim in range(lower_bound, max_dim + 1, step_size):
        total_alt = [0] * len(cross_overs)
        total_64 = 0
        
        for _ in range(trials):
            #Generate matrices, multiply and sum time taken with each crossover
            A, B = get_matrices(dim)
            for i, k in enumerate(cross_overs):
                total_alt[i] += time_hybrid(A, B, k)
            total_64 += time_hybrid(A, B, 64)
            
        min_alt_avg = min(total_alt) / trials #Choose the best crossover
        av_64 = total_64 / trials
        
        #Save both averages
        best_alt_avgs.append(min_alt_avg)
        avgs_64.append(av_64)

    #Graph results
    graphing_function([best_alt_avgs, avgs_64],
                      ['Best of Candidates', '64'],
                      ['Dimension', 'Time (sec)'],
                      list(range(lower_bound, max_dim + 1, step_size)),
                      'Comparing Cross Over Values')

if __name__ == "__main__":
    
    #Compare various multiplication methods
    strassen_vs_matmul(750, 5, 10)
    hybrid_vs_matmul(750, 5, 10)
    
    #Compare 2 different cross-over values to 64
    #note: this will take a few minutes
    candidates = [10, 130]
    compare_crossovers(candidates, 1250, 5, 20)


