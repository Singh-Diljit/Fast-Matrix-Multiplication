# Strassen and np.matmul

## Strassen's Algorithm

Strassen's algorithm is a divide and conquer algorithm for multiplying two square matrices of dimension n by n where n is a power of 2. The runtime is O(n^log_2(7)) which beats the naive O(n^3) approach but loses to more recent record breaking matrix multiplication algorithms. For reference, log_2(7) is approximately 2.80735492206 while the record exponent as of 2020 is 2.3728596 (Josh Alman and Virginia Williams). However many sub-O(n^log_2(7)) matrix multiplication algorithms come with astronomical constants making them impractical for use even on massive matrices. In general, Strassen can be easily implemented to beat optimized traditional multiplication methods at reasonably sized dimensions (complexity of implementation, compiler, and computer play a big role in determining crossover dimension). 

A common way to expand the domain of Strassen from 2^m x 2^m matrices to square or rectangular matrices is by 'padding' your matrix with rows and columns of zeros until the result is a square matrix with power of 2 rows and columns. Unexpectedly this comes with overhead but it is easy to implement, and as we will see, still pretty fast. Exciting implementations aimed at optimizing both speed and memory exist but our implementation uses the standard ‘padding’ method.

For reference, the crossover between optimized traditional multiplication methods and Strassen varies based on a number of factors but with clever enough implementations Strassen can start to outperform starting around 500 x 500 matrices.

[1] A preprint of Josh Alman and Virginia Williams’ 2020 paper can be found here: https://arxiv.org/abs/2010.05846. As of 2023 this is still the best exponent among peer-reviewed papers though two preprints claim to beat this.

[2] An optimized implementation of Strassen: https://www.researchgate.net/publication/315365781_Strassen%27s_Algorithm_Reloaded

## NumPy's matmul vs Strassen

NumPy vector and matrix functions are based on BLAS, a highly optimized matrix library. The implementation and optimizations used are very interesting but not quite in the scope of this document. For smaller matrices NumPy's matmul function is far faster than our implementation of Strassen, but surprisingly for larger matrices our Strassen starts to dominate matmul. The actual trade-off starts to occur around 200 x 200 but it becomes more noticeable at higher dimensions. This speed increase is not free, our implementation of Strassen takes substantially more memory and is not as numerically stable as matmul. But it is still very interesting to see the difference in speeds.

![image](https://github.com/Singh-Diljit/Fast-Matrix-Multiplication/blob/main/matmul%20vs%20strassen.png)

## Strassen with matmul

Because Strassen is a divide and conquer algorithm, having a faster way to do small chunks of the problem can greatly increase total speed. The goal of this program was to combine Strassen and NumPy's matmul function so large matrices are broken up via Strassen and small chunks are multiplied with matmul. The first question to arise was what dimension to swap Strassen with matmul. By doing a grid-search of crossover dimensions and refining the search when it seemed to improve speeds, I arrived at 64 x 64 to be the point matmul would help the most. Included is a small comparison of 64 vs min_time(k=10 and k=105) (though many other values were tested and can be using the benchmark file).

![image](https://github.com/Singh-Diljit/Fast-Matrix-Multiplication/blob/main/Comparing%20Cross%20Over%20Values.png)

The methodology was to randomly generate two K x K matrices, time how long it took to multiply the matrices, repeat this for 100 trials, and save the average time. The dimension of matrices, K, ranged from 0 to 4096. To keep runtimes low the matrices were populated with integers between 0 and 100.

It is interesting to see just how much faster this hybrid Strassen is compared to either Strassen or matmul, but as I mentioned before there are a few reasons one would expect this - namely with our implementation of this hybrid it is space, numerical stability (due to Strassen overall), and the general robustness of matmul vs this particular implementation of Strassen.

![image](https://github.com/Singh-Diljit/Fast-Matrix-Multiplication/blob/main/Matmul%20vs%20Hybrid%20Runtimes.png)

