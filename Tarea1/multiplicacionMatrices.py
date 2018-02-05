import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 1024
M = 1024
L = 1024
THREADS_PER_BLOCK = 32

mod = SourceModule("""
    #define N 1024
    #define M 1024
    #define L 1024  

    __global__ void multiplicacionMatrices(float *a, float *b, float *c) {
        int i = threadIdx.x + blockIdx.x*blockDim.x; 
        int j = threadIdx.y + blockIdx.y*blockDim.y; 
            
        c[j+i*L] = 0;

        for(int k=0 ; k < M ; k++ ){
            c[j+i*L] += a[k+i*M] * b[j+k*L];
        }
    }
""")

a = np.array([1 for i in range(N*M)]).astype(np.float32) 
b = np.array([1 for i in range(M*L)]).astype(np.float32)
c = np.array([0 for i in range(N*L)]).astype(np.float32)

multiplicacion_matrices = mod.get_function("multiplicacionMatrices")

multiplicacion_matrices(cuda.In(a), cuda.In(b), cuda.Out(c),
    block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), grid=(N//THREADS_PER_BLOCK, L//THREADS_PER_BLOCK, 1))

print(c.reshape((N,L)))