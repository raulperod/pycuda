import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 1024
M = 1024
THREADS_PER_BLOCK = 32

mod = SourceModule("""
    #define N 1024
    #define M 1024

    __global__ void sumaMatrices(float *a, float *b, float *c) {
        int x = threadIdx.x + blockIdx.x * blockDim.x; // 0 1023
        int y = threadIdx.y + blockIdx.y * blockDim.y; // 0 1023
        int indice = y+x*M;
        if(indice < N*M){
            c[indice] = a[indice] + b[indice];
        }  
    }
""")

a = np.array([1 for i in range(N*M)]).astype(np.float32) 
b = np.array([1 for i in range(N*M)]).astype(np.float32)
c = np.array([0 for i in range(N*M)]).astype(np.float32)

suma_matrices = mod.get_function("sumaMatrices")

suma_matrices(cuda.In(a), cuda.In(b), cuda.Out(c),
    block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), grid=(N//THREADS_PER_BLOCK, M//THREADS_PER_BLOCK, 1))

print( c.astype(np.int32).reshape((N,M)) )