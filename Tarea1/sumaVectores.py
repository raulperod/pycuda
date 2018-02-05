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

    __global__ void sumaVectores(float *a, float *b, float *c) {
        int i = threadIdx.x + blockIdx.x * blockDim.x; 
        int j = threadIdx.y + blockIdx.y * blockDim.y; 
        int indice = j+i*M;

        if(indice < N*M){
            c[indice] = a[indice] + b[indice];
        }  
    }
""")

a = np.array([1 for i in range(N*M)]).astype(np.float32) 
b = np.array([1 for i in range(N*M)]).astype(np.float32)
c = np.array([0 for i in range(N*M)]).astype(np.float32)

suma_vectores = mod.get_function("sumaVectores")

suma_vectores(cuda.In(a), cuda.In(b), cuda.Out(c),
    block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), grid=(N//THREADS_PER_BLOCK, M//THREADS_PER_BLOCK, 1))

print( c.astype(np.int32) )