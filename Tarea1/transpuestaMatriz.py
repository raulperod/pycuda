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

    __global__ void transpuestaMatriz(float *a, float *b) {
        int x = threadIdx.x + blockIdx.x * blockDim.x; // 0 1023
        int y = threadIdx.y + blockIdx.y * blockDim.y; // 0 1023
        int indice_a = y+x*M;
        int indice_b = x+y*N;
        if(indice_a < N*M){
            b[indice_b] = a[indice_a];
        }  
    }
""")

a = np.array([i for i in range(N*M)]).astype(np.float32) 
b = np.array([0 for i in range(M*N)]).astype(np.float32)
transpuesta_matriz = mod.get_function("transpuestaMatriz")

transpuesta_matriz(cuda.In(a), cuda.Out(b), block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), 
    grid=(M//THREADS_PER_BLOCK, N//THREADS_PER_BLOCK, 1))

print( b.astype(np.int32).reshape((M,N)) )