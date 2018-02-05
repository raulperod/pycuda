import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from numpy.random import random as azar

N = 4096
M = 4096
THREADS_PER_BLOCK = 32

mod = SourceModule("""
    #define N 1024
    #define M 1024

    __global__ void aproximarPi(float *a, float *b, float *c) {
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        int index = j + i*M;
        
        if( (a[index] * a[index] + b[index] * b[index]) <= 1.0f){
            atomicAdd(c, 1);
        }
    }
""")

a = np.array(azar(N*M)).astype(np.float32) 
b = np.array(azar(N*M)).astype(np.float32)
c = np.array([0]).astype(np.float32)

aprox_pi = mod.get_function("aproximarPi")

aprox_pi(cuda.In(a), cuda.In(b), cuda.Out(c), 
    block=(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1), grid=(N//THREADS_PER_BLOCK, M//THREADS_PER_BLOCK, 1) )

pi = ((4.0 * c[0]) / (N*M))
print("Pi: {}".format(pi))