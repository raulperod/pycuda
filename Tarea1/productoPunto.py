import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 1024
THREADS_PER_BLOCK = 32

mod = SourceModule("""
    #define THREADS_PER_BLOCK 32

    __global__ void productoPunto( float *a, float *b, float *c ) {
        __shared__ float temp[THREADS_PER_BLOCK];
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        temp[threadIdx.x] = a[index] * b[index];
        __syncthreads();
        if( 0 == threadIdx.x ) {
            float sum = 0;
            for( int i = 0; i < THREADS_PER_BLOCK; i++ ){
                sum += temp[i];
            }
            atomicAdd( c , sum );
        }
    }
""")

a = np.array([1 for i in range(N)]).astype(np.float32) 
b = np.array([1 for i in range(N)]).astype(np.float32)
c = np.array([0]).astype(np.float32)

producto_punto = mod.get_function("productoPunto")

producto_punto(cuda.In(a), cuda.In(b), cuda.Out(c), 
    block=(THREADS_PER_BLOCK, 1, 1), grid=(N//THREADS_PER_BLOCK, 1, 1) )

print( c )
