import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

TPB = 16
N = 16
N2 = N**2

mod = SourceModule("""
    #define N 16
    #define N2 256

    __global__ void reverse(float *array, float *reverse_array) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int index = j+i*N; // indice en el array
        
        if (index < N2) {
            int new_index = N2-1-index;
            reverse_array[new_index] = array[index];
        }
    }

""")

def parallel():
    # get kernel
    reverse_kernel = mod.get_function("reverse")    
    # initialize arrays  
    array = np.array([i for i in range(N2)]).astype(np.float32)
    reverse_array = np.zeros_like(array).astype(np.float32)
    # run kernel
    reverse_kernel(cuda.In(array), cuda.InOut(reverse_array),
        block=(TPB, TPB, 1), grid=( N//TPB, N//TPB, 1) )
    print(reverse_array)

if __name__ == '__main__':
    parallel()
