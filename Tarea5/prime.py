import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

TPB = 16
N = 1024
N2 = N**2

mod = SourceModule("""
    #include <stdio.h>
    #define N 1024
    #define N2 1048576

    __global__ void prime(int *array, int *prime_mask) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(array[i] > 1){
            if(prime_mask[i] == 0){
                int inc = array[i];
                for(int j = i+inc ; j < N2 ; j += inc ){
                    prime_mask[j] = 1;
                }            
            }
        }
    }

""")

def p_prime(array, prime_mask):
    for i in range(1, N2):
        if prime_mask[i] == 0:
            print("El numero {} es primo".format(array[i]))
            
def serial():
    # initialize arrays  
    array = np.array([i+1 for i in range(N2)]).astype(np.int32)
    prime_mask = np.zeros_like(array).astype(np.int32)
    # run kernel
    for i in range(1, N+1):
        if prime_mask[i] == 0:
            inc = array[i];
            for j in range(i+inc, N2, inc):
                prime_mask[j] = 1
    # print prime numbers
    p_prime(array, prime_mask)

def parallel():
    # get kernel
    prime_kernel = mod.get_function("prime")    
    # initialize arrays  
    array = np.array([i+1 for i in range(N2)]).astype(np.int32)
    prime_mask = np.zeros_like(array).astype(np.int32)
    # run kernel
    prime_kernel(cuda.In(array), cuda.InOut(prime_mask),
        block=( TPB, 1, 1), grid=( N//TPB, 1, 1) )
    p_prime(array, prime_mask)

if __name__ == '__main__':
    parallel()
    #serial()
