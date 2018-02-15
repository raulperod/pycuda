import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

NC = 16
M = 16
N_BLOCKS = 16 # numero de bloques de la matriz
TPB = 8 # numero de hilos por bloque

mod = SourceModule("""
    const int N_BLOCKS = 16;
    const int NC = 16;
    const int M = 16;
    __global__ void blocks(float *matriz, float *u, float *resultado){
        int index1 = threadIdx.x + blockIdx.x*blockDim.x; 
        int index2 = threadIdx.y + blockIdx.y*blockDim.y;
        float suma = 0;
        for(int i=0 ; i < N_BLOCKS ; i++){ 
            suma = 0;   
            for(int l=0 ; l < NC ; l++){
                suma += matriz[l+index1*NC] * u[index2+M*(l+i*NC)];
            }
            resultado[index2 + M*(index1+i*NC)] = suma;
        }
    }
""")

def serial_blocks(A, B, C):
    for i in range(N_BLOCKS):
        for j in range(NC):
            for k in range(M):
                suma = 0
                for l in range(NC):
                    suma += A[l + j*NC] * B[k + M*(l+i*NC)]
                C[k + M*(j+i*NC)] = suma

def parallel_blocks(A, B, C):           
    blocks = mod.get_function("blocks")
    blocks(cuda.In(A), cuda.In(B), cuda.Out(C), 
        block=(TPB, TPB, 1), grid=(NC//TPB , M//TPB, 1))

def execute(A, B, C, parallel=True):
    parallel_blocks(A, B, C) if parallel else serial_blocks(A, B, C)
    print(C.reshape((NC*N_BLOCKS, M)))

if __name__ == '__main__':
    A = np.array([i+1 for i in range(NC*NC)]).astype(np.float32)
    B = np.array([i+1 for i in range(NC*N_BLOCKS*M)]).astype(np.float32)
    C = np.array([0 for i in range(NC*N_BLOCKS*M)]).astype(np.float32)
    execute(A, B, C, parallel=True)