import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N, M, L = 32, 32, 32
threads_per_block = 8

mod = SourceModule("""
    const int L = 32;
    __global__ void CRS(float *val, int *col_ind, int *row_ptr, float *u, float *resultado){
        int i = threadIdx.x + blockIdx.x*blockDim.x; 
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        float suma = 0.0;
        for(int k = row_ptr[i]-1; k < row_ptr[i+1]-1; k++){
            suma += val[k] * u[j + ( (col_ind[k]-1) * L) ];
        }
        resultado[j+i*L] = suma;
    }
""")

def initialize_vector(length, value=0):
    return np.array([value for i in range(length)]).astype(np.float32) 

def count_not_zeros(matriz):
    return sum([ (1 if matriz[j+i*M] != 0 else 0) for i in range(N) for j in range(M)]) 

def get_crs_matrices(matriz, nnz):
    # inicializo val, col_ind y row_ptr
    val = initialize_vector(nnz)
    col_ind = initialize_vector(nnz)
    row_ptr = initialize_vector(N+1)
    # obtener val y col_ind
    contador = 0
    for i in range(N):
        for j in range(M):
            if matriz[j+i*M] != 0:
                val[contador], col_ind[contador] = matriz[j+i*M], j+1
                contador += 1
    # obtengo row_ptr
    contador, indice, fila = 0, 0, -1
    for i in range(N):
        for j in range(M):
            if matriz[j+i*M] != 0:
                contador+=1
                if i != fila:
                    row_ptr[indice], fila = contador, i
                    indice += 1
    row_ptr[N] = nnz+1
    return np.array(val).astype(np.float32), np.array(col_ind).astype(np.int32), np.array(row_ptr).astype(np.int32)

if __name__ == '__main__':
    A = [ (1 if j >= i else 0) for i in range(N) for j in range(M) ]
    B = initialize_vector(M*L, value=1)
    C = initialize_vector(N*L)
    nnz = count_not_zeros(A)
    val, col_ind, row_ptr = get_crs_matrices(A, nnz)
    # get kernel
    CRS = mod.get_function("CRS")
    # mult
    CRS(cuda.In(val), cuda.In(col_ind), cuda.In(row_ptr), cuda.In(B), cuda.Out(C), 
        block=(threads_per_block, threads_per_block, 1), grid=(N//threads_per_block, L//threads_per_block, 1))
    
    print(C.reshape((N,L)))