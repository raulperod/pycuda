import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

cNX = 256
cNY = 256
n_steps = 1000
GAMA = 0.001
cnthx = 32
cnthy = 32
dt = 0.001
h = 1.0/cNX 
h2 = h*h

# para los hilos
N = 1024
M = 1024
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

if __name__ == '__main__':
    # arreglos iniciales    
    u0 = np.array([ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) for j in range(ny) ]).astype(np.float32)
    u = np.array([0 for i in range(nx*ny)]).astype(np.float32)
    # Number of timesteps
    nsteps = 100
    # Output 4 figures at these timesteps
    mfig = [0, 10, 50, 99]
    fignum = 0
    fig = plt.figure()

    for m in range(nsteps):
        u0, u = do_timestep(u0, u)
        if m in mfig:
            fignum += 1
            print(m, fignum)
            ax = fig.add_subplot(220 + fignum)
            im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
            ax.set_axis_off()
            ax.set_title('{:.1f} ms'.format(m*dt*1000))
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    