import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

dx = dy = 0.1
D = 4.
Tcool, Thot = 300, 700
nx, ny = 128, 128
threads_per_block = 32
dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

mod = SourceModule("""
    const int nx = 128;
    const int ny = 128;
    const float D = 4.0;
    const float dx2 = 0.01;
    const float dy2 = 0.01;
    const float dt = dx2 * dy2 / (2 * D * (dx2 + dy2));

    __global__ void euler(float* u0, float* u){
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j = threadIdx.y + blockIdx.y*blockDim.y;

        if(i*(nx-i-1)*j*(ny-j-1)!=0){
            u[i+ny*j] = u0[i+ny*j] + (D * dt/dx2) * (u0[i-1+ny*j]+u0[i+1+ny*j]+u0[i+(j-1)*ny]+u0[i+(j+1)*ny]
                            -4.0*u0[i+ny*j]);
        }
    }
""")

if __name__ == "__main__":
    # obtener kernel
    euler = mod.get_function("euler")
    # Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
    r, cx, cy = 3, 5, 5 
    r2 = r**2

    a0 = [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) for j in range(ny) ]
    a = [ 0 for i in range(nx) for i in range(ny) ]

    u0 = np.array(a0).astype(np.float32)
    u = np.array(a).astype(np.float32)
    # Number of timesteps
    nsteps = 150
    # Output 4 figures at these timesteps
    mfig = [0, 50, 100, 149]
    fignum = 0
    fig = plt.figure()

    for m in range(nsteps):
        # kernel
        euler(cuda.In(u0), cuda.Out(u), 
            block=(threads_per_block, threads_per_block, 1), grid=(nx//threads_per_block, ny//threads_per_block, 1) )
        u0 = u.copy()
        if m in mfig:
            fignum += 1
            print(m, fignum)
            ax = fig.add_subplot(220 + fignum)
            im = ax.imshow(u.reshape((nx,ny)).copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
            ax.set_axis_off()
            ax.set_title('{:.1f} ms'.format(m*dt*1000))

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('graph.png')	
