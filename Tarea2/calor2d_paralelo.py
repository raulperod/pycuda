import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def animate(data, im):
    im.set_data(data)

def step(u0, u, nsteps):
    euler = mod.get_function("euler")
    for i in range(nsteps):
        euler(cuda.In(u0), cuda.Out(u), 
            block=(threads_per_block, threads_per_block, 1), grid=(nx//threads_per_block, ny//threads_per_block, 1) )
        u0 = u.copy()
        yield u0.reshape((nx,ny))

if __name__ == "__main__":
    # condiciones iniciales
    r, cx, cy = 4, 7, 6 
    r2 = r**2

    a0 = [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) for j in range(ny) ]
    a = [ 0 for i in range(nx) for i in range(ny) ]

    u0 = np.array(a0).astype(np.float32)
    u = np.array(a).astype(np.float32)
    # Numero de iteraciones
    nsteps = 150
    # Configuracion
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    im = ax.imshow(u.reshape((nx,ny)), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
    fig.colorbar(im, cax=cbar_ax)
    ax.set_axis_off()
    ax.set_title("Mapa de Calor")

    ani = FuncAnimation( fig, animate, step(u0, u, nsteps), 
        interval=1, save_count=nsteps, repeat=True,repeat_delay=1, fargs=(im,))
    ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
