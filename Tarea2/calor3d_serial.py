import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dx = dy = dz = 0.1
D = 4.
Tcool, Thot = 300, 700
nx, ny, nz = 128, 128, 128
dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
dt = dx2 * dy2 * dz2 / (3 * D * (dx2 + dy2 + dz2))

def do_timestep(u0, u):
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i*(nx-i-1)*j*(ny-j-1)*k*(nz-k-1) != 0:
                    u[k+nz*(j+i*ny)] = u0[k+nz*(j+i*ny)] + (D * dt/dx2) * (u0[k+nz*(j+(i+1)*ny)] 
                        + u0[k+nz*(j+(i-1)*ny)] + u0[k+nz*((j-1)+i*ny)] + u0[k+nz*((j+1)+i*ny)]
                        + u0[(k+1)+nz*(j+i*ny)] + u0[(k-1)+nz*(j+i*ny)] - 6.0*u0[k+nz*(j+i*ny)])

    u0 = u.copy()
    return u0, u

def animate(data, im):
    im.set_data(data)

def step(u0, u, nsteps):
    for i in range(nsteps):
        u0, u = do_timestep(u0, u)
        yield get_plane(u, z=32)

def get_plane(u, pos=0, fix='z'):
    return ([ [u[pos+nz*(j+i*ny)] for j in range(ny) ] for i in range(nx) ] if fix is 'z'
        else [ [u[k+nz*(pos+i*ny)] for k in range(nz) ] for i in range(nx) ] if fix is 'y'
        else [ [u[k+nz*(j+pos*ny)] for k in range(nz) ] for j in range(ny) ])

def execute(u0, u, nsteps=10):
    # Config
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    im = ax.imshow(get_plane(u, z=32), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
    fig.colorbar(im, cax=cbar_ax)
    ax.set_axis_off()
    ax.set_title("Mapa de Calor")

    ani = FuncAnimation( fig, animate, step(u0, u, nsteps), 
        interval=1, save_count=nsteps, repeat=True,repeat_delay=1, fargs=(im,))
    ani.save('calor3d.mp4', fps=20, writer="ffmpeg", codec="libx264")
    
if __name__ == "__main__":
    r, cx, cy, cz = 3, 5, 5, 5 
    r2 = r**2

    a0 = [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2 + (k*dz-cz)**2) < r2 else Tcool) for i in range(nx) for j in range(ny) for k in range(nz) ]
    a = [ 0 for i in range(nx) for i in range(ny) for k in range(nz) ]

    u0 = np.array(a0).astype(np.float32)
    u = np.array(a).astype(np.float32)
    
    execute(u0, u)