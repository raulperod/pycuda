import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dx = dy = dz = 0.1
D = 4.
Tcool, Thot = 300, 700
nx, ny, nz = 32, 32, 32
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

def execute(u0, u, nsteps=100):
    for _ in range(nsteps):
        u0, u = do_timestep(u0, u)
    
    print(u0.reshape((nx, ny, nz)))
    
if __name__ == "__main__":
    r, cx, cy, cz = 2, 3, 3, 3 
    r2 = r**2

    a0 = [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2 + (k*dz-cz)**2) < r2 else Tcool) for i in range(nx) for j in range(ny) for k in range(nz) ]
    a = [ 0 for i in range(nx) for i in range(ny) for k in range(nz) ]

    u0 = np.array(a0).astype(np.float32)
    u = np.array(a).astype(np.float32)
    
    execute(u0, u)