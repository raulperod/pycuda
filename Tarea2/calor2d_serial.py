import numpy as np
import matplotlib.pyplot as plt

dx = dy = 0.1
D = 4.

Tcool, Thot = 300, 700

nx, ny = 128, 128

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    for i in range(nx):
        for j in range(ny):
            if i*(nx-i-1)*j*(ny-j-1) != 0:
                u[i][j] = u0[i][j] +  (D * dt/dx2) * ( u0[i-1][j]+u0[i+1][j]+u0[i][j-1]+u0[i][j+1]
                                - 4.0*u0[i][j] )
            
    u0 = u.copy()
    return u0, u

if __name__ == "__main__":

    # Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
    r, cx, cy = 3, 5, 5 
    r2 = r**2

    u0 = np.array([ [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) ] for j in range(ny) ])
    u = np.array([ [0 for i in range(nx)] for i in range(ny) ]) 
    # Number of timesteps
    nsteps = 150
    # Output 4 figures at these timesteps
    mfig = [0, 50, 100, 149]
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
    plt.savefig('graph.png')	
