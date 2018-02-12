import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dx = dy = 0.1
D = 4.
Tcool, Thot = 300, 700
nx, ny = 128, 128
dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

def do_timestep(u0, u):
    for i in range(nx):
        for j in range(ny):
            if i*(nx-i-1)*j*(ny-j-1) != 0:
                u[i][j] = u0[i][j] +  (D * dt/dx2) * ( u0[i-1][j]+u0[i+1][j]+u0[i][j-1]+u0[i][j+1]
                                - 4.0*u0[i][j] )
            
    u0 = u.copy()
    return u0, u

def animate(data, im):
    im.set_data(data)

def step(u0,u):
    for i in range(1000):
        u0, data = do_timestep(u0, u)
        yield data

if __name__ == "__main__":
    # condiciones iniciales
    r, cx, cy = 4, 7, 6 
    r2 = r**2

    u0 = np.array([ [ (Thot if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2 else Tcool) for i in range(nx) ] for j in range(ny) ])
    u = np.array([ [0 for i in range(nx)] for i in range(ny) ]) 
    # Numero de iteraciones
    nsteps = 150
    # Configuracion
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    im = ax.imshow(u, cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
    fig.colorbar(im, cax=cbar_ax)
    ax.set_axis_off()
    ax.set_title("Mapa de Calor")

    ani = animation.FuncAnimation( fig, animate, step(u0,u), 
        interval=1, repeat=True,repeat_delay=1, fargs=(im,))
    ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
