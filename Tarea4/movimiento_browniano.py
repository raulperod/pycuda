import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.curandom import rand as curand, XORWOWRandomNumberGenerator
import numpy as np
from math import sqrt
from pylab import plot, show, grid, xlabel, ylabel, savefig
import matplotlib.pyplot as plt

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = generar_numeros_normal(size=x0.shape + (n,), desv=delta*sqrt(dt) )
    # if `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    return out

def generar_numeros_normal(size, desv):
    n, m = size
    generador = XORWOWRandomNumberGenerator()
    array = generador.gen_normal(shape=n*m, dtype=np.float32) 
    array = array.reshape((n, m)).get()  
    return array

if __name__ == "__main__":
    # The Wiener process parameter.
    delta = 2
    # Total time.
    T = 10.0
    # Number of steps.
    N = 1000
    # Time step size
    dt = T/N
    # Number of realizations to generate.
    m = 1000
    # Create an
    x = np.empty((m,N+1))
    # Initial values of x.
    x[:, 0] = 50
   
    brownian(x[:,0], N, dt, delta, out=x[:,1:])
    # brownian
    t = np.linspace(0.0, N*dt, N+1)
    for k in range(m):
        plot(t, x[k])
    xlabel('t', fontsize=16)
    ylabel('x', fontsize=16)
    grid(True)
    savefig("brownian.png")
    plt.close()
    # histograma
    subplots = [1, 2, 3, 4]
    histograms = [5, 100, 500, -1]
    
    fig = plt.figure()
    
    for i, j in zip(subplots, histograms):
        plt.subplot(2, 2, i)
        plt.hist(x[:,j],bins='auto', facecolor='b')
        # titulos 
        plt.title('Media') if i in (1, 2) else None
        plt.xlabel("Valor") if i in (3, 4) else None
        plt.ylabel('Frecuencia') if i in (1, 3) else None
    
    plt.savefig("histograma.png")
    plt.close()
