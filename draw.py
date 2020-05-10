#adapted from mandelbrot set sample code and https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
import numpy as np
from numpy import pi
from colorsys import hls_to_rgb
import matplotlib.animation as animation
from numpy.random import default_rng
import matplotlib.pyplot as plt
import itertools
np.random.seed(19680801)

rng = default_rng()
print(rng._bit_generator.state)

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c


def brownian(tstep):
    cur = 0
    while True:
        cur+= rng.normal(scale=(tstep**0.5))
        yield cur


def zeta(t,r=brownian(0.001)):
    """The driving function"""
    #Currently cheats, because we know it's called in once per timestep
    return next(r)

def sle(xmin, xmax, ymax, xn, yn, tmax=1, tn=1000):
    Gt = slesetup(xmin, xmax, ymax, xn, yn)
    delta = tmax/tn
    for t in np.arange(0,tmax,tmax/tn):
        Gt += delta*(2/(Gt-zeta(t)))
    return Gt

def slesetup(xmin, xmax, ymax, xn, yn):
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(0, ymax, yn).astype(np.float32)
    return X[:, None] + Y * 1j

def slepart(Gt, tstart,tstop,tstep=0.001):
    for t in np.arange(tstart,tstop,tstep):
        Gt += delta*(2/(Gt-zeta(t)))
    return Gt

xmin, xmax, xn = -2,2, 400
ymax, yn = xmax-xmin, xn
framerate=1

Z = sle(xmin, xmax, ymax, xn, yn)
img = colorize(Z)
plt.imshow(img,origin="lower",extent=(xmin,xmax,0,ymax))
plt.show()


