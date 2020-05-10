#adapted from mandelbrot set sample code
# and https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
# and http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
import numpy as np
from numpy import pi
from colorsys import hls_to_rgb
import matplotlib.animation as animation
from numpy.random import default_rng
import matplotlib.pyplot as plt
import itertools

def main():
    """Stuff you can easily change"""
    xmin, xmax, xn = -2,2, 400
    ymax, yn = xmax-xmin, xn
    framerate=10
    timestep = 0.001
    colour = colIm # Change to 'colorize' to see argument and modulus as hue and luminosity
    def brownian(tstep):
        cur = 0
        while True:
            cur+= rng.normal(scale=(tstep**0.5))
            yield cur
    def zeta(t,r=brownian(timestep)):
        """The driving function"""
        #Currently cheats, because we know it's called in once per timestep
        return next(r)
    draw_sle(zeta, colour, xmin,xmax,xn, ymax,yn, timestep,framerate )


rng = default_rng()


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

def colIm(z):
    h = np.log(z.imag)*0.3
    l = 0.5+0.1*z.real
    s = 1

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

def slesetup(xmin, xmax, ymax, xn, yn):
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(0, ymax, yn).astype(np.float32)
    return X[:, None] + Y * 1j

def slepart(Gt, zeta, tstart,tstop,tstep=0.001):
    for t in np.arange(tstart,tstop,tstep):
        Gt += tstep*(2/(Gt-zeta(t)))
    return Gt


def draw_sle(zeta, colour, xmin=-2, xmax=2, xn=2, ymax=None, yn=None, timestep=0.001, framerate=10):
    if ymax is None:
        ymax = xmax-xmin
    if yn is None:
        yn=xn
    im = plt.imshow([[0]],origin="lower",extent=(xmin,xmax,0,ymax))
    Gt=None
    def init():
        nonlocal Gt,im
        Gt = slesetup(xmin, xmax, ymax, xn, yn)
        img = colorize(Gt)
        im.set_data(img)
        return [im]

    def dostep(i):
        nonlocal Gt
        print(i)
        Gt=slepart(Gt, zeta, i, i + 1/framerate)
        img=colIm(Gt) #.imag*100
        im.set_data(img)
        return [im]
    init()
    animation.FuncAnimation(im.figure,dostep,init_func=init,
                            frames=np.arange(0,4,1/framerate),
                            interval=20,
                            blit=True, repeat=False)
    plt.show()

main()
