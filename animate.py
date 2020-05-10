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
import matplotlib
from time import strftime


def main():
    """Stuff you can easily change"""
    xmin, xmax, xn = -2,2, 300
    ymax, yn = xmax-xmin, xn
    
    runtime = 4
    timestep = 0.001
    framerate=10
    
    colour = colIm # Change to `colourize` to see argument and modulus as hue and luminosity

    
    kappa=1
    def brownian(tstep):
        cur = 0
        while True:
            cur+= rng.normal(scale=((kappa*tstep)**0.5))
            yield cur
    def zeta(t,r=brownian(timestep)):
        """The driving function"""
        #Currently cheats, because we know it's called in once per timestep
        return next(r)
    draw_sle(zeta, colour, xmin,xmax,xn, ymax,yn,
             runtime, timestep,framerate, save=False )


rng = default_rng()
print(rng._bit_generator.state)



def colIm(z):
    """Returns a colour where log(Im(z)) is represented by hue.
    This makes it easy to see where Im(z) converges to 0"""
    h = np.log(z.imag)*pi
    l = np.clip(0.5+0.05*z.real,0.1,0.9)
    s = 1
    c = hsl2rgb(h,s,l)
    return c

def colConformal(z):
    """Highlights the fact that the mapping is conformal"""
    h = np.maximum((np.log(z.imag))%1 , (z.real*3)%1)**2
    l = 0.5
    s = 1
    c = hsl2rgb(h,s,l)
    return c


def colourize(z):
    """colour a complex """
    r = np.abs(z)
    arg = np.angle(z) 

    h = arg
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8
    c = hsl2rgb(h,s,l)
    return c

def hsl2rgbslow(h,s,l):
    c = np.vectorize(hls_to_rgb) ((h/(2*pi))%1,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    return c.swapaxes(0,2)
def hsl2rgb(h,s,l):
    """Computes rgb efficiently
    https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB_alternative"""
    a=s*np.minimum(l,1-l)
    def f(n):
        k=(n+h/(pi/6))%12
        return l-a*np.clip(np.minimum(k-3,9-k),-1,1)
    return np.array((f(0),f(8),f(4))).swapaxes(0,-1)
    

def slesetup(xmin, xmax, ymax, xn, yn):
    """Return an appropriately dimensioned numpy array"""
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(0, ymax, yn).astype(np.float32)
    return X[:, None] + Y * 1j

def slepart(Gt, zeta, tstart,tstop,tstep):
    """Advance the sle driven by zeta.
    Given that Gt describes g at tstart,
    returns an array describing g at tstep"""
    for t in np.arange(tstart,tstop,tstep):
        Gt += tstep*(2/(Gt-zeta(t)))
    return Gt


def draw_sle(zeta, colour, xmin=-2, xmax=2, xn=2,
             ymax=None, yn=None,
             runtime=4, timestep=None, framerate=10,
             save=False):
    if save:
        matplotlib.use("Agg")
        
    if ymax is None:
        ymax = xmax-xmin
    if yn is None:
        yn=xn
    im = plt.imshow([[0]],origin="lower",extent=(xmin,xmax,0,ymax))
    Gt=None
    def init():
        return [im]

    def dostep(i):
        nonlocal Gt
        Gt=slepart(Gt, zeta, i, i + 1/framerate,timestep)
        img=colour(Gt) #.imag*100
        im.set_data(img)
        return [im]
    #init()
    Gt = slesetup(xmin, xmax, ymax, xn, yn)
    img = colour(Gt)
    im.set_data(img)
    ani = animation.FuncAnimation(im.figure,dostep,init_func=init,
                            frames=np.arange(0,runtime,1/framerate),
                            # breaks if this is below 10 and framerate is low.
                            # I don't know why
                            interval=0,
                            blit=True, repeat=False)
    if save:
        if not isinstance(save,str):
            save="sle-"+strftime("%Y%m%d-%H%M%S")+".mp4"
        mywriter = animation.FFMpegWriter(fps=30)
        ani.save(save,writer=mywriter)
    else:
        plt.show()

main()
