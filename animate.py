#! /usr/bin/env python3
import numpy as np
from numpy import pi
import matplotlib.animation as animation
from numpy.random import default_rng, SeedSequence
import matplotlib.pyplot as plt
import itertools
import matplotlib
from time import strftime
import math #for eval'd command line arguments

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

def hsl2rgb(h,s,l):
    """Computes rgb efficiently
    https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB_alternative"""
    a=s*np.minimum(l,1-l)
    def f(n):
        k=(n+h/(pi/6))%12
        return l-a*np.clip(np.minimum(k-3,9-k),-1,1)
    rgb = np.array((f(0),f(8),f(4))).swapaxes(0,-1)
    rgb[np.any(np.isnan(rgb),axis=-1)]=[1,1,1]
    #rgb=rgb
    return rgb
    

def slesetup(xmin, xmax, ymax, xn, yn):
    """Return an xn by yn numpy array of complex numbers from xmin to xmax and 0 to ymax"""
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(ymax/yn, ymax, yn).astype(np.float32)
    return X[:, None] + Y * 1j

def slepart(Gt, zeta, tstart,tsteps,tstep,scale=lambda x:x**2):
    """Advance the sle driven by zeta.
    Given that Gt describes g at scale(tstart),
    returns an array describing g at scale(tstart+tsteps*tstep)"""
    if scale:
        prev=scale(tstart-tstep)
        for n in range(tsteps):
            cur = scale(tstart+n*tstep)
            Gt += (cur-prev)*(2/(Gt-zeta(cur)))
            prev=cur
    else:
        for t in np.arange(tstart,tstop,tstep):
            Gt += tstep*(2/(Gt-zeta(t)))
    return Gt



def reproducible_brownian(kappa=1, size=10, resolution=10**-7, rng_=None):
    """probably faster and more statistically sound than the one which makes a tree of rngs"""
    if rng_ is None:
        rng_ = rng
    pts = rng.standard_normal(int(size/resolution))*(resolution*kappa)**0.5
    pts=np.cumsum(pts)
    pts=np.insert(pts,0,0)
    def f(t):
        #if t<resolution: print(t)
        assert 0<=t<=size
        t/=resolution
        if t==(it:=int(t)):
            return pts[it]
        else:
            return pts[it]*(it+1-t) + pts[it+1]*(t-it)
            #could interpolate (+rng_.normal(scale=((t-it)*(it+1-t))**0.5)*scale), but that would lose reproducability
    return f

def brownian(kappa=1):
    """probably faster and more statistically sound than the one which makes a tree of rngs"""
    global rng
    cur=0
    last=0
    def f(t):
        nonlocal cur,last
        if t==0: return 0
        assert t>last
        cur+=rng.normal()*((t-last)*kappa)**0.5
        last=t
        return cur
    return f

def levy(kappa=1, jumpfreq=1, jumpdist=lambda rng: rng.normal()):
    global rng
    cur=0
    last=0
    def f(t):
        nonlocal cur,last
        if t==0: return 0
        assert t>last
        dt = t-last
        cur+=rng.normal()*(dt*kappa)**0.5
        for i in range(rng.poisson(jumpfreq*dt)):
            k=jumpdist(rng)
            print(k)
            cur+=k
        last=t
        return cur
    return f

def draw_sle(zeta=None, kappa=1, colour=colIm, xmax=2, xmin=None, xn=300,
             ymax=None, yn=None, seed=None,
             runtime=2, scale=(lambda t:t**2), timestep=0.001, perframe=100,
             save=False, figsize=10, borders=True):
    """Compute and display the Schramm-Loewner evolution driven by brownian motion

    zeta     : function of t driving the sle (default brownian(kappa))
    kappa    : scales brownian motion
    colour   : function to turn complex numbers into pixels
        - 'colIm' (default) ~ hue = log(Im(z)), lum = Re(z)
        - 'colConformal'    ~ Highlights the fact that the mapping is conformal
        - 'colourize'       ~ hue = arg(z) lum = |z|
    xmax     : greatest x coordinate (default 2)
    xn       : Horizontal resolution(default 300)
    seed     : seed for rng
    runtime  : length of time in arbitrary units (default 2)
    scale    : dynamically adjust the size of simulation steps
    timestep : size of steps through runtime (default 0.001)
    perframe : steps to compute per frame
    save     : either True or quoted destination filename  (default False)
    borders  : set to False to hide axes and borders (default True)"""
    # Initialize unspecified arguments
    seed=SeedSequence(seed)
    print(seed)
    global rng
    rng=default_rng(seed)
    # Setup G0
    if zeta is None:
        zeta=brownian(kappa)
    if save:
        matplotlib.use("Agg")
    if xmin is None:
        xmin = -xmax
    if ymax is None:
        ymax = xmax-xmin
    if yn is None:
        yn=xn
    plt.figure(figsize=(figsize,figsize))
    im = plt.imshow([[0]],origin="lower",extent=(xmin,xmax,0,ymax))#,interpolation="antialiased")
    if not borders:
        plt.axis("off")
        plt.tight_layout(0)
    global Gt
    Gt = slesetup(xmin, xmax, ymax, xn, yn)
    img = colour(Gt)
    im.set_data(img)
    def init():
        return [im]
    def dostep(i):
        global Gt
        r=runtime/10
        if i/r-int(i/r) < timestep: print(i)
        Gt=slepart(Gt, zeta, i, min(perframe, int((runtime-i)/timestep)),timestep,scale=scale)
        img=colour(Gt) #.imag*100
        im.set_data(img)
        return [im]
    ani = animation.FuncAnimation(im.figure,dostep,init_func=init,
                            frames=np.arange(0,runtime,perframe*timestep),
                            interval=20,
                            blit=True, repeat=False)
    if save:
        if not isinstance(save,str):
            save="sle-"+strftime("%Y%m%d-%H%M%S")+".mp4"
        mywriter = animation.FFMpegWriter(fps=30)
        ani.save(save,writer=mywriter)
    else:
        plt.show()

if __name__=="__main__":
    import argumentclinic
    argumentclinic.entrypoint(draw_sle,lambda x:eval(x)) #trick to allow colConformal
    

#parts taken from:
#mandelbrot set matplotlib sample code
#https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
#http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
