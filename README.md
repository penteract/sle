Schramm-Loewner Evolution Visualiser
====================================

A Python program for displaying [Schramm-Loewner Evolution](https://en.wikipedia.org/wiki/Schramm%E2%80%93Loewner_evolution). Set up to use the driving function of brownian motion.
Does very dumb approximation of differential equations.

Can be run from the command line as `./animate.py` or within Python using either `animate.main()` or `animate.draw_sle()`

`levy()` uses brownian motion with randomly added jumps (the sizes of the jumps follow a normal distribution).


It should be easy to change the resolution (both space and time), the scale of the brownian motion driving it, and the driving function itself. The colouring shouldn't be too hard to change.

note: draw.py is out of date. You're better off adapting animate.py.
