Schramm-Loewner Evolution Visualiser
====================================

A Python program for displaying [Schramm-Loewner Evolution](https://en.wikipedia.org/wiki/Schramm%E2%80%93Loewner_evolution). Set up to use the driving function of brownian motion.
Does very dumb approximation of differential equations.

Can be run from the command line as `./animate.py` or within Python using `animate.draw_sle()`

`levy()` uses brownian motion with randomly added jumps (the sizes of the jumps follow a normal distribution).


It should be easy to change the resolution (both space and time), the scale of the brownian motion driving it, and the driving function itself. The colouring shouldn't be too hard to change.

sample commands:
`python3 animate.py`
see a full list of options
`python3 animate.py --help`
adjust the brownian motion scale
`python3 animate.py --kappa=2`

increase time resolution
`python3 animate.py --dt 0.0001`
increase spatial resolution 
`python3 animate.py --xn 500`

enter a custom driving function
`python3 animate.py --zeta "lambda t: t**0.5"`

driven by a levy process with no brownian motion component
`python3 animate.py --zeta "levy(kappa=0, jumpfreq=3)" --scale "lambda t:t" --dt 0.0002`

another levy process
`python3 animate.py --zeta "levy(kappa=0, jumpfreq=5, jumpdist=lambda rng:rng.exponential())" --scale "lambda t:t" --dt 0.0002 --xmin -0.1 --xmax 10 --ymax 2 --xn 500`


note: draw.py is out of date. You're better off adapting animate.py.
