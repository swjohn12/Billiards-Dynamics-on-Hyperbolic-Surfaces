from sympy import Symbol, Matrix, diff, simplify, pretty
from numpy import arange, array, cos, sin, pi, reshape, sqrt, linspace
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint
from random import random

def RK4(x, f, t1, t2):
    t_start = t1
    t_end = t2
    h = t2-t1
    k1 = h*f(x)
    k2 = h*f(x+0.5*k1)
    k3 = h*f(x+0.5*k2)
    k4 = h*f(x+k3)
    x = x + (1.0/6)*(k1+2*k2+2*k3+k4)

    return array(x)
