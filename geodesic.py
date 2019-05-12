from sympy import Symbol, Matrix, diff, simplify, pretty
from numpy import arange, array, cos, sin, pi, reshape, sqrt, linspace
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint
from random import random

def geodesic_RK4(x, f, t1, t2):
    t_start = t1
    t_end = t2
    h = t2-t1
    k1 = h*f(x)
    k2 = h*f(x+0.5*k1)
    k3 = h*f(x+0.5*k2)
    k4 = h*f(x+k3)
    x = x + (1.0/6)*(k1+2*k2+2*k3+k4)

    return array(x)
#
# #u(t) plot
# plt.figure()
# plt.plot(tpoints[:len(p)],p[:,0])
# plt.xlabel("t")
# plt.ylabel("u(t)")
# plt.show()
#
# #v(t) plot
# plt.figure()
# plt.plot(tpoints[:len(p)],p[:,1])
# plt.xlabel("t")
# plt.ylabel("v(t)")
# plt.show()
#
# #u(t)-v(t) plot
# plt.figure()
# ax = plt.gca()
# circle = plt.Circle((0, 0), 1, fill=False)
# ax.add_artist(circle)
# ax.set_aspect(1.0)
# plt.plot(p[:,0],p[:,1])
# plt.xlabel("u(t)")
# plt.ylabel("v(t)")
# plt.xlim((-1,1))
# plt.ylim((-1,1))
# plt.show()
