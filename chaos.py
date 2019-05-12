##Author Sungwon Kim

from sympy import Symbol, Matrix, diff, simplify, pretty
from numpy import arange, array, cos, sin, pi, reshape, sqrt, linspace, asarray, arctan2, append
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint
from random import random
from matplotlib.animation import FuncAnimation
from RK4 import RK4

#global constant (threshold)
global e
e = 0.1

#Billiard class
class Billiard:
    def __init__(self,
                 init_state = [0, 0, 0, 0],
                 metric = Matrix([[0,0], [0,0]])):
        self.init_state = asarray(init_state, dtype='float')
        self.initial_energy = self.init_state[2]**2+self.init_state[3]**2
        self.time_elapsed = 0
        self.state = self.init_state.copy()
        self.trajectory = [[self.init_state[0], self.init_state[1]]]
        self.metric = metric
        self.energy = self.state[2]**2+self.state[3]**2

        #Calculate Christoffel Symbols
        g = metric
        g_inv = g.inv('LU')
        dgdu = diff(g, u)
        dgdv = diff(g, v)

        dg = []
        dg.append(dgdu)
        dg.append(dgdv)

        r = []

        for i in range(2):
            for k in range(2):
                r_vec = []
                for l in range(2):
                    component = 0 # component indexed by m
                    for m in range(2):
                        component += (0.5*g_inv[i,m]*(dg[l][m,k] + dg[k][m,l] - dg[m][k, l]))
                    r_vec.append(simplify(component))
                r.append(r_vec)

        r = reshape(r, (2,2,2))

        self.R = r

        #Define Christoffel Symbols as functions
    def r1_11(self,state):
        return self.R[0][0][0].subs([(u, state[0]), (v, state[1])])
    def r1_12(self,state):
        return self.R[0][0][1].subs([(u, state[0]), (v, state[1])])
    def r1_21(self,state):
        return self.R[0][1][0].subs([(u, state[0]), (v, state[1])])
    def r1_22(self,state):
        return self.R[0][1][1].subs([(u, state[0]), (v, state[1])])
    def r2_11(self,state):
        return self.R[1][0][0].subs([(u, state[0]), (v, state[1])])
    def r2_12(self,state):
        return self.R[1][0][1].subs([(u, state[0]), (v, state[1])])
    def r2_21(self,state):
        return self.R[1][1][0].subs([(u, state[0]), (v, state[1])])
    def r2_22(self,state):
        return self.R[1][1][1].subs([(u, state[0]), (v, state[1])])

        #Geodesic equation as a system of ODEs
        """state change for each time step"""
    def dstate_dt(self, state):
        u_dot = state[2]
        v_dot = state[3]
        u_ddot = -self.r1_11(state)*(state[2]**2)-2*self.r1_12(state)*state[2]*state[3]-self.r1_22(state)*(state[3]**2)
        v_ddot = -self.r2_11(state)*(state[2]**2)-2*self.r2_12(state)*state[2]*state[3]-self.r2_22(state)*(state[3]**2)
        return array([u_dot, v_dot, u_ddot, v_ddot], float)

    def step(self, dt):
        """execute one time step of length dt and update state"""
        print("time:", self.time_elapsed)

        self.state = RK4(self.state, self.dstate_dt, 0, dt)
        self.energy = self.state[2]**2+self.state[3]**2
        self.time_elapsed += dt
        self.trajectory.append([self.state[0], self.state[1]])

        """threshold distance for collision"""
        e = 0.1
        if self.state[0]**2+self.state[1]**2 > (1-e)**2:
            """scale factors"""
            scale_x = sqrt(self.init_state[0]**2+self.init_state[1]**2)/sqrt(self.state[0]**2+self.state[1]**2)
            scale_rebound = 1.0/(self.state[0]**2+self.state[1]**2)

            """orthogonal reflection"""
            rebound = scale_rebound*(self.state[2]*self.state[0]+self.state[3]*self.state[1])
            new_v_x = self.state[2]-2*rebound*self.state[0]
            new_v_y = self.state[3]-2*rebound*self.state[1]
            scale_v = sqrt(self.initial_energy)/sqrt(new_v_x**2+new_v_y**2)

            """update state, energy, time_elapsed, and trajectory"""
            self.state = array([scale_x*self.state[0], scale_x*self.state[1], scale_v*new_v_x, scale_v*new_v_y], dtype=float)
            self.energy = self.state[2]**2+self.state[3]**2
            self.time_elapsed += dt
            self.trajectory.append([self.state[0], self.state[1]])

            """until the particle moves away completely from boundary"""
            while (self.state[0]**2+self.state[1]**2 > (1-e)**2):
                self.state = RK4(self.state, self.dstate_dt, 0, dt)
                self.trajectory.append([self.state[0], self.state[1]])
                self.time_elapsed += dt

# set up initial state and global variables

theta = 2*pi*random()
theta2 = 2*pi*random()

"""initial conditions in polar coordinates"""
r = linspace(0.8, 0.9, 10)
theta = linspace(pi/2.0, pi/2.5, 10)
u = Symbol('u')
v = Symbol('v')
g = Matrix([[4/((1-(u**5+v**5))**2), 0], [0, 4/((1-(u**5+v**5))**2)]]) #5-Poincare
#g = Matrix([[1, 0], [0, 1]]) #flat
#g = Matrix([[4/((1-(u**3+v**3))**2), 0], [0, 4/((1-(u**3+v**3))**2)]]) #3-Poincare

"""100 particles"""
billiards = []
for i in range(len(r)):
    for j in range(len(theta)):
        ball = Billiard([r[i]*cos(theta[j]), r[i]*sin(theta[j]), 0.9*cos(theta[0]+pi/5), 0.9*sin(theta[0]+pi/5)], g)
        billiards.append(ball)

dt = 1./10 # time-step

# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
circle = plt.Circle((0, 0), 1-e, fill=False) #boundary of the table
ax.add_artist(circle)
traj, = ax.plot([], [], 'bo', ms=3)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    traj.set_data([], [])
    time_text.set_text('')
    return traj, time_text

def animate(i):
    """perform animation step"""
    global billiards, dt
    for ball in billiards:
        ball.step(dt)
        traj.set_data([ball.state[0] for ball in billiards], [ball.state[1] for ball in billiards])
    time_text.set_text('time = %.1f' % billiards[0].time_elapsed)
    return traj, time_text

#choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

#animation
ani = FuncAnimation(fig, animate, frames=6000, interval = interval, blit=False, init_func=init)
#ani.save('5555.mp4', fps = 50, extra_args=['-vcodec', 'libx264'])

plt.show()
