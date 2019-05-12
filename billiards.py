## Author Sungwon Kim
## Reference: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

from sympy import Symbol, Matrix, diff, simplify, pretty
from numpy import arange, array, cos, sin, pi, reshape, sqrt, linspace, asarray, arctan2, append
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint
from random import random
from matplotlib.animation import FuncAnimation
from RK4 import RK4

#class for Billiards
class Billiards:
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
        self.count = 0 # For evaluating the accuracy of geodesic

        #Christoffel Symbols
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
                        component += (0.5*g_inv[i,m]*(dg[l][m,k] + dg[k][m,l] - dg[m][k, l])) #summed over index m
                    r_vec.append(simplify(component))
                r.append(r_vec)

        r = reshape(r, (2,2,2))

        self.R = r

        #Print Christoffel Symbols
        for i in range(len(r)):
            for j in range(len(r)):
                for k in range(len(r)):
                    print("r", (i+1,j+1,k+1), "=", r[i][j][k])

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

        """accuracy evaluation"""
        if abs(self.time_elapsed - 1.0) < dt and self.count<1:
            position = [self.state[0], self.state[1]]
            print("position at t=1", position)
            """distance is only measured once"""
            self.count += 1

        """update state, energy, time, and trajectory"""
        self.state = RK4(self.state, self.dstate_dt, 0, dt)
        self.energy = self.state[2]**2+self.state[3]**2
        self.time_elapsed += dt
        self.trajectory.append([self.state[0], self.state[1]])

        e = 0.1 #threshold for collision
        if self.state[0]**2+self.state[1]**2 > (1-e)**2:
            """scale factors"""
            scale_x = sqrt(self.init_state[0]**2+self.init_state[1]**2)/sqrt(self.state[0]**2+self.state[1]**2)
            scale_rebound = 1.0/(self.state[0]**2+self.state[1]**2)

            """orthogonal reflection"""
            rebound = scale_rebound*(self.state[2]*self.state[0]+self.state[3]*self.state[1])
            new_v_x = self.state[2]-2*rebound*self.state[0]
            new_v_y = self.state[3]-2*rebound*self.state[1]

            """energy conservation"""
            scale_v = sqrt(self.initial_energy)/sqrt(new_v_x**2+new_v_y**2)
            self.state = array([scale_x*self.state[0], scale_x*self.state[1], scale_v*new_v_x, scale_v*new_v_y], dtype=float)
            self.energy = self.state[2]**2+self.state[3]**2
            self.trajectory.append([self.state[0], self.state[1]])

            """until the particle leaves the boundary"""
            while (self.state[0]**2+self.state[1]**2 > (1-e)**2):
                self.state = RK4(self.state, self.dstate_dt, 0, dt)
                self.trajectory.append([self.state[0], self.state[1]])
                self.time_elapsed += dt

# set up initial state and global variables

theta = 2*pi*random()
theta2 = 2*pi*random()
e = 0.1
initial_state = array([(1-e)*cos(pi/2), (1-e)*sin(pi/2), -0.4*cos(pi/2+pi/50), -0.4*sin(pi/2+pi/50)], float)
"""metric as symbolic expressions"""
u = Symbol('u')
v = Symbol('v')
g = Matrix([[4/((1-(u**2+v**2))**2), 0], [0, 4/((1-(u**2+v**2))**2)]]) #Poincare
#g = Matrix([[1, 0], [0, 1]]) #flat
#g = Matrix([[4/((1-(u**3+v**3))**2), v], [u, 4/((1-(u**3+v**3))**2)]]) #3-Poincare
#g = Matrix([[4/((1-(u**5+v**5))**2), v], [u, 4/((1-(u**5+v**5))**2)]]) #5-Poincare
billiards = Billiards(initial_state, g) #initialize class
dt = 1./10 # time-step

# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
circle = plt.Circle((0, 0), 1-e, fill=False) # boundary disk
ax.add_artist(circle)
line, = ax.plot([], [], 'o-', lw=2)
traj, = ax.plot([], [], 'b-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
init_position = ax.text(0.02, 0.85, '', transform=ax.transAxes)
init_velocity = ax.text(0.02, 0.80, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    init_position.set_text('')
    init_velocity.set_text('')
    return line, time_text, energy_text

def animate(i):
    """perform animation step"""
    global billiards, dt
    billiards.step(dt)
    line.set_data(billiards.state[0], billiards.state[1])
    traj.set_data([position[0] for position in billiards.trajectory] , [position[1] for position in billiards.trajectory])
    time_text.set_text('time = %.1f' % billiards.time_elapsed)
    energy_text.set_text('energy = %.1f' % billiards.energy)
    return line, time_text, energy_text

#choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

#animation
ani = FuncAnimation(fig, animate, frames=3000, interval = interval, blit=False, init_func=init)
#ani.save('simulation.mp4', fps = 50, extra_args=['-vcodec', 'libx264'])

plt.show()
