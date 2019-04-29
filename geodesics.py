from sympy import Symbol, Matrix, diff, simplify, pretty
from numpy import arange, array, cos, sin, pi, reshape, sqrt, linspace
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint

u = Symbol('u')
v = Symbol('v')

#def metric(u, v):

g = Matrix([[4/((1-(u**2+v**2))**2), 0], [0, 4/((1-(u**2+v**2))**2)]])
g_inv = g.inv('LU')
dgdu = diff(g, u)
dgdv = diff(g, v)

dg = []
dg.append(dgdu)
dg.append(dgdv)

print("g=", pretty(g),"\n")
print("g_inv=", pretty(g_inv), "\n")
print("g_partials=", pretty(dg), "\n")

#Christoffel Symbols (as symbolic expressions)

r = []

for i in range(2):
    for j in range(2):
        r_vec = []
        for m in range(2):
            component = 0 # component indexed by m
            for k in range(2):
                component += (0.5*g_inv[k,m]*(dg[j][i,k] + dg[i][j,k] - dg[k][i, j]))
            r_vec.append(simplify(component))
        r.append(r_vec)

r = reshape(r, (2,2,2))

for i in range(len(r)):
    for j in range(len(r)):
        for k in range(len(r)):
            print("r at index", (i+1,j+1,k+1), "=", r[i][j][k])

# Christoffel Symbols (as functions)
def r1_11(x,y):
    return r[0][0][0].subs([(u, x), (v, y)])
def r1_12(x,y):
    return r[0][0][1].subs([(u, x), (v, y)])
def r1_21(x,y):
    return r[0][1][0].subs([(u, x), (v, y)])
def r1_22(x,y):
    return r[0][1][1].subs([(u, x), (v, y)])
def r2_11(x,y):
    return r[1][0][0].subs([(u, x), (v, y)])
def r2_12(x,y):
    return r[1][0][1].subs([(u, x), (v, y)])
def r2_21(x,y):
    return r[1][1][0].subs([(u, x), (v, y)])
def r2_22(x,y):
    return r[1][1][1].subs([(u, x), (v, y)])

#Geodesic equation

def f(x,t):
    u_dot = x[2]
    v_dot = x[3]
    u_ddot = -r1_11(x[0],x[1])*(x[2]**2)-2*r1_12(x[0],x[1])*x[2]*x[3]-r1_22(x[0],x[1])*(x[3]**2)
    v_ddot = -r2_11(x[0],x[1])*(x[2]**2)-2*r2_12(x[0],x[1])*x[2]*x[3]-r2_22(x[0],x[1])*(x[3]**2)
    print(x)
    return array([u_dot, v_dot, u_ddot, v_ddot], float)

#initial conditions
x = array([cos(pi/4)-0.1, sin(pi/4), -cos(pi/4), -sin(pi/4)], float) #u, v, u_dot, v_dot

# Runge-Kutta 4th order

N = 1000 # number of steps for RK4
t1 = 0
t2 = 1

h = (t2-t1)/N # step-size

tpoints = linspace(t1,t2,N+1) # time array
xpoints = [] # x array

e = 0.1

for t in tqdm(tpoints):
    if abs((x[0]**2+x[1]**2) - 1) < e**2:
        break # break if "sufficiently" near boundary
    xpoints.append(x)
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1,t+0.5*h)
    k3 = h*f(x+0.5*k2,t+0.5*h)
    k4 = h*f(x+k3,t+h)
    x = x + (1.0/6)*(k1+2*k2+2*k3+k4)

#scipy.integrate.odeint
#xpoints = odeint(f, x, tpoints)

p = array(xpoints) #converting the results to a 2D-array
print("xpoints:", p)

#u(t) plot
plt.figure()
plt.plot(tpoints[:len(p)],p[:,0])
plt.xlabel("t")
plt.ylabel("u(t)")
plt.show()

#v(t) plot (fox)
plt.figure()
plt.plot(tpoints[:len(p)],p[:,1])
plt.xlabel("t")
plt.ylabel("v(t)")
plt.show()

#u(t)-v(t) plot
plt.figure()
ax = plt.gca()
circle = plt.Circle((0, 0), 1, fill=False)
ax.add_artist(circle)
ax.set_aspect(1.0)
plt.plot(p[:,0],p[:,1])
plt.xlabel("u(t)")
plt.ylabel("v(t)")
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.show()
