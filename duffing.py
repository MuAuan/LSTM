from scipy.integrate import odeint, simps
import numpy as np
import matplotlib.pyplot as plt

def duffing(var, t, gamma, a, b, F0, omega, delta):
    """
    var = [x, p]
    dx/dt = p
    dp/dt = -gamma*p + 2*a*x - 4*b*x**3 + F0*cos(omega*t + delta)
    """
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])

# parameter
F0, gamma, omega, delta = 10, 0.1, np.pi/3, 1.5*np.pi
a, b = 1/4, 1/2
#F0, gamma, omega, delta = 2.0, 0.1, 2.4, 0            #np.pi/3, 1.5*np.pi
#a, b = 1/5, 1/2
var, var_lin = [[0, 1]] * 2

#timescale
t = np.arange(0, 20000, 2*np.pi/omega)
#t = np.linspace(0, 200, 3000)   #10000)
t_lin = np.linspace(0, 100, 10000)

# solve
var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
#var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))
var_lin = odeint(duffing, [0.5, 0], t_lin, args=(gamma, a, b, F0, omega, delta))

x, p = var.T[0], var.T[1]
x_lin, p_lin = var_lin.T[0], var_lin.T[1]

plt.plot(t[:100], x[:100])
plt.pause(3)
#plt.close()

plt.plot(t[:100], p[:100])
plt.pause(3)
plt.savefig('plot_D_xpvst0.png', dpi=60)
plt.close()

plt.plot(x, p, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_D_xp0.png', dpi=60)
plt.close()

# is chaotic?
plt.plot(t_lin, x_lin)
plt.pause(3)

#var_lin = odeint(duffing, [0.1, 1], t_lin, args=(gamma, a, b, F0, omega, delta))
var_lin = odeint(duffing, [0.5001, 0], t_lin, args=(gamma, a, b, F0, omega, delta))
x_lin, p_lin = var_lin.T[0], var_lin.T[1]
plt.plot(t_lin, x_lin)
plt.pause(3)
plt.savefig('plot_D_xpvst1.png', dpi=60)
plt.close()
plt.plot(x_lin, p_lin,".", markersize=4)
plt.pause(3)
plt.savefig('plot_D_xp1.png', dpi=60)
plt.close()

"""
scipy.integrate.odeint
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

The second order differential equation for the angle theta of a pendulum acted on by gravity with friction can be written:
theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0

theta'(t) = omega(t)
omega'(t) = -b*omega(t) - c*sin(theta(t))
"""

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

b = 0.025
c = 5.0
y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 100, 1001)
sol = odeint(pend, y0, t, args=(b, c))

x, p = sol.T[0], sol.T[1]
plt.plot(x, p, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_xp.png', dpi=60)
plt.close()

plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.xlabel('t')
plt.grid()
plt.legend(loc='best')
plt.pause(3)
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.pause(3)
plt.savefig('plot_thomegavst.png', dpi=60)
plt.close()


def lorenz(var, t, p, r, b):
    #{\frac  {dx}{dt}}=-px+py
    #{\frac  {dy}{dt}}=-xz+rx-y
    #{\frac  {dz}{dt}}=xy-bz
    """
    var = [x,y,z, x_dot,y_dot,z_dot]
    dx/dt = -px+py
    dy/dt = -xz+rx-y
    dz/dt = xy-bz
    """
    x_dot = -p*var[0]+p*var[1]
    y_dot = -var[0]*var[2]+r*var[0]-var[1]
    z_dot = var[0]*var[1]-b*var[2]
    #x_dot= -p*x_dot +p*y_dot
    #y_dot= -x_dot*var[2] - var[0]*z_dot +r*x_dot -y_dot
    #z_dot= x_dot*var[1] + var[0]*y_dot -b*z_dot
    return x_dot,y_dot,z_dot  #_dot,py_dot,pz_dot

p = 10
b = 8/3
r = 28
y0 = [0.1,0, 0.0]  #,0]
t = np.linspace(0, 20, 3001)
sol = odeint(lorenz, y0, t, args=(p, r, b))

x, y, z = sol.T[0], sol.T[1], sol.T[2]
plt.plot(x, y, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_xy.png', dpi=60)
plt.close()

plt.plot(y, z, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_yz.png', dpi=60)
plt.close()

plt.plot(z, x, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_zx.png', dpi=60)
plt.close()

#plt.savefig('plot_xyyzzx.png', dpi=60)
#plt.close()

plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.xlabel('t')
plt.grid()
plt.legend(loc='best')
plt.savefig('plot_xt.png', dpi=60)
plt.pause(3)
plt.close()

plt.plot(t, sol[:, 1], 'g', label='y(t)')
plt.xlabel('t')
plt.grid()
plt.legend(loc='best')
plt.savefig('plot_yt.png', dpi=60)
plt.pause(3)
plt.close()

plt.plot(t, sol[:, 2], 'g', label='z(t)')
plt.xlabel('t')
plt.grid()
plt.legend(loc='best')
plt.savefig('plot_zt.png', dpi=60)
plt.pause(3)
plt.close()

#plt.savefig('plot_xyzvst.png', dpi=60)
#plt.close()

