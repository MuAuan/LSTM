from scipy.integrate import odeint, simps
import numpy as np
import matplotlib.pyplot as plt

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


def dualPendulum(y, t, b, c):
    """
    (𝑚1 +𝑚2)𝑙1^2𝜃1̈  = -(+𝑚2𝑙1𝑙2(𝜃2̈ cos(𝜃1 −𝜃2)+𝜃2̇ ^2 sin(𝜃1 −𝜃2)) +(𝑚1 +𝑚2)𝑔𝑙1 sin𝜃1)
    𝑚2𝑙2^2𝜃2̈  = -(+𝑚2𝑙1𝑙2(𝜃1̈ cos(𝜃1 −𝜃2)−𝜃1̇ 2 sin(𝜃1 −𝜃2)) +𝑚2𝑔𝑙2 sin𝜃2)
    𝜃1'' = - (a*(𝜃2''*cos(𝜃1-𝜃2)+𝜃2'^2*sint(𝜃1-𝜃2)) + b*sint𝜃1)
    𝜃2'' = -(c*(𝜃1''*cos(𝜃1-𝜃2)-𝜃1'^2*sin(𝜃1-𝜃2))+d*sin𝜃1)
    
    𝜃1'(t) = ω1(t)
    𝜃2'(t) = ω2(t)
    ω1'(t) = - (a*(ω2'*cos(𝜃1-𝜃2)+ω2^2*sin(𝜃1-𝜃2)) + b*sin𝜃1)
    ω2'(t) = - (c*(ω1'*cos(𝜃1-𝜃2)-ω1^2*sin(𝜃1-𝜃2)) + d*sin𝜃1)
    ω1'(t) = -((a*c*ω1^2*sin(𝜃1-𝜃2)) - a*d*sin𝜃1)*cos(𝜃1-𝜃2)+a*ω2^2*sin(𝜃1-𝜃2)) + b*sin𝜃1)/(1 - a*c*cos(𝜃1-𝜃2)^2)
    ω1'(t) = -(A*ω1^2*sin(𝜃1-𝜃2)- B*sin𝜃1)*cos(𝜃1-𝜃2)+ a*ω2^2*sin(𝜃1-𝜃2)+b*sin𝜃1)/(1 - A*cos(𝜃1-𝜃2)^2)
    
    ω1'(t) = -b*omega1(t) - c*sin(theta1(t)) - k*theta21
    ω2'(t) = -b*omega2(t) - c*sin(theta2(t)) + k*theta21
    
    theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
    theta'(t) = omega(t) 
    omega'(t) = -b*omega(t) - c*sin(theta(t))
    """
    theta1, omega1,theta2, omega2 = y
    #theta21=theta2-theta1
    dydt = [omega1, -b*omega1 - c*np.sin(theta1)-0.06*(theta2-theta1),omega2, -b*omega2 - c*np.sin(theta2)+0.06*(theta2-theta1)]
    return dydt

b = 0.001  #0.025
c = 5.0
y0 = [np.pi - 0.1, 0.0,0, 0.0]
t = np.linspace(0, 10000, 4001)
sol = odeint(dualPendulum, y0, t, args=(b, c))

x1, p1,x2,p2 = sol.T[0], sol.T[1], sol.T[2], sol.T[3]
plt.plot(x1, p1, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_xp1omega2006-10000-001-m1.png', dpi=180)
plt.close()
plt.plot(x2, p2, ".", markersize=4)
plt.pause(3)
plt.savefig('plot_xp2omega2006-10000-001-m1.png',dpi=180)
plt.close()

plt.plot(t, sol[:, 0], 'b', label='theta1(t)')
plt.xlabel('t')
plt.xlim(3000,4000)
plt.ylim(-0.5,0.5)
plt.grid()
plt.legend(loc='best')
plt.pause(3)
plt.plot(t, sol[:, 2], 'g', label='theta2(t)')
plt.legend(loc='best')
plt.pause(3)
plt.savefig('plot_theta2vst1006-10000g-001-m1.png', dpi=180)
plt.close()

plt.plot(t, sol[:, 1], 'b', label='omega1(t)')
plt.xlabel('t')
plt.xlim(3000,4000)
plt.ylim(-1,1)
plt.grid()
plt.legend(loc='best')
plt.pause(3)
plt.plot(t, sol[:, 3], 'g', label='omega2(t)')
plt.legend(loc='best')
plt.pause(3)
plt.savefig('plot_thomega2vst2006-10000g-001-m1.png', dpi=180)
plt.close()
