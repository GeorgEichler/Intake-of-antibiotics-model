import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

'''
#Saturable absorption doesn't seem that promising as concentration level just rises
def two_stage_Michaelis_Menten_ode(t, y, B, Vmax, km, a):
    b, s = y
    dbdt = B*a*s - Vmax/(km + b)*b
    dsdt = -a*s

    return [dbdt, dsdt]
'''

def two_stage_Michaelis_Menten_ode(t, y, B, Vmax, km, k):
    b, s = y
    dbdt = B*Vmax/(km + s)*s - k*b 
    dsdt = -Vmax/(km + s)*s

    return [dbdt, dsdt]


def simulate_two_stage_Michaelis_Menten_ode(y0, tau, D, B, Vmax, km, k, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    
    #nondimensionalize
    Vmax = Vmax*tau/D
    km = km/D
    k = k*tau
    dose_times = 1/tau*dose_times
    end_time = end_time/tau

    times = np.concatenate(([0], dose_times, [end_time] ))
    
    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(two_stage_Michaelis_Menten_ode, (times[i], times[i+1]), y0, t_eval=t_eval, args=(B, Vmax, km, k))
        t = np.concatenate((t, sol.t))
        b = np.concatenate((b, sol.y[0]))
        s = np.concatenate((s, sol.y[1]))
        y0 = [b[-1], s[-1] + 1] #get new initial value and apply pulse for new pill, 1 is nondimensionalized dose

    return t, b, s

y0 = [0,0]
D = 100
tau = 24
B = 0.8
Vmax = 10
km = 2
T12 = 1.5
k = np.log(2)/T12
dose_times = np.array([6,12,18,24])
end_time = 48
 

t, b, s = simulate_two_stage_Michaelis_Menten_ode(y0, tau, D, B, Vmax, km, k, dose_times, end_time)


plt.plot(t,b, label = 'Bloodstream')
plt.plot(t,s, label = 'Stomach')

plt.xlabel('t')
plt.ylabel('b,s')
plt.legend()
plt.show()