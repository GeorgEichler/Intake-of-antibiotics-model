#Model includes three compartment: stomach, bloodstream and tissue interationg due to first-order interactions

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp





def three_stage_model(t,y,F,k12,k23,k32,k):
    #unpack state variables
    S, B, T = y

    #compute derivatives
    dSdt = -k12*S
    dBdt = F*k12*S + k32*T - (k23 + k)*B
    dTdt = k23*B - k32*T
    return [dSdt,dBdt,dTdt]

def simulate_three_stage_model(y0,F,k12,k23,k32,k,tau,dose_times, end_time):
    t = np.array([])
    S = np.array([])
    B = np.array([])
    T = np.array([])

    #nondimensionalize
    k12 = tau*k12
    k23 = tau*k23
    k32 = tau*k32
    k = tau*k
    dose_times = (1/tau)*dose_times
    end_time = (1/tau)*end_time
    times = np.concatenate(([0], dose_times, [end_time] ))

    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(three_stage_model, (times[i], times[i+1]), y0, t_eval=t_eval, args=(F,k12, k23, k32, k))
        t = np.concatenate((t, sol.t))
        S = np.concatenate((S, sol.y[0]))
        B = np.concatenate((B, sol.y[1]))
        T = np.concatenate((T, sol.y[2]))
        y0 = [S[-1] + 1, B[-1], T[-1]] #get new initial value and apply pulse for new pill

    return t, S, B, T

#Define parameters
y0 = [0,0,0]
k12 = 1
k23 = 1
k32 = 1
k   = 1
tau = 24

dose_times = np.array([6, 12, 18])

#bioavailibility
F = 0.8

t, S, B, T = simulate_three_stage_model(y0, F, k12, k23, k32, k, tau, 
                                        dose_times, end_time = 24)

plt.plot(t, S, label = 'Stomach')
plt.plot(t, B, label = 'Bloodstream')
plt.plot(t, T, label = 'Tissue')

plt.xlabel('t')
plt.ylabel('S,B,T')
plt.legend()
plt.show()
