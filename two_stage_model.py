import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

T12 = 1.5         #half-life of antibiotica in [h]
k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
a = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)
#Nondimensionalization would be k*tau, d*tau

D = 250           #dosis of an antibiotics pill in [mg]
tau = 24          #rescaling [in 24h]

dose_times = np.array([6, 12, 18]) #administrative times for pill in [h]
         
y0 = [0,0] #initial condotions
def two_stage_ode(t, y, a, k):
    b, s = y
    dbdt = a*s - k*b
    dsdt = -a*s
    return [dbdt, dsdt]

def simulate_two_stage_ode(y0, a, k, tau, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    #non-dimensionalize the input
    a = tau*a
    k = tau*k
    dose_times = (1/tau)*dose_times 
    end_time = (1/tau)*end_time
    times = np.concatenate(([0], dose_times, [end_time] ))
    
    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(two_stage_ode, (times[i], times[i+1]), y0, t_eval=t_eval, args=(a, k))
        t = np.concatenate((t, sol.t))
        b = np.concatenate((b, sol.y[0]))
        s = np.concatenate((s, sol.y[1]))
        y0 = [b[-1], s[-1] + 1] #get new initial value and apply pulse for new pill, 1 is nondimensionalized dose

    return t, b, s

def two_stage_Michaelis_Menten_ode(t, y, B, a, k, km):
    b, s = y
    dbdt = B*a*s - k*b/(km + b)*b
    dsdt = -a*s

    return [dbdt, dsdt]

def simulate_two_stage_Michaelis_Menten_ode(y0, D, B, a, k, km, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    
    times = np.concatenate(([0], dose_times, [end_time] ))
    
    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(two_stage_Michaelis_Menten_ode, (times[i], times[i+1]), y0, t_eval=t_eval, args=(B, a, k, km))
        t = np.concatenate((t, sol.t))
        b = np.concatenate((b, sol.y[0]))
        s = np.concatenate((s, sol.y[1]))
        y0 = [b[-1], s[-1] + D] #get new initial value and apply pulse for new pill, 1 is nondimensionalized dose

    return t, b, s

B = 0.8 #bioavailibility
km = 10
print("k:", k)
print('km:', km)

#t, b, s = simulate_two_stage_ode(y0, a, k, tau, dose_times, end_time = 24)
t, b, s = simulate_two_stage_Michaelis_Menten_ode(y0, D, B, a, k, km, dose_times, end_time = 24)

plt.plot(t, b, label = 'Antibiotics in blood flow')
plt.plot(t, s, label = 'Antibiotics in stomach')
plt.xlabel('t')
plt.ylabel('s,b')
plt.title('Two stage model')
plt.legend()
plt.show()

