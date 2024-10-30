import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson


def two_stage_ode(t, y, B, a, k):
    b, s = y
    dbdt = B*a*s - k*b
    dsdt = -a*s
    return [dbdt, dsdt]

def simulate_two_stage_ode(y0, B, a, k, tau, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    dose_interval = dose_times[1] - dose_times[0]

    #Measure variables
    maxima =       []
    minima =       []
    b_auc  =       []
    time_to_peak = []

    #non-dimensionalize parameters
    a = tau*a
    k = tau*k
    dose_interval = (1/tau)*dose_interval
    dose_times = (1/tau)*dose_times 
    end_time = (1/tau)*end_time
    
    times = np.concatenate(([0], dose_times, [end_time] ))
    
    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(two_stage_ode, (times[i], times[i+1]), y0, t_eval=t_eval, args=(B, a, k))
        t0 = times[i]

        b_auc.append(simpson(sol.y[0], x = sol.t))

        peak_index = np.argmax(sol.y[0])
        peak_time = sol.t[peak_index]
        time_to_peak.append(peak_time - t0)

        maxima.append(np.max(sol.y[0]))
        minima.append(np.min(sol.y[0]))


        t = np.concatenate((t, sol.t))
        b = np.concatenate((b, sol.y[0]))
        s = np.concatenate((s, sol.y[1]))
        y0 = [b[-1], s[-1] + 1] #get new initial value and apply pulse for new pill, 1 is nondimensionalized dose

    b_auc = np.array(b_auc)
    means = 1/dose_interval * b_auc
    
    return t, b, s, minima, maxima, b_auc, means, time_to_peak


T12 = 1.5         #half-life of antibiotica in [h]
k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
B = 0.8           #bioavailibility
a = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)

D = 100           #dosis of an antibiotics pill in [mg]
tau = 24          #rescaling with respect to [24h]

dose_times = np.arange(6, 24*3 + 1, 6)
end_time = 24*4
         
y0 = [0,0] #initial condotions

t, b, s, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(y0, B, a, k, tau, dose_times, end_time)

plt.plot(t, b, label = 'Antibiotics in blood flow')
plt.plot(t, s, label = 'Antibiotics in stomach',alpha = 0.2)


plt.xlabel('t')
plt.ylabel('s,b')
plt.title('Two stage model')
plt.legend()
print(f'Minima: {minima}')
print(f'Maxima: {maxima}')
print(f'Area under the curve {b_auc}')
print(f'Means: {means}')
print(f'Time to peak {time_to_peak}')
'''
plt.figure()

timesteps = np.arange(1, 25)
minimas = []
maximas = []

for step in timesteps:
    dose_times = np.arange(1, 24*7, step)

    _, _, _, minima, maxima = simulate_two_stage_ode(y0, B, a, k, tau, dose_times, end_time)
    minimas.append(minima)
    maximas.append(maxima)


plt.scatter(timesteps, maximas, label = 'maxima')
plt.scatter(timesteps, minimas, label = 'minima')
plt.xlabel('Timesteps')
plt.ylabel('Max/Min concentration')
plt.legend()
'''
plt.show()
