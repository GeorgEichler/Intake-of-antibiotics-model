import numpy as np
from scipy.integrate import solve_ivp


def two_stage_Michaelis_Menten_ode(t, y, B, a, k, km):
    b, s = y
    dbdt = B*a*s - k/(km + b)*b
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