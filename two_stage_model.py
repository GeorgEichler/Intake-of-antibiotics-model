import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

T12 = 0.5         #half-life of antibiotica in [h]
k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
D = 250           #dosis of an antibiotics pill in [mg]
d = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)

mu = 1            #nondimensionalized parameter with relation mu = D*d

dose_times_with_units = np.array([6, 12, 18]) #administrative times for pill in [h]

dose_times = k*dose_times_with_units          #nondimensionalized version

def two_stage_ode(t, y, mu):
    b, s = y
    dbdt = s - b
    dsdt = -s
    return [dbdt, dsdt]
