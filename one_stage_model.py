"""
The one stage model assumes an instantaneous ingestion of antibiotics to the body and is modeled by
db/dt = D \sum_{i=1}^N \delta(t - t_i) - k * b
where the delta functions represent the intake of a pill

b   ... amount of antibiotics in bloodstream (in mg)
D   ... dosis of one pill (in mg)
t_i ... time point of pill ingestion 
T12 ... half-life of antibiotics (in h)
k   ... degredation rate of antibiotica (in 1/h)
"""

import numpy as np
import matplotlib.pyplot as plt

#Parameter
D = 250
T12 = 2
#degredation rate - half-life relation is a consequance of exponential decay e^{-kt}
k = np.log(2)/T12
b0 = 0 #starting amount
intake_times = [6,12,18,24]
end_time = 50

time_steps = 1001
t = np.linspace(0,end_time, time_steps)
b = np.zeros_like(t)

for dosis_time in intake_times:
    b += D * np.heaviside(t - dosis_time, 1) * np.exp(-k*(t - dosis_time))

plt.plot(t, b, label = 'Antibiotics')
plt.xlabel('t in h')
plt.ylabel('b in mg')
plt.title('One stage model for antibiotics ingestion')
plt.legend()

plt.show()