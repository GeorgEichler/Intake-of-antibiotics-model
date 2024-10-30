from first_order_two_stage_model import simulate_two_stage_ode
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure()

timesteps = np.arange(1, 25)
minimas = []
maximas = []

for step in timesteps:
    dose_times = np.arange(1, 24*7, step)

    _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(y0, B, a, k, tau, dose_times, end_time)

minimas = range(1,25)
maximas = range(1,25)

plt.scatter(timesteps, maximas, label = 'maxima')
plt.scatter(timesteps, minimas, label = 'minima')
plt.xlabel('Timesteps')
plt.ylabel('Max/Min concentration')
plt.legend()
plt.show()