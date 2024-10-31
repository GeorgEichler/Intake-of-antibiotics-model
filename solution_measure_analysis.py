from first_order_two_stage_model import simulate_two_stage_ode
from michaelis_menten_two_stage_model import simulate_two_stage_Michaelis_Menten_ode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_concentration(y0, **params):
    #Input:
    #y0 - initial conditions
    # params - dictionary containing additional parameters (e.g., tau, dose_interval, end_time, method)

    if params["method"] == "First order":
        dose_times = np.arange(6, 24*3, params["dose_interval"])
        t, b, s, _, _, _, _, _ = simulate_two_stage_ode(y0, params["tau"], params["B"], params["a"],
                                                        params["k"], dose_times, params["end_time"])
    elif params["method"] == "Michaelis-Menten":
        dose_times = np.arange(6, 24*3, params["dose_interval"])
        t, b, s, _, _, _, _, _ = simulate_two_stage_Michaelis_Menten_ode(y0, params["tau"], params["B"], params["Vmax"],
                                                                        params["km"], params["k"], dose_times, params["end_time"])
    
    plt.figure()
    plt.plot(t, b, label = 'Antibiotics in blood flow')
    plt.plot(t, s, label = 'Antibiotics in stomach',alpha = 0.2)


    plt.xlabel('t')
    plt.ylabel('s,b')
    plt.title('Two stage model')
    plt.legend()

    
def analyze_model(variable_name, variable_values, y0, **params):
    #Input:
    # variable_name: str, the variable to analyze (e.g., "B", "Vmax", "km")
    # variable_values: list, the values of the changing parameter to analyze
    # y0: list, initial conditions for the ODE
    # params: dict, additional parameters needed for the ODE (e.g., "tau", "dose_interval", "end_time", method)
    # method parameter is either First order or Michaelis-Menten
    minima_list = []
    maxima_list = []
    means_list  = []

    for value in variable_values:
        params[variable_name] = value
        
        dose_times = np.arange(1, 24*3, params["dose_interval"])

        if params["method"] == "First order":
            _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(
                y0, params["tau"], params["B"], params["a"], params["k"], dose_times, params["end_time"]
            )
        elif params["method"] == "Michaelis-Menten":
            _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_Michaelis_Menten_ode(
                y0, params["tau"], params["B"], params["Vmax"], params["km"], params["k"], dose_times, params["end_time"]
            )

        minima_list.append(minima[-2])
        maxima_list.append(maxima[-2])
        means_list.append(means[-2])
    
    plt.figure()
    plt.scatter(variable_values, minima_list, label='minima')
    plt.scatter(variable_values, means_list, label='mean')
    plt.plot(variable_values, means_list)
    plt.scatter(variable_values, maxima_list, label='maxima')
    plt.xlabel(f'{variable_name} values')
    plt.ylabel('Max/Min concentration')
    plt.legend()
    plt.title(f'Concentration Analysis vs {variable_name}')

#Set some parameters
T12 = 1.5         #half-life of antibiotica in [h]
k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
B = 0.8           #bioavailibility
a = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)

Vmax = 10
km = 5

D = 200           #dosis of an antibiotics pill in [mg]
tau = 24          #rescaling with respect to [24h]

#non-dimensionalize
a = tau*a
k = tau*k
Vmax = Vmax*tau/D
km = km/D

dose_times = np.arange(6, 24*3 + 1, 6)
end_time = 24*4

#Create dictionaries for the two models to analyze
params_first_order = {
    "method": "First order",
    "tau": tau,
    "B": B,
    "a": a,
    "k": k,
    "dose_interval": 6,
    "end_time": end_time
}

params_Michaelis_Menten = {
    "method": "Michaelis-Menten",
    "tau": tau,
    "B": B,
    "Vmax": Vmax,
    "km": km,
    "k": k,
    "dose_interval": 6,
    "end_time": end_time,
}

y0 = [0,0] #initial condotions


plot_model_concentration(y0, **params_first_order)
plt.show()
exit()

dose_intervals = np.arange(1, 12, 1)
analyze_model("dose_interval", dose_intervals, y0, **params_first_order)




dose_intervals = np.arange(1, 12, 1)
dose_amounts = np.arange(50, 500, 50)

minima_values = np.zeros((len(dose_amounts), len(dose_intervals)))
maxima_values = np.zeros((len(dose_amounts), len(dose_intervals)))
mean_values   = np.zeros((len(dose_amounts), len(dose_intervals)))

for i, dose_amount in enumerate(dose_amounts):
    for j, interval in enumerate(dose_intervals):
        dose_times = np.arange(1, 24*3, interval)

        _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(y0, tau, B, a, k, dose_times, end_time)
        
        #Extract values where a steady state has been obtained
        minima_values[i, j] = minima[-2]
        maxima_values[i, j] = maxima[-2]
        mean_values[i, j]   = means[-2]

#Plot heatmaps with seaborn
sns.set_theme(style='white',font_scale=1.0) # Seaborn style setting

#Minima heatmap
plt.figure()
sns.heatmap(minima_values, annot=True, fmt=".2f", cmap ='viridis',
            xticklabels=np.round(dose_intervals, 1), yticklabels=dose_amounts,
            cbar_kws={'label': 'Minima concentration'})
plt.title('Minima concentration')
plt.xlabel('Dose interval')
plt.ylabel('Dose amount')

#Maxima heatmap
plt.figure()
sns.heatmap(maxima_values, annot=True, fmt=".2f", cmap ='viridis',
            xticklabels=np.round(dose_intervals, 1), yticklabels=dose_amounts,
            cbar_kws={'label': 'Maxima concentration'})
plt.title('Maxima concentration')
plt.xlabel('Dose interval')
plt.ylabel('Dose amount')

#Mean heatmap
plt.figure()
sns.heatmap(mean_values, annot=True, fmt=".2f", cmap ='viridis',
            xticklabels=np.round(dose_intervals, 1), yticklabels=dose_amounts,
            cbar_kws={'label': 'Mean concentration'})
plt.title('Mean concentration')
plt.xlabel('Dose interval')
plt.ylabel('Dose amount')

plt.show()