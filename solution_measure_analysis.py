from first_order_two_stage_model import simulate_two_stage_ode
from michaelis_menten_two_stage_model import simulate_two_stage_Michaelis_Menten_ode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_concentration(y0, MIC, **params):
    #Input:
    #y0 - initial conditions
    #MIC - inhibitory conditional analysis
    # params - dictionary containing additional parameters (e.g., tau, dose_interval, end_time, method)

    if params["method"] == "First order":
        dose_times = np.arange(6, 24*3, params["dose_interval"])
        t, b, s, min, _, _, _, _ = simulate_two_stage_ode(y0, params["tau"], params["B"], params["$k_{abs}$"],
                                                        params["$k_{el}$"], dose_times, params["end_time"])
    elif params["method"] == "Michaelis-Menten":
        dose_times = np.arange(6, 24*3, params["dose_interval"])
        t, b, s, _, _, _, _, _ = simulate_two_stage_Michaelis_Menten_ode(y0, params["tau"], params["B"], params["$V_{max}$"],
                                                                        params["$k_{m}$"], params["$k_{el}$"], dose_times, params["end_time"])
    
    plt.figure()
    plt.plot(t, b, label = 'Antibiotics in bloodstream', color = 'red')
    plt.plot(t, s, label = 'Antibiotics in stomach',alpha = 0.2, color = 'blue')
    plt.axhline(y=MIC, color='magenta', linestyle = '--', label = 'MIC')
    plt.xlabel('t')
    plt.ylabel('s,b')
    plt.title(f'Two stage model {params["method"]}')
    plt.legend()

def sensitive_analysis(variable_name, variable_values, y0, MIC, **params):
    #Input:
    # y0 - initial conditions
    # variable_name: str, the variable to analyze (e.g., "B", "Vmax", "km")
    # variable_values: list, the values of the changing parameter to analyze
    # params - dictionary containing additional parameters (e.g., tau, dose_interval, end_time, method)
    # MIC - minimum inhibitory concentration
    params["dose_interval"] = 8
    dose_times = np.arange(8, 24*3+1, params["dose_interval"])
    params["end_time"] = 24*4 + 12
    plt.figure()

    for value in variable_values:
        params[variable_name] = value

        if params["method"] == "First order":
            t, b, s, _, _, _, _, _ = simulate_two_stage_ode(
                y0, params["tau"], params["B"], params["$k_{abs}$"], params["$k_{el}$"], dose_times, params["end_time"]
            )
        elif params["method"] == "Michaelis-Menten":
            t, b, s, _, _, _, _, _= simulate_two_stage_Michaelis_Menten_ode(
                y0, params["tau"], params["B"], params["$V_{max}$"], params["$k_{m}$"], params["$k_{el}$"], dose_times, params["end_time"]
            )

        # Get the next color from the color cycle
        color = plt.gca()._get_lines.get_next_color()

        plt.plot(t, b, label = f'{variable_name} = {np.round(value, 2)}', color = color)
        #plt.plot(t, s, label = f'{variable_name} = {np.round(value,2)}', linestyle = '--', color = color)
    
    xticks = np.arange(0, 4, 0.25)  # Adjust the range to fit your x-axis limits
    #plt.xticks(xticks, labels=[f'{tick:.2f}' for tick in xticks], rotation=45)
    #plt.axhline(y = MIC, label = "MIC", linestyle = '--', color = 'magenta')
    plt.xlabel('t')
    plt.ylabel('s,b')
    plt.title(f'Two stage model ({params["method"]})')
    plt.legend()

def analyze_model(variable_name, variable_values, y0, MIC, **params):
    #Input:
    # variable_name: str, the variable to analyze (e.g., "B", "Vmax", "km")
    # variable_values: list, the values of the changing parameter to analyze
    # y0: list, initial conditions for the ODE
    # params: dict, additional parameters needed for the ODE (e.g., "tau", "dose_interval", "end_time", method)
    # method parameter is either First order or Michaelis-Menten
    minima_list = []
    maxima_list = []
    means_list  = []
    time_to_peak_list = []
    params["end_time"] = 24*7 + 6

    for value in variable_values:
        params[variable_name] = value
        
        dose_times = np.arange(1, 24*7, params["dose_interval"])
        
        if params["method"] == "First order":
            _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(
                y0, params["tau"], params["B"], params["$k_{abs}$"], params["$k_{el}$"], dose_times, params["end_time"]
            )
        elif params["method"] == "Michaelis-Menten":
            _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_Michaelis_Menten_ode(
                y0, params["tau"], params["B"], params["$V_{max}$"], params["$k_{m}$"], params["$k_{el}$"], dose_times, params["end_time"]
            )
        epsilon = 1e-3
        if (np.isclose(minima[-2], minima[-3], atol=epsilon) and np.isclose(maxima[-2], maxima[-3], atol=epsilon) and
            np.isclose(means[-2], means[-3], atol=epsilon) and np.isclose(time_to_peak[-2], time_to_peak[-3], atol=epsilon) ):
            minima_list.append(minima[-2])
            maxima_list.append(maxima[-2])
            means_list.append(means[-2])
            time_to_peak_list.append(time_to_peak[-2])
        else:
            print("\033[91mWarning: The steady state has not been reached yet. \033[0m")
            print(f'{variable_name} value is {params[variable_name]}')
            minima_list.append(minima[-2])
            maxima_list.append(maxima[-2])
            means_list.append(means[-2])
            time_to_peak_list.append(time_to_peak[-2])
    
    plt.figure()
    plt.scatter(variable_values, minima_list, label='minima',color='magenta')
    plt.scatter(variable_values, means_list, label='mean',color='orange')
    plt.plot(variable_values, means_list,color='orange')
    plt.scatter(variable_values, maxima_list, label='maxima',color='blue')
    plt.axhline(y=MIC, color='red', linestyle = '--', label = 'MIC')
    plt.xlabel(f'{variable_name} values')
    plt.ylabel('Max/Min concentration')
    plt.legend()
    plt.title(f'Analysis ({params["method"]}) of {variable_name}')

    plt.figure()
    plt.plot(variable_values, time_to_peak_list)
    plt.xlabel(f'{variable_name} values')
    plt.ylabel('Time to peak')
    plt.title(f'Peak times ({params["method"]})')


#Set some parameters
T12 = 1.5         #half-life of antibiotica in [h]
k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
B = 0.8           #bioavailibility
a = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)

Vmax = 1220
km = 287
MIC = 0.2

D = 250           #dosis of an antibiotics pill in [mg]
tau = 24          #rescaling with respect to [24h]

#non-dimensionalize
a = tau*a
a = 10
k = tau*k
Vmax = Vmax*tau/D
km = km/D

dose_times = np.arange(1, 24*3 + 1, 6)
end_time = 24*4

#Create dictionaries for the two models to analyze
#Dictionary with parameters for the first order model
params_first_order = {
    "method": "First order",
    "tau": tau,
    "B": B,
    "$k_{abs}$": a,
    "$k_{el}$": k,
    "dose_interval": 6,
    "end_time": end_time
}

#Dictionary with parameters for the Michaelis-Menten Kinetics model
params_Michaelis_Menten = {
    "method": "Michaelis-Menten",
    "tau": tau,
    "B": B,
    "$V_{max}$": Vmax,
    "$k_{m}$": km,
    "$k_{el}$": k,
    "dose_interval": 6,
    "end_time": end_time,
}

y0 = [0,0] #initial condotions

analyzing = True
if analyzing:
    #plot_model_concentration(y0, MIC, **params_first_order)

    #plot_model_concentration(y0, MIC, **params_Michaelis_Menten)

    B_values = np.arange(0.1, 1, 0.2)
    #sensitive_analysis("B", B_values, y0, MIC, **params_first_order)

    absorption_values = np.array([0.1, 1, 10, 100, 1000])
    #sensitive_analysis("$k_{abs}$", absorption_values, y0, MIC, **params_first_order)

    elimination_values = np.array([0.1, 1, 10, 50, 100])
    sensitive_analysis("$k_{el}$", elimination_values, y0, MIC, **params_first_order)

    dose_intervals = np.arange(1, 12, 1)
    #analyze_model("dose_interval", dose_intervals, y0, MIC, **params_first_order)

    absorption_values = np.arange(1, 100, 1)
    #analyze_model("$k_{abs}$",absorption_values, y0, MIC, **params_first_order)

    elimination_values = np.arange(0.1, 10, 0.1)
    #analyze_model("k_{abs}", elimination_values,y0, MIC, **params_first_order)

    Vmax_values = np.arange(0.1, 10, 0.1)
    #analyze_model("$V_{max}$",Vmax_values,y0, MIC, **params_Michaelis_Menten)

    km_values = np.arange(10, 200, 10)
    #analyze_model("$k_{m}$", km_values,y0, MIC, **params_Michaelis_Menten)

    elimination_values = np.arange(0.1, 10, 0.1)
    #analyze_model("$k_{el}$", elimination_values,y0, MIC, **params_Michaelis_Menten)


heatmap = False
if heatmap:


    #Varying absorption and elimination rate
    annot = False

    interval = 6
    elimination_rates = np.arange(0.1, 1, 0.1)
    absorption_rates = np.arange(0.1, 1, 0.1)

    minima_values = np.zeros((len(absorption_rates), len(elimination_rates)))
    maxima_values = np.zeros((len(absorption_rates), len(elimination_rates)))
    mean_values   = np.zeros((len(absorption_rates), len(elimination_rates)))

    for i, a in enumerate(absorption_rates):
        for j, k in enumerate(elimination_rates):
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
    sns.heatmap(minima_values, annot=annot, fmt=".2f", cmap ='viridis',
                xticklabels=np.round(elimination_rates, 1), yticklabels=np.round(absorption_rates, 1),
                cbar_kws={'label': 'Minima concentration'})
    plt.title('Minima concentration')
    plt.xlabel('Elimination rate')
    plt.ylabel('Absorption rate')

    #Maxima heatmap
    plt.figure()
    sns.heatmap(maxima_values, annot=annot, fmt=".2f", cmap ='viridis',
                xticklabels=np.round(elimination_rates, 1), yticklabels=np.round(absorption_rates, 1),
                cbar_kws={'label': 'Maxima concentration'})
    plt.title('Maxima concentration')
    plt.xlabel('Elimination rate')
    plt.ylabel('Absorption rate')

    #Mean heatmap
    plt.figure()
    sns.heatmap(mean_values, annot=annot, fmt=".2f", cmap ='viridis',
                xticklabels=np.round(elimination_rates, 1), yticklabels=np.round(absorption_rates, 1),
                cbar_kws={'label': 'Mean concentration'})
    plt.title('Mean concentration')
    plt.xlabel('Elimination rate')
    plt.ylabel('Absorption rate')


    Vmax_values = np.arange(1, 100, 1)
    km_values = np.arange(1, 100, 1)

    minima_values = np.zeros((len(km_values), len(Vmax_values)))
    maxima_values = np.zeros((len(km_values), len(Vmax_values)))
    mean_values   = np.zeros((len(km_values), len(Vmax_values)))

    for i, km in enumerate(km_values):
        for j, Vmax in enumerate(Vmax_values):
            dose_times = np.arange(1, 24*3, interval)

            _, _, _, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_Michaelis_Menten_ode(y0, tau, B, Vmax, km, k,
                                                                                                        dose_times,end_time)
            
            #Extract values where a steady state has been obtained
            minima_values[i, j] = minima[-2]
            maxima_values[i, j] = maxima[-2]
            mean_values[i, j]   = means[-2]

    #Plot heatmaps with seaborn
    sns.set_theme(style='white',font_scale=1.0) # Seaborn style setting
    vmin = 0.0
    vmax = 5.0 

    #ticks would be Vmax and km
    #Minima heatmap
    plt.figure()
    sns.heatmap(minima_values, vmin = vmin, vmax = vmax, annot=annot, fmt=".2f", cmap ='viridis',
                cbar_kws={'label': 'Minima concentration'})
    plt.title('Minima concentration')
    plt.xlabel('$V_{max}$')
    plt.ylabel('$K_m$')

    #Maxima heatmap
    plt.figure()
    sns.heatmap(maxima_values, vmin = vmin, vmax = vmax, annot=annot, fmt=".2f", cmap ='viridis',
                cbar_kws={'label': 'Maxima concentration'})
    plt.title('Maxima concentration')
    plt.xlabel('$V_{max}$')
    plt.ylabel('$K_m$')

    #Mean heatmap
    plt.figure()
    sns.heatmap(mean_values, annot=annot, vmin = vmin, vmax = vmax, fmt=".2f", cmap ='viridis',
                cbar_kws={'label': 'Mean concentration'})
    plt.title('Mean concentration')
    plt.xlabel('$V_{max}$')
    plt.ylabel('$K_m$')



plt.show()