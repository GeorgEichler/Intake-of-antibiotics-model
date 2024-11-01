import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

'''
#Saturable absorption doesn't seem that promising as concentration level just rises
def two_stage_Michaelis_Menten_ode(t, y, B, Vmax, km, a):
    b, s = y
    dbdt = B*a*s - Vmax/(km + b)*b
    dsdt = -a*s

    return [dbdt, dsdt]
'''

#Michaelis-Menten kinetics for the absorption rate
def two_stage_Michaelis_Menten_ode(t, y, B, Vmax, km, k):
    b, s = y
    dbdt = B*Vmax/(km + s)*s - k*b 
    dsdt = -Vmax/(km + s)*s

    return [dbdt, dsdt]


def simulate_two_stage_Michaelis_Menten_ode(y0, tau, B, Vmax, km, k, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    dose_interval = dose_times[1] - dose_times[0]
    
    #measure variables
    maxima =       []
    minima =       []
    b_auc =        []
    time_to_peak = []

    
    dose_times = 1/tau*dose_times
    dose_interval = (1/tau)*dose_interval
    end_time = end_time/tau

    times = np.concatenate(([0], dose_times, [end_time] ))
    
    for i in range(len(times) - 1):
        t_eval = np.linspace(times[i], times[i+1], 1001)

        sol = solve_ivp(two_stage_Michaelis_Menten_ode, (times[i], times[i+1]), y0, t_eval=t_eval, args=(B, Vmax, km, k))
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

if __name__ == '__main__':
    y0 = [0,0]
    D = 250
    tau = 24
    B = 0.8
    Vmax = 1220
    km = 287
    T12 = 1.5
    k = np.log(2)/T12

    #non-dimensionalize parameters
    Vmax = Vmax*tau/D
    km = km/D
    k = k*tau

    dose_times = np.arange(6, 24*3+1, 6)
    end_time = 24*4
    

    t, b, s, _, _, _, _, _ = simulate_two_stage_Michaelis_Menten_ode(y0, tau, B, Vmax, km, k, dose_times, end_time)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    b_line, = plt.plot(t, b, label = 'Antibiotics in bloodstream')
    s_line, = plt.plot(t, s, label = 'Antibiotics in stomach', alpha = 0.2)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()

    #Add sliders for parameters
    axcolor = 'lightgoldenrodyellow'
    ax_Vmax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_km = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_k = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_Vmax = Slider(ax_Vmax, 'Maximal absorption (Vmax)', 0.1, 1000, valinit=Vmax)
    slider_km = Slider(ax_km, 'Half concentration (km)', 0.01, 10, valinit=km)
    slider_k = Slider(ax_k, 'Elimination rate (k)', 0.1, 100, valinit=k)

    #Update function for sliders
    def update(val):
        new_Vmax = slider_Vmax.val 
        new_km = slider_km.val
        new_k = slider_k.val
        t, new_b_values, new_s_values, _, _, _, _, _ = simulate_two_stage_Michaelis_Menten_ode(y0, tau, B, new_Vmax, new_km,
                                                                                               new_k, dose_times, end_time)
        b_line.set_ydata(new_b_values)
        s_line.set_ydata(new_s_values)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    #Attach update function to slider
    slider_Vmax.on_changed(update)
    slider_km.on_changed(update)
    slider_k.on_changed(update)

    plt.show()