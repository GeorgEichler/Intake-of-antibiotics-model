import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from matplotlib.widgets import Slider

def two_stage_ode(t, y, B, a, k):
    b, s = y
    dbdt = B*a*s - k*b
    dsdt = -a*s
    return [dbdt, dsdt]

def simulate_two_stage_ode(y0, tau, B, a, k, dose_times, end_time):
    t = np.array([])
    b = np.array([])
    s = np.array([])
    dose_interval = dose_times[1] - dose_times[0]

    #Measure variables
    maxima =       []
    minima =       []
    b_auc  =       []
    time_to_peak = []

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


if __name__ == '__main__':

    T12 = 1.5         #half-life of antibiotica in [h]
    k = np.log(2)/T12 #degredation rate of antibiotica in [1/h]
    B = 0.8           #bioavailibility
    a = 1             #absorption rate of antibiotics in the stomach [in 1/h] (need same units as k)

    D = 100           #dosis of an antibiotics pill in [mg]
    tau = 24          #rescaling with respect to [24h]

    #non-dimensionalize parameters
    a = tau*a
    k = tau*k

    dose_times = np.arange(6, 24*3 + 1, 6)
    end_time = 24*4
            
    y0 = [0,0] #initial condotions

    t, b, s, minima, maxima, b_auc, means, time_to_peak = simulate_two_stage_ode(y0, tau, B, a, k, dose_times, end_time)

    # Plotting setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    b_line, = plt.plot(t, b, label='Antibiotics in bloodstream', color = 'red')
    s_line, = plt.plot(t, s, label='Antibiotics in stomach', alpha=0.2, color = 'blue')
    plt.xlabel('t')
    plt.ylabel('b,s')
    plt.legend()

    # Add sliders for parameters
    axcolor = 'lightgoldenrodyellow'
    ax_B = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_a = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_k = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_B = Slider(ax_B, 'Bioavailability (B)', 0.1, 1.5, valinit=B)
    slider_a = Slider(ax_a, 'Absorption rate (a)', 0.1, 200, valinit=a)
    slider_k = Slider(ax_k, 'Elimination rate (k)', 0.01, 200, valinit=k)

    # Update function for sliders
    def update(val):
        new_B = slider_B.val
        new_a = slider_a.val
        new_k = slider_k.val
        t, new_b_values, new_s_values, _, _, _, _, _ = simulate_two_stage_ode(y0, tau, new_B, new_a, new_k, dose_times, end_time)
        b_line.set_ydata(new_b_values)
        s_line.set_ydata(new_s_values)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    # Attach update function to slider
    slider_B.on_changed(update)
    slider_a.on_changed(update)
    slider_k.on_changed(update)

    plt.show()
