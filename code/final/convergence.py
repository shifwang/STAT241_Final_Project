import numpy as np
# import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm
def convergence_time(one_trajectory, modes, epsilon_neighbor = 1e-2):
    n_mode = len(modes)
    converge_t_by_mode = []

    for it in range(0,n_mode):
        time_reach = np.where(abs(np.array(one_trajectory) - modes[it]) < epsilon_neighbor)[0]
        if len(time_reach) == 0:
            # indicate not converge to the mode with -1
            time_here = -1
        else:
            time_here = time_reach[0]
        converge_t_by_mode.append(time_here)

    if all(np.array(converge_t_by_mode) > 0):
        # if converge to all modes,
        # we return the convergence time as the time trajectory reaches all modes
        return max(converge_t_by_mode)
    else:
        # if not converge, return -1
        return -1

# convergence_time(one_trajectory = one_trajectory, modes = [100], epsilon_neighbor=1e-2)
def convergence_time_all(all_trajectory, modes, epsilon_neighbor = 1e-2):
    n_total = all_trajectory.shape[1]
    time_all = np.empty(shape = n_total)
    for it in range(0,n_total):
        time_all[it] = convergence_time(one_trajectory = all_trajectory[:,it], modes = modes, epsilon_neighbor=epsilon_neighbor)
    return time_all