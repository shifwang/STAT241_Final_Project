import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm

plt.interactive(True)

import ipyparallel

import proof_of_concept as sgd_base
import mixing
import convergence
import simulation
# %run mixing
# %run convergence
%run simulation


# EG1: f(x) = -x**4
degree = 4
maxIter = int(1e5)
def func(x, degree = degree):
    return -np.power(x, degree)
def grad(x, degree = degree):
    return -degree * np.power(x, degree - 1)
def station_func(x):
    return np.exp(func(x))
modes = [0]

# trajectory, image, haltIter = sgd_base.GD(func, grad, initialPoint=1., stepsize=1e-2/2,
#                                  noiseLevel=1e-1, maxIter=maxIter, desiredObj=100)
# plt.hist(trajectory[haltIter/2:haltIter], 100)
# ha = trajectory[haltIter/2:haltIter]

# all_simu = simulation.simu_all_parallel(n_sim = 1e1, func = func, grad = grad, initialPoint=1., stepsize=1e-2/2, noiseLevel=1e-1, maxIter=int(1e5), desiredObj=100)
all_traject = simu_all_parallel(n_sim = 1e2, func = func, grad = grad, initialPoint=1., stepsize=1e-2/2, noiseLevel=1e-1, maxIter=int(1e5), desiredObj=100)

mix_time = mixing.mixing_time(all_trajectory = all_traject, station_func = station_func, epsilon_norm=1e-1, a=-3, b=3, dx=.01)

conv_time_all = convergence.convergence_time_all(all_trajectory = all_traject, modes = modes, epsilon_neighbor=1e-2)


np.median(conv_time_all)
mix_time
