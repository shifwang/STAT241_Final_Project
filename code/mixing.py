import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm

def graph(formula, x_range = range(-3, 3)):
    '''
    graph any function within a range
    '''
    # x = np.array(x_range)
    x = np.linspace(start = x_range[0], stop = x_range[len(x_range)-1], num=50, endpoint=True)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)
    plt.show()
# ====================================================================================
def pdf_to_cdf(pdf_func, a = -3, b = 3, dx = .01):
    # WILSON: from -inf to a: the mass is missing
    X  = np.arange(a,b,dx)
    Y  = pdf_func(X)
    Y /= (dx*Y).sum()
    CY = np.cumsum(Y*dx)
    step = sm.distributions.empirical_distribution.StepFunction(x = X, y = CY,side='left')
    return step

def cdf_l1(cdf1, cdf2, a=-3, b=3, dx=.01):
    X  = np.arange(a,b,dx)
    val1 = cdf1(X)
    val2 = cdf2(X)
    return np.mean(abs(val1 - val2))
# ====================================================================================
# plt.figure()
# graph(ecdf, range(-3, 3))
# graph(station_cdf, range(-3, 3))

def is_mix(emp_samples, station_func, epsilon_norm = 1e-2, a=-3, b=3, dx=.01):
    ecdf = sm.distributions.empirical_distribution.ECDF(emp_samples)
    station_cdf = pdf_to_cdf(pdf_func = station_func, a=a, b=b, dx=dx)

    l1_norm = cdf_l1(cdf1 = ecdf, cdf2 = station_cdf, a=a, b=b, dx=dx)
    if l1_norm <= epsilon_norm:
        return True
    else:
        return False
# ====================================================================================

def mixing_time(all_trajectory, station_func, epsilon_norm = 1e-2):
    pass


