import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm

def graph(formula, x_range = range(-3, 3)):
    '''graph any function within a range

    Parameters
    ----------
    formula : TYPE
        The formula
    x_range : TYPE, optional
        The x range
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

def is_mix(emp_samples, station_func, epsilon_norm = 1e-2, a= -3, b=3, dx=.01):
    """determine from a sample, whether it is mixing with the true distribution

    Determines if mix.

    Parameters
    ----------
    emp_samples : vec
        The emp samples
    station_func : func
        The stationart distribution
    epsilon_norm : float, optional
        The epsilon tolerance for norm
    a : int, optional
        Description
    b : int, optional
        Description
    dx : float, optional
        Description

    Returns
    -------
    bool
        True if mix, False otherwise.
    """
    ecdf = sm.distributions.empirical_distribution.ECDF(emp_samples)
    station_cdf = pdf_to_cdf(pdf_func = station_func, a=a, b=b, dx=dx)

    l1_norm = cdf_l1(cdf1 = ecdf, cdf2 = station_cdf, a=a, b=b, dx=dx)
    if l1_norm <= epsilon_norm:
        return True
    else:
        return False
# ====================================================================================
def mixing_time(all_trajectory, station_func, epsilon_norm = 1e-1, a=-3, b=3, dx=.01):
    """calculate mixing time

    Parameters
    ----------
    all_trajectory : TYPE
        Description
    station_func : TYPE
        Description
    epsilon_norm : float, optional
        Description
    a : TYPE, optional
        Description
    b : int, optional
        Description
    dx : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    N_total = all_trajectory.shape[0]
    is_mix_all = np.empty(shape = N_total, dtype = 'bool')
    for it in range(0,N_total):
        is_mix_all[it] = is_mix(all_trajectory[it,:], station_func = station_func, epsilon_norm=epsilon_norm, a=a, b=b, dx=dx)
    mix_index = np.where(is_mix_all)[0]
    if len(mix_index) == 0:
        return -1
    else:
        return mix_index[0]

