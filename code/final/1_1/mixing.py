import numpy as np
# import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm
import pickle

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
# def pdf_to_cdf(pdf_func, a = -3, b = 3, dx = .01):
#     # WILSON: from -inf to a: the mass is missing
#     X  = np.arange(a,b,dx)
#     Y  = pdf_func(X)
#     Y /= (dx*Y).sum()
#     CY = np.cumsum(Y*dx)
#     step = sm.distributions.empirical_distribution.StepFunction(x = X, y = CY,side='left')
#     return step

def pdf_normalize(pdf_func, a = -3, b = 3, dx = .01):
    # WILSON: from -inf to a: the mass is missing
    X  = np.arange(a,b,dx)
    Y  = pdf_func(X)
    Y /= (dx*Y).sum()
    step = sm.distributions.empirical_distribution.StepFunction(x = X, y = Y,side='left')
    return step

# def l1(cdf1, cdf2, a=-3, b=3, dx=.01):
#     X  = np.arange(a,b,dx)
#     val1 = cdf1(X)
#     val2 = cdf2(X)
#     return np.mean(abs(val1 - val2))
# ====================================================================================
# plt.figure()
# graph(ecdf, range(-3, 3))
# graph(station_cdf, range(-3, 3))
def pdf_l1(emp_samples, station_func, a= -3, b=3, dx=.01):
    # ecdf = sm.distributions.empirical_distribution.ECDF(emp_samples)
    X  = np.arange(a,b,dx)
    kde = sp.stats.gaussian_kde(emp_samples, bw_method=None)
    e_pdf = kde.evaluate(X) # vector of pdf value

    station_pdf = pdf_normalize(pdf_func = station_func, a=a, b=b, dx=dx)
    t_pdf = station_pdf(X)

    return np.mean(abs(e_pdf - t_pdf))

# def is_mix(emp_samples, station_func, epsilon_norm = 1e-2, a= -3, b=3, dx=.01):
#     """determine from a sample, whether it is mixing with the true distribution

#     Determines if mix.

#     Parameters
#     ----------
#     emp_samples : vec
#         The emp samples
#     station_func : func
#         The stationart distribution
#     epsilon_norm : float, optional
#         The epsilon tolerance for norm
#     a : int, optional
#         Description
#     b : int, optional
#         Description
#     dx : float, optional
#         Description

#     Returns
#     -------
#     bool
#         True if mix, False otherwise.
#     """

#     ecdf = sm.distributions.empirical_distribution.ECDF(emp_samples)
#     station_cdf = pdf_to_cdf(pdf_func = station_func, a=a, b=b, dx=dx)

#     l1_norm = cdf_l1(cdf1 = ecdf, cdf2 = station_cdf, a=a, b=b, dx=dx)
#     if l1_norm <= epsilon_norm:
#         return True
#     else:
#         return False
# ====================================================================================
def mixing_time(all_traject, station_func, epsilon_norm = 1/4, a=-3, b=3, dx=.01):
    """calculate mixing time

    Parameters
    ----------
    all_traject : TYPE
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
    N_total = all_traject.shape[0]
    l1_all = np.empty(shape = N_total, dtype = 'float')

    for it in range(0,N_total):
        l1_all[it] = pdf_l1(emp_samples = all_traject[it,:], station_func = station_func, a=a, b=b, dx=dx)

    # plt.plot(l1_all)
    # plt.savefig('yo.pdf', format='pdf')

    pickle.dump(obj = l1_all, file = open('./l1_all.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # l1_all = pickle.load('l1_all.pickle')

    is_mix = l1_all <= epsilon_norm

    mix_index = np.where(is_mix)[0]
    if len(mix_index) == 0:
        return -1
    else:
        return mix_index[0]

