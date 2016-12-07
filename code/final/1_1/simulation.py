import numpy as np
# import matplotlib.pyplot as plt
import random
import scipy as sp
import statsmodels as sm
import ipyparallel

def simu_all(n_sim, func, grad, initialPoint=1., stepsize=1e-2/2, noiseLevel=1e-1, maxIter = int(1e5), desiredObj = 100, verbose = False):
    trajectory_all = []
    image_all = []
    haltIter_all = []
    for it in range(0,n_sim):
        print(it)
        trajectory, image, haltIter = sgd_base.GD(func, grad, initialPoint=initialPoint, stepsize=stepsize,
                                 noiseLevel=noiseLevel, maxIter=maxIter, desiredObj=desiredObj, verbose = verbose)
        trajectory_all.append(trajectory)
        image_all.append(image)
        haltIter_all.append(haltIter)
    return trajectory_all, image_all, haltIter_all


# $ ipcluster start -n 3
# $ ipcluster start -n 24
def simu_all_parallel(n_sim, func, grad, initialPoint=1., stepsize=1e-2/2, noiseLevel=1e-1, maxIter = int(1e5), desiredObj = 100, burn_in = 1e3):
    clients = ipyparallel.Client()
    dview = clients.direct_view()

    # # sync_imports does not support
    # # import foo as bar
    # # therefore need rewrite module names
    # with dview.sync_imports():
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import random
    #     import proof_of_concept as sgd_base
    #     import mixing as mixing
    #     import convergence as convergence

    # parallel cannot use lambda function either

    dview.execute('import numpy as np')
    dview.execute('import matplotlib.pyplot as plt')
    dview.execute('import random')
    dview.execute('import proof_of_concept as sgd_base')
    dview.execute('import mixing as mixing')
    dview.execute('import convergence as convergence')

    dview.push(dict(func = func, grad = grad))

    n_core = len(clients.ids)
    n_sim_each = int(np.floor(n_sim/n_core))

    results = dview.map_sync(simu_all, [n_sim_each]*n_core, [func]*n_core, [grad]*n_core)

    all_traject = np.empty(shape = maxIter)
    for it_tuple in results:
        all_traject = np.column_stack((all_traject, np.concatenate(it_tuple[0], axis=1)))

    # remove the empty column
    all_traject = np.delete(all_traject, 0, axis=1)
    # remove burn in period
    all_traject = all_traject[int(burn_in):,:]

    return all_traject

# %time all_traject = simu_all_parallel(n_sim = 1e2, func = func, grad = grad, initialPoint=1., stepsize=1e-2/2, noiseLevel=1e-1, maxIter=int(1e5), desiredObj=100)
