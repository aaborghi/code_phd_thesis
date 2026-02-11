import numpy as np
from numpy import exp
from numpy import identity
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from match_1pevp import train, evaluate, match
from match_1pevp.nonparametric import beyn, loewner
from helpers_test import runTest
from match_1pevp.helpers import interp1d
import pandas as pd

def define_problem(method : str):
    radius = 2
    center = 0+0j
    
    
    L_notshifted = lambda z, p: np.array([
        [z-1, 1j*p],
        [1j*p, z+1]
    ])
    L = lambda z, p: L_notshifted(z+center,p)
    
    # define parameter range
    p_range = [0.3,1.7]
    
    # define parameters for training
    l_sketch = 3 # number of sketching directions 
    lhs = np.random.randn(l_sketch, 2) + 1j * np.random.randn(l_sketch, 2) # left sketching matrix
    rhs = np.random.randn(2, l_sketch) + 1j * np.random.randn(2, l_sketch) # right sketching matrix
    
    if method == "beyn":
        train_nonpar = lambda L: beyn(L, 0., radius, lhs, rhs, 40, 1e-10, 1)
        lint = 0
        rint = 0
    elif method == "loewner":
        lint = 0 + 2*radius * np.exp(1j * np.linspace(1.4 * np.pi/2, 2.6 * np.pi/2, l_sketch)) # left interpolation points
        rint = 0 + 2*radius * np.exp(1j * np.linspace(-0.6 * np.pi/2, 0.6 * np.pi/2, l_sketch)) # right interpolation points
        train_nonpar = lambda L: loewner(L, 0., radius, lhs, rhs, lint, rint, 40, 1e-10)
    
    cutoff = lambda x: x[np.abs(x) <= radius]
    bounds = [[np.real(center) - radius, np.real(center) + radius],
              [np.imag(center) - radius, np.imag(center) + radius]]
    
    contour = center + radius*exp(np.linspace(0,2*np.pi,100)*1j)
    return p_range, L, train_nonpar, [], cutoff, center, bounds, [lint, rint, contour]

if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('match_1pevp').setLevel(logging.INFO)

    np.random.seed(42)

    method = "loewner"
    
    p_range, L, train_nonpar, train_nonpar_args, cutoff, center, bounds, parameters = define_problem(method)
    tol = 1e-2 # tolerance for outer adaptive loop
    interp_kind = "linear" # interpolation strategy (piecewise-linear hat functions)
    min_patch_deltap = 0.1 # minimum width of interpolation patches in case of bifurcations
    
    # train
    model, ps_train = train(L, train_nonpar, train_nonpar_args, cutoff, interp_kind,
                            None, p_range, tol, d_thresh = 0.01, min_patch_deltap = min_patch_deltap)
    

    # test
    
    ps_n = np.linspace(-0.7, -0.3, 100, endpoint=False) 
    ps_neg = -np.logspace(np.log10(0.3), -8, 100, endpoint=False) 
    ps_pos = np.logspace(-8, np.log10(0.3), 100, endpoint=False)  
    ps_p = np.linspace(0.3, 0.7, 100)
    ps = np.concatenate((ps_n, ps_neg, ps_pos, ps_p))+1
    getApprox = lambda p: evaluate(model, ps_train, p, interp_kind, None, cutoff)
    
    def getExact(p): # exact solution
        val1 = lambda p: np.sqrt(1 + (1j*p)**2)
        val2 = lambda p: -np.sqrt(1 + (1j*p)**2)
        v_ref = np.array([val1(p), val2(p)])
        return cutoff(v_ref)
    
    val_app, val_ref, error = runTest(ps, 1, getApprox, getExact)
    val_app, val_ref = val_app + center, val_ref + center

    EP = 0
    plt.figure(figsize = (5, 5))
    plt.plot(np.real(val_app), np.imag(val_app), 'r-')
    plt.plot(np.real(EP), np.imag(EP), 'bo')
    plt.plot(np.real(parameters[0]+center), np.imag(parameters[0]+center), 'kP')
    plt.plot(np.real(parameters[1]+center), np.imag(parameters[1]+center), 'kP')
    plt.plot(np.real(parameters[2]), np.imag(parameters[2]), 'k-')
    plt.xlabel("Re($\lambda$)"), plt.ylabel("Im($\lambda$)")
    plt.tight_layout(), plt.show()
    
    plt.figure(figsize = (7, 5))
    plt.subplot(141)
    plt.plot(np.real(val_app), ps, 'r-')
    plt.xlabel("Re($\lambda$)")
    plt.ylabel("$p$")
    plt.subplot(142)
    plt.plot(np.imag(val_app), ps, 'b-')
    plt.xlabel("Im($\lambda$)")
    plt.subplot(143)
    plt.plot(np.linspace(1,len(ps_train),len(ps_train)), ps_train, 'k.')
    plt.xlabel("$p_{train}$")
    plt.subplot(144)
    plt.semilogx(error, ps, 'k-')
    plt.xlabel("error")
    plt.tight_layout(), plt.show()

    # printing out identification of bifurcation (see where 2 pops out)
    idbif = np.zeros(len(model[1][:])+1)
    for j in range(0,len(model[1][:])):
        for c in model[1][j]:
            print(len(c))
            idbif[j] = len(c)
    
    
    