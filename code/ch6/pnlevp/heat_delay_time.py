import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from match_1pevp import train, evaluate
from match_1pevp.nonparametric import beyn, loewner
from helpers_test import runTest
import pandas as pd
import time

def define_problem(method : str,M):
    center, radius = -1., 1.
    N, l, k, f0, f1, t1, t2 = M, np.pi, .02, -.1, .05, 1., 2. # scalar constants appearing in the problem
    stiff = sparse.diags([-2 * np.ones(N - 1)] + [np.ones(N - 2)] * 2, [0, -1, 1], format = "csc") * (N / l) ** 2
    eye = sparse.eye(N - 1, format = "csc", dtype = complex) # identity
    # parametric matrix L(z,p) that defines the pEVP
    L_base = lambda z, p: k * stiff + (f0 - z - f1 * np.exp(-t1 * z) - p * np.exp(-t2 * z)) * eye
    L = lambda z, p: L_base(z + center, p) # center at z=0
    
    # define parameter range
    p_range = [0.01, .14]
    
    # define parameters for training
    l_sketch = 30 # number of sketching directions in Beyn's method
    lhs = np.random.randn(l_sketch, N - 1) + 1j * np.random.randn(l_sketch, N - 1) # left sketching matrix
    rhs = np.random.randn(N - 1, l_sketch) + 1j * np.random.randn(N - 1, l_sketch) # right sketching matrix
    
    if method == "beyn":
        train_nonpar = lambda L: beyn(L, 0., radius, lhs, rhs, 1000, 1e-10, 5)
    elif method == "loewner":
        lint = 0 + 2*radius * np.exp(1j * np.linspace(-0.9 * np.pi/2, 0.9 * np.pi/2, l_sketch)) # left interpolation points
        rint = 0 + 3*radius * np.exp(1j * np.linspace(-0.5 * np.pi/2, 0.5 * np.pi/2, l_sketch)) # right interpolation points
        train_nonpar = lambda L: loewner(L, 0., radius, lhs, rhs, lint, rint, 1000, 1e-10)
    
    cutoff = lambda x: x[np.abs(x) <= radius]
    bounds = [[-2, 0], [-1, 1]]
    
    return p_range, L, train_nonpar, [], cutoff, center, bounds

if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('match_1pevp').setLevel(logging.INFO)

    np.random.seed(42)

    # method = "beyn"
    method = "loewner"
    Mlist = np.array([100, 400, 800, 1600, 3000])
    trainingMace = np.zeros(Mlist.shape)
    runningMace = np.zeros(Mlist.shape)
    Loewnertime = np.zeros(Mlist.shape)
    for i in range(Mlist.shape[0]):
        p_range, L, train_nonpar, train_nonpar_args, cutoff, center, bounds = define_problem(method, Mlist[i])
        tol = 1e-7 # tolerance for outer adaptive loop
        interp_kind = "spline7" # interpolation strategy (degree-7 splines)
        patch_width = 11 # minimum width of interpolation patches in case of bifurcations
        # train
        ps = np.linspace(*p_range, 500)
        startmatchtrain = time.time()
        model, ps_train = train(L, train_nonpar, train_nonpar_args, cutoff, interp_kind,
                                patch_width, p_range, tol)
        endmatchtrain = time.time()
        getApprox = lambda p: evaluate(model, ps_train, p, interp_kind, patch_width, cutoff)
        def getExact(p): # reference solution
            return train_nonpar(lambda z: L(z, p), *train_nonpar_args)
        # check timings
        startmatch = time.time()
        for j,p in enumerate(ps):
            getApprox(p)
        endmatch = time.time()
        startloewner = time.time()
        for j,p in enumerate(ps):
            getExact(p)
        endloewner = time.time()
        
        trainingMace[i] = endmatchtrain - startmatchtrain
        runningMace[i] = endmatch - startmatch
        Loewnertime[i] = endloewner - startloewner
        print("The time it took for training MACE is", trainingMace[i])
        print("The time it took for MACE is", runningMace[i])
        print("The time it took for Loewner is", Loewnertime[i])
    
    # plot approximation and error
    plt.figure(figsize = (15, 5))
    plt.subplot(141)
    plt.semilogy(Mlist, Loewnertime, 'k')
    plt.semilogy(Mlist, trainingMace, 'r-')
    plt.semilogy(Mlist, runningMace, 'r--')
    plt.xlabel("M"), plt.ylabel("time")
    plt.tight_layout(), plt.show()
    
    