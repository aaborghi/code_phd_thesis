import numpy as np
from numpy import exp
from numpy import identity
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from match_1pevp import train, evaluate, match
from match_1pevp.nonparametric import beyn, loewner
from helpers_test import runTest
import pandas as pd

def define_problem(method : str):
    radius = 1.5
    center = 1.35+0.15j
    Ri = 1
    Ro = -1
    n = 4.5
    
    mu = 0.35476895
    tau = 2.62945941
    def dD(vars):
        x, y = vars 
        z = x + 1j*y
        val = 1j*Ri*Ro*exp(2j*z)*(n*exp(1j*z*tau)*(tau+2)+4) - 1j*n*exp(1j*z*tau)*(tau+(mu**2)*(tau+1)*(Ro-Ri)*exp(1j*z))
        return [val.real, val.imag]
    root = fsolve(dD, [1.34,0.15], xtol=1e-12)

    def dD_(vars):
        x, y, m, t = vars 
        z = x + 1j*y
        val = 1j*Ri*Ro*exp(2j*z)*(n*exp(1j*z*t)*(t+2)+4) - 1j*n*exp(1j*z*t)*(t+(m**2)*(t+1)*(Ro-Ri)*exp(1j*z))
        val2 = (m**2)*n*exp(1j*z*t)*exp(1j*z)*(Ro-Ri) - (n*exp(1j*z*t)+2)*(Ro*Ri*exp(2*1j*z)-1)
        return [val.real, val.imag, val2.real, val2.imag]
    root = fsolve(dD_, [1.35,0.15, 0.35, 2.63], xtol=1e-12)
    print(root) # EP
    
    L_notshifted = lambda z, p: np.array([
        [-1, Ri*exp(1j*z), 0, 0, 0, p*Ri*exp(1j*z)/2],
        [0, -1, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, p/2],
        [0, 0, Ro*exp(1j*z), -1, 0, 0],
        [p, -p, 0, 0, -1, -0.5],
        [0, 0, 0, 0, n*exp(1j*z*tau), -1]
    ])
    L = lambda z, p: L_notshifted(z+center,p)
    
    # define parameter range
    p_range = [0, 1]
    
    # define parameters for training
    l_sketch = 3 # number of sketching directions 
    lhs = np.random.randn(l_sketch, 6) + 1j * np.random.randn(l_sketch, 6) # left sketching matrix
    rhs = np.random.randn(6, l_sketch) + 1j * np.random.randn(6, l_sketch) # right sketching matrix
    
    if method == "beyn":
        train_nonpar = lambda L: beyn(L, 0., radius, lhs, rhs, 25, 1e-10, 1)
    elif method == "loewner":
        lint = 0 + 3*radius * np.exp(1j * np.linspace(1.4 * np.pi/2, 2.6 * np.pi/2, l_sketch)) # left interpolation points
        rint = 0 + 3*radius * np.exp(1j * np.linspace(-0.6 * np.pi/2, 0.6 * np.pi/2, l_sketch)) # right interpolation points
        train_nonpar = lambda L: loewner(L, 0., radius, lhs, rhs, lint, rint, 80, 1e-10)
    
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
    tol = 1e-4 # tolerance for outer adaptive loop
    interp_kind = "spline7" # interpolation strategy (piecewise-linear hat functions)
    patch_width = 11# minimum width of interpolation patches in case of bifurcations

    # train
    model, ps_train = train(L, train_nonpar, train_nonpar_args, cutoff, interp_kind,
                            None, p_range, tol)
    
    p_lim = [0,1]
    ps_n = np.linspace(0, 0.33, 100, endpoint=False) 
    ps_neg = np.logspace(np.log10(0.33), np.log10(0.35476895), 100, endpoint=False) 
    ps_pos = np.logspace(np.log10(0.35476895), np.log10(0.36), 100, endpoint=False)  
    ps_p = np.linspace(0.37, 1.1, 100)
    ps = np.concatenate((ps_n, ps_neg, ps_pos, ps_p))
    ps_coarse = ps[::10] # coarse testing grid
    getApprox = lambda p: evaluate(model, ps_train, p, interp_kind, patch_width, cutoff)
    def getExact(p): # reference solution
        return train_nonpar(lambda z: L(z, p), *train_nonpar_args)
    val_app, val_ref, error = runTest(ps, 10, getApprox, getExact) # run testing routine
    val_app, val_ref = val_app, val_ref # shift to original range
    print(ps)
    
    
    Ri = 1
    Ro = -1
    n = 4.5
    tau = 2.62945941
    F = lambda w: n*exp(1j*w*tau)
    def detL(vars,p):
        x, y = vars 
        z = x + 1j*y
        val = (p**2)*F(z)*exp(1j*z)*(Ro-Ri) - (F(z)+2)*(Ro*Ri*exp(2*1j*z)-1)
        return [val.real, val.imag]
    root = np.empty((2,30), dtype = complex)
    p_prime = np.linspace(*p_range, 100)
    p_ = p_prime[::10]
    for i in range(0,p_.size):
        root[:,i] = fsolve(detL, [1.1,0.6], args= (p_[i],), xtol=1e-12)
        root[:,i+p_.size] = fsolve(detL, [1.5,-0.5], args= (p_[i],), xtol=1e-12)
        root[:,i+2*p_.size] = fsolve(detL, [1,-0.5], args= (p_[i],), xtol=1e-12)

    
    
    val_app = val_app + center
    EP = 1.34556417 + 0.14803129j # Exceptional point
    plt.figure(figsize = (5, 5))
    plt.plot(np.real(val_app), np.imag(val_app), 'r-')
    plt.plot(root[0,:], root[1,:],'k.')
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
    plt.ylim(*p_lim) 
    plt.subplot(142)
    plt.plot(np.imag(val_app), ps, 'b-')
    plt.xlabel("Im($\lambda$)")
    plt.ylim(*p_lim) 
    plt.subplot(143)
    plt.plot(np.linspace(1,len(ps_train),len(ps_train)), ps_train, 'k.')
    plt.xlabel("$p_{train}$")
    plt.ylim(*p_lim) 
    plt.subplot(144)
    plt.semilogx(error, ps_coarse)
    plt.semilogx([tol] * 2, p_range, 'k:')
    plt.ylim(*p_lim) 
    plt.xlabel("lambda error"), plt.ylabel("p")
    plt.tight_layout(), plt.show()

    # printing out identification of bifurcation (see where 2 pops out)
    idbif = np.zeros(len(model[1][:])+1)
    for j in range(0,len(model[1][:])):
        for c in model[1][j]:
            print(len(c))
            idbif[j] = len(c)
    