import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from match_1pevp import match

def runTest(ps, coarsen, get_approx, get_exact):
    """
    This function runs a test to compare the approximation of a function with its exact value.

    Parameters:
    ps (numpy array): The array of p-values at which the function is evaluated.
    coarsen (int): The coarsening factor for the p-values.
    get_approx (function): The function that computes the approximation.
    get_exact (function): The function that computes the exact value.

    Returns:
    val_app (numpy array): The approximated values.
    val_ref (numpy array): The exact values.
    error (numpy array): The error between the approximated and exact values.
    """
    ps_coarse = ps[::coarsen]
    val_app = np.empty((len(ps), 0), dtype = complex)
    val_ref = np.empty((len(ps_coarse), 0), dtype = complex)
    error = np.empty((len(ps_coarse), 0))
    val_ref[:], val_app[:], error[:] = np.inf, np.inf, np.nan
    for j, p in enumerate(ps):
        v_app = get_approx(p)
        Napp = len(v_app)
        dN = Napp - val_app.shape[1]
        if dN > 0: # enlarge arrays for storage
            val_app = np.pad(val_app, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
        elif dN < 0:
            v_app = np.pad(v_app, (0, - dN), constant_values = np.inf + 1j*np.inf)
            Napp = val_app.shape[1]
        # sort v to follow trajectories (note: here val_app.shape[1] == len(v_app) so the match problem is square)
        if j > 0:
            p_opt, _ = match(val_app[j - 1, :], v_app)
            if len(v_app): v_app = v_app[p_opt[1]]
        val_app[j, :] = v_app
        
        # compute error
        if not j % coarsen:
            print(":" * (j // coarsen % 5 == 0) + "." * (j // coarsen % 5 != 0), end = "")
            v_ref = get_exact(p)
            Nref = len(v_ref)
            dN = Nref - val_ref.shape[1]
            if dN > 0: # enlarge arrays for storage
                val_ref = np.pad(val_ref, [(0, 0), (0, dN)], constant_values = np.inf + 1j*np.inf)
                error = np.pad(error, [(0, 0), (0, dN)], constant_values = np.nan)
            elif dN < 0:
                v_ref = np.pad(v_ref, (0, - dN), constant_values = np.inf + 1j*np.inf)
                Nref = val_ref.shape[1]
            # sort v to follow trajectories (note: here val_ref.shape[1] == len(v_ref) so the match problem is square)
            if j > 0:
                p_opt, _ = match(val_ref[j // coarsen - 1, :], v_ref)
                if len(v_ref): v_ref = v_ref[p_opt[1]]
            val_ref[j // coarsen, :] = v_ref

            if Nref == Napp:
                p_opt, d_opt = match(v_ref, v_app)
            elif Nref < Napp:
                p_opt, d_opt = match(np.pad(v_ref, [(0, Napp - Nref)], constant_values = np.inf), v_app)
                p_opt, d_opt = (p_opt[0][: Nref], p_opt[1][: Nref]), d_opt[: Nref] # remove extras
            else: #if Nref > Napp:
                p_opt, d_opt = match(v_ref, np.pad(v_app, [(0, Nref - Napp)], constant_values = np.inf))
            if len(d_opt): error[j // coarsen, p_opt[0]] = d_opt[p_opt[0], p_opt[1]]
    print()
    return val_app, val_ref, error

def getEigenvectors(vals, ps, L, size = None):
    if size is None: size = L(0., 1.).shape[1]
    vecs = np.empty(vals.shape + (size,), dtype = complex)
    vecs[:] = np.nan
    for i, (vs, p) in enumerate(zip(vals, ps)):
        for j, v in enumerate(vs):
            if not np.isnan(v):
                vecs[i, j] = np.linalg.svd(L(v, p))[2][-1].conj()
    return vecs

def getIndicatorVal(vals, ps, L, basisv, basisnw, r = 1):
    ind = np.empty(vals.shape, dtype = float)
    ind[:] = np.nan
    for i, (vs, p) in enumerate(zip(vals, ps)):
        for j, v in enumerate(vs):
            if not np.isnan(v):
                vec = np.linalg.svd(L(v, p))[2][-1].conj()
                funv = lambda x: sum([vecj * b(x, v, p)
                                for vecj, b in zip(vec, basisv)])
                funnw = lambda x: sum([vecj * b(x, v, p)
                                for vecj, b in zip(vec, basisnw)])
                fun = lambda x: x * (np.abs(funv(x)) ** 2
                                   - np.abs(funnw(x)) ** 2)
                ind[i, j] = fun(r)
    return ind

def getIndicator(vals, ps, L, basisv, basisnw, r_min = 0, r_max = 1):
    ind = np.empty(vals.shape, dtype = float)
    ind[:] = np.nan
    for i, (vs, p) in enumerate(zip(vals, ps)):
        for j, v in enumerate(vs):
            if not np.isnan(v):
                vec = np.linalg.svd(L(v, p))[2][-1].conj()
                funv = lambda x: sum([vecj * b(x, v, p)
                                for vecj, b in zip(vec, basisv)])
                funnw = lambda x: sum([vecj * b(x, v, p)
                                for vecj, b in zip(vec, basisnw)])
                fun = lambda x: x * (np.abs(funv(x)) ** 2
                                   - np.abs(funnw(x)) ** 2)
                ind[i, j] = quad(fun, r_min, r_max)[0]
    return ind

def plotResults(val_app, val_ref, error, ps, stride_coarse,
                ps_train, tol, bounds):
    # plot approximation and exact
    plt.figure(figsize = (10, 5))
    plt.subplot(121)
    plt.plot(np.real(val_ref), np.imag(val_ref), 'k,')
    plt.xlim(*bounds[0]), plt.ylim(*bounds[1])
    plt.xlabel("Re(lambda)"), plt.ylabel("Im(lambda)"), plt.title('exact')
    plt.subplot(122)
    plt.plot(np.real(val_app), np.imag(val_app))
    plt.xlim(*bounds[0]), plt.ylim(*bounds[1])
    plt.xlabel("Re(lambda)"), plt.ylabel("Im(lambda)"), plt.title('approx')
    plt.tight_layout(), plt.show()
    
    # plot approximation and error
    plt.figure(figsize = (15, 5))
    plt.subplot(141)
    plt.plot(np.real(val_ref[:, 0]), ps[::stride_coarse], 'ro')
    plt.plot(np.real(val_app[:, 0]), ps, 'b:')
    plt.plot(np.real(val_ref), ps[::stride_coarse], 'ro')
    plt.plot(np.real(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[0]), plt.ylim(ps[0], ps[-1])
    plt.xlabel("Re(lambda)"), plt.ylabel("p")
    plt.subplot(142)
    plt.plot(np.imag(val_ref[:, 0]), ps[::stride_coarse], 'ro')
    plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
    plt.plot(np.imag(val_ref), ps[::stride_coarse], 'ro')
    plt.plot(np.imag(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[1]), plt.ylim(ps[0], ps[-1])
    plt.xlabel("Im(lambda)"), plt.ylabel("p")
    plt.subplot(143)
    plt.plot([0] * len(ps_train), ps_train, 'bx')
    plt.ylim(ps[0], ps[-1]), plt.ylabel("sample p-points")
    plt.subplot(144)
    plt.semilogx(error, ps[::stride_coarse])
    plt.semilogx([tol] * 2, [ps[0], ps[-1]], 'k:')
    plt.ylim(ps[0], ps[-1]), plt.xlabel("lambda error"), plt.ylabel("p")
    plt.tight_layout(), plt.show()