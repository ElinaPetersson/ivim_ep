"""
Methods for Cramer-Rao lower bounds optmization of b-value schemes.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds, curve_fit
from ivim.models import sIVIM, diffusive, ballistic, sBallistic, intermediate, sIVIM_jacobian, diffusive_jacobian, ballistic_jacobian, sBallistic_jacobian, check_regime, NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, SBALLISTIC_REGIME, INTERMEDIATE_REGIME
from ivim.seq.sde import calc_c, G_from_b, calc_interm_pars, MONOPOLAR, BIPOLAR
from itertools import combinations


def crlb(D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], regime: str, 
         bmax: float = 1000, nbvals: int = 14,
         fitK: bool = False, K: npt.NDArray[np.float64] | None = None, 
         minbias: bool = False, bias_regime: str = DIFFUSIVE_REGIME, 
         SNR: float = 100, adjustSNR: bool = False,
         bthr: float = 200,
         Dstar: npt.NDArray[np.float64] | None = None,
         vd: npt.NDArray[np.float64] | None = None, usr_input: dict | None = None,
         v: npt.NDArray[np.float64] | None = None, tau: npt.NDArray[np.float64] | None = None):
    """
    Optimize b-values (and possibly c-values) using Cramer-Rao lower bounds optmization.

    Arguments:
        D:           diffusion coefficients to optimize over [mm2/s]
        f:           perfusion fractions to optimize over (same size as D)
        regime:      IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bmax:        (optional) the largest b-value that can be returned by the optimization
        nbvals:      (optional) total number of bvalues
        fitK:        (optional) if True, optimize with the intention to be able to fit K in addition to D and f
        minbias:     (optional) if True, include a bias term in cost function. Requires some of the remaining optional arguments
        bias_regime: (optional) specifies model to use for bias term
        K:           (optional) kurtosis coefficients to optimize over if fitK and for bias term if minbias
        SNR:         (optional) expected SNR level at b = 0 to be used to scale the influence of the bias term
        adjust_SNR:  (optional) if True, adjust SNR based on highest b-value and optimize with regards to system limitations
        usr_input:   
    ---- no regime ----
        bthr:        (optional) the smallest non-zero b-value that can be returned by the optimization
    ---- diffusive regime ----
        Dstar:       (optional) pseudodiffusion coefficients for optimization and/or bias term [mm/s]
    ---- ballistic regime ----
        vd:          (optional) velocity dispersion coefficient for optimization and/or bias term [mm/s]
        usr_input:   (optinoal) dict with maximum gradient strength, epi redout duration, 180 pulse duration and rise time
    ---- intermediate regime ----
        v:           (optional) velocities (same size as D or scalar)
        tau:         (optional) correlation times [s] (same size as D or scalar)
    
    Output:
        b:           optimized b-values
        a:           fraction of total acquisition time to spend at each b-value in b 
    ---- ballistic regime ----
        c:           optimized c-values
    """

    def cost(x, n0 = 0, nfc = 0):
        """ 
        x: vector with b-values and possibly fractions 
        n0: number of b = 0 acquisitions (only relevant for regime = 'no') 
        nfc: number of b-values with flow compensated gradients (only relevant for regime = 'ballistic' and seq = 'bipolar')
        """

        nb = (n0 + x.size) // 2 
        b = np.zeros(nb)
        b[n0:] = x[:-nb]
        a = x[-nb:]

        if (regime == BALLISTIC_REGIME) or (regime == SBALLISTIC_REGIME): 
            seq = BIPOLAR
        else:
            seq = MONOPOLAR
        
        Delta,delta,c,k,T = calc_interm_pars(b,usr_input,seq,nfc)
        

        S0 = np.ones_like(D)
        if regime == DIFFUSIVE_REGIME:
            if fitK:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0, K = K)
            else:
                J = diffusive_jacobian(b, D, f, Dstar, S0 = S0)
        elif regime == BALLISTIC_REGIME:
            if fitK:
                J = ballistic_jacobian(b, c, D, f, vd, S0 = S0, K = K)
            else: 
                J = ballistic_jacobian(b, c, D, f, vd, S0 = S0)
        elif regime == SBALLISTIC_REGIME:
            if fitK:
                J = sBallistic_jacobian(b, c, D, f, S0 = S0, K = K)
            else:
                J = sBallistic_jacobian(b, c, D, f, S0 = S0)
        else: # NO_REGIME
            if fitK:
                J = sIVIM_jacobian(b, D, f, S0 = S0, K = K)
            else:
                J = sIVIM_jacobian(b, D, f, S0 = S0)
        
        F = ((a*nbvals)[np.newaxis,np.newaxis,:]*J.transpose(0,2,1))@J
        try:
            Finv = np.linalg.inv(F)
        except:
            print('Unable to compute inv(F)')
            return np.inf
        
        if adjustSNR:       
            SNR0 = 70*np.exp(75e-3/80e-3)
            if regime == DIFFUSIVE_REGIME or regime == NO_REGIME:
                TE_lim = 2*delta + usr_input['t_180'] + usr_input['t_epi'] + 2*usr_input['t_rise']
            elif regime == BALLISTIC_REGIME or regime == SBALLISTIC_REGIME:
                TE_lim = 4*delta + usr_input['t_180'] + usr_input['t_epi'] + 4*usr_input['t_rise']
            SNR_TE_adj = SNR0*np.exp(-TE_lim*(1/usr_input['T2d']+1/usr_input['T2p']))/((1-f)*np.exp(-TE_lim/usr_input['T2p'])+f*np.exp(-TE_lim/usr_input['T2d']))
            Finv = Finv/SNR_TE_adj[:,np.newaxis,np.newaxis]**2
        else:
            Finv = Finv/SNR**2
                 
        # C = np.sum(np.sqrt(Finv[:, 0, 0])/D + np.sqrt(Finv[:, 1, 1])/f)
        C = np.sum(np.sqrt(Finv[:, 1, 1])/f)
        if regime == DIFFUSIVE_REGIME:
            C += np.sum(np.sqrt(Finv[:, 2, 2])/Dstar)
            idxK = 4
        elif regime == BALLISTIC_REGIME:
            C += np.sum(np.sqrt(Finv[:, 2, 2])/vd)
            idxK = 4
        else: # NO_REGIME
            idxK = 3 
        if fitK:
            C += np.sum(np.sqrt(Finv[:, idxK, idxK])/K)
        
        if minbias:
            if bias_regime == DIFFUSIVE_REGIME:
                if Dstar is None:
                    raise ValueError('Dstar must be set to calculate bias term.')
                else:
                    Y = diffusive(b, D, f, Dstar, K = K)
            elif bias_regime == BALLISTIC_REGIME:
                if vd is None:
                    raise ValueError('vd must be set to calculate bias term.')
                else:
                    Y = ballistic(b, c, D, f, vd, K = K)
            elif bias_regime == INTERMEDIATE_REGIME:
                if (v is None) or (tau in None):
                    raise ValueError('v and tau must be set to calculate bias term.')
                else:
                    Y = intermediate(b, np.ones_like(b)*delta, np.ones_like(b)*Delta, D, f, v, tau, T=T,k=k,K=K,seq=seq)
            elif bias_regime == SBALLISTIC_REGIME:
                Y = sBallistic(b, c, D, f, K=K)
            else:
                Y = sIVIM(b, D, f, K=K)
            p0 = np.array([1e-3, 0.1, 1])
            bounds = np.array([[1e-4, 0, 0], [3e-3, 1, 2]])
            if regime == DIFFUSIVE_REGIME:
                p0 = np.insert(p0, 2, 10e-3)
                bounds = np.insert(bounds, 2, np.array([3e-3, 1e-1]), axis = 1)
                x = b
                if fitK:
                    def fn(x, D, f, Dstar, S0, K):    
                        return diffusive(x, D, f, Dstar, S0, K).squeeze()
                    def jac(x, D, f, Dstar, S0, K):
                        return diffusive_jacobian(x,D,f,Dstar,S0,K).squeeze()
                else:
                    def fn(x, D, f, Dstar, S0):    
                        return diffusive(x, D, f, Dstar, S0).squeeze()
                    def jac(x, D, f, Dstar, S0):
                        return diffusive_jacobian(x,D,f,Dstar,S0).squeeze()
            elif regime == BALLISTIC_REGIME:
                p0 = np.insert(p0, 2, 2)
                bounds = np.insert(bounds, 2, np.array([1e-1, 5]), axis = 1)
                x = np.stack((b, c), axis=1)
                if fitK:
                    def fn(x, D, f, vd, S0, K):
                        b = x[:, 0]
                        c = x[:, 1]    
                        return ballistic(b, c, D, f, vd, S0, K).squeeze()
                    def jac(x, D, f, vd, S0, K):
                        b = x[:, 0]
                        c = x[:, 1] 
                        return ballistic_jacobian(b,c,D,f,vd,S0,K).squeeze()
                else:
                    def fn(x, D, f, vd, S0):    
                        b = x[:,0]
                        c = x[:,1]
                        return ballistic(b, c, D, f, vd, S0).squeeze()
                    def jac(x, D, f, vd, S0):
                        b = x[:, 0]
                        c = x[:, 1] 
                        return ballistic_jacobian(b,c,D,f,vd,S0).squeeze()
            elif regime == SBALLISTIC_REGIME:
                x = np.stack((b, c), axis=1)
                if fitK:
                    def fn(x, D, f, S0, K):   
                        b = x[:, 0]
                        c = x[:, 1]    
                        return sBallistic(b, c, D, f, S0, K).squeeze()
                    def jac(x,D,f,S0,K):
                        b = x[:, 0]
                        c = x[:, 1] 
                        return sBallistic_jacobian(b,c,D,f,S0,K).squeeze()
                else:
                    def fn(x, D, f, S0):   
                        b = x[:, 0]
                        c = x[:, 1]  
                        return sBallistic(b, c, D, f, S0).squeeze()
                    def jac(x,D,f,S0):
                        b = x[:, 0]
                        c = x[:, 1] 
                        return sBallistic_jacobian(b,c,D,f,S0).squeeze()
            else: # NO_REGIME
                x = b
                if fitK:
                    def fn(x, D, f, S0, K):    
                        return sIVIM(x, D, f, S0, K).squeeze()
                    def jac(x, D, f, S0, K):
                        return sIVIM_jacobian(x, D, f, S0, K).squeeze()
                else:
                    def fn(x, D, f, S0):    
                        return sIVIM(x, D, f, S0).squeeze()
                    def jac(x, D, f, S0):
                        return sIVIM_jacobian(x, D, f, S0).squeeze()
            if fitK:
                p0 = np.append(p0, 1)
                bounds = np.append(bounds, np.array([1e-1, 5])[:, np.newaxis], axis = 1)
            P = np.full((Y.shape[0], p0.size), np.nan)
            for i, y in enumerate(Y):
                try:
                    P[i, :],_ = curve_fit(fn, x, y, p0=p0, bounds=bounds,jac=jac)
                except:
                    P[i, :] = 1e5
            # C += np.sum(np.abs(D - P[:, 0])/D + np.abs(f - P[:, 1])/f)
            C += np.sum(np.abs(f - P[:, 1])/f)
            if regime == DIFFUSIVE_REGIME:
                C += np.sum(np.abs(Dstar - P[:, 2])/Dstar)
                idxK = 4
            elif regime == BALLISTIC_REGIME:
                C += np.sum(np.abs(vd - P[:, 2])/vd)
                idxK = 4
            else:
                idxK = 3
            if fitK:
                C += np.sum(np.abs(K - P[:, idxK])/K)

        return C

    check_regime(regime)

    if bias_regime not in [DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME]:
        raise ValueError(f'bias_regime must be "{DIFFUSIVE_REGIME}", "{BALLISTIC_REGIME}" or"{INTERMEDIATE_REGIME}"')

    nb = 4 + fitK - 2*(regime == NO_REGIME) - (regime == SBALLISTIC_REGIME) - (regime == SBALLISTIC_REGIME)*(bthr > 0)
    na = 4 + fitK - (regime == NO_REGIME) - (regime == SBALLISTIC_REGIME)
    n0 = (regime == NO_REGIME) + (regime == SBALLISTIC_REGIME)*(bthr > 0)
    bmin = bthr*((regime == NO_REGIME) + (regime == SBALLISTIC_REGIME))

    mincost = np.inf
    for nfc in range(1+nb*(regime == BALLISTIC_REGIME or regime == SBALLISTIC_REGIME)): # 1-4
        print('nfc: ' + str(nfc))
        # EP: Give combintions from picking nfc from [0,1,2,3,4] for ballistic or [0,1,2,3] for sBallistic
        if regime == BALLISTIC_REGIME:
            combs = [i for i in combinations(np.linspace(0,nb-1,nb,dtype=int),nfc)] # EP: this is only necessary if there are limitations on any of the bvalues # TODO
        else:
            combs = [()]

        for fc_idx in range(len(combs)):
            lb = bmin * np.ones(nb+na)
            lb[nb:] = 0.01 # lower bound for fraction of bvalue
            ub = bmax * np.ones(nb+na) # upper bound for bvalues 
            ub[nb:] = 1.0 # upper bound for fraction of bvalues

            if regime == DIFFUSIVE_REGIME:
                ub[0] = 100
            elif regime == BALLISTIC_REGIME:
                ub[0] = 10 

            # EP: rearrange to include all cases since FC are always the first indices
            if fc_idx != 0 and (regime == BALLISTIC_REGIME):
                fc_list = np.ones(nb+na) 
                fc_list[list(combs[fc_idx])]=0 # EP: Set the current fc position to 0 
                order = np.argsort(fc_list) # EP: Sort so that e.g., [0,1,0] -> [0,0,1]
                ub = ub[order]
                lb = lb[order]

            bounds = Bounds(lb, ub, keep_feasible = np.full_like(lb, True))
            constraints = ({'type':'eq',   'fun':lambda x: np.sum(x[nb:]) - 1}) # sum(a) = 1
            for seed_idx in range(20):
                print('seed ' + str(seed_idx))
                x0 = 1/na * np.ones(nb + na)
                x0[:nb] = [np.random.uniform(lb[i],ub[i],size=1)[0] for i in range(nb)]
                cost_regime = lambda x: cost(x, n0, nfc)
                res = minimize(cost_regime, x0, bounds = bounds, constraints = constraints, method = 'SLSQP')
                if res.fun < mincost:
                    b = np.zeros(nb+n0)
                    b[n0:] = res.x[:nb]
                    a = res.x[nb:]
                    if regime == BALLISTIC_REGIME or regime == SBALLISTIC_REGIME:
                        fc = np.full(b.size, False)
                        fc[:nfc] = True
                    mincost = res.fun   
                    
    if (a == 1/na * np.ones(na)).all():
        print('It is likely that an optimum was not found')

    idx = np.argsort(b)

    if regime == BALLISTIC_REGIME or regime == SBALLISTIC_REGIME:
        c = np.zeros(nb)
        _,_,c,_,_ = calc_interm_pars(b,usr_input,BIPOLAR,np.sum(fc))
        return b[idx], a[idx], c[idx], mincost
    else:
        return b[idx], a[idx], mincost