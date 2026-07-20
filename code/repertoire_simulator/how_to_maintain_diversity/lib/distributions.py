import numpy as np

from scipy.special import hyp2f1
from scipy.stats import levy_stable, norm

import repertoire_simulator.lib as lib

#################################################################################
# Mean field predictions for clone size distribution
#################################################################################

def number_density_mean_field(c, alpha, theta, D, C0=1):
    
    small_c = 1/c
    big_c = C0**alpha * c**(-(1 + alpha))
    
    n_c = np.zeros(len(c))

    n_c[(c < C0) & (c > 0)] = small_c
    n_c[c >= C0] = big_c
    
    return theta / (alpha * D) * n_c

def unnormalized_survival_function_mean_field(c, alpha):
    
    return c**(-alpha)

def T_star_mean_field(alpha, theta, D, C0=1):
    
    return (theta * C0) / (D * (alpha - 1))

def alpha_mean_field(D, d, gamma):

    return (D + d * gamma) / (D * (1 + gamma))

def get_g_from_cmax_mean_field(c_t, alpha, burn_frac=0.02, return_c_star=False):
    
    cmax = lib.results_after_burn(np.array([np.nanmax(clones) for clones in c_t]), burn_frac=burn_frac)
    
    A_fit = alpha * len(cmax) / np.sum(cmax**(-alpha))
    Lambda_cmax = (A_fit / alpha) * cmax**(-alpha)
    c_star_fit = (A_fit / alpha)**(1 / alpha)
    
    if return_c_star:
        return -np.log(Lambda_cmax), c_star_fit
     
    return -np.log(Lambda_cmax)

#################################################################################
# Cavity-like method predictions for clone size distribution
#################################################################################

def unnormalized_number_density_cavity_like(c, alpha, D, d, T_b, C0):
    
    small_c = 1/c
    big_c = c**(-(1 + alpha)) * (1 + c / T_b)**(-(d / D - alpha))
    
    n_c = np.zeros(len(c))

    n_c[(c < C0) & (c > 0)] = small_c
    n_c[c >= C0] = big_c
    
    return n_c

def unnormalized_survival_function_cavity_like(c, alpha, D, d, T_b):
    
    delta = d / D
    beta = delta - alpha
    
    return (T_b**beta) / (delta * c**delta) * hyp2f1(delta, beta, delta + 1, -T_b / c)

def get_g_from_cmax_cavity_like(c_t, alpha, theta, D, d, C0=1, burn_frac=0.02):

    cmax =  lib.results_after_burn(np.array([np.nanmax(clones) for clones in c_t]), burn_frac=burn_frac)

    T_star = T_star_mean_field(alpha, theta, D, C0=C0)
    F_cmax = unnormalized_survival_function_cavity_like(cmax, alpha, D, d, T_star)

    A_fit = len(F_cmax) / np.sum(F_cmax)
    Lambda_cmax = A_fit * F_cmax

    return -np.log(Lambda_cmax)


#################################################################################
# Mean field diversity fluctuation distributions
#################################################################################
def diversity_fluctuation_density(x, alpha):

    beta = alpha / 2
    levy_stable.parameterization = "S1"
    
    # For 1 < alpha < 2 the limiting density has a finite scale 
    if 1 < alpha < 2:
        scale = np.cos(np.pi * beta / 2) ** (1 / beta)
        mean_log_z = np.euler_gamma * (1 / beta - 1)
        z = np.exp(mean_log_z - x)

        return z * levy_stable.pdf(z, beta, 1, loc=0, scale=scale)

    # For alpha > 2 the limiting density shrinks in width so returns the unit scale limiting distribution
    if 2 < alpha < 4:
        
        return levy_stable.pdf(-x, beta, 1, loc=0, scale=1)

    if alpha > 4:
        
        return norm.pdf(x, loc=0, scale=1)

    raise ValueError("alpha = 2 and alpha = 4 require marginal theories.")

def sample_diversity_fluctuations(alpha, size=int(1e6)):
  
    beta = alpha / 2

    levy_stable.parameterization = "S1"

    if 1 < alpha < 2:
        scale = np.cos(np.pi * beta / 2) ** (1 / beta)
        mean_log_z = np.euler_gamma * (1 / beta - 1)

        z = levy_stable.rvs(beta, 1, loc=0, scale=scale, size=size)

        return -np.log(z) + mean_log_z

    if 2 < alpha < 4:
        z = levy_stable.rvs(beta, 1, loc=0, scale=1, size=size)

        return -z

    if alpha > 4:
        return np.random.normal(loc=0, scale=1, size=size)

    raise ValueError("alpha = 2 and alpha = 4 require marginal theories.")