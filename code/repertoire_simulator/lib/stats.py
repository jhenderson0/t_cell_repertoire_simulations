import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

#################################################################################
# Compute CDFs
#################################################################################
def empirical_cdf(x):
    
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    cdf = (np.arange(1, n + 1) - 0.5) / n
    
    return x, cdf

def conditional_tail_cdf(x, tail="lower", threshold=1.0):
    
    if tail == "lower":
        tail_x = x[x < threshold]

    elif tail == "upper":
        tail_x = x[x > threshold]

    tail_x = np.sort(tail_x)
    n = len(tail_x)
    if tail == "lower":
        tail_probability = (np.arange(1, n + 1) - 0.5) / n

    else:
        tail_probability = (n - np.arange(n) - 0.5) / n

    return tail_x, tail_probability

#################################################################################
# Extreme value calculations
#################################################################################
def mean_over_threshold(x, thresholds, tail="lower"):
    
    if tail == "lower":
        means = [np.mean(x[x < threshold]) if np.any(x < threshold) else np.nan for threshold in thresholds]

    else:
        means = [np.mean(x[x > threshold]) if np.any(x > threshold) else np.nan for threshold in thresholds]

    return np.asarray(means)


#################################################################################
# Simpson's diversity
#################################################################################
def get_global_simpsons_diversity(c):
    
    species_totals = c.sum(axis=1)
    total_abundance = species_totals.sum()
    if total_abundance <= 0:
        return 0.0

    p = species_totals / total_abundance
    return 1.0 / np.sum(p ** 2)

def get_average_simpsons_diversity(c):

    location_totals = c.sum(axis=0)
    nonempty = location_totals > 0
    if not np.any(nonempty):
        return 0.0

    p = c[:, nonempty] / location_totals[nonempty]
    local_simpson = 1.0 / np.sum(p ** 2, axis=0)

    return np.mean(local_simpson)

#################################################################################
# Expectations after a burn in period
#################################################################################
def results_after_burn(x, burn_frac=0.2, xmin=0):
    
    x = np.asarray(x)
    burn = int(burn_frac * len(x))
    x = x[burn:]
    x = x[np.isfinite(x) & (x > xmin)]
 
    return x

def mean_after_burn(x, burn_frac=0.2, xmin=0):
    
    x = np.asarray(x)
    burn = int(burn_frac * len(x))
    x = x[burn:]
    x = x[np.isfinite(x) & (x > xmin)]
 
    return np.mean(x)

def median_after_burn(x, burn_frac=0.2, xmin=0):
    
    x = np.asarray(x)
    burn = int(burn_frac * len(x))
    x = x[burn:]
    x = x[np.isfinite(x) & (x > xmin)]
 
    return np.median(x)

def geometric_mean_after_burn(x, burn_frac=0.2, xmin=0):
    
    x = np.asarray(x)
    burn = int(burn_frac * len(x))
    x = x[burn:]
    x = x[np.isfinite(x) & (x > xmin)]
 
    return np.exp(np.mean(np.log(x)))

#################################################################################
# Power spectral densities
#################################################################################
def psd_model(f, A, fc, beta, floor):
    
    return floor + A / (1 + (f / fc)**beta)

#################################################################################
# Model fitting
#################################################################################
def fit_power_law_prefactor(x, y, exponent):
    
    logx = np.log(x)
    logy = np.log(y)

    log_prefactor = np.mean(logy - exponent * logx)
    
    return np.exp(log_prefactor)

def fit_power_law(x, y):
    
    logx = np.log10(x)
    logy = np.log10(y)
    
    slope, intercept, r_value, p_value, std_err = linregress(logx, logy)

    prefactor = 10**intercept
    exponent = slope
    
    return prefactor, exponent

def fit_psd(f, Pxx, fit_log=True, maxfev=10000):
    m = (f > 0) & np.isfinite(Pxx) & (Pxx > 0)
    f = f[m]
    Pxx = Pxx[m]

    p0 = [np.max(Pxx), f[len(f)//4], 2.0, np.min(Pxx)]

    bounds = ([0, f.min(), 0.1, 0], [np.inf, f.max(), 5.0, np.inf])

    if fit_log:
        def model_to_fit(f, A, fc, beta, floor):
            return np.log(psd_model(f, A, fc, beta, floor))

        y = np.log(Pxx)
    
    else:
        model_to_fit = psd_model
        y = Pxx

    popt, pcov = curve_fit(model_to_fit, f, y, p0=p0, bounds=bounds, maxfev=maxfev)

    A, fc, beta, floor = popt
    tau = 1 / (2 * np.pi * fc)

    return A, fc, beta, floor, tau