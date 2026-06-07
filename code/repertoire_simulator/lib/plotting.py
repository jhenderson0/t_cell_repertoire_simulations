import matplotlib.pyplot as plt
import pyrepseq as prs
import numpy as np
import scipy

################################################################
# Clone size distributions
################################################################
def plot_referencescaling(ax=None, x=[4e-5, 4e-2], factor=1.0, color='k', exponent=-1.0, label=True, **kwargs):
    """
    Adapted from https://github.com/andim/paper-tcellimprint

    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x)
    ax.plot(x, factor*x**exponent, color=color, **kwargs)
    if label:
        xt = scipy.stats.gmean(x)
        xt = xt*1.18
        yt = factor*xt**exponent*1.18
        ax.text(xt, yt, rf'$-\alpha={-exponent}$', va='bottom', ha='left', color=color)
        
################################################################
# Two-point inequality
################################################################      
def get_empirical_log_ratios(abundances_i, abundances_j, threshold=10, base=10):
    
    counts_1 = abundances_i
    counts_2 = abundances_j
    
    #set abundances less than 1 to be zero
    counts_1[counts_1 < 1] = 0.0
    counts_2[counts_2 < 1] = 0.0
    
    to_keep = ((counts_1 + counts_2) > threshold) & (counts_1 != 0) & (counts_2 != 0)
    
    counts_1 = counts_1[to_keep]
    counts_2 = counts_2[to_keep]
    
    log_ratios = np.log(counts_1/counts_2) / np.log(base)
    
    return log_ratios

################################################################
# Live simulation plotting 
################################################################


            