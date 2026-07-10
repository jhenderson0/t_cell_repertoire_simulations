import pyrepseq as prs
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon


################################################################
# Richness and diversity tests
################################################################
def plot_richness_and_diversity_differences(group_1_metrics, group_2_metrics, metrics=['chao1', 'Seff'], metric_labels=[ "Richness (chao1)", "Diversity (Simpson)"],
                                            box_colors=["skyblue", "moccasin"], x_labels=["Children", "Adults"], ylim=[1e1, 1e7], test="MWU"):
    
    fig, ax = plt.subplots(1, 2, figsize=(3, 1.8), layout='tight', sharey=True)

    if test == "MWU":
        stat_metric1, p_metric1 = mannwhitneyu(group_1_metrics[metrics[0]], group_2_metrics[metrics[0]], alternative="two-sided")
        stat_metric2, p_metric2 = mannwhitneyu(group_1_metrics[metrics[1]], group_2_metrics[metrics[1]], alternative="two-sided")
        
    if test == "WCR":
        stat_metric1, p_metric1 = wilcoxon(group_1_metrics[metrics[0]], group_2_metrics[metrics[0]], alternative="two-sided")
        stat_metric2, p_metric2 = wilcoxon(group_1_metrics[metrics[1]], group_2_metrics[metrics[1]], alternative="two-sided")
    

    plots = [(metrics[0], metric_labels[0], p_metric1), (metrics[1], metric_labels[1], p_metric2)]
    for i, (metric, label, p_value) in enumerate(plots):
        data = [group_1_metrics[metric].dropna(), group_2_metrics[metric].dropna()]
        bp = ax[i].boxplot(data, tick_labels=x_labels, medianprops=dict(color="black"), flierprops=dict(marker="o",markersize=3),  widths=0.3,
                    patch_artist=True)
    
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)

        ax[i].set_title(f"p = {p_value:.2g}", fontsize=7)
        ax[i].set_ylabel(label)
        ax[i].set_yscale('log')
    
    ax[0].set_ylim(ylim)
    ax[0].yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax[0].yaxis.set_minor_locator(mticker.LogLocator(base=10,subs=np.arange(2, 10), numticks=100))
    
    return fig, ax

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
def get_empirical_log_ratios(abundances_i, abundances_j, threshold=10, base=10, cmin=1):
    
    counts_1 = abundances_i
    counts_2 = abundances_j
    
    #set abundances less than c_min to be zero
    counts_1[counts_1 < cmin] = 0.0
    counts_2[counts_2 < cmin] = 0.0
    
    to_keep = ((counts_1 + counts_2) > threshold) & (counts_1 != 0) & (counts_2 != 0)
    
    counts_1 = counts_1[to_keep]
    counts_2 = counts_2[to_keep]
    
    log_ratios = np.log(counts_1/counts_2) / np.log(base)
    
    return log_ratios

################################################################
# Live simulation plotting 
################################################################


            