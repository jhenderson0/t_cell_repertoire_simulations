import numpy as np


def get_average_simpsons_diversity(c):
    
    c_copy = c.copy()
    c_copy[c_copy < 1] = 0.0
    N_cells = c_copy.sum(axis=0)
    if np.all(N_cells) > 0:
        Seff = 1 / np.nanmean(((c / N_cells)**2).sum(axis=0))
        
    else:
        Seff = 0.0
        
    return Seff