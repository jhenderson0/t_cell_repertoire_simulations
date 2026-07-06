import numpy as np

def get_global_simpsons_diversity(c):
    
    c = c.copy()
    c[c < 1] = 0.0  

    species_totals = c.sum(axis=1)
    total_abundance = species_totals.sum()
    if total_abundance <= 0:
        return 0.0

    p = species_totals / total_abundance
    return 1.0 / np.sum(p ** 2)


def get_average_simpsons_diversity(c):
    
    c = c.copy()
    c[c < 1] = 0.0  

    location_totals = c.sum(axis=0)
    nonempty = location_totals > 0
    if not np.any(nonempty):
        return 0.0

    p = c[:, nonempty] / location_totals[nonempty]
    local_simpson = 1.0 / np.sum(p ** 2, axis=0)

    return np.mean(local_simpson)