import numpy as np
from functools import partial

import repertoire_simulator.lib as lib
import repertoire_simulator.models as models

#################################################################################
# Model setup
#################################################################################

# Simulation time
t_start = 0
t_end = 2
dt = 5e-4

# Dimensions
S = int(1e5)
R = S
N = 2

# Parameters controlling the single-species distribution
M = 1.0
D = 5.0
lamb = 1.0

# Parameters controlling the total population
d = D + M  
T_star = 1e7
b = (d - D) * T_star

# Every clone starts with equal abundance in both patches
c_initial = T_star / S
a_initial = 0.0

# No recruitment or artificial extinction during the recovery test
theta_c = 10
c_new = 0.0
a_new = 0.0
c_cutoff = 0.0

initial_param_state = {}

# migration rates
base_migration_matrix = np.full((N, N), M / N)
np.fill_diagonal(base_migration_matrix, 0.0)
migration_matrix = np.broadcast_to(base_migration_matrix[None, :, :], (1, N, N)).copy()

# clonal dynamics functions
initial_param_state['homeostatic_control'] = {'b': b, 'T0': 1}
homeostatic_control_func = models.local_homeostatic_competition_func

initial_param_state['death'] = {'d': d}
death_func = models.simple_death_func

initial_param_state['antigen_response'] = {}
antigen_response_func = models.simple_antigen_response_func 

initial_param_state['migration'] = {'migration_matrix': migration_matrix}
migration_func = models.simple_migration_func

demographic_stochasticity='no'

# antigen dynamics 
initial_param_state['antigen'] = {'D': D, 'lamb': lamb}
antigen_update_func = models.ou_antigen_update

# update method
continuum_update_method="euler" 

#rng
seed = 12345 + int(M + D + lamb + b)

#################################################################################
# Run the simulation
#################################################################################
print("Starting repertoire simulation...")

c, a, param_state, records = lib.simulate_repertoire(homeostatic_control_func=homeostatic_control_func,
                                                death_func=death_func,
                                                antigen_response_func=antigen_response_func,
                                                migration_func=migration_func,
                                                antigen_update_func=antigen_update_func,
                                                initial_param_state=initial_param_state,
                                                t_start=t_start, t_end=t_end, dt=dt,
                                                theta_c=theta_c, S=S, R=R, N=N,
                                                c_initial=c_initial, a_initial=a_initial, c_new=c_new, a_new=a_new,
                                                c_cutoff=c_cutoff,
                                                continuum_update_method=continuum_update_method,
                                                demographic_stochasticity=demographic_stochasticity, verbose=True, sample_dt=0.05, seed=seed)

print("Simulation complete! Saving results...")

np.savez_compressed(f"../../../../data/sim_results/two_patches_weak_coloured_noise.npz", 
                        **{key: np.array(value, dtype=object) for key, value in records.items()})