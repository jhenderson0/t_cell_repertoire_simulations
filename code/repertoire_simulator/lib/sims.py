import numpy as np
from tqdm.auto import tqdm
from .stats import get_average_simpsons_diversity

#################################################################################
# Update rates
#################################################################################
def get_clonal_rates(c, a, param_state, t, homeostatic_control_func, death_func, antigen_response_func, migration_func):

    #Compute antigenic contribution to clonal birth/death rate
    antigen_env = antigen_response_func(c, a, param_state['antigen_response'], t)
    
    birth_c = homeostatic_control_func(c, param_state['homeostatic_control'], t) + np.maximum(antigen_env, 0.0)
    death_c = death_func(c, param_state['death'], t) + np.maximum(-antigen_env, 0.0)
    mig_c = migration_func(c, a, param_state['migration'], t)
    
    return birth_c, death_c, mig_c

#################################################################################
# Clonal dynamics 
#################################################################################

#Get the deterministic part of dc/dt
def deterministic_clonal_evolution(c, a, param_state, t,
                                   homeostatic_control_func, death_func, antigen_response_func, migration_func):

    birth_c, death_c, mig_c = get_clonal_rates(c, a, param_state, t,
                                          homeostatic_control_func, death_func, antigen_response_func, migration_func)

    return (birth_c - death_c - mig_c.sum(axis=1)) * c + np.einsum("kij,kj->ki", mig_c, c)

#Get the birth-death noise part of dc/dt
def birth_death_noise_increment(c, a, param_state, t, dt, 
                                homeostatic_control_func, death_func, antigen_response_func, migration_func,
                                prng=np.random):
 
    birth_c, death_c, _mig_c = get_clonal_rates(c, a, param_state, t, 
                                          homeostatic_control_func, death_func, antigen_response_func, migration_func)

    return np.sqrt((birth_c + death_c) * c * dt) * prng.normal(size=c.shape)

#################################################################################
# Evolutions step
#################################################################################
def continuum_evolution_steps(c, a, param_state, t, dt, theta_c,
                    homeostatic_control_func, death_func, antigen_response_func, migration_func,
                    c_cutoff, antigen_update_func, method="euler", demographic_stochasticity="no", prng=np.random):

    S, N = c.shape
    
    # determine next intro event
    t_old = t
    t_delta = prng.exponential(1.0 / theta_c)
    
    #progress clonal abundances and antigenic levels
    while t < t_old + t_delta:
        # make time step fit next intro event time
        if t + dt > t_old + t_delta:
            thisdt = t_old + t_delta - t
        else:
            thisdt = dt
            
        ####################################
        #Growth and migration updates
        ####################################
        a_for_rates, a_next = antigen_update_func(a, param_state['antigen'], t, thisdt, prng=prng)
        
        #Simple Euler method for ODE integration
        if method == "euler":
            
            dc = deterministic_clonal_evolution(c, a_for_rates, param_state, t,
                                      homeostatic_control_func, death_func, antigen_response_func, migration_func)
            
            deterministic_increment = dc * thisdt
            
        
        #Method for updating SDEs with Stratonovich noise - also works as an rk2 update 
        elif method == "heun":
            
            k1 = deterministic_clonal_evolution(c, a_for_rates, param_state, t,
                                       homeostatic_control_func, death_func, antigen_response_func, migration_func)


            k2 = deterministic_clonal_evolution(c + thisdt * k1, a_for_rates, param_state, t + thisdt,
                                       homeostatic_control_func, death_func, antigen_response_func, migration_func)

            deterministic_increment = 0.5 * (k1 + k2) * thisdt
        
        #More accurate ODE integrator
        elif method == "rk4":

            k1 = deterministic_clonal_evolution(c, a_for_rates, param_state, t,
                                                homeostatic_control_func,
                                                death_func,
                                                antigen_response_func,
                                                migration_func)

            k2 = deterministic_clonal_evolution(c + 0.5 * thisdt * k1, a_for_rates, param_state, t + 0.5 * thisdt,
                                                homeostatic_control_func,
                                                death_func,
                                                antigen_response_func,
                                                migration_func)

            k3 = deterministic_clonal_evolution(c + 0.5 * thisdt * k2, a_for_rates, param_state, t + 0.5 * thisdt,
                                                homeostatic_control_func,
                                                death_func,
                                                antigen_response_func,
                                                migration_func)

            k4 = deterministic_clonal_evolution(c + thisdt * k3, a_for_rates, param_state, t + thisdt,
                                                homeostatic_control_func,
                                                death_func,
                                                antigen_response_func,
                                                migration_func)

            deterministic_increment = (thisdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        ####################################
        #Demographic noise
        ####################################
        if demographic_stochasticity == "yes":
            demographic_noise = birth_death_noise_increment(c, a_for_rates, param_state, t, thisdt,
                                                            homeostatic_control_func, death_func, antigen_response_func,migration_func, prng=prng)
            
        elif demographic_stochasticity == "no":
            demographic_noise = 0.0
            
        ####################################
        #Apply updates
        ####################################
        c += deterministic_increment + demographic_noise
        c = np.maximum(c, 0.0)
        a = a_next
        t += thisdt

        ## clones lower than threshold everywhere are zeroed
        below_cutoff_everywhere = np.all(c < c_cutoff, axis=1)
        c[below_cutoff_everywhere] = 0.0
        
    return c, a, t_old + t_delta

#################################################################################
# Introduction events
#################################################################################
def introduce_clone(c, c_new, c_cutoff, prng=np.random):

    S, N = c.shape
    
    # Choose a location for the new clone
    location = prng.randint(N)

    # Make new clonal cross site vector
    c_new_vec = np.zeros(N, dtype=c.dtype)
    c_new_vec[location] = c_new
    
    # Find a globally rare clone to replace
    clone_totals = c.sum(axis=1)
    below_cutoff_everywhere = np.all(c < c_cutoff, axis=1)
    
    if np.any(below_cutoff_everywhere):
        candidates = np.flatnonzero(below_cutoff_everywhere)
        clone_totals = c.sum(axis=1)
        clone_index = candidates[np.argmin(clone_totals[candidates])]
        replace = True

        c[clone_index, :] = c_new_vec
    
    # Or extend the clonal array
    else:
        clone_index = S
        replace = False

        c = np.concatenate([c, c_new_vec[None, :]], axis=0)
        
    return c, clone_index, replace

def introduce_antigen(a, clone_index, replace, a_new=0.0):
    
    R, N = a.shape

    if replace:
        a[clone_index, :] = a_new

    else:
        a_buffer = np.empty((R + 1, N), dtype=a.dtype)
        a_buffer[:R, :] = a
        a_buffer[R, :] = a_new
        a = a_buffer

    return a
    
def execute_introduction_event(c, a, param_state, c_new, c_cutoff, a_new,
                               fixed_antigen_pool=False, param_introduction_func=None, 
                               prng=np.random):
    
    #Introduce a new clone
    c, clone_index, replace = introduce_clone(c, c_new, c_cutoff, prng=prng)
   
    # Choose whether to introduce a new antigen
    if not fixed_antigen_pool:
        a = introduce_antigen(a, clone_index=clone_index, replace=replace, a_new=a_new)
        
    #Update parameters - to be added
    if param_introduction_func is not None:
        S, N = c.shape
        R, _ = a.shape
        param_state = param_introduction_func(param_state, S=S, N=N, R=R, 
                                              clone_index=clone_index, replace=replace, fixed_antigen_pool=fixed_antigen_pool, prng=prng)
    
    return c, a, param_state

#################################################################################
# Run simulation
#################################################################################
def print_progress(c, t, t_end, pbar, progress_t,  c_cutoff):

    new_progress_t = min(t, t_end)
    delta = max(0.0, new_progress_t - progress_t)

    pbar.update(delta)
    
    N_cells = c.sum(axis=0)
    Seff = get_average_simpsons_diversity(c)
    pbar.set_postfix({"t": f"{t:.3g}",
                      "clones": int(np.sum(np.all(c > c_cutoff, axis=1))),
                      "Seff/patch": f"{Seff:.1f}",
                      "cells/patch": f"{np.mean(N_cells):.2e}",
                      "cmax": f"{np.max(c):.2e}"})

    return new_progress_t

def record_sample(records, c, a, param_state, t, next_sample_t, sample_dt):

    if sample_dt is None:
        return next_sample_t

    if t >= next_sample_t:
        records['t'].append(t)
        records['c'].append(c.copy())

        while next_sample_t <= t:
            next_sample_t += sample_dt

    return next_sample_t

def simulate_repertoire(homeostatic_control_func, death_func, antigen_response_func, migration_func, 
                        antigen_update_func, initial_param_state,
                        fixed_antigen_pool=False, param_introduction_func=None,
                        t_start=0, t_end=1, dt=0.01, theta_c=1.0,
                        S=1, R=1, N=1, 
                        c_initial=0.0, a_initial=0.0, c_new=1.0, a_new=0.0,
                        c_cutoff=0.1,
                        continuum_update_method="euler", demographic_stochasticity='no', seed=1996,  prng=np.random,
                        verbose=False, sample_dt=None):

    #set rng
    prng.seed(seed=seed)
        
    # initialize clonal and antigen array
    c = np.ones((S, N)) * c_initial # S clones in N locations
    
    if fixed_antigen_pool:
        a = np.ones((R, N)) * a_initial # R antigen in N locations
    else:
        a = np.ones((S, N)) * a_initial # R = S (number of clones)
    
    # Choose model parameters 
    param_state = initial_param_state
        
    #progress bar
    pbar = tqdm(total=t_end - t_start, desc="Simulation", unit="time", disable=not verbose, 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}{postfix}]")
    progress_t = t_start
    
    # main loop
    t = t_start
    records = {"t": [], "c": []}
    next_sample_t = record_sample(records, c, a, param_state, t, t_start, sample_dt)
    while t < t_end:
        introevent = False
        
        c, a, t = continuum_evolution_steps(c, a, param_state, t, dt, theta_c,
                    homeostatic_control_func, death_func, antigen_response_func, migration_func,
                    c_cutoff, antigen_update_func, method=continuum_update_method,
                    demographic_stochasticity=demographic_stochasticity, prng=prng)
        
        next_sample_t = record_sample(records, c, a, param_state, t, next_sample_t, sample_dt)
        
        progress_t = print_progress(c, t, t_end, pbar, progress_t,  c_cutoff)
        
        introevent = True
        if introevent:
            c, a, param_state = execute_introduction_event(c, a, param_state, c_new, c_cutoff, a_new, 
                                                           fixed_antigen_pool, param_introduction_func,
                                                           prng=prng)
            
    pbar.close()

    return c, a, param_state, records