import numpy as np

############################################################
# Homeostatic control
############################################################
def simple_homeostatic_competition_func(c, params, _t):
        
    return params['b']

def local_homeostatic_competition_func(c, params, _t):
    
    effective_birth_rate = params['b'] / (params['T0'] + c.sum(axis=0))
    
    return effective_birth_rate

def focal_clone_homeostatic_competition_func(c, params, _t):
    
    effective_birth_rate = params['b'] / params['T_star']
    
    return effective_birth_rate

############################################################
# Clonal death
############################################################
def simple_death_func(c, params, _t):
    
    return params['d'] 

############################################################
# Antigen response
############################################################
def simple_antigen_response_func(_c, a, _params, _t):
    
    return a

def self_inhibition_antigen_response_func(c, a, params, _t):
    
    return a * (1 + params['epsilon']) / (1 + params['epsilon'] * c)

def abundant_antigen_response_func(_c, a, params, _t):
    
    return params['K'] @ a

def limited_antigen_response_fun(c, a, params, availability_function, _t):
    
    antigen_demand = params['K'].T @ c
    free_antigen = a * availability_function(antigen_demand)
    
    return params['K'] @ free_antigen

############################################################
# Migration
############################################################
def simple_migration_func(_c, _a, params, _t):
    
    return params["migration_matrix"]