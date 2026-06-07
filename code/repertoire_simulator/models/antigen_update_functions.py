import numpy as np

#No antigen signal - for a fully neutral repertoire 
def no_noise_antigen_update(a, _params, _t, _dt, prng=np.random):
    
    a_for_rates = a * 0
    a_next = a_for_rates
   
    return a_for_rates, a_next

#White noise
def white_noise_antigen_update(a, params, _t, dt, prng=np.random):
    
    D = params["D"]
   
    a_for_rates = np.sqrt(2 * D / dt) * prng.normal(size=a.shape)

    a_next = a

    return a_for_rates, a_next

#Ornstein-Uhlenbeck process
def ou_antigen_update(a, params, _t, dt, mean=0.0, prng=np.random):
    
    D = params["D"]
    lamb = params['lamb']

    decay = np.exp(-lamb * dt)
    a_next = mean + decay * (a - mean) + np.sqrt(D * lamb * (1.0 - decay**2)) * prng.normal(size=a.shape)

    #Representative antigen level during the timestep
    a_for_rates = 0.5 * (a + a_next)

    return a_for_rates, a_next

#Shot-noise antigen
def shot_noise_antigen_update(a, params, t, dt, event_threshold=10, prng=np.random):
    
    A = params['A']
    lamb = params['lamb']
    encounter_rate = params['encounter_rate']
    
    expected_events = encounter_rate * dt
    
    #If lots of events in timestep use a Gaussian approximation
    if expected_events >= event_threshold:
        
        mean = A * encounter_rate / lamb
        D_ou = A**2 * encounter_rate / (2.0 * lamb**2)

        return ou_antigen_update(a, {'D': D_ou, 'lamb': lamb}, t, dt, mean=mean, prng=prng)
 
    # Existing antigen decay
    decay = np.exp(-lamb * dt)
    a_next = decay * a

    # Timestep average of existing decaying antigen
    avg_decay_factor = -np.expm1(-lamb * dt) / (lamb * dt)
    a_for_rates = avg_decay_factor * a

    # New encounter events during this timestep
    n_events = prng.poisson(encounter_rate * dt, size=a.shape)

    a_next_flat = a_next.ravel()
    a_for_rates_flat = a_for_rates.ravel()
    n_events_flat = n_events.ravel()
    for idx in np.flatnonzero(n_events_flat):
        n = int(n_events_flat[idx])

        u = prng.uniform(0.0, dt, size=n)

        remaining = dt - u

        endpoint_weights = np.exp(-lamb * remaining)
        average_weights = -np.expm1(-lamb * remaining) / (lamb * dt)

        a_next_flat[idx] += A * endpoint_weights.sum()
        a_for_rates_flat[idx] += A * average_weights.sum()

    return a_for_rates, a_next
