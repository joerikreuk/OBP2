import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

def calculate_availability(lambd, mu, n, k, s, warm_standby):
    """Exact availability calculation using birth-death process"""
    Q = np.zeros((n + 1, n + 1))
    
    for i in range(n + 1):
        working = n - i
        
        if warm_standby:
            # WARM STANDBY: All components can fail
            birth_rate = (n - i) * lambd if i < n else 0
        else:
            # COLD STANDBY: Only active components fail when system is up
            active = min(working, k) if working >= k else 0
            birth_rate = active * lambd if i < n else 0
        
        # Repair transitions (death rates)
        death_rate = min(i, s) * mu if i > 0 else 0
        
        # Fill Q matrix
        if i < n:
            Q[i, i+1] = birth_rate  # Birth rate (failures)
        if i > 0:
            Q[i, i-1] = death_rate  # Death rate (repairs)
        Q[i, i] = -np.sum(Q[i, :])  # Diagonal
    
    # Solve for stationary distribution
    A = Q.T
    A[-1, :] = 1  # Normalization condition
    b = np.zeros(n + 1)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
        
    availability = 0
    for j in range(n + 1):
        if (n - j) >= k:
            availability += pi[j]  

    return pi, availability

def optimize_with_ml(lambd, mu, k, warm_standby, cost_component, cost_repairman, cost_downtime, max_n=40, max_s=30):
    """
    Optimize using Bayesian Optimization (Gaussian Processes)
    Returns: (optimal_n, optimal_s, optimal_cost)
    """
    # Define search space
    space = [
        Integer(k, max_n, name='n'),  # n must be >= k
        Integer(1, max_s, name='s')   # s must be >= 1
    ]
    
    # Cost function to minimize
    @use_named_args(space)
    def objective(n, s):
        pi, avail = calculate_availability(lambd, mu, n, k, s, warm_standby)
        cost = (cost_component * n + 
               cost_repairman * s + 
               cost_downtime * (1 - avail))
        return cost
    
    # Run Bayesian optimization
    res = gp_minimize(
        objective,
        space,
        n_calls=50,                  # Number of evaluations
        random_state=42,
        acq_func='EI',               # Expected Improvement
        n_initial_points=10          # Random exploration first
    )
    
    # Get best solution
    optimal_n = res.x[0]
    optimal_s = res.x[1]
    optimal_cost = res.fun
    
    # Verify by local search around the solution
    for delta_n in [-1, 0, 1]:
        for delta_s in [-1, 0, 1]:
            n = optimal_n + delta_n
            s = optimal_s + delta_s
            if n >= k and s >= 1:
                cost = objective([n, s])
                if cost < optimal_cost:
                    optimal_n, optimal_s, optimal_cost = n, s, cost
    
    return optimal_n, optimal_s, optimal_cost

# Example usage
if __name__ == "__main__":
    # Your parameters
    params = {
        'lambd': 0.5,
        'mu': 1.0,
        'k': 2,
        'warm_standby': True,
        'cost_component': 4.0,
        'cost_repairman': 0.0,
        'cost_downtime': 4.0,
        'max_n': 40,
        'max_s': 30
    }
    
    try:
        n, s, cost = optimize_with_ml(**params)
        print(f"Optimal Solution: n={n}, s={s}, Total Cost={cost:.2f}")
        
        # Calculate exact availability for the solution
        avail = calculate_availability(params['lambd'], params['mu'], n, params['k'], s, params['warm_standby'])
        print(f"Availability: {avail:.4f} ({avail*100:.2f}%)")
        print(f"Component Cost: {params['cost_component']*n:.2f}")
        print(f"Repairman Cost: {params['cost_repairman']*s:.2f}")
        print(f"Downtime Cost: {params['cost_downtime']*(1-avail):.2f}")
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")