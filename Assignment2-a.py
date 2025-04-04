import streamlit as st
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

def calculate_stationary_distribution(lambd, mu, n, k, s, warm_standby):
    """
    Calculate stationary distribution for both warm and cold standby cases
    """
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
        Integer(k, max_n, name='n2'),  # n must be >= k
        Integer(1, max_s, name='s2')   # s must be >= 1
    ]
    
    # Cost function to minimize
    @use_named_args(space)
    def objective(n2, s2):
        pi, avail = calculate_stationary_distribution(lambd, mu, n2, k, s2, warm_standby)
        cost = (cost_component * n2 + 
               cost_repairman * s2 + 
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
            n2 = optimal_n + delta_n
            s2 = optimal_s + delta_s
            if n2 >= k and s2 >= 1:
                cost = objective([n2, s2])
                if cost < optimal_cost:
                    optimal_n, optimal_s, optimal_cost = n2, s2, cost
    
    return optimal_n, optimal_s, optimal_cost



def main():
    st.title("K-out-of-N System Analysis")
    
    # Create tabs for different exercises
    tab1, tab2 = st.tabs(["Exercise A: System Availability", "Exercise B: System's cost optimization"])
    
    with tab1:
        st.markdown("""
            **Calculate the fraction of time a k-out-of-n system is up, where:**
            - Components fail exponentially (rate λ) 
            - Repairs take exponential time (rate μ)
            - You can specify warm/cold standby behavior
            - System needs at least *k* working components from *n* total
            - Limited repair capacity (*s* repairmen)
            """)        
        # Input parameters
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Total components (n)", min_value=1, value=4, key="a_n")
            k = st.number_input("Required working components (k)", min_value=1, max_value=n, value=2, key="a_k")
            s = st.number_input("Repairmen (s)", min_value=1, max_value=n, value=1, key="a_s")
            warm_standby = st.checkbox("Warm standby mode", value=True, key="a_warm")
        with col2:
            lambd = st.number_input("Failure rate (λ)", min_value=0.0, value=0.5, step=0.1, key="a_lambd")
            mu = st.number_input("Repair rate (μ)", min_value=0.0, value=1.0, step=0.1, key="a_mu")
        
        if st.button("Calculate Availability", key="a_calc"):
            try:
                pi, availability = calculate_stationary_distribution(lambd, mu, n, k, s, warm_standby)
                st.success(f"Fraction of time system is up: {availability:.4f} ({availability*100:.2f}%)")
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")
    
    with tab2:
        st.markdown("""
        **Optimal Resource Allocation**  
        Finds cost-optimal number of components and repairmen considering:
        - Component costs (C<sub>c</sub>)
        - Repairman costs (C<sub>r</sub>)
        - Downtime costs (C<sub>d</sub>)
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Maximum number of components to consider ($n_{max}$)", min_value=1, value=50, key="b_n")
            k = st.number_input("Required working components (k)", min_value=1, max_value=n, value=2, key="b_k")
            s = st.number_input("Maximum number of repairmen to consider ($s_{max}$)", min_value=1, value=50, key="b_s")
            warm_standby = st.checkbox("Warm standby mode", value=True, key="b_warm")
        with col2:
            lambd = st.number_input("Failure rate (λ)", min_value=0.0, value=0.5, step=0.1, key="b_lambd")
            mu = st.number_input("Repair rate (μ)", min_value=0.0, value=1.0, step=0.1, key="b_mu")
        
        # st.markdown('**Costs input:**')

        col3, col4, col5 = st.columns(3)
        with col3:
            cost_component = st.number_input("Cost per component", min_value=0.0, value=4.0, step=0.1, key="b_cc")

        with col4:
            cost_repairman = st.number_input("Cost per repairman", min_value=0.0, value=4.0, step=0.1, key="b_cr")
        
        with col5:
            cost_downtime = st.number_input("Downtime cost", min_value=0.0, value=4.0, step=0.1, key="b_ct")


        if st.button("Find the optimal number of components and repairmen", key="b_calc"):
            try:
                solution = optimize_with_ml(lambd, mu, k, warm_standby, cost_component, cost_repairman, cost_downtime, n, s)
                st.success(f"Number of components: {solution[0]}, number of repairmen: {solution[1]}, total cost: {solution[2]:.2f}")
           
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")

if __name__ == "__main__":
    main()