import streamlit as st
import numpy as np
from math import comb, factorial

def calculate_stationary_distribution(lambd, mu, n, k, s, warm_standby):
    """
    Calculate stationary distribution for both warm and cold standby cases
    """
    # Calculate pi(0) first
    if warm_standby:
        # Warm standby - all components can fail
        sum_part1 = sum(comb(n, j) * (lambd / mu) ** j for j in range(0, s + 1))
        sum_part2 = sum((factorial(n)/(factorial(n-j)*factorial(s)*(s**(j-s)))) * (lambd/mu)**j 
                      for j in range(s+1, n+1))
        pi0 = 1 / (sum_part1 + sum_part2)
        
        pi = np.zeros(n + 1)
        pi[0] = pi0
        
        for j in range(1, s + 1):
            pi[j] = comb(n, j) * (lambd / mu) ** j * pi0
        
        for j in range(s + 1, n + 1):
            pi[j] = (factorial(n)/(factorial(n-j)*factorial(s)*(s**(j-s)))) * (lambd/mu)**j * pi0
    else:
        # Cold standby - only active components can fail and only when system is up
        Q = np.zeros((n + 1, n + 1))
        
        for i in range(n + 1):
            working = n - i
            # Active components are min(working, k) ONLY when system is up (working >= k)
            active = min(working, k) if working >= k else 0
            
            # Failure transitions (i → i+1) - only happen when system is up
            if i < n and active > 0:
                Q[i, i+1] = active * lambd
            
            # Repair transitions (i → i-1) - always possible if there are failed components
            if i > 0:
                Q[i, i-1] = min(i, s) * mu
        
        # Fill diagonal elements
        for i in range(n + 1):
            Q[i, i] = -np.sum(Q[i, :])
        
        # Solve for stationary distribution
        A = Q.T
        A[-1, :] = 1
        b = np.zeros(n + 1)
        b[-1] = 1
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        
    availability = 0
    for j in range(n + 1):
        if (n - j) >= k:
            availability += pi[j]  

    return pi, availability

def get_transition_rates(lambd, mu, n, k, s, warm_standby):
    """Generate transition rates for visualization"""
    birth_rates = []
    death_rates = []
    
    for j in range(n + 1):
        working = n - j
        
        if warm_standby:
            # Warm standby: all components can fail
            birth_rate = (n - j) * lambd if j < n else 0
        else:
            # Cold standby: only active components can fail, and only when system is up
            active = min(working, k) if working >= k else 0
            birth_rate = active * lambd if j < n else 0
        
        death_rate = min(j, s) * mu if j > 0 else 0
        
        birth_rates.append(birth_rate)
        death_rates.append(death_rate)
    
    return birth_rates, death_rates

def main():
    st.title("K-out-of-N System Analysis")
    # st.markdown("""
    # **Analyze system reliability with different standby configurations:**
    # - **Warm standby**: All components can fail (even unused ones)
    # - **Cold standby**: Only active components can fail (system stops failing when down)
    # """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Total components (n)", min_value=1, value=4)
        k = st.number_input("Required working components (k)", min_value=1, max_value=n, value=2)
        s = st.number_input("Repairmen (s)", min_value=1, max_value=n, value=1)
        warm_standby = st.checkbox("Warm standby mode", value=True)
    with col2:
        lambd = st.number_input("Failure rate (λ)", min_value=0.0, value=0.5, step=0.1)
        mu = st.number_input("Repair rate (μ)", min_value=0.0, value=1.0, step=0.1)
    
    if st.button("Analyze System"):
        try:
            # Calculate results
            pi, availability = calculate_stationary_distribution(lambd, mu, n, k, s, warm_standby)
            birth_rates, death_rates = get_transition_rates(lambd, mu, n, k, s, warm_standby)
            # Display results
            st.success(f"Fraction of time system is up: {availability:.4f} ({availability*100:.2f}%)")

            
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")

if __name__ == "__main__":
    main()