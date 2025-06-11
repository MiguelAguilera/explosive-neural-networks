"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import njit

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

# Define parameters
B = 100001  # Number of beta values to evaluate
betas = np.linspace(0, 20, B)  # Range of beta values

C = 0.25  # Coupling parameter
dt = 0.04  # Time step for integration

# Function to compute the equilibrium magnetization for each beta
@njit
def calculate_m(betas):
    """
    Computes the equilibrium magnetization for each beta value using an iterative method.
    The system is solved using a numerical update rule until convergence.
    
    Args:
        betas (numpy array): Array of beta values to evaluate.
    
    Returns:
        numpy array: Array of magnetization values for each beta.
    """
    M = np.zeros((B, 2))  # Storage for magnetization values
    m0 = np.array([0.25, 1.0])  # Initial conditions for m
    
    # Iterate over beta values in reverse order
    for ib in range(len(betas) - 1, -1, -1):
        beta = np.round(betas[ib], 6)  # Ensure numerical precision
        m = m0.copy()
        error = 1.0  # Initialize error for convergence check
        
        # Initialize state
        m = np.ones(2)
        m[0] = 0.25
        
        # Iterative update until convergence
        while error > 1E-8:
            dm = np.zeros(2)
            
            # Compute updates using the dynamical rule
            dm[0] = -m[0] + np.tanh(beta * (m[0] + m[1])) * (1 + C) / 2 + np.tanh(beta * (m[0] - m[1])) * (1 - C) / 2 
            dm[1] = -m[1] + np.tanh(beta * (m[0] + m[1])) * (1 + C) / 2 + np.tanh(beta * (m[1] - m[0])) * (1 - C) / 2 
            
            # Update state
            m += dt * dm
            
            # Compute error for stopping criterion
            error = np.max(np.abs(dm))
        
        # Store the computed magnetization values
        print(beta, m)
        M[ib, :] = m
        
        # Update initial condition for the next iteration
        m0 = m.copy()
    
    return M

# Compute magnetization values
M = calculate_m(betas)

# Save results to a compressed numpy file
filename = 'data/Fig3b.npz'
np.savez_compressed(filename, betas=betas, m=M)

# Plot results
plt.figure(figsize=(4, 3))
plt.plot(betas, M, 'k')
plt.axis([np.min(betas), np.max(betas), -0.05, 1.05])
plt.ylabel(r'$\bm m$', fontsize=18, rotation=0, labelpad=16)
plt.xlabel(r'$\beta$', fontsize=18)
plt.show()

