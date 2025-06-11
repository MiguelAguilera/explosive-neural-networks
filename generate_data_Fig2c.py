"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define parameters
B = 100001  # Number of beta values to evaluate
betas = np.linspace(0, 30, B)  # Range of beta values

dt = 0.04  # Time step for integration

# Function to compute the equilibrium magnetization for each beta
@njit
def calculate_m(betas):
    """
    Computes the equilibrium magnetization (m) for each beta value using an iterative method.
    The system evolves dynamically until convergence.
    
    Args:
        betas (numpy array): Array of beta values to evaluate.
    
    Returns:
        numpy array: Array of magnetization values for each beta.
    """
    M = np.zeros(B)  # Storage for magnetization values
    
    # Iterate over beta values in reverse order
    for ib in range(len(betas) - 1, -1, -1):
        beta = np.round(betas[ib], 6)  # Ensure numerical precision
        
        if ib > 0:
            if beta <= 1:
                M[ib] = 0  # Below critical beta, magnetization is zero
            else:
                error = 1.0  # Initialize error for convergence check
                m = 0.9  # Initial guess for magnetization
                
                # Iterative update until convergence
                while error > 1E-8:
                    dm = -m + np.tanh(beta * m)  # Compute update step
                    m += dt * dm  # Apply update
                    error = np.abs(dm)  # Check convergence condition
                
                M[ib] = m  # Store the computed magnetization
        
        print(ib, beta, M[ib])  # Output progress
    
    return M

# Compute magnetization values
m = calculate_m(betas)

# Compute derivative dm/dβ
dm = np.gradient(m, betas)

# Create figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(betas, m, 'k', label='Magnetization')
ax[1].plot(betas, dm, 'k', label='Derivative')

# Set labels and axes limits
ax[0].set_ylabel(r'$m$', rotation=0, labelpad=16)
ax[0].set_xlabel(r'$\beta$')
ax[0].axis([betas[0], np.max(betas), 0, 1])
ax[1].set_ylabel(r'$\dfrac{dm}{d\beta}$', rotation=0, labelpad=16)
ax[1].set_xlabel(r'$\beta$')
ax[1].axis([betas[0], np.max(betas), 0, 2.5])

# Adjust layout
plt.subplots_adjust(wspace=0.02, hspace=0.02)
fig.tight_layout(h_pad=0.0, w_pad=0.7, rect=[0, 0, 1, 0.975])

# Save computed data
filename = 'data/Fig2c.npz'
np.savez_compressed(filename, betas=betas, m=m)

# Show the plot
plt.show()

