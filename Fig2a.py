"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
import matplotlib.pyplot as plt

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['Latin Modern Roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define the equation whose roots we are interested in
def equation(m, beta, gamma):
    """
    Defines the self-consistent equation for magnetization m.
    """
    return np.tanh(beta * m / (1 + (gamma / 2) * m**2)) - m

# Define range for beta values
Nb = 100001  # Number of beta values to evaluate
betas = np.linspace(0., 1.5, Nb)

# Loop over different gamma values
for gamma in [-0.5, -1.5]:
    print(f"Processing gamma = {gamma}")
    
    # Arrays to store solutions
    m1 = np.full(Nb, np.nan)  # First root
    m2 = np.full(Nb, np.nan)  # Second root
    m3 = np.full(Nb, np.nan)  # Third root
    
    N = 1000  # Number of m values to evaluate
    
    # Iterate over beta values
    for ib, beta in enumerate(betas):
        m_values = np.linspace(0, 1, N)  # Range of m values to check
        y_values = equation(m_values, beta, gamma)
        
        # Identify sign changes (roots of the equation)
        sign_changes = np.sign(y_values)
        
        # Locate points where sign changes from + to - (first root)
        inds1 = np.where(np.logical_and(sign_changes[1:] == -1, sign_changes[:-1] == 1))[0]
        if len(inds1):
            m1[ib] = m_values[inds1[0]]
        else:
            m2[ib] = 0  # Default value when no root is found
        
        # Locate points where sign changes from - to + (third root)
        inds2 = np.where(np.logical_and(sign_changes[1:] == 1, sign_changes[:-1] == -1))[0]
        if len(inds2):
            m2[ib] = 0  # Default value
            m3[ib] = m_values[inds2[0]]
    
    # Plot results
    plt.figure(figsize=(4, 3))
    plt.plot(betas, m1, 'k', label='First root')
    plt.plot(betas, m2, 'k', label='Second root')
    plt.plot(betas, m3, 'k:', label='Third root')
    
    plt.axis([np.min(betas), np.max(betas), -0.05, 1.05])
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$m$', rotation=0, labelpad=16)
    plt.title(rf"$\gamma'={gamma}$", size=20)
    
    # Save figure
    plt.savefig(f'img/Fig2a_{gamma}.pdf', bbox_inches='tight')

# Display all plots
plt.show()

