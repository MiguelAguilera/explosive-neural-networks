"""
GPLv3 2025 Miguel Aguilera

This code computes and plots the phase diagram of a magnetization model.
It iteratively solves for gamma values as a function of beta, identifying the transition points.
The results are visualized in a phase diagram and saved as a PDF file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import math
from numba import njit

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['Latin Modern Roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define range for beta values
Nb = 10001  # Number of beta values to sample
betas = np.linspace(0, 1, Nb + 1)[:-1]  # Exclude last value for consistency

Ng = 1001  # Number of gamma values to sample

N = 10000  # Number of m values to evaluate for root finding

# Function to compute gamma values iteratively for each beta
@njit(parallel=True)
def compute_gamma(betas):
    """
    Iterates over beta values to determine the critical gamma values at which 
    the self-consistent equation for magnetization loses solutions.
    """
    gammas = np.ones(Nb) * 2  # Initialize gamma values with a default of 2
    for b, beta in enumerate(betas):
        if beta == 0:
            gammas[b] = 2
        else:
            gammas_ = np.linspace(0, gammas[b], Ng)  # Sample gamma values
            for g in range(len(gammas_) - 1, -1, -1):  # Iterate in reverse order
                gamma = np.round(gammas_[g], 8)
                m = np.linspace(0, 1, N)  # Sample m values
                y = np.tanh(beta * m / (1 - gamma / 2 * m**2)) - m
                s = np.sign(y)
                inds = np.where(np.logical_and(s[1:] == 1, s[:-1] == -1))[0]  # Find sign change
                if not len(inds):
                    gammas[b] = gamma  # Update gamma if no sign change detected
                    break
        print(b/Nb, beta, gammas[b])  # Output progress
    return gammas

# Compute the gamma values over the range of betas
gammas = compute_gamma(betas)

# Define additional beta range for plotting
betas2 = np.linspace(2/3, 2, Nb)

# Create and configure the figure
plt.figure(figsize=(4, 3))
plt.plot(betas[gammas > 0], -gammas[gammas > 0], 'k', lw=1)  # Phase boundary
plt.plot([1, 1], [0, -2], 'k', lw=1)  # Vertical reference line at beta=1
plt.axis([0, 2, 0, -2])  # Set plot axis limits

# Label axes and add annotations
plt.xlabel(r'$\beta$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=16)
plt.text(0.4, -0.9, r'P', size=20)  # Annotate critical points
plt.text(1.5, -0.9, r'M', size=20)
plt.text(0.55, -1.62, r'Exp', size=20)

# Save the figure
plt.savefig('img/Fig2_b.pdf', bbox_inches='tight')

# Display the plot
plt.show()

