"""
GPLv3 2025 Miguel Aguilera

This code computes and plots the phase diagram of a magnetization model.
It iteratively solves for gamma values as a function of beta, identifying the transition points.
Results are plotted and saved as a PDF file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import math

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['Latin Modern Roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define the self-consistent equation for magnetization
def equation(m):
    return np.tanh(beta * m / (1 - gamma / 2 * m**2)) - m

# Define range for beta values
Nb = 10001  # Number of beta values
betas = np.linspace(0, 1, Nb + 1)[:-1]  # Exclude last value for consistency

gammas = np.ones(Nb) * 2  # Initialize gamma values
Ng = 1001  # Number of gamma values

N = 10000  # Number of m values to evaluate

# Compute gamma values iteratively
for b, beta in enumerate(betas):
    if beta == 0:
        gammas[b] = 2
    else:
        gammas_ = np.linspace(0, gammas[b], Ng)
        for g, gamma in reversed(list(enumerate(gammas_))):
            gamma = np.round(gamma, 8)
            m = np.linspace(0, 1, N)
            y = equation(m)
            s = np.sign(y)
            inds = np.where(np.logical_and(s[1:] == 1, s[:-1] == -1))[0]
            if not len(inds):
                gammas[b] = gamma
                break
    print(beta, gammas[b])

# Define additional beta range for plotting
betas2 = np.linspace(2/3, 2, Nb)

# Create and configure figure
plt.figure(figsize=(4, 3))
plt.plot(betas[gammas > 0], -gammas[gammas > 0], 'k', lw=1)
plt.plot([1, 1], [0, -2], 'k', lw=1)
plt.axis([0, 2, 0, -2])

# Label axes and annotations
plt.xlabel(r'$\beta$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=16)
plt.text(0.4, -0.9, r'P', size=20)
plt.text(1.5, -0.9, r'M', size=20)
plt.text(0.55, -1.62, r'Exp', size=20)

# Save figure
plt.savefig('img/Fig2_b.pdf', bbox_inches='tight')

# Display plot
plt.show()
