"""
GPLv3 2025 Miguel Aguilera

This code solves the self-consistent equation for magnetization in a homogeneous explosive neural network.
Results are plotted and saved as compressed numpy data files.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define the equation whose roots we are solving
def equations(m):
    """
    Self-consistent equation for magnetization m.
    """
    return np.tanh(beta * m) - m

# Define range for beta values
B = 100001  # Number of beta values
betas = np.linspace(0, 30, B)

# Arrays to store results
m = np.zeros(B)
dm = np.zeros(B)

m0 = 0.9  # Initial guess for magnetization

# Compute magnetization for each beta
for ib, beta in reversed(list(enumerate(betas))):
    beta = np.round(beta, 6)
    if ib > 0:
        if beta <= 1:
            m[ib] = 0
        else:
            res1 = root(equations, m0, method='lm')
            m[ib] = res1.x[0]
    print(ib, beta, m[ib])

# Compute derivative dm/dβ
dm = np.gradient(m, betas)

# Create figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(betas, m, 'k')
ax[1].plot(betas, dm, 'k')

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

# Save data
filename = 'data/magnetization.npz'
np.savez_compressed(filename, betas=betas, m=m)

# Show plot
plt.show()

