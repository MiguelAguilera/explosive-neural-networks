"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
from matplotlib import pyplot as plt

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

# Define system parameter
gamma = -1.25  # Interaction parameter

# Load computed magnetization data
filename = 'data/Fig3_b.npz'
data = np.load(filename)
betas = data['betas']  # Beta values
m = data['m']  # Magnetization values

# Compute effective beta transformation
beta0 = betas * (1 + gamma * 0.5 * (m[:, 0]**2 + m[:, 1]**2))

# Identify regions where the transformation is monotonic
inds = np.gradient(beta0, betas) >= 0
m1 = m.copy()
m2 = m.copy()
m1[~inds] = np.nan  # NaN out non-monotonic regions
m2[inds] = np.nan

# Create first plot: Raw magnetization vs beta
plt.figure(figsize=(4, 3))
plt.plot(betas, m, 'k', label='Magnetization')
plt.plot(betas, -m, 'k')
plt.axis([0.5, 1.6, -1.05, 1.05])
plt.title(r"$\gamma' = 0.0$", size=20)
plt.ylabel(r'$\bm{m}$', fontsize=18, rotation=0, labelpad=16)
plt.xlabel(r'$\beta$', fontsize=18)
plt.savefig('img/Fig3_b_1.pdf', bbox_inches='tight')

# Create second plot: Effective beta transformation vs magnetization
plt.figure(figsize=(4, 3))
plt.plot(beta0, m1, 'k', label='Stable branch')
plt.plot(beta0, m2, ':k', label='Unstable branch')
plt.plot(beta0, -m1, 'k')
plt.plot(beta0, -m2, ':k')
plt.axis([0.68, 0.9, -1.05, 1.05])
plt.title(rf"$\gamma' = {gamma}$", size=20)
plt.ylabel(r'$\bm{m}$', fontsize=18, rotation=0, labelpad=16)
plt.xlabel(r'$\beta$', fontsize=18)
plt.savefig('img/Fig3_b_2.pdf', bbox_inches='tight')

# Show plots
plt.show()

