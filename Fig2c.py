"""
GPLv3 2025 Miguel Aguilera

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, least_squares, root
import math

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Create 3D figure
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(projection='3d')

gamma = -1.2  # Parameter for deformation

# Load precomputed magnetization data
filename = 'data/magnetization.npz'
data = np.load(filename)
betas = data['betas']
m = data['m']

# Compute transformed beta values
beta0 = betas * (1 + gamma * m**2 / 2)

# Find index where beta is close to 3.6
mind = np.argmin((betas - 3.6) ** 2)

# Identify monotonicity changes
inds = np.gradient(beta0, betas) >= 0
m1 = m.copy()
m0 = m.copy()
m0[~inds] = np.nan
m1[inds] = np.nan

# Plot projection lines
ax.plot(beta0[:mind+1] * 0, betas[:mind+1], m[:mind+1], 'gray')
ax.plot(beta0[:mind+1], betas[:mind+1] * 0, m0[:mind+1], 'gray')
ax.plot(beta0[:mind+1], betas[:mind+1] * 0, m1[:mind+1], ':', color='gray')
ax.plot(beta0[:mind+1], betas[:mind+1], m[:mind+1] * 0, 'gray')

# Plot main 3D curve
ax.plot(beta0[:mind+1], betas[:mind+1], m[:mind+1], color='k')

# Adjust axes limits and labels
ax.axes.set_ylim3d(top=betas[mind], bottom=betas[0])
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\beta'$")
ax.set_zlabel(r'$m$', labelpad=5)

# Set background color to white
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# Adjust layout
fig.tight_layout()
fig.subplots_adjust(top=1.4, bottom=-0.3, left=-0.0, right=0.85)

# Save figure
plt.savefig('img/Fig2_c.pdf')

# Display plot
plt.show()
