"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
import matplotlib.pyplot as plt

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size':20}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':16})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# === Parameters (must match saved data) ===
beta = 2
G = 41
N = 32 * 32 * 3  # Image size
Ms = np.arange(1, 15) * 5
nM=len(Ms)
gammas = -np.linspace(-1, 1, G)

# === Load data ===
filename = f'data/Fig5.npz'
data = np.load(filename)
m = data['m']
m2 = data['m2']

# === Plot: Mean Overlap ===
plt.figure(figsize=(6, 4),layout='constrained')
plt.imshow(
    m,
    extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]],
    aspect='auto',
    interpolation='none',
    origin='lower',
    cmap='inferno_r'
)
plt.colorbar()
plt.ylabel(r"$\gamma'$",rotation=0, labelpad=16)
plt.xlabel(r'$\alpha$')
plt.title(r'$m=\langle o\rangle$',pad=8)
plt.savefig('img/memory-capacity-cifar.pdf', dpi=300)

# === Plot: Overlap Variance ===
plt.figure(figsize=(6, 4),layout='constrained')
plt.imshow(
    m2 - m ** 2,
    extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]],
    aspect='auto',
    interpolation='none',
    origin='lower',
    cmap='inferno_r'
)
plt.colorbar()
plt.ylabel(r"$\gamma'$",rotation=0, labelpad=16)
plt.xlabel(r'$\alpha$')
plt.title(r'$\langle o^2\rangle - \langle o\rangle^2$',pad=8)
plt.savefig('img/spurious-memories-cifar.pdf', dpi=300)

plt.show()

