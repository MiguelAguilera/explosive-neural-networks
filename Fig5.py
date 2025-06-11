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
nM = 14
G = 41
N = 32 * 32 * 3  # Image size
Ms = np.arange(1, 15) * 5
gammas = -np.linspace(-1, 1, G)

# === Load data ===
filename = f'data/Fig5.npz'
data = np.load(filename)
m = data['m']
m2 = data['m2']

# === Plot: Mean Overlap ===
plt.figure()
plt.imshow(
    m,
    extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]],
    aspect='auto',
    interpolation='none',
    origin='lower'
)
plt.xlabel('Memory Load $M/N$')
plt.ylabel(r'Scaling Parameter $\gamma$')
plt.colorbar(label='Mean Overlap')
plt.title('Mean Overlap vs $M/N$ and $\gamma$')
plt.tight_layout()

# === Plot: Overlap Variance ===
plt.figure()
plt.imshow(
    m2 - m ** 2,
    extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]],
    aspect='auto',
    interpolation='none',
    origin='lower'
)
plt.xlabel('Memory Load $M/N$')
plt.ylabel(r'Scaling Parameter $\gamma$')
plt.colorbar(label='Overlap Variance')
plt.title('Variance of Overlap')
plt.tight_layout()

plt.show()

