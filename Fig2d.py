"""
GPLv3 2025 Miguel Aguilera

This code simulates and visualizes the explosive dynamics of a magnetization model.
It integrates the dynamical equation iteratively for different values of gamma and plots the evolution of magnetization.
Results are saved as PDF files for further analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 20}
plt.rc('font', **font)
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define the magnetization function
def f(m, beta, gamma):
    """
    Defines the self-consistent equation for magnetization.
    """
    return np.tanh(beta * m / (1 + gamma / 2 * m**2))

# Define parameters
T = 60000  # Total integration time
B = 11  # Number of gamma values to test
dt = 0.1  # Time step for integration
beta = 1.001  # Fixed beta value

gammas = -np.linspace(0, 1.5, B)  # Range of gamma values
gammas[0] = 0.0  # Ensure first gamma is zero

# Define colormap for plotting
cmap = plt.get_cmap('plasma_r')
colors = [cmap(i / (B - 1)) for i in range(B)]

# Create first figure for magnetization dynamics
plt.figure(figsize=(4, 3))
for ind, gamma in reversed(list(enumerate(gammas))):
    gamma = round(gamma, 3)
    m = np.ones(T) * 0.01  # Initial condition for m
    for t in range(1, T):
        m[t] = m[t-1] + dt * (-m[t-1] + f(m[t-1], beta, gamma))
    if ind in [0, B//2, B-1]:
        plt.plot(np.arange(T) * dt, m, color=colors[ind], label=rf"$\gamma'={gamma}$")
    else:
        plt.plot(np.arange(T) * dt, m, color=colors[ind])

plt.axis([0, T * dt, 0, 1.05])
plt.legend(loc='center right', bbox_to_anchor=(1.01, 0.37), borderpad=0.2)
plt.xlabel(r'$t$')
plt.ylabel(r'$m$', rotation=0, labelpad=16)
plt.savefig('img/Fig2d_1.pdf', bbox_inches='tight')

# Create second figure for effective beta dynamics
plt.figure(figsize=(4, 3))
for ind, gamma in enumerate(gammas):
    gamma = round(gamma, 3)
    m = np.ones(T) * 0.01
    for t in range(1, T):
        m[t] = m[t-1] + dt * (-m[t-1] + f(m[t-1], beta, gamma))
    if ind in [0, B//2, B-1]:
        plt.plot(np.arange(T) * dt, beta / (1 + gamma / 2 * m**2), '-', color=colors[ind], label=rf"$\gamma'={gamma}$")
    else:
        plt.plot(np.arange(T) * dt, beta / (1 + gamma / 2 * m**2), color=colors[ind])

plt.axis([0, T * dt, 0.5, np.max(beta / (1 + gamma / 2 * m**2)) * 1.05])
plt.xlabel(r'$t$')
plt.ylabel(r"$\beta'$", rotation=0, labelpad=16)
plt.savefig('img/Fig2d_2.pdf', bbox_inches='tight')
plt.show()

## Create phase transition plots
#plt.figure(figsize=(4, 3))
#plt.plot(betas, m1, 'k')
#plt.axis([np.min(betas), np.max(betas), -0.05, 1.05])
#plt.xlabel(r'$\beta$')
#plt.ylabel(r'$m$', rotation=0, labelpad=16)
#plt.savefig('img/phase-transition.pdf', bbox_inches='tight')

plt.show()

