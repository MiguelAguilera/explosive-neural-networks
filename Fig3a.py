"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Function to compute effective beta
def beta1(beta, H, J, m, gamma):
    """ Computes the effective inverse temperature beta1. """
    b1 = beta / (1 - gamma * beta * (H * m + J / 2 * m**2))
    b1[gamma * beta * (H * m + J / 2 * m**2) > 1] = 1E8  # Prevent divergence
    return b1

# Define pairs of (gamma, beta) for looping
gamma_beta_pairs = [
    (0.0, 0.5), (0.0, 1.0), (0.0, 1.5),
    (-1.2, 0.8), (-1.2, 0.825), (-1.2, 0.9)
]

# Loop over gamma, beta pairs
for gamma, beta in gamma_beta_pairs:
    C = 0.6  # Coupling coefficient

    # Define mesh grid for visualization
    N = 100
    m1 = np.linspace(-1, 1, N)
    m2 = np.linspace(-1, 1, N)
    M1, M2 = np.meshgrid(m1, m2)

    beta1 = beta / (1 + gamma * 0.5 * (M1**2 + M2**2))

    # Compute potential landscape
    if gamma == 0:
        phi = 0.5 * beta1 * (M1**2 + M2**2) - C * np.log(2 * np.cosh(beta1 * (M1 + M2))) - (1 - C) * np.log(2 * np.cosh(beta1 * (M1 - M2)))
    else:
        phi = beta / gamma * np.log(beta1 / beta) + beta1 * (M1**2 + M2**2) - C * np.log(2 * np.cosh(beta1 * (M1 + M2))) - (1 - C) * np.log(2 * np.cosh(beta1 * (M1 - M2)))
    phi[phi < -1000] = 1
    phi = np.nan_to_num(phi, nan=1000)

    # Create figure for each (gamma, beta) pair
    plt.figure(figsize=(4, 3))

    # Plot potential landscape
    plt.imshow(phi, cmap='inferno', origin='lower', norm=colors.PowerNorm(gamma=0.15, vmin=np.min(phi), vmax=1), extent=[-1, 1, -1, 1])
    plt.colorbar(ticks=[np.min(phi), np.min(phi) + (1 - np.min(phi)) * 0.02, np.min(phi) + (1 - np.min(phi)) * 0.2, 1], extend='max')

    # Compute vector field
    N_vec = 16
    m1_vec = np.linspace(-1, 1, N_vec) * 0.95
    m2_vec = np.linspace(-1, 1, N_vec) * 0.95
    M1_vec, M2_vec = np.meshgrid(m1_vec, m2_vec)

    beta1_vec = beta / (1 + gamma * 0.5 * (M1_vec**2 + M2_vec**2))
    dM1 = -M1_vec + C * np.tanh(beta1_vec * (M1_vec + M2_vec)) + (1 - C) * np.tanh(beta1_vec * (M1_vec - M2_vec))
    dM2 = -M2_vec + C * np.tanh(beta1_vec * (M1_vec + M2_vec)) + (1 - C) * np.tanh(beta1_vec * (-M1_vec + M2_vec))
    dM = np.sqrt(dM1**2 + dM2**2)

    plt.quiver(M1_vec, M2_vec, dM1 / dM * 0.1, dM2 / dM * 0.1, pivot='tip', width=0.015, units='inches', scale=1 / 1.4, alpha=0.3)

    # Calculate stable points
    if (gamma==0 and beta<1) or (gamma<0 and beta<=0.8):
        m1=np.array([0.1])
        m2=np.array([0.1])
    elif (gamma==0) or (gamma<0 and beta>1):
        m1=np.array([-1,0.25,0.25,1])
        m2=np.array([-0.25,-1,1,0.25])
    else:
        m1=np.array([-1,-0.5,-0.25,0.01,0.25,0.5,1])
        m2=np.array([-0.25,-0.5,-1,0.01,1,0.5,0.25])

    #m1=np.array([-1,0.25,0.25,1])
    #m2=np.array([-0.25,-1,1,0.25])

    #m1=np.array([0.1])
    #m2=np.array([0.1])
    dt=0.1
    T=10000
    for t in range(T-1):
        beta1 = beta/(1 + gamma*0.5*(m1**2+ m2**2))
        m1 = m1 + dt*(-m1  + C* np.tanh(beta1*(m1+m2)) +  (1-C)* np.tanh(beta1*(m1-m2)))
        m2 = m2 + dt*(-m2  + C* np.tanh(beta1*(m1+m2)) +  (1-C)* np.tanh(beta1*(-m1+m2)))
    # Plot stable points
    plt.plot(m1, m2, 'wx', markersize=5)

    # Set title and labels
    plt.title(rf"$\gamma'={np.round(10 * gamma) / 10},\, \beta={beta}$")
    plt.xlabel(r'$m_1$')
    plt.ylabel(r'$m_2$', rotation=0)

    # Save figure
    plt.savefig(f'img/Fig3a_{str(gamma)[0:4]}_b{beta}.pdf', bbox_inches='tight')

# Show the plot
plt.show()

