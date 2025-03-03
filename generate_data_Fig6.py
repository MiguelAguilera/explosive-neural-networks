import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def Gtanh2(x, g, sqrtD):
    """
    Function for the Gaussian integral, representing (1 - tanh^2) weighted by a Gaussian distribution.
    
    Parameters:
        x (float or array): Integration variable.
        g (float): Mean input value.
        sqrtD (float): Standard deviation of Gaussian noise.
    
    Returns:
        float: Weighted function value.
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (1 - np.tanh(g + x * sqrtD)**2)

def Gaussian_integral(F, g, sqrtD, Nint=10000, zmax=5, z0=0):
    """
    Computes a numerical integral using Gaussian quadrature.
    
    Parameters:
        F (function): Function to integrate.
        g (float): Mean value for the function.
        sqrtD (float): Standard deviation scaling factor.
        Nint (int): Number of integration points.
        zmax (float): Maximum value for integration.
        z0 (float): Shift for integration range.
    
    Returns:
        float: Numerical integral result.
    """
    if sqrtD == 0:
        return np.sqrt(2 * np.pi) * sqrtD * F(0, g, sqrtD)
    else:
        z = np.linspace(-1., 1., Nint) * zmax + z0
        return np.sum(F(z, g, sqrtD)) * (z[1] - z[0])

def equations(p):
    """
    Defines the self-consistent equation for q.
    
    Parameters:
        p (float): The variable q to solve for.
    
    Returns:
        float: Difference between the function and q.
    """
    q = p
    q = q * int(q > 0)  # Ensure q is non-negative
    return 1 - Gaussian_integral(Gtanh2, 0, beta * np.sqrt(q), Nint=10000, zmax=6) - q

# Define range for beta values
B = 100001  # Number of beta values
betas = np.linspace(0, 30, B)

# Arrays to store results
q1 = np.zeros(B)  # Stores q values

# Initial conditions for root solving
q0 = 0.9  # Initial guess for q
b0 = betas[-1]

# Solve for q across beta values in reverse order
for ib, beta in reversed(list(enumerate(betas))):
    beta = np.round(beta, 6)
    if ib > 0:
        if beta <= 1:
            q1[ib] = 0
        else:
            res1 = root(equations, (q0), method='lm')  # Solve self-consistent equation
            q1[ib] = res1.x[0]
            q0 = res1.x[0]
    print(ib, beta, q1[ib])

# Compute the derivative dq/dβ
dq1 = np.gradient(q1, betas)

# Create figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(8, 3))

# Plot q vs beta
ax[0].plot(betas, q1, 'k')
ax[0].set_ylabel(r'$q$', rotation=0, labelpad=16)
ax[0].set_xlabel(r'$\beta$')
ax[0].axis([betas[0], np.max(betas), 0, 1])

# Plot dq/dβ vs beta
ax[1].plot(betas, dq1, 'k')
ax[1].set_ylabel(r'$\dfrac{dq}{d\beta}$', rotation=0, labelpad=16)
ax[1].set_xlabel(r'$\beta$')
ax[1].axis([betas[0], np.max(betas), 0, 2.5])

# Adjust layout
plt.subplots_adjust(wspace=0.02, hspace=0.02)
fig.tight_layout(h_pad=0.0, w_pad=0.7, rect=[0, 0, 1, 0.975])

# Save data
filename = 'data/Fig6.npz'
np.savez_compressed(filename, betas=betas, q=q1)

# Show plot
plt.show()

