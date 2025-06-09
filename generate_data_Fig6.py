import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root

# --- Plotting configuration for LaTeX-style text ---
plt.rc('text', usetex=True)
plt.rc('font', size=16)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# --- Define integrand function: (1 - tanh²) under Gaussian noise ---
def Gtanh2(x, g, sqrtD):
    """
    Computes the Gaussian-weighted (1 - tanh^2) function.
    
    Parameters:
        x (float or np.array): Integration variable.
        g (float): Mean input to tanh.
        sqrtD (float): Std. dev. of Gaussian noise.
    
    Returns:
        np.array: Weighted function values.
    """
    arg = g + x * sqrtD
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (1 - np.tanh(arg)**2)

# --- Perform numerical integration over Gaussian distribution ---
def Gaussian_integral(F, g, sqrtD, Nint=10000, zmax=5, z0=0):
    """
    Approximates a Gaussian integral of a given function.
    
    Parameters:
        F (callable): Function to integrate (must accept vectorized inputs).
        g (float): Mean input for function.
        sqrtD (float): Std. dev. for noise term.
        Nint (int): Number of sampling points.
        zmax (float): Integration bound (symmetric around 0).
        z0 (float): Offset for integration center.
    
    Returns:
        float: Approximate integral value.
    """
    if sqrtD == 0:
        return np.sqrt(2 * np.pi) * sqrtD * F(0, g, sqrtD)
    else:
        z = np.linspace(-1., 1., Nint) * zmax + z0
        dz = z[1] - z[0]
        return np.sum(F(z, g, sqrtD)) * dz

# --- Self-consistency equation for q ---
def equations(p):
    """
    Defines the self-consistent equation for overlap q.
    
    Parameters:
        p (float): Trial value for q.
    
    Returns:
        float: Residual of fixed-point equation.
    """
    q = max(0, p)  # Ensure q is non-negative
    return 1 - Gaussian_integral(Gtanh2, 0, beta * np.sqrt(q), Nint=10000, zmax=6) - q

# --- Parameter sweep: beta values ---
B = 100001  # Number of beta points
betas = np.linspace(0, 30, B)  # Range of beta

# --- Allocate result array ---
q1 = np.zeros(B)

# --- Initial condition for root-finding ---
q0 = 0.9  # Initial guess
b0 = betas[-1]  # Start from high beta

# --- Solve q(beta) using fixed-point root-finding (backward sweep) ---
for ib, beta in reversed(list(enumerate(betas))):
    beta = np.round(beta, 6)  # Improve numerical stability
    if ib > 0:
        if beta <= 1:
            q1[ib] = 0  # Below critical beta: no memory
        else:
            res1 = root(equations, q0, method='lm')
            q1[ib] = res1.x[0]
            q0 = res1.x[0]  # Update for next beta
    print(ib, beta, q1[ib])  # Progress output

# --- Compute derivative dq/dβ ---
dq1 = np.gradient(q1, betas)

# --- Plotting results ---
fig, ax = plt.subplots(1, 2, figsize=(8, 3))

# Plot q vs beta
ax[0].plot(betas, q1, 'k')
ax[0].set_xlabel(r'$\beta$')
ax[0].set_ylabel(r'$q$', rotation=0, labelpad=16)
ax[0].axis([betas[0], betas[-1], 0, 1])

# Plot dq/dβ vs beta
ax[1].plot(betas, dq1, 'k')
ax[1].set_xlabel(r'$\beta$')
ax[1].set_ylabel(r'$\dfrac{dq}{d\beta}$', rotation=0, labelpad=16)
ax[1].axis([betas[0], betas[-1], 0, 2.5])

# Tidy plot layout
plt.subplots_adjust(wspace=0.02)
fig.tight_layout(h_pad=0.0, w_pad=0.7, rect=[0, 0, 1, 0.975])

# --- Save results to file ---
np.savez_compressed('data/Fig6.npz', betas=betas, q=q1)

# --- Display plot ---
plt.show()

