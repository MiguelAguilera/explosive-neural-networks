"""
GPLv3 2025 Miguel Aguilera

This script computes the self-consistent magnetization equations for a spin system using both NumPy and PyTorch.
It iteratively solves for different values of alpha and beta, leveraging GPU acceleration when available.
The results are stored in compressed NumPy files for further analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define parameter grids
A = 1001  # Number of alpha values
B = 10001  # Number of beta values
A = 41  # Number of alpha values
B = 1001  # Number of beta values
betas = 1 / np.linspace(0, 1.5, B)[::-1]  # Inverse temperature values
alphas = np.linspace(0.0, 0.2, A)  # Range of alpha values
J = 1  # Interaction strength
Nint = 10001  # Integration grid size

# Time step parameters for numerical iteration
dt = 0.05
dt2 = 0.05
dt3 = 0.01

# Function to compute magnetization using PyTorch
def calc_torch(alpha, p0, betas, Nint, J, dt):
    """
    Computes the equilibrium magnetization using an iterative self-consistent method.
    
    Args:
        alpha (float): Scaling parameter.
        p0 (tuple): Initial conditions (magnetization, overlap, squared overlap).
        betas (torch.Tensor): Beta values.
        Nint (int): Number of integration points.
        J (float): Coupling parameter.
        dt (float): Time step for convergence.
    
    Returns:
        tuple: Computed values (magnetization, overlap, squared overlap, correction factor).
    """
    B = len(betas)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = betas.clone().detach().to(device)
    
    if p0 is None:
        m = torch.ones(B, device=device)
        for _ in range(50):
            m = torch.tanh(betas * m)
        q = m**2
        q2 = torch.zeros(B, dtype=float, device=device)
        q2_mask = 1 / betas <= (1 + alpha**0.5)
        q2[q2_mask] = (1 - (1 / betas) / (1 + alpha**0.5))[q2_mask]
    else:
        m, q, q2 = [x.clone().detach().to(device) for x in p0]
    
    mask_m0 = (betas <= 1 / (1 + alpha**0.5))
    m[mask_m0] = 0
    
    zmax = 6
    x = torch.linspace(-1., 1., Nint, device=device) * zmax
    dx = x[1] - x[0]
    pdf = (1 / (2 * torch.pi)**0.5 * torch.exp(-0.5 * x**2) * dx)
    
    indices = torch.arange(0, B, device=device)
    error_val = 1E-6
    
    while len(indices) > 0:
        g = betas * m * J
        dm = -m[indices] + torch.tanh(g[indices])
        m[indices] += dt * dm
        mask1 = (torch.abs(dm) > error_val)
        if mask1.sum() / len(indices) <= 0.5:
            indices = indices[mask1]
    
    q = m**2
    C = torch.cosh(g)**(-4)
    
    return m.cpu(), q.cpu(), q2.cpu(), C.cpu()

# Function to compute initial magnetization for alphas
def calc_torch0(alphas, Nint, J, dt):
    """
    Computes the initial magnetization for given alpha values.
    
    Args:
        alphas (torch.Tensor): Alpha values.
        Nint (int): Number of integration points.
        J (float): Coupling parameter.
        dt (float): Time step for convergence.
    
    Returns:
        torch.Tensor: Computed initial magnetization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphas = alphas.clone().detach().to(device)
    A = len(alphas)
    y = torch.ones(A, device=device)
    
    zmax = 6
    x = torch.linspace(-1., 1., Nint, device=device) * zmax
    dx = x[1] - x[0]
    pdf = (1 / (2 * torch.pi)**0.5 * torch.exp(-0.5 * x**2) * dx)
    
    error = 1.0
    while error > 1E-5:
        dy = -y + torch.erf(y / (2 * alphas)**0.5) - 2 * y / (2 * alphas)**0.5 / torch.pi**0.5 * torch.exp(-y**2 / (2 * alphas))
        y += dt * dy
        error = torch.max(torch.abs(dy)).item()
    
    m = torch.erf(y / (2 * alphas)**0.5)
    return m.cpu()

# Compute magnetization for all alpha values
m = torch.ones(B)
q = torch.ones(B)
q2 = torch.ones(B)
C = torch.ones(B)

m0 = calc_torch0(torch.from_numpy(alphas[1:]), Nint, J, dt)
m0 = np.concatenate(([1], m0.numpy()))
filename = f'data/Fig4/beta0_A_{A}_B_{B}_N_{Nint}.npz'
np.savez(filename, m0=m0, alphas=alphas)

# Iterative computation for each alpha value
overwrite = True
for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)
    print(ia,alpha)
    
    filename = 'data/Fig4/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
    if os.path.exists(filename):
        if not overwrite:
            print(f"{filename} exists, skipping!")
            data=np.load(filename)
            m=torch.from_numpy(data['m'])
            q=torch.from_numpy(data['q'])
            q2=torch.from_numpy(data['qsg'])
            continue
        else:
            print(f"{filename} exists, overwriting!")
    if ia>0:
        m[-1]=m0[ia-1]
    if ia==0:
        m[:-1],q[:-1],q2[:-1],C[:-1]=calc_torch(alpha, None, torch.from_numpy(betas[:-1]), Nint, J, dt)
    else:
        m[:-1],q[:-1],q2[:-1],C[:-1]=calc_torch(alpha, (m[:-1],q[:-1],q2[:-1]), torch.from_numpy(betas[:-1]), Nint, J, dt)
    np.savez(filename, m=m.numpy(), q=q.numpy(), qsg=q2.numpy(), C=C.numpy(), betas=betas)

