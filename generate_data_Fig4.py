# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import torch
import os

# Matplotlib configuration for LaTeX-rendered text
plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define grid sizes and parameters
A = 1001
B = 10001  # Number of beta values

# Define inverse temperature grid (increasing)
betas = 1 / np.linspace(0, 1.5, B)[::-1]

# Define alpha (load) values
alphas = np.linspace(0.0, 0.2, A)

# Coupling strength
J = 1

# Number of integration points for Gaussian quadrature
Nint = 10001

# Time step parameters for numerical integration
dt=0.04
dt2=0.04
dt3=0.01

# Overwrite files if already calculated
overwrite = True


# Main function to compute order parameters using mean-field iteration
def calc_torch(alpha, p0, betas, Nint, J, dt):
    B = len(betas)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move betas to device
    betas = betas.clone().detach().to(device)
    
    # Initialize from fixed point or use given initial conditions
    if p0 is None:
        m = torch.ones(B, device=device)
        for _ in range(50):
            m = torch.tanh(betas * m)
        q = m**2
        q2 = torch.zeros(B, dtype=float, device=device)
        q2_mask = 1 / betas <= (1 + alpha**0.5)
        q2[q2_mask] = (1 - (1 / betas) / (1 + alpha**0.5))[q2_mask]
    else:
        m = p0[0].clone().detach().to(device)
        q = p0[1].clone().detach().to(device)
        q2 = p0[2].clone().detach().to(device)

    # Enforce paramagnetic region
    mask_m0 = (betas <= 1 / (1 + alpha**0.5))
    m[mask_m0] = 0

    # Initialize q2 from analytic approximation if alpha > 0
    q2 = torch.zeros(B, dtype=float, device=device)
    if alpha > 0:
        q2_mask = 1 / betas <= (1 + alpha**0.5)
        q2[q2_mask] = (1 - (1 / betas) / (1 + alpha**0.5)).to(q2.dtype)[q2_mask]
    
    # Ensure q >= q2
    q = torch.maximum(q, q2)

    # Prepare Gaussian integration grid
    zmax = 6
    x = torch.linspace(-1., 1., Nint, device=device) * zmax
    dx = x[1] - x[0]
    pdf = (1 / (2 * torch.pi)**0.5 * torch.exp(-0.5 * x**2) * dx)

    t = 0
    indices1 = torch.arange(0, B).to(device)
    indices2 = torch.arange(0, B).to(device)
    error_val = 1E-6

    # Iterative update loop
    if alpha > 0:
        while (len(indices1) > 0 or len(indices2) > 0):
            t += 1
            g = betas * m * J

            # Compute renormalized variances
            r = q / (1 - betas * J * (1 - q))**2
            r2 = q2 / (1 - betas * J * (1 - q2))**2

            # Compute effective noise
            sqrtD = betas * J * torch.sqrt(alpha * r)
            sqrtD2 = betas * J * torch.sqrt(alpha * r2)

            # Compute Gaussian integrals
            z = g[indices1, None] + x[None, :] * sqrtD[indices1, None]
            z2 = x[None, :] * sqrtD2[indices2, None]
            F1 = pdf[None, :] * torch.tanh(z)
            F2 = pdf[None, :] * torch.tanh(z)**2
            F3 = pdf[None, :] * torch.tanh(z2)**2

            # Compute updates
            dm = -m[indices1] + torch.sum(F1, dim=1)
            dq = -q[indices1] + torch.sum(F2, dim=1)
            dq2 = -q2[indices2] + torch.sum(F3, dim=1)

            # Adaptive time step modulation
            dt0 = 1. + 0.99 * torch.cos(2 * torch.pi * torch.tensor(t, dtype=torch.float32) / 10000)

            # Update values
            m[indices1] += dt * dm * dt0
            q[indices1] += dt2 * dq * dt0
            q2[indices2] += dt3 * dq2 * dt0
            m[mask_m0] = 0.

            # Clamp to avoid numerical issues
            m = m.clamp(min=0.)
            q = q.clamp(min=0.)
            q2 = q2.clamp(min=0.)

            # Select which indices still need updating
            mask1 = (torch.abs(dm) > error_val) | (torch.abs(dq) > error_val)
            if mask1.sum() / len(indices1) <= 0.5:
                indices1 = indices1[mask1]
            mask2 = torch.abs(dq2) > error_val
            if mask2.sum() / len(indices2) <= 0.5:
                indices2 = indices2[mask2]

        # Compute susceptibility
        z = g[:, None] + x[None, :] * sqrtD[:, None]
        F4 = pdf[None, :] * torch.cosh(z)**(-4)
        C = torch.sum(F4, dim=1)
    else:
        # For alpha = 0 (no interactions), fixed-point solution is tanh(beta * m)
        while len(indices1) > 0:
            t += 1
            g = betas * m * J
            dm = -m[indices1] + torch.tanh(g[indices1])
            m[indices1] += dt * dm
            mask1 = (torch.abs(dm) > error_val)
            if mask1.sum() / len(indices1) <= 0.5:
                indices1 = indices1[mask1]
        q = m**2
        C = torch.cosh(g)**(-4)

    return m.cpu(), q.cpu(), q2.cpu(), C.cpu()

# Compute analytical solution m0 for alpha > 0 (for initialization)
def calc_torch0(alphas, Nint, J, dt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphas = alphas.clone().detach().to(device)
    A = len(alphas)
    y = torch.ones(A, device=device)

    zmax = 6
    x = torch.linspace(-1., 1., Nint, device=device) * zmax
    dx = x[1] - x[0]
    pdf = (1 / (2 * torch.pi)**0.5 * torch.exp(-0.5 * x**2) * dx)

    error = 1.0
    t = 0
    while error > 1E-5:
        t += 1
        dy = -y + torch.erf(y / (2 * alphas)**0.5) - \
             2 * y / (2 * alphas)**0.5 / torch.pi**0.5 * torch.exp(-y**2 / (2 * alphas))
        y += dt * dy
        error = torch.max(torch.abs(dy)).item()
    m = torch.erf(y / (2 * alphas)**0.5)
    return m.cpu()

# === MAIN EXECUTION ===

# Initialize memory and order parameters
m = torch.ones(B)
q = torch.ones(B)
q2 = torch.ones(B)
C = torch.ones(B)

# Compute and save m0 for initialization
m0 = calc_torch0(torch.from_numpy(alphas[1:]), Nint, J, dt)
m0 = np.concatenate(([1], m0.numpy()))
filename = 'data/Fig4/beta0_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) + '.npz'
np.savez(filename, m0=m0, alphas=alphas)

# Run simulation over all alpha values
for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)
    print(ia, alpha)
    
    filename = 'data/Fig4/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) + '.npz'
    if os.path.exists(filename):
        if not overwrite:
            print(f"{filename} exists, skipping!")
            data = np.load(filename)
            m = torch.from_numpy(data['m'])
            q = torch.from_numpy(data['q'])
            q2 = torch.from_numpy(data['qsg'])
            continue
        else:
            print(f"{filename} exists, overwriting!")

    # Use m0 as initialization for the last beta
    if ia > 0:
        m[-1] = m0[ia - 1]

    # Compute fixed point
    if ia == 0:
        m[:-1], q[:-1], q2[:-1], C[:-1] = calc_torch(alpha, None, torch.from_numpy(betas[:-1]), Nint, J, dt)
    else:
        m[:-1], q[:-1], q2[:-1], C[:-1] = calc_torch(alpha, (m[:-1], q[:-1], q2[:-1]), torch.from_numpy(betas[:-1]), Nint, J, dt)

    # Save results
    np.savez(filename, m=m.numpy(), q=q.numpy(), qsg=q2.numpy(), C=C.numpy(), betas=betas)

