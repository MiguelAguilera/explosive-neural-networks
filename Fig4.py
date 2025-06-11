"""
GPLv3 2025 Miguel Aguilera
"""

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from numba import njit               # Just-In-Time compilation for numerical performance
import numba
from scipy.interpolate import interp1d
import scipy as sp
import os

# Configure matplotlib for LaTeX-style text rendering
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from matplotlib import cm
cmap = cm.get_cmap('hot_r')

# Configure base directory for saving free energy values
base_dir = os.path.expanduser("~/explosive-nn-data/data-Fig4")


# Number of integration points for numerical integration (e.g., in Gaussian integrals)
Nint = 10001

# Number of alpha samples (e.g., memory load parameter)
A = 1001

# Number of beta samples (e.g., inverse temperature values)
B = 10001

# Coupling constant (used in the Hamiltonian or dynamics)
J = 1

# Nonlinearity parameters for two models or conditions
gamma = -0.8
gamma2 = 0.8

# Flag to determine whether to overwrite existing data files
overwrite = False


# Define numerically stable log(cosh(x)) function
@njit
def log2cosh(x):
    ax = np.abs(x)
    return ax + np.log(1 + np.exp(-2 * ax))

# Compute the G-term in the free energy, involving Gaussian integral with log2cosh
@njit
def G(m, q, alpha, b, gamma):
    r = q / (1 - b * J * (1 - q))**2
    R = (1 / (b * J) - (1 - 2 * q)) / (1 - b * J * (1 - q))**2
    g = b * m * J
    sqrtD = b * J * np.sqrt(alpha * r)
    zmax = 6
    z0 = 0
    Nint = 1001
    x = np.linspace(-1., 1., Nint) * zmax + z0  # Integration support
    Gx = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * log2cosh(g + x * sqrtD)  # Integrand
    return np.sum(Gx) * (x[1] - x[0])  # Riemann sum

# Compute the free energy of the system depending on m, q, and various parameters
@njit
def free_energy(m, q, b, alpha, beta, gamma):
    if q < 0:
        return 100  # Penalize unphysical values of q
    if alpha == 0 and gamma == 0:
        # Classical mean-field (no memory load, no nonlinearity)
        phi = -beta * J / 2 * m**2 + np.log(2 * np.cosh(beta * m * J))
    elif alpha == 0:
        # No memory load, but with nonlinearity
        if (1 + gamma / 2 * J * m**2) <= 0:
            return 100
        phi = beta / gamma * np.log(1 + gamma / 2 * J * m**2) - b * J * m**2 + np.log(2 * np.cosh(b * m * J))
    elif gamma == 0:
        # With memory load, no nonlinearity
        if (1 - beta * J * (1 - q)) < 0:
            return 100
        r = q / (1 - beta * J * (1 - q))**2
        R = (1 / (beta * J) - (1 - 2 * q)) / (1 - beta * J * (1 - q))**2
        Gmx = G(m, q, alpha, beta, gamma)
        phi = (-0.5 * beta * J * m**2
               - 0.5 * alpha * (beta * J) * (1 + (beta * J) * r * (1 - q))
               - 0.5 * alpha * (np.log(1 - beta * J * (1 - q)) - beta * J * np.sqrt(r * q))
               + Gmx)
    else:
        # Full model with both memory load and nonlinearity
        if (1 - b * J * (1 - q)) <= 0:
            return 100
        if (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - b * J * (1 - q**2)) / (1 - b * J * (1 - q))**2 - 1)) <= 0:
            return 100
        r = q / (1 - b * J * (1 - q))**2
        R = (1 - b * J * (1 - 2 * q)) / (b * J) / (1 - b * J * (1 - q))**2
        Gmx = G(m, q, alpha, b, gamma)
        if (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((b * J) * (R - r * q) - 1)) <= 0:
            return 100
        phi = (beta / gamma * np.log(1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((b * J) * (R - r * q) - 1))
               - b * J * m**2
               - 0.5 * alpha * (b * J) * (1 + (b * J) * r * (1 - q) + (b * J) * (R - r * q))
               - 0.5 * alpha * (np.log(1 - b * J * (1 - q)) - b * J * np.sqrt(r * q))
               + Gmx)
    return -phi  # Return negative free energy (convention)

# Wrapper to compute or retrieve free energy values for m, q, and q_sg (spin glass)
def get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=False):
    """
    Compute or load free energy values f1 and f2.

    Parameters:
        m, q, qsg: Arrays of m, q, and q_sg values.
        betas, alpha, betas0, betas0_: Arrays of beta values.
        gamma: Scalar gamma value.
        A, B: Integers representing relevant parameters for the file naming.
        overwrite: If True, always compute and overwrite files. Otherwise, load if available.

    Returns:
        f1, f2: Computed or loaded free energy arrays.
    """
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Construct filename
    filename = os.path.join(base_dir, f"free_energy_alpha_{alpha:.6f}_gamma_{gamma:.6f}_B_{B}.npz")

    if not overwrite and os.path.exists(filename):
        # Load precomputed f1 and f2 if file exists
        data = np.load(filename)
        f1 = data['f1']
        f2 = data['f2']
        print(f"Loaded cached free energy for alpha={alpha:.6f}, gamma={gamma:.6f}, B={B}")
    else:
        # Compute f1 and f2 and save to file
        f1 = calc_f(m, q, betas, alpha, betas0, gamma)
        f2 = calc_f(m * 0, qsg, betas, alpha, betas0_, gamma)

        np.savez(filename, f1=f1, f2=f2)
        print(f"Saved computed free energy for alpha={alpha:.6f}, gamma={gamma:.6f}, B={B}")

    return f1, f2



# Generate inverse temperature values (descending order), avoiding division by zero
betas = 1 / np.linspace(0.0001, 1.5, B)[::-1]
betas = 1 / np.linspace(0, 1.5, B)[::-1]  # Overrides previous line; starts from 0 (may include inf)

# Generate alpha values (e.g., memory load) from 0.0 to 0.2
alphas = np.linspace(0.0, 0.2, A)


# Arrays to store critical beta values or related order parameters for different models
bM = np.ones(A) * np.nan       # Beta of magnetization transition (gamma = 0)
bM1 = np.ones(A) * np.nan      # Beta of magnetization transition (positive gamma)
bM2 = np.ones(A) * np.nan      # Beta of magnetization transition (negative gamma)

bM__ = np.ones(A) * np.nan     # Duplicate sets of critical betas for a different gamma
bM1__ = np.ones(A) * np.nan
bM2__ = np.ones(A) * np.nan

# Arrays to store maximum beta values before instability or divergence (with gamma)
bg = np.ones(A)
bg1 = np.ones(A)
bg2 = np.ones(A)

# Arrays to store beta values of entropy-like transitions (or energy features)
bE = np.zeros(A) * np.nan
bE1 = np.zeros(A) * np.nan
bE2 = np.zeros(A) * np.nan

# Arrays to store beta values where transitions or crossings are detected
bc = np.ones(A) * np.nan
bc1 = np.ones(A) * np.nan
bc2 = np.ones(A) * np.nan
bc_ = np.ones(A) * np.nan
bc1_ = np.ones(A) * np.nan
bc2_ = np.ones(A) * np.nan

# Arrays to store critical beta values from adaptive thresholds or numerical bifurcations
bAT = np.ones(A) * np.nan
bAT1 = np.ones(A) * np.nan
bAT2 = np.ones(A) * np.nan

# Arrays to store magnetization and overlap values at critical or interesting points
mM = np.zeros(A)   # Magnetization at memory transition
qM = np.zeros(A)   # Overlap at memory transition
mc = np.zeros(A)   # Magnetization at crossing
qc = np.zeros(A)   # Overlap at crossing
mE = np.zeros(A)   # Magnetization at entropy-like event
qE = np.zeros(A)   # Overlap at entropy-like event

# Placeholder for minimum beta at which a memory state is observed
bM_ = np.ones(A) * np.inf


###    CALCULATE DETERMINISTIC MODEL    ###

# Load previously computed deterministic model data from file
filename = 'data/Fig4_/beta0_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
data = np.load(filename)
m0 = data['m0']          # Magnetization values from deterministic model
alphas = data['alphas']  # Corresponding alpha values (e.g. memory load)

# Find the index of the largest alpha for which magnetization is significantly non-zero
inds = np.where(m0 > 0.001)[0]
a0M = np.max(alphas[inds])       # Maximum alpha supporting non-zero magnetization
ia0M = np.argmax(alphas[inds])   # Index of that alpha

# Compute first free energy estimate f1 from m0 using analytical expression
x = sp.special.erfinv(m0)
f1 = 0.5 * (sp.special.erf(x))**2 + 1 / np.pi * np.exp(-x**2) \
     - 2 / np.pi * (np.exp(-x**2) + np.sqrt((alphas * np.pi) / 2)) \
     * (x * np.sqrt(np.pi) * sp.special.erf(x) + np.exp(-x**2))

# Compute second free energy estimate f2 assuming zero magnetization (x = 0)
x = x * 0
f2 = 0.5 * (sp.special.erf(x))**2 + 1 / np.pi * np.exp(-x**2) \
     - 2 / np.pi * (np.exp(-x**2) + np.sqrt((alphas * np.pi) / 2)) \
     * (x * np.sqrt(np.pi) * sp.special.erf(x) + np.exp(-x**2))

# Print value of the inverse error function at 0 for reference/debugging
print(sp.special.erfinv(0))

# Determine the largest alpha where the zero-magnetization free energy is lower than the nonzero one
inds = np.where(f2 > f1)[0]
a0c = np.max(alphas[inds])       # Critical alpha at which the zero state becomes more favorable
ia0c = np.argmax(alphas[inds])   # Index of this critical alpha
print(a0c, a0M)

# Numba-accelerated function to compute free energy across a vector of betas
@njit
def calc_f(m, q, b, alpha, betas, gamma):
    f = np.zeros(len(betas))
    for ib in range(len(betas)):
        f[ib] = free_energy(m[ib], q[ib], b[ib], alpha, betas[ib], gamma)
    return f
    
    
###    RUN OVER BETA VALUES    ###

# Arrays to store critical alpha values for each beta
aM = np.zeros(B)       # Max alpha where magnetization is non-zero
ac = np.zeros(B)       # Min alpha where F1 > F2 (non-trivial state energetically favored)
b0M = np.zeros(B)      # Placeholder for potential future beta-related quantity

# Arrays to store m, q, q_sg, and free energy across alpha-beta grid
M = np.zeros((A, B))       # Magnetization m
Q = np.zeros((A, B))       # Order parameter q
Qsg = np.zeros((A, B))     # Spin-glass order parameter q_sg
F1 = np.zeros((A, B))      # Free energy for the magnetized solution
F2 = np.zeros((A, B))      # Free energy for the spin-glass (zero magnetization) solution

# Loop over all alpha values
for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)

    # Load data for given alpha from file
    filename = 'data/Fig4_/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) + '.npz'
    data = np.load(filename) 
    m = data['m']
    q = data['q'] 
    qsg = data['qsg']

    # Store in corresponding arrays
    M[ia, :] = m
    Q[ia, :] = q
    Qsg[ia, :] = qsg

    # Compute or load free energies F1 (magnetized) and F2 (spin-glass) for this alpha
    f1, f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas, betas, 0, B, overwrite=overwrite)
    F1[ia, :] = f1
    F2[ia, :] = f2

# Loop over all beta values (except the last one)
for ib, beta in enumerate(betas[:-1]):
    beta = round(beta, 6)

    # Extract m, q, q_sg, f1, f2 profiles across all alphas for current beta
    m = M[:, ib]
    q = Q[:, ib]
    qsg = Qsg[:, ib]
    f1 = F1[:, ib]
    f2 = F2[:, ib]

    # Analyze beta > 1 region (non-trivial phase possible)
    if beta > 1:
        # Find maximum alpha where magnetization is non-zero
        inds = np.where(m > 0.001)[0]
        if len(inds):
            aM[ib] = np.max(alphas[inds])

        # Find minimum alpha where magnetized state is energetically preferred
        inds = np.where((f1 - f2) > 1E-5)[0]
        if len(inds):
            ac[ib] = np.min(alphas[inds])

# Manually set last value of critical alpha from deterministic model
ac[-1] = a0c

###    RUN OVER ALPHA VALUES    ###

for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)  # Round alpha for consistent filename
    filename = 'data/Fig4_/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) + '.npz'
    data = np.load(filename) 
    m = data['m']
    q = data['q'] 
    qsg = data['qsg']
    C = data['C']
    
    # Compute theoretical lower bound for q_sg using stability threshold
    q1 = np.zeros(B)
    q1[1/betas <= (1 + np.sqrt(alpha))] = (1 - (1/betas) / (1 + np.sqrt(alpha)))[1/betas <= (1 + np.sqrt(alpha))]
    if ia == 1:
        qsg = q1.copy()
    qsg[qsg < q1] = q1[qsg < q1]  # Enforce minimum q_sg
    betas = data['betas']
    
    # Compute modified inverse temperature arrays under nonlinear reparameterization
    betas0 = betas * (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))
    betas0[-1] = np.inf  # Avoid overflow in the last element
    betas02 = betas * (1 + gamma2 / 2 * J * m**2 + gamma2 / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))
    betas02[-1] = np.inf

    # Compute Almeida-Thouless line (spin glass instability condition)
    inds = (C * alpha * betas**2 / (1 - betas * (1 - q))**2 < 1).nonzero()
    if len(inds):
        bAT[ia] = np.max(betas[inds])
        bAT1[ia] = np.max(betas0[inds])
        bAT2[ia] = np.max(betas02[inds])
            
    # Find beta value corresponding to magnetization onset
    inds = np.where(m > 0.001)[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bM[ia] = betas[inds][i]
        bM1[ia] = betas0[inds][i]
        bM2[ia] = betas02[inds][i]
        mM[ia] = m[inds][i]
        qM[ia] = q[inds][i]
        
        # Detect discontinuity in magnetization (first-order transition)
        s = np.zeros(B)
        s[inds] = 1   
        ds = np.diff(s)
        inds1 = (ds < 0)
        inds2 = (ds > 0)
        if np.sum(inds2) > 1:
            bM__[ia] = betas[:-1][inds1][0]
            bM1__[ia] = betas0[:-1][inds1][0]
            bM2__[ia] = betas02[:-1][inds1][0]

        if alpha > 0 and alpha < a0c * 3:
            # Compute free energies to detect phase transitions
            f1, f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas, betas, 0, B, overwrite=False)
            inds = np.where(f1 - f2 > 1E-4)[0]
            s = np.zeros(B)
            s[inds] = 1   
            ds = np.diff(s)
            inds1 = (ds < 0)
            inds2 = (ds > 0)
            if len(inds1):
                bc[ia] = (np.min(betas[:-1][inds1]) + np.min(betas[1:][inds1])) / 2

            # Adjusted inverse temperature for q_sg (frozen spin glass state)
            betas0_ = betas * (1 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))   
            betas0_[-1] = np.inf

            # Compute free energy again under modified beta schedule
            f1, f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=overwrite)
            interp_func1 = interp1d(betas0_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated1 = interp_func1(betas0)
            db = np.gradient(betas, betas0)
            inds = np.where((f1 - f2_interpolated1 > 1E-4))[0]
            
            s = np.zeros(B)
            s[inds] = 1   
            ds = np.diff(s)
            inds1 = (ds < 0)
            inds2 = (ds > 0)
            
            if len(inds1):
                bc1[ia] = (np.min(betas0[:-1][inds1]) + np.min(betas0[1:][inds1])) / 2
                bc1[ia] = max(bc1[ia], bM1[ia])
            elif alpha < a0c:
                bc1[ia] = bM1[ia]
            else:
                bc1[ia] = np.nan

            if np.sum(inds2) > 1:
                bc1_[ia] = betas0[:-1][np.where(inds2)[0][1]]

            # Repeat same analysis for gamma2
            betas02_ = betas * (1 + gamma2 / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))   
            betas02_[-1] = np.inf
            f1, f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas02, betas02_, gamma2, B, overwrite=overwrite)

            mask_keep = betas02 >= bM2[ia]
            betas02_keep = betas02[mask_keep]
            m_keep = m[mask_keep]
            q_keep = q[mask_keep]
            f1_keep = f1[mask_keep]

            mask_add = betas02_ < bM2[ia]
            betas02_add = betas02_[mask_add]
            m_add = np.zeros_like(betas02_add)
            q_add = qsg[mask_add]
            f2_add = f2[mask_add]

            betas02a = np.concatenate([betas02_add, betas02_keep])
            ma = np.concatenate([m_add, m_keep])
            qa = np.concatenate([q_add, q_keep])
            f1a = np.concatenate([f2_add, f1_keep])

            interp_func1 = interp1d(betas02_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated1 = interp_func1(betas02a)
            inds = (f1a - f2_interpolated1 > 1E-4).nonzero()[0]

            s = np.zeros(len(f1a))
            s[inds] = 1   
            ds = np.diff(s)
            inds1 = (ds < 0)
            inds2 = (ds > 0)
            
            if len(betas02a[1:][inds1]):
                bc2[ia] = np.min(betas02a[1:][inds1])
            else:
                bc2[ia] = np.nan

            if np.sum(inds2) > 2:
                bc2_[ia] = betas02a[:-1][np.where(inds2)[0][1]]

        elif alpha == 0:
            bc[ia] = 1
            bc1[ia] = 1
            bc2[ia] = 1

    # Store transition beta where m > 0
    inds = (m > 0.001).nonzero()[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bE[ia] = betas[inds][i]
        bE1[ia] = betas0[inds][i]
        mE[ia] = m[inds][i]
        qE[ia] = q[inds][i]

    # Store transition beta where q > 0 (non-zero correlation)
    inds = np.where(q > 0.001)[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bg[ia] = betas[inds][i]
        bg1[ia] = betas0[inds][i]
        bg2[ia] = betas02[inds][i]

# Find index of alpha closest to critical value a0c
ind0c = np.argmin((alphas - a0c)**2)

# Replace infinite values in bc1 and bc1_ with NaN
bc1[np.isinf(bc1)] = np.nan   
bc1_[np.isinf(bc1_)] = np.nan

# Invert bc1_, but avoid division by zero (replace resulting inf/nan with 0)
inv_bc1_ = 1 / bc1_
inv_bc1_[np.isnan(bc1_)] = 0

# Find index of maximum inverse value (smallest bc1_)
ind = np.argmax(inv_bc1_)

print('ind', ind, alphas[ind], 1 / bc1_[ind])

# Average bc1 and bc1_ at this index and set the next point to the average
bc1m = (bc1[ind] + bc1_[ind]) / 2
bc1_[ind + 1] = bc1m
bc1[ind + 1] = bc1m

# Invalidate data points after the smoothing index
bc1_[ind + 2:] = np.nan
bc1[ind + 2:] = np.nan

# Remove too-small values (thresholded) for cleaner plotting
bc1_[1 / bc1_ < 1E-5] = np.nan
bc1[1 / bc1 < 1E-3] = np.nan

# Fix invalid starting value in bc2
bc2[0] = np.nan
inv_bc2 = 1 / bc2
inv_bc2[np.isnan(bc2)] = 0

# Find index of max valid inverse value in bc2
ind = np.argmax(inv_bc2)
bc2[ind - 1] = bM2[ind - 1]
bc2[:ind - 1] = np.nan  # Invalidate points before that

print(alphas[ind0c - 1])
print(alphas[ind0c])

# ================================
#         Plotting Section
# ================================
plt.figure(figsize=(6, 4), layout='constrained')

# Mask low-temperature regime in ac
ac[betas < 1] = np.nan

# Plot critical line ac and trivial limit line
plt.plot(ac, 1 / betas, '--', color=cmap(1.0))
plt.plot(alphas, 1 + np.sqrt(alphas), '--', color=cmap(1.0))

# Axis labels and region labels
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$T$", rotation=0, labelpad=15)
plt.text(0.02, 1.5, r'P', size=25)   # Paramagnetic
plt.text(0.02, .25, r'F', size=25)   # Ferromagnetic
plt.text(0.08, .1, r'M', size=25)    # Memory
plt.text(0.085, 0.8, r'SG', size=25) # Spin Glass

# Plot inverted boundaries for various gamma values
plt.plot(alphas, 1 / bg1, color=cmap(0.66))
plt.plot(alphas, 1 / bg2, color=cmap(0.33))
plt.plot(alphas, 1 / bc1_, color=cmap(0.66))
plt.plot(alphas, 1 / bc2_, color=cmap(0.33))

# Remove invalid points in bc2 below AT line
bc2[1 / bc2 < 1 / bAT2] = np.nan
plt.plot(alphas, 1 / bc2, color=cmap(0.33))
plt.plot(alphas, 1 / bc1, color=cmap(0.66))

# Mask unstable regions in AT lines above memory capacity
bAT[1 / bAT > 1 / bM] = np.nan
bAT1[1 / bAT1 > 1 / bM1] = np.nan
bAT2[1 / bAT2 > 1 / bM2] = np.nan

# Plot AT stability boundaries with dashed line styles
plt.plot(alphas, 1 / bAT, linestyle=(0, (1, 3)), color=cmap(1.))
plt.plot(alphas, 1 / bAT1, linestyle=(1, (1, 3)), color=cmap(0.66))
plt.plot(alphas, 1 / bAT2, linestyle=(2, (1, 3)), color=cmap(0.33))

# Truncate memory capacity curves at low-temperature collapse
ind = np.where(1 / bM < 1E-2)[0][0]
bM[ind + 1:] = np.nan
bM1[ind + 1:] = np.nan
bM2[ind + 1:] = np.nan

# Plot memory capacity curves for gamma=0, gamma2, gamma
plt.plot(alphas, 1 / bM, '--', color=cmap(1.), label=r"$\gamma'=0$")
plt.plot(alphas, 1 / bM2, color=cmap(0.33), label=fr"$\gamma'={gamma2}$")
plt.plot(alphas, 1 / bM1, color=cmap(0.66), label=fr"$\gamma'={gamma}$")

# Set plot limits and legend
plt.axis([0, 0.2, 0, 1 / bg1[-1]])
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$T$", rotation=0, labelpad=15)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.43), labelspacing=0.2)

# Save figure
plt.savefig('img/phase-diagram-memory-capacity.pdf', bbox_inches='tight')


# Number of gamma values to test
GN = 101

# Define range of gamma values from -1 to 1
gammas = -np.linspace(-1., 1.0, GN)

# Arrays to store transition points as a function of gamma
aM = np.zeros(GN)   # Memory onset transition
ac = np.zeros(GN)   # Critical transition (free energy difference)

# Loop over all gamma values
for ig, gamma in enumerate(gammas):
    gamma = round(gamma, 6)

    # Adjust beta values according to gamma using synaptic scaling expression
    bM_ = bM * (1 + gamma / 2 * J * mM**2 + gamma / 2 * J * alpha * ((1 - bM * J * (1 - qM**2)) / (1 - bM * J * (1 - qM))**2 - 1))
    bc_ = bc * (1 + gamma / 2 * J * mc**2 + gamma / 2 * J * alpha * ((1 - bc * J * (1 - qc**2)) / (1 - bc * J * (1 - qc))**2 - 1))

    # Find alpha for which beta crosses 2 in the transformed curve
    ind = np.argmin((bM_[:np.argmax(bM_)] - 2)**2)
    aM[ig] = alphas[ind]

    # Reset bc_ array to zero for current gamma
    bc_ = np.zeros(A)

    # Loop over all alpha values
    for ia, alpha in enumerate(alphas):
        alpha = round(alpha, 6)

        # Load precomputed fixed point values for given alpha
        filename = f'data/Fig4_/alpha={alpha}_A_{A}_B_{B}_N_{Nint}.npz'
        data = np.load(filename)
        m = data['m']
        q = data['q']
        qsg = data['qsg']
        betas = data['betas']

        # Adjust inverse temperature arrays for mean field corrections
        betas0 = betas * (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))
        betas0_ = betas * (1 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))

        # Consider only cases with non-zero magnetization
        inds = np.where(m > 1E-5)[0]
        if len(inds) and alpha > 0:
            # Compute or retrieve free energies for both m > 0 and m = 0 (spin-glass)
            f1, f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=overwrite)

            # For positive gamma, adjust m=0 spin-glass solution range if needed
            if gamma > 0:
                # Keep only part of f1 corresponding to betas0 >= threshold
                mask_keep = betas0 >= bM2[ia]
                betas0_keep = betas0[mask_keep]
                m_keep = m[mask_keep]
                q_keep = q[mask_keep]
                f1_keep = f1[mask_keep]

                # Add part of f2 where betas0_ < threshold (spin-glass)
                mask_add = betas0_ < bM2[ia]
                betas0_add = betas0_[mask_add]
                m_add = np.zeros_like(betas0_add)
                q_add = qsg[mask_add]
                f2_add = f2[mask_add]

                # Combine both segments into f1 and associated quantities
                betas0 = np.concatenate([betas0_add, betas0_keep])
                m = np.concatenate([m_add, m_keep])
                q = np.concatenate([q_add, q_keep])
                f1 = np.concatenate([f2_add, f1_keep])

            # Interpolate both free energies at beta = 2.0
            interp_func1 = interp1d(betas0, f1, bounds_error=False, fill_value="extrapolate")
            f1_interpolated = interp_func1(2.0)
            interp_func2 = interp1d(betas0_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated = interp_func2(2.0)

            # If free energy of m > 0 is larger than spin-glass solution, store transition
            if f1_interpolated - f2_interpolated > 1E-5 or alpha >= aM[ig]:
                ac[ig] = alphas[ia - 1]
                print(gamma, alphas[ia - 1])
                break

# ======================
# Final Plotting
# ======================

plt.figure(figsize=(6, 4))

# Plot memory onset and free energy critical transition as function of gamma
plt.plot(aM, gammas, 'k')   # Memory capacity transition
plt.plot(ac, gammas, 'k')   # Free energy transition

# Set axis limits and labels
plt.axis([0, np.max(aM) * 1.0, np.min(gammas), np.max(gammas)])
plt.ylim(max(gammas), min(gammas))
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=15)

# Add region labels to the phase diagram
plt.text(0.012, -.5, r'F', size=25)
plt.text(0.05, -.25, r'M', size=25)
plt.text(0.085, .0, r'SG', size=25)

# Save figure to PDF
plt.savefig('img/memory-capacity.pdf', bbox_inches='tight')

# Show plot
plt.show()

