import numpy as np
from matplotlib import pyplot as plt

# --- Configure LaTeX-style plotting for consistent formatting ---
plt.rc('text', usetex=True)
plt.rc('font', size=20, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=18)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# === PART 1: GAMMA-DEPENDENT ANALYSIS AND PLOTTING ===
for gamma in [-0.8, -1.0, -1.2]:
    # Load original data
    data = np.load('data/Fig6.npz')
    betas = data['betas']
    q = data['q']

    # Compute transformed inverse temperature with gamma correction
    beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))

    # Identify regions where beta0 is increasing (i.e., stability criterion)
    increasing_inds = np.gradient(beta0, betas) > 0

    # Prepare q and derivative arrays
    q1 = q.copy()  # Backup original q
    dq = np.gradient(q, beta0)  # Derivative of q w.r.t. beta0

    # Apply masks to highlight stable regions
    q[~increasing_inds] = np.nan      # Unstable: mask q
    q1[increasing_inds] = np.nan      # Stable: mask in backup for dotted line
    dq[~increasing_inds] = np.nan     # Mask unstable dq
    dq[q < 1E-7] = np.nan             # Mask near-zero overlap
    dq[dq <= 0] = np.nan              # Only consider increasing regions

    # Identify transition point: sharpest increase in q
    imax = np.nanargmax(dq)
    dq[:imax] = np.nan  # Mask values before maximum
    transition_inds = np.where(dq > 5)[0]
    if len(transition_inds):
        dq[transition_inds[0] - 1] = np.nan  # Smooth visualization

    # Compute derivative of beta0 for diagnostics
    Db0 = np.gradient(beta0, betas)

    # Get valid transition index for vertical line
    valid_inds = np.where(~np.isnan(dq))[0]
    print(valid_inds)

    # === Plot results for current gamma ===
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle(fr"$\gamma' = {gamma}$", y=0.85)

    # Plot q vs beta0
    ax[0].plot(beta0, q, 'k')
    ax[0].plot(beta0, q1, 'k:')
    ax[0].set_ylabel(r'$q$', rotation=0, labelpad=16)
    ax[0].set_xlabel(r'$\beta$')
    ax[0].axis([beta0[0], 1.2, 0, 1])
    ax[0].yaxis.set_label_coords(-0.18, 0.45)

    # Plot dq/dbeta0 vs beta0
    ax[1].plot(beta0, dq, 'k')
    if len(valid_inds):
        ax[1].plot(beta0[valid_inds[0] - 1:valid_inds[0] + 1], [0, dq[valid_inds[0]]], 'k:')
    ax[1].set_ylabel(r'$\dfrac{dq}{d\beta}$', rotation=0, labelpad=16)
    ax[1].set_xlabel(r'$\beta$')
    ax[1].axis([beta0[0], 1.2, 0, 8])
    ax[1].yaxis.set_label_coords(-0.2, 0.3)

    # Save figure and adjust layout
    plt.subplots_adjust(wspace=0.2)
    fig.tight_layout(h_pad=0.0, w_pad=0.7, rect=[0, 0, 1, 0.975])
    plt.savefig(f'img/Fig6_gamma_{gamma}.pdf', bbox_inches='tight')

# === PART 2: PHASE DIAGRAM ANALYSIS ===

# Reset plotting configuration for general use
plt.rc('text', usetex=True)
plt.rc('font', size=16)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Reload base data
data = np.load('data/Fig6.npz')
betas = data['betas']
q = data['q']

# Define range of gamma values (negative only)
G = 1000
gammas = -np.linspace(1.0, 1.25, G + 1)[1:]  # Avoid exact -1.0 to prevent numerical issues
b = np.zeros(G)  # Store critical beta values

# Find beta0 at last decreasing point (end of stability)
for i, gamma in enumerate(gammas[1:]):
    beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))
    ind = np.where(np.gradient(beta0, betas) < 0)[0][-1]
    b[i] = beta0[ind]

# Analytical prediction of boundary: beta = gamma/2 + 1
N = 1000
gamma = np.linspace(0, -2, N)
beta = gamma / 2 + 1

# === Plot phase diagram ===
plt.figure(figsize=(6, 4))
plt.plot(b, gammas, 'k', label="Computed transition line")
plt.plot(beta, gamma, 'k', label="Analytical boundary")

# Set plot limits and annotate phase regions
plt.axis([0, 0.8, -0.9, -1.25])
plt.text(0.2, -1.05, r'P', size=25)     # Paramagnetic
plt.text(0.6, -1.05, r'SG', size=25)    # Spin glass
plt.text(0.25, -1.21, r'Exp', size=25)  # Expansion region

# Axis labels
plt.xlabel(r'$\beta$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=16)

# Save and show phase diagram
plt.savefig(f'img/Fig6_d.pdf', bbox_inches='tight')
plt.show()

