import numpy as np
from matplotlib import pyplot as plt

# Configure LaTeX-style plotting for better readability
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define gamma values for analysis
for gamma in [-0.8, -1.0, -1.2]:
    
    # Load data from file
    filename = 'data/Fig6.npz'
    data = np.load(filename)
    betas = data['betas']
    q = data['q']
    
    # Compute the modified beta values with gamma correction
    beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))
    
    # Identify regions where beta0 is increasing
    increasing_inds = np.gradient(beta0, betas) > 0
    q1 = q.copy()
    
    # Compute the derivative dq/d(beta0)
    dq = np.gradient(q, beta0)
    
    # Mask invalid values for visualization
    q[~increasing_inds] = np.nan
    q1[increasing_inds] = np.nan
    dq[~increasing_inds] = np.nan
    dq[q < 1E-7] = np.nan
    dq[dq <= 0] = np.nan
    
    # Identify critical transition point
    imax = np.nanargmax(dq)
    dq[:imax] = np.nan
    transition_inds = np.where(dq > 5)[0]
    if len(transition_inds):
        dq[transition_inds[0] - 1] = np.nan
    
    # Compute beta derivative
    Db0 = np.gradient(beta0, betas)
    
    # Find valid transition points
    valid_inds = np.where(~np.isnan(dq))[0]
    print(valid_inds)
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle(fr"$\gamma' = {gamma}$", y=0.85)
    
    # Plot q vs beta0
    ax[0].plot(beta0, q, 'k')
    ax[0].plot(beta0, q1, 'k:')
    ax[0].set_ylabel(r'$q$', rotation=0, labelpad=16)
    ax[0].set_xlabel(r'$\beta$')
    ax[0].axis([beta0[0], 1.2, 0, 1])
    ax[0].yaxis.set_label_coords(-0.18, 0.45)
    
    # Plot dq/d(beta0) vs beta0
    ax[1].plot(beta0, dq, 'k')
    if len(valid_inds):
        ax[1].plot(beta0[valid_inds[0] - 1:valid_inds[0] + 1], [0, dq[valid_inds[0]]], 'k:')
    ax[1].set_ylabel(r'$\dfrac{dq}{d\beta}$', rotation=0, labelpad=16)
    ax[1].set_xlabel(r'$\beta$')
    ax[1].axis([beta0[0], 1.2, 0, 8])
    ax[1].yaxis.set_label_coords(-0.2, 0.3)
    
    # Adjust layout and save figure
    plt.subplots_adjust(wspace=0.2, hspace=0.02)
    fig.tight_layout(h_pad=0.0, w_pad=0.7, rect=[0, 0, 1, 0.975])
    plt.savefig(f'img/Fig6_gamma_{gamma}.pdf', bbox_inches='tight')

# Reset plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', **{'size': 16})
plt.rc('legend', **{'fontsize': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Load data again for phase diagram
filename = 'data/Fig6.npz'
data = np.load(filename)
betas = data['betas']
q = data['q']

# Define gamma range for analysis
G = 1000
gammas = -np.linspace(1.0, 1.25, G + 1)[1:]  # Exclude first value to avoid singularities

# Initialize array to store critical beta values
b = np.zeros(G)

# Compute stability transition points for different gamma values
for i, gamma in enumerate(gammas[1:]):  
    beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))
    ind = np.where(np.gradient(beta0, betas) < 0)[0][-1]  # Find last decreasing index
    b[i] = beta0[ind]  # Store corresponding beta value

# Generate analytical phase boundary curve
N = 1000
gamma = np.linspace(0, -2, N)  # Define gamma values
beta = gamma / 2 + 1  # Compute beta from gamma

# Create phase diagram
plt.figure(figsize=(6, 4))
plt.plot(b, gammas, 'k', label="Computed transition line")  # Computed transition line
plt.plot(beta, gamma, 'k', label="Analytical boundary")  # Analytical boundary

# Set axis limits
plt.axis([0, 0.8, -0.9, -1.25])

# Label different phase regions
plt.text(0.2, -1.05, r'P', size=25)    # Paramagnetic phase
plt.text(0.6, -1.05, r'SG', size=25)   # Spin-glass phase
plt.text(0.25, -1.21, r'Exp', size=25) # Expanding region

# Label axes
plt.xlabel(r'$\beta$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=16)

# Save and display phase diagram
plt.savefig(f'img/Fig6_d.pdf', bbox_inches='tight')
plt.show()

