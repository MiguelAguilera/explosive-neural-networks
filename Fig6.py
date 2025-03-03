import numpy as np
from matplotlib import pyplot as plt

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define gamma parameter
gamma = -1.2

# Load data from the stored file
filename = 'data/Fig6.npz'
data = np.load(filename)
betas = data['betas']
q = data['q']

# Compute the modified beta values using the gamma correction factor
beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))

# Define gamma range for analysis
G = 1000
gammas = -np.linspace(1.0, 1.25, G + 1)[1:]  # Exclude first value to avoid singularities

# Initialize array to store critical beta values
b = np.zeros(G)

# Iterate over different gamma values to compute stability transition points
for i, gamma in enumerate(gammas[1:]):  
    beta0 = betas * (1 + gamma * (betas * (1 - q**2) / 2))
    ind = np.where(np.gradient(beta0, betas) < 0)[0][-1]  # Find last decreasing index
    b[i] = beta0[ind]  # Store corresponding beta value

# Generate phase boundary curve
N = 1000
gamma = np.linspace(0, -2, N)  # Define gamma values
beta = gamma / 2 + 1  # Compute beta from gamma

# Create the phase diagram
plt.figure(figsize=(6, 4))
plt.plot(b, gammas, 'k')  # Plot computed transition line
plt.plot(beta, gamma, 'k')  # Plot analytical boundary

# Set axis limits
plt.axis([0, 0.8, -0.9, -1.25])

# Label different phase regions
plt.text(0.2, -1.05, r'P', size=25)    # Paramagnetic phase
plt.text(0.6, -1.05, r'SG', size=25)   # Spin-glass phase
plt.text(0.25, -1.21, r'Exp', size=25) # Expanding region

# Label axes
plt.xlabel(r'$\beta$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=16)

# Save the phase diagram as a PDF file
plt.savefig('img/phase-diagram-SK.pdf', bbox_inches='tight')

# Display the figure
plt.show()

