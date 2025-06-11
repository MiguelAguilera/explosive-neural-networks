"""
GPLv3 2025 Miguel Aguilera
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import numba
from numba import njit, prange

# Configure LaTeX-style plotting
plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Set the number of parallel threads
numba.set_num_threads(4)

# ======================
#  Data Processing Functions
# ======================

def unpickle(file):
    """Load CIFAR-10 dataset from a file using pickle."""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def image(data, i):
    """
    Extracts a single image from the CIFAR-10 dataset.
    Returns a 32x32 RGB image normalized between 0 and 1.
    """
    im = np.zeros((32, 32, 3))
    im[:, :, 0] = np.reshape(data[b'data'][i, :1024], (32, 32))
    im[:, :, 2] = np.reshape(data[b'data'][i, 2048:], (32, 32))
    return im / 256


def bw_im(im):
    """Converts an RGB image to grayscale by averaging the channels."""
    return np.mean(im, axis=2)


# ======================
#  Simulation Parameters
# ======================

# Memory sizes to test (number of stored patterns)
Ms = np.arange(1, 15) * 5  
nM = len(Ms)  # Number of different memory sizes

# Gamma values for synaptic scaling
G = 41
gammas = -np.linspace(-1, 1, G)

# Temperature (Inverse of synaptic noise)
beta = 2

# Number of repetitions for statistical robustness
R = 500

# Image and neural network size
L = 32  # CIFAR images are 32x32
N = L * L * 3  # Number of neurons (for RGB images)

# ======================
#  Glauber Algorithm Functions
# ======================

@njit
def exp_g(x, gamma1):
    """Computes an exponential function with synaptic scaling correction."""
    return np.exp((1 / gamma1) * np.log(1 + gamma1 / 1 * x))


@njit
def update_beta1(xi, s, gamma, beta):
    """Dynamically updates the inverse temperature (beta) based on overlap."""
    size = len(s)
    overlap = np.dot(xi, s) / size
    return beta / (1 + gamma * (np.sum(overlap ** 2) / 2))


@njit
def SequentialGlauberStep(s, H, J, beta, gamma, xi, T):
    """
    Performs T steps of the Glauber algorithm, updating spins sequentially.
    
    Parameters:
    - s: Spin configuration
    - H: External field (zero in this case)
    - J: Interaction matrix (Hebbian)
    - beta: Inverse temperature
    - gamma: Synaptic scaling
    - xi: Stored patterns
    - T: Number of Glauber updates
    """
    N = len(s)
    for _ in range(N * T):
        i = np.random.randint(N)  # Select a random spin
        h = H[i] + np.dot(J[i, :], s)  # Local field
        if gamma == 0:
            # Standard Glauber dynamics
            s[i] = int(np.random.rand() * 2 - 1 < np.tanh(beta * h)) * 2 - 1
        else:
            # Glauber dynamics with synaptic scaling
            beta1 = update_beta1(xi, s, gamma, beta)
            tanh_g = (1 - exp_g(-2 * beta1 * h, gamma / N)) / (1 + exp_g(-2 * beta1 * h, gamma / N))
            s[i] = int(np.random.rand() * 2 - 1 < tanh_g) * 2 - 1
    return s


@njit
def calculate_overlap(gamma, beta, M, xi0, inds_cifar):
    """
    Computes the memory retrieval overlap for a given gamma, beta, and memory size M.
    
    Parameters:
    - gamma: Synaptic scaling
    - beta: Inverse temperature
    - M: Number of stored patterns
    - xi0: Initial set of CIFAR patterns
    - inds_cifar: CIFAR indices for pattern selection

    Returns:
    - Overlap value with the first stored pattern.
    """
    M0 = xi0.shape[0]
    inds = np.arange(M0)
    np.random.shuffle(inds)

    # Select M patterns randomly
    inds_cifar = inds_cifar[inds[:M]]
    xi = xi0[inds[:M], :]

    # Compute Hebbian interaction matrix
    J = np.dot(xi.T, xi) / N
    np.fill_diagonal(J, 0.)

    # Initialize state with the first pattern
    s = xi[0, :].copy()
    T = N // 100  # Number of updates per step
    s = SequentialGlauberStep(s, np.zeros(N), J, beta, gamma, xi, T)

    # Compute overlap
    return np.dot(xi[0, :], s) / N


# ======================
#  Simulation Execution
# ======================

# Load CIFAR patterns
data = np.load('data/cifar-patterns-200.npz')
xi0 = data['xi']
inds_cifar0 = data['inds'].astype(int)

@njit(parallel=True)
def calculate():
    # Initialize result arrays
    ms = np.zeros((G, nM, R))  # Individual trial overlaps
    m = np.zeros((G, nM))  # Mean overlaps
    m2 = np.zeros((G, nM))  # Squared overlaps for variance

    # Loop over memory sizes and gamma values
    for im in prange(nM):
        M=Ms[im]
        for ig in range(G):
            gamma = np.round(gammas[ig], 6)
            for r in range(R):
                m_ = calculate_overlap(gamma, beta, M, xi0.copy(), inds_cifar0.copy())
                m[ig, im] += m_ / R
                m2[ig, im] += m_ ** 2 / R
                ms[ig, im, r] += m_

            print("Gamma", gamma, "Memory Fraction", M/N, "Mean Overlap:",m[ig, im])
    return ms, m, m2
    
ms, m, m2 = calculate()
# Save results
filename = f'data/Fig5.npz'
np.savez(filename, m=m, m2=m2, ms=ms)

# ======================
#  Visualization
# ======================

plt.figure()
plt.imshow(m, extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]], aspect='auto', interpolation="none", origin="lower")
plt.colorbar(label="Mean Overlap")

plt.figure()
plt.imshow(m2 - m ** 2, extent=[Ms[0] / N, Ms[-1] / N, gammas[0], gammas[-1]], aspect='auto', interpolation="none", origin="lower")
plt.colorbar(label="Overlap Variance")

plt.show()

