"""
GPLv3 2025 Miguel Aguilera
"""

import os
import urllib.request
import tarfile
import numpy as np
from matplotlib import pyplot as plt
import pickle
from numba import njit

# URL of the CIFAR-100 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
filename = url.split("/")[-1]
dataset_dir = "cifar-100-python"

# Download the dataset
if not os.path.exists(dataset_dir):
    print("Downloading CIFAR-100 dataset...")
    urllib.request.urlretrieve(url, filename)
    
    # Extract the tarball
    print("Extracting the dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    
    # Delete the tarball
    print("Deleting the tarball...")
    os.remove(filename)
else:
    print("CIFAR-100 dataset already exists. Skipping download.")

# Function to unpickle the dataset files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to convert data to image format
def image(data, i):
    im = np.zeros((32, 32, 3))
    im[:, :, 0] = np.reshape(data_array[i, :1024], (32, 32))
    im[:, :, 1] = np.reshape(data_array[i, 1024:2048], (32, 32))
    im[:, :, 2] = np.reshape(data_array[i, 2048:], (32, 32))
    return im / 256

# Function to binarize the image
def binarize(im):
    imd = np.zeros_like(im)
    for i in range(3):
        imd[:, :, i] = (im[:, :, i] > np.median(im[:, :, i])).astype(int) * 2 - 1
    return imd

# Load the training data
train_file = os.path.join(dataset_dir, 'train')
data = unpickle(train_file)
data_array=data[b'data']

# Number of images in the dataset
M0 = len(data[b'filenames'])
print(f"Total images in the dataset: {M0}")

# Randomly select an image index
ind = np.random.randint(M0)
print(f"Processing image index: {ind}")

# Process the selected image
I = image(data, ind)

# Image dimensions
L = 32
N = L * L * 3

# Shuffle and select M images
M = 200  # Number of images to process
inds = np.arange(M0)
np.random.shuffle(inds)
im_inds = np.zeros(M, int)
xi = np.zeros((M, N))

for a in range(M):
    ind = inds[a]
    im = image(data, ind)
    imd = binarize(im)
    xi[a, :] = imd.flatten()
    im_inds[a] = ind

# Set a reference correlation threshold
REFc = np.sqrt(1 / N) * 10

# Replace images with high correlation
def replace_images(xi):
    cond = True
    r = 0
    while cond:
        cond = False
        count = 0
        A = np.arange(M)
        np.random.shuffle(A)
        for a in A:
            C = np.einsum('i,ai->a', xi[a, :], xi, optimize=True) / N
            C[a]=0
            C_over_threshold = np.sum(np.abs(C)>REFc)

            if C_over_threshold > 0:
                cond = True
                # Try a new image
                ind = np.random.randint(M0)
                im = image(data, ind)
                imd = binarize(im)
                new_x = imd.flatten()

                # Correlation of new image with current set
                C_new = np.einsum('i,ai->a', new_x, xi, optimize=True) / N
                C_new[a]=0
                C_new_over_threshold = np.sum(np.abs(C_new)>REFc)

                # Accept only if new image is better
                if C_new_over_threshold <= C_over_threshold:
                    xi[a, :] = new_x
                    im_inds[a] = ind
                count += 1
        r += 1
        print(f"Iteration {r}, {count} large-correlation images remaining.")
    return xi
    
xi = replace_images(xi)

# Save the processed data
output_file = f'data/cifar-patterns-{M}.npz'
os.makedirs('data', exist_ok=True)
np.savez_compressed(output_file, xi=xi, inds=im_inds)
print(f"Processed data saved to {output_file}")

# Compute and display the covariance matrix
Cov = np.einsum('ai,bi->ab', xi, xi, optimize=True) / N
plt.figure()
plt.imshow(Cov)
plt.colorbar()
plt.title("Covariance Matrix")

# Display histogram of covariance values
plt.figure()
plt.hist(Cov.flatten(), bins=1000)
plt.title("Histogram of Covariance Values")
plt.show()

