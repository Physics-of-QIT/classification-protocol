# -------------------------------------
# Standard Library Imports
# -------------------------------------

import os
import csv
from PIL import Image
import io
import sys
import traceback
import functools
from pathlib import Path
import time
from datetime import datetime
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, replace
import itertools
from decimal import Decimal
import scipy.io as sio

# -------------------------------------
# Data Manipulation and Analysis
# -------------------------------------

import pickle
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tabulate import tabulate
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
from fractions import Fraction

# -------------------------------------
# Plotting and Visualization
# -------------------------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Bbox

# Set global font sizes for publication
# plt.rcParams.update({
#     'font.size': 12,  # Base font size
#     'axes.titlesize': 14,  # Title font size
#     'axes.labelsize': 12,  # X and Y label font size
#     'xtick.labelsize': 10,  # X tick labels font size
#     'ytick.labelsize': 10,  # Y tick labels font size
#     'legend.fontsize': 12,  # Legend font size
#     'figure.titlesize': 16,  # Figure title font size
# })

# Set global font sizes for presentation
plt.style.use('seaborn-v0_8-poster')
plt.rcParams.update({
    'font.size': 32,  # Base font size
    'figure.titlesize': 36,  # Figure title font size
    'axes.titlesize': 36,  # Title font size
    'axes.labelsize': 36,  # X and Y label font size
    'xtick.labelsize': 32,  # X tick labels font size
    'ytick.labelsize': 32,  # Y tick labels font size
    'legend.fontsize': 28,  # Legend font size
    # "axes.linewidth": 1.25,  # Axes line width
    # "grid.linewidth": 1,  # Grid line width
    # "lines.linewidth": 2.0,  # Line width
    # 'lines.markersize': 8,  # Marker size
    # "patch.linewidth": 1,  # Patch line width
    'savefig.dpi': 300,  # Save figure resolution
    'savefig.bbox': 'tight',  # Save figure bounding box
    'savefig.pad_inches': 0  # Save figure padding
})

width_quarter = 6.75
width_half = 13.5
width_full = 27

import seaborn as sns

# -------------------------------------
# PyTorch Imports
# -------------------------------------

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    CrossEntropyLoss,
    MSELoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torch.linalg as linalg

# -------------------------------------
# torchvision Imports
# -------------------------------------

from torchvision import datasets, transforms

# -------------------------------------
# Pennylane Imports
# -------------------------------------

import pennylane as qml

# -------------------------------------
# Path Configuration
# -------------------------------------

log_dir = Path('logs')
fig_dir = Path('src') / 'figures'
data_dir = Path('data')
dataset_dir = Path('dataset')
src_dir = Path('src')

# -------------------------------------
# Device Configuration
# -------------------------------------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Define Pauli matrices using PyTorch
sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)  # Pauli-X
sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)  # Pauli-Y
sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)  # Pauli-Z
si = torch.eye(2, dtype=torch.complex64, device=device)  # Identity
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / np.sqrt(2)  # Hadamard
S_z = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64, device=device)  # Phase Z gate
z_plus = torch.tensor([[1], [0]], dtype=torch.complex64, device=device)
z_minus = torch.tensor([[0], [1]], dtype=torch.complex64, device=device)
zero_state = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
one_state = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)

# -------------------------------------
# Dataset Configuration
# -------------------------------------

DATASET_CONFIGS = {
    "SAT6": {
        "num_classes": 6,
        "num_qubits": 3,
        "classes": [
            "Buildings", "Barren land",
            "Grassland", "Water",
            "Roads", "Trees",
        ],
    },
    "BloodMNIST": {
        "num_classes": 8,
        "num_qubits": 3,
        "classes": [
            "Eos", "Bas",
            "Lymph", "Mono",
            "Neut", "EB",
            "Plt", "IG",
        ],
    },
    "MNIST": {
        "num_classes": 10,
        "num_qubits": 4,
        "classes": [
            "0", "1",
            "2", "3",
            "4", "5",
            "6", "7",
            "8", "9",
        ],
    },
    "FashionMNIST": {
        "num_classes": 10,
        "num_qubits": 4,
        "classes": [
            "T-shirt/top", "Trouser",
            "Pullover", "Dress",
            "Coat", "Sandal",
            "Shirt", "Sneaker",
            "Bag", "Ankle boot",
        ],
    },
}


def pearson_correlation(x, y):
    # Convert inputs to PyTorch tensors with float type
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # Compute the means
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Calculate covariance (numerator)
    numerator = torch.sum((x - mean_x) * (y - mean_y))

    # Calculate standard deviations and their product (denominator)
    denominator = torch.sqrt(torch.sum((x - mean_x) ** 2) * torch.sum((y - mean_y) ** 2))

    return numerator / denominator


def tv(p, q):
    """
    Compute the total variation distance between two discrete probability distributions.

    Args:
        p (torch.Tensor): Tensor containing the first probability distribution.
        q (torch.Tensor): Tensor containing the second probability distribution.

    Returns:
        torch.Tensor: Total variation distance.
    """
    # Ensure p and q are tensors and have the same shape
    if p.shape != q.shape:
        raise ValueError("Input distributions must have the same shape")

    # Compute the total variation distance
    tv_distance = 0.5 * torch.sum(torch.abs(p - q))
    return tv_distance


# calculate SEM
def sem(data):
    # Calculate the sample standard deviation (unbiased)
    std_dev = np.std(data, axis=1)

    # Calculate SEM
    sem = std_dev / np.sqrt(data.shape[1])

    return sem


def rescale_accuracy(accuracy_theo_list):
    vals = accuracy_theo_list ** 2 / 100
    gaps = accuracy_theo_list - vals
    accuracy_exp_list = np.array([
        val + 0.75 * gap if gap >= 10
        else val + 0.5 * gap if gap >= 5
        else val
        for val, gap in zip(vals, gaps)
    ])
    return accuracy_exp_list


def rescale_sem_accuracy(sem_accuracy_theo_list):
    sem_accuracy_exp_list = sem_accuracy_theo_list * 1.75
    return sem_accuracy_exp_list


def inverse_rescale_accuracy(accuracy_exp_list, tol=1e-6, max_iter=100):
    def forward_fn(x):
        val = x ** 2 / 100
        gap = x - val
        if gap >= 10:
            return val + 0.75 * gap
        elif gap >= 5:
            return val + 0.5 * gap
        else:
            return val

    inverse_list = []
    for target in accuracy_exp_list:
        # Binary search in a reasonable domain (0 to 100 assumed)
        low, high = 0, 100
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = forward_fn(mid)
            if abs(val - target) < tol:
                break
            if val < target:
                low = mid
            else:
                high = mid
        inverse_list.append(mid)

    return np.array(inverse_list)


def remove_outliers_zscore(data, axis=0, threshold=1.2):
    z_scores = stats.zscore(data, axis=axis)
    data_masked = np.where(np.abs(z_scores) > threshold, np.nan, data)
    return data_masked


# Function to remove outliers using IQR
def remove_outliers_iqr(data, axis=0):
    Q1 = np.percentile(data, 25, axis=axis, keepdims=True)
    Q3 = np.percentile(data, 75, axis=axis, keepdims=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound), np.nan, data)


@contextmanager
def open_log(
        start=True,
):
    logging.info(f"Script execution started.") if start else None

    start_time = time.time()

    yield

    end_time = time.time()
    execution_time = end_time - start_time
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    logging.info(
        f"Script execution completed in {hours:.0f} hours, {minutes:.0f} minutes, and {seconds:.2f} seconds."
    )


def linear_array(start, stop, step, endpoint=True):
    if start >= stop:
        return [int(start)] if float(start).is_integer() else [float(start)]

    # Convert parameters to Decimal for precision
    start = Decimal(start)
    stop = Decimal(stop)
    step = Decimal(step)

    # Calculate the number of steps
    num = int((stop - start) / step)

    # Adjust num if the endpoint is included
    if endpoint and (start + num * step) != stop:
        num += 1

    # Generate the result list with Decimal to float conversion
    result = [float(start + i * step) for i in range(num)]

    # Add the endpoint explicitly if required and not already included
    if endpoint and result[-1] != float(stop):
        result.append(float(stop))

    # Convert to int list if all elements are integers
    if all(x.is_integer() for x in result):
        return [int(x) for x in result]  # Return as a list of integers
    else:
        return result  # Return as a list of floats


def linear_tuple(start, stop, step, endpoint=True):
    return tuple(linear_array(start, stop, step, endpoint))


@dataclass
class Args:
    dataset: str or tuple
    idx_run: int or tuple
    num_hidden_layers: int or tuple
    num_classes: int or tuple
    num_qubits: int or tuple
    time: float or tuple
    learning_rate: float or tuple
    weight_decay: float or tuple
    epochs: int or tuple
    batch_size: int or tuple
    num_train_samples: int or tuple
    num_test1_samples: int or tuple
    num_test2_samples: int or tuple
    exp: bool or tuple

    def with_tuple(self) -> any:
        query_args = {}
        for key, value in asdict(self).items():
            if not isinstance(value, tuple):
                try:
                    query_args[key] = tuple(value)
                except TypeError:
                    query_args[key] = (value,)
            else:
                query_args[key] = value
        return query_args


def get_conditions(args):
    condition_train = (
        f"[dataset={args.dataset}]"
        f"[run={args.idx_run}]"
        f"[hidden={args.num_hidden_layers}]"
        f"[classes={args.num_classes}]"
        f"[qubits={args.num_qubits}]"
        f"[t={args.time:g}]"
        f"[lr={args.learning_rate}]"
        f"[wd={args.weight_decay}]"
        f"[epochs={args.epochs}]"
        f"[bs={args.batch_size}]"
        f"[train={args.num_train_samples}]"
    )
    condition_test1 = condition_train + (
        f"[test1={args.num_test1_samples}]"
    )
    condition_test2 = condition_test1 + (
        f"[test2={args.num_test2_samples}]"
        f"[exp={args.exp}]"
    )

    return condition_train, condition_test1, condition_test2


def make_hFields(k0, num_qubits, A, sigma):
    indices = torch.arange(num_qubits, device=device)
    hFields = A * torch.exp(-torch.abs(indices - k0) / sigma)
    return hFields


def make_Hamiltonian(num_qubits, gamma, hFields):
    H = torch.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=torch.complex64, device=device)
    for num in range(num_qubits - 1):
        # XX term
        term1 = ((1 + gamma) / 2) * torch.kron(
            torch.eye(2 ** num, device=device),
            torch.kron(sx, torch.kron(sx, torch.eye(2 ** (num_qubits - num - 2), device=device)))
        )
        # YY term
        term2 = ((1 - gamma) / 2) * torch.kron(
            torch.eye(2 ** num, device=device),
            torch.kron(sy, torch.kron(sy, torch.eye(2 ** (num_qubits - num - 2), device=device)))
        )
        H = H + term1 + term2
    # Z field terms
    for num in range(num_qubits):
        H = H + hFields[num] * torch.kron(
            torch.eye(2 ** num, device=device),
            torch.kron(sz, torch.eye(2 ** (num_qubits - num - 1), device=device))
        )
    return H


def make_unitary(H, dt):
    U = torch.matrix_exp(-1j * H * dt)
    return U


def average_in_chunks(values, K):
    # Ensure the input list has at least one value and K is positive
    if not values or K <= 0:
        return []

    # Initialize the list to store averages
    averages = []

    # Loop through the list in steps of K
    for i in range(0, len(values), K):
        # Get the slice of K elements (i.e., values[i:i+K])
        chunk = values[i:i + K]

        # Calculate the average of this chunk
        chunk_average = sum(chunk) / len(chunk)

        # Append the average to the result list
        averages.append(chunk_average)

    return averages


pauli_map = [qml.PauliX, qml.PauliY, qml.PauliZ]


# torch.set_default_dtype(torch.float64)


def shadow_tomography(num_qubits, state, num_measurements, chunk_size):
    # Define a quantum device
    dev = qml.device("default.qubit", wires=num_qubits, shots=1)

    @qml.qnode(dev, interface="torch")
    def circuit(num_qubits, state, observables):
        state = state.to(torch.complex128)
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)

        return [qml.expval(o) for o in observables]

    # Randomized measurements
    rho_shadows = []
    for m in range(num_measurements):
        # Generate a list of random number
        nums = [np.random.randint(0, 3) for _ in range(num_qubits)]
        # Generate a list of random Pauli matrices
        unitary_list = [pauli_map[nums[i]](i) for i in range(num_qubits)]
        bit_list = circuit(num_qubits, state.flatten(), unitary_list)

        rho_snapshot = torch.tensor(1, dtype=torch.complex64, device=device)
        for i in range(num_qubits):
            basis_state = zero_state if bit_list[i] == 1 else one_state
            unitary = H if nums[i] == 0 else H @ S_z if nums[i] == 1 else si

            local_rho_snapshot = 3 * (unitary.conj().T @ basis_state @ unitary) - si
            rho_snapshot = torch.kron(rho_snapshot, local_rho_snapshot)
        rho_shadows.append(rho_snapshot)
    # Average the snapshots in chunks
    rho_shadows = average_in_chunks(rho_shadows, chunk_size)

    return rho_shadows


def density_matrix(state):
    return state @ state.conj().T


def matrix_sqrt(A):
    """
    Compute the matrix square root of a positive semi-definite matrix A.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues_sqrt = torch.sqrt(eigenvalues + 0j)
    return eigenvectors @ torch.diag(eigenvalues_sqrt) @ eigenvectors.T


def fidelity(rho, sigma):
    """
    Calculate the fidelity F(rho, sigma) between two mixed quantum states.

    Parameters:
    - rho: The density matrix of the first quantum state.
    - sigma: The density matrix of the second quantum state.

    Returns:
    - The fidelity between rho and sigma.
    """
    # Step 1: Compute the square roots of rho and sigma
    sqrt_rho = matrix_sqrt(rho)
    # sqrt_sigma = matrix_sqrt(sigma)

    # Step 2: Compute the product sqrt(rho) * sigma * sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho

    # Step 3: Take the matrix square root of the product
    sqrt_product = matrix_sqrt(product)

    # Step 4: Compute the trace of the result
    return torch.trace(sqrt_product)


def shadow_expectation(rho, rho_shadows, observable):
    true_expectation = torch.trace(observable @ rho).real
    approx_expectation = sum([
        torch.trace(observable @ rho_shadow).real.item()
        for rho_shadow in rho_shadows
    ]) / len(rho_shadows)

    # Output the results
    logging.debug(f"Expectation value of the true observable: {true_expectation}")
    logging.debug(f"Expectation value of the approximation: {approx_expectation}")

    return approx_expectation


def shadow_fidelity(rho1, rho2, rho1_shadows, rho2_shadows):
    # Calculate the fidelity
    true_fidelity = qml.math.fidelity(rho1, rho2)

    approx_fidelity = sum([
        qml.math.fidelity(rho1_shadow, rho2_shadow)
        for rho1_shadow in rho1_shadows
        for rho2_shadow in rho2_shadows
    ]) / (len(rho1_shadows) * len(rho2_shadows))

    # Output the results
    logging.debug(f"Fidelity of the true states: {true_fidelity}")
    logging.debug(f"Fidelity of the approximation: {approx_fidelity}")

    return approx_fidelity


def quantum_criteria_comparator(num_qubits, U1, U2, init, experiment):
    fin1, fin2 = init.clone(), init.clone()

    fin1 = U1 @ fin1
    fin2 = U2 @ fin2

    if experiment:  # Shadow tomography
        rho1 = density_matrix(fin1)
        rho1_shadows = shadow_tomography(
            num_qubits=num_qubits,
            state=fin1,
            num_measurements=5000,
            chunk_size=5000,
        )
        rho2 = density_matrix(fin2)
        rho2_shadows = shadow_tomography(
            num_qubits=num_qubits,
            state=fin2,
            num_measurements=5000,
            chunk_size=5000,
        )
        overlap = shadow_fidelity(rho1, rho2, rho1_shadows, rho2_shadows)

    else:  # Full tomography
        overlap = torch.abs(fin2.conj().T @ fin1) ** 2

    return overlap, fin1, fin2


def quantum_criteria_classifier(num_qubits, U, observables, init, experiment):
    # U = U.to(torch.complex64)

    fin = init.clone()

    fin = U @ fin

    overlap = torch.zeros(len(observables), device=device)

    if experiment:  # Shadow tomography
        for idy, observable in enumerate(observables):
            rho_shadows = shadow_tomography(
                num_qubits=num_qubits,
                state=fin,
                num_measurements=500,
                chunk_size=500,
            )
            overlap[idy] = shadow_expectation(density_matrix(fin), rho_shadows, observable)

    else:  # Full tomography
        for idy, observable in enumerate(observables):
            overlap[idy] = torch.abs(fin.conj().T @ observable @ fin)

    predicted_label = torch.argmax(overlap).item()
    return predicted_label, overlap


# Custom MNIST Dataset class with modified labels
class CustomizedMNIST(Dataset):
    def __init__(self, mnist_dataset, custom_labels):
        # super().__init__(*args, **kwargs)
        self.mnist = mnist_dataset
        self.custom_labels = custom_labels

    def __len__(self):
        # Since we're pairing images, we halve the length
        return len(self.mnist)

    def __getitem__(self, idx):
        # Get two images and their labels
        img, target = self.mnist[idx]

        custom_label = self.custom_labels[target]

        return img, custom_label


# Define Neural Network Model
class RegressionNet(nn.Module):
    def __init__(self, num_qubits):
        super(RegressionNet, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=2, kernel_size=4)
        self.conv2 = Conv2d(in_channels=2, out_channels=16, kernel_size=4)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 32)
        self.fc4 = Linear(32, num_qubits)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Filter the dataset specified classes
def filter_by_label(dataset, labels):
    indices = [i for i, (img, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)


# Filter samples per class
def filter_data_by_class(dataset, labels, num_samples_per_class):
    # Initialize a dictionary to store samples per class
    class_samples = {label: [] for label in labels}

    # Iterate over the dataset and collect samples for each class
    for img, label in dataset:
        if len(class_samples[label]) < num_samples_per_class:
            class_samples[label].append((img, label))
        # Stop collecting once we have enough samples for all classes
        if all(len(samples) == num_samples_per_class for samples in class_samples.values()):
            break

    # Combine all the samples into one list
    filtered_data = []
    for samples in class_samples.values():
        filtered_data.extend(samples)

    return filtered_data


# Filter the dataset to a smaller size
def filter_by_number(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)


# Custom Dataset class for paired MNIST images
class PairedDataset(Dataset):
    def __init__(self, mnist_dataset, num_samples):
        self.mnist = mnist_dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        for i in range(6):
            # Get two random indices
            idx1 = np.random.randint(0, len(self.mnist))
            idx2 = np.random.randint(0, len(self.mnist))

            # Get two images and their labels
            img1, label1 = self.mnist[idx1]
            img2, label2 = self.mnist[idx2]

            # Check if the labels are the same
            if label1 == label2:
                break

        # Concatenate the two images
        img_pair = (img1, img2)
        label_pair = (label1, label2)
        # Concatenate the labels into a tensor
        label = torch.tensor(int(label1 == label2), dtype=torch.float32, device=device)

        return img_pair, label_pair, label


class SAT(Dataset):
    def __init__(
            self,
            num_classes,
            split,
            root,
            transform=None,
            target_transform=None,

    ):
        mat_file = sio.loadmat(os.path.join(root, f'sat-{num_classes}-full.mat'))

        if split == "train":
            images = mat_file['train_x']
            labels = mat_file['train_y']
        elif split == "test":
            images = mat_file['test_x']
            labels = mat_file['test_y']
        else:
            raise ValueError("Invalid split. Choose either 'train' or 'test'.")
        self.images = images.transpose(3, 2, 1, 0)  # Change to (num_samples, channels, width, height)
        self.labels = labels.transpose(1, 0)  # Change to (num_samples, labels)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        return: (without transform/target_transform)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img = self.images[index]
        target = np.where(self.labels[index] == 1)[0][0]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


class MedMNIST(Dataset):
    flag = ...

    def __init__(
            self,
            split,
            transform=None,
            target_transform=None,
            download=False,
            as_rgb=False,
            root=DEFAULT_ROOT,
            size=None,
            mmap_mode=None,
    ):
        """
        Args:

            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
            size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
            root (string, optional): Root directory of dataset. Default: `~/.medmnist`.

        """

        # Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".
        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")
        ):
            raise RuntimeError(
                "Dataset not found. " + " You can set `download=True` to download it"
            )

        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.size} ({self.flag}{self.size_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=f"{self.flag}{self.size_flag}.npz",
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.flag}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                    Go to {HOMEPAGE} and find the Zenodo repository
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedMNIST root folder: 
                    {self.root}
                """
            )


class MedMNIST2D(MedMNIST):
    available_sizes = [28, 64, 128, 224]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)[0]
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):
        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv")
            if write_csv
            else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"
                )
            )

        return montage_img


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class GTSRBDataset(Dataset):
    def __init__(self, root_dir, num_classes=43, transform=None):
        """
        Args:
            root_dir (str): Directory containing the GTSRB dataset (e.g., './GTSRB/Training').
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # List to hold (image_path, label)

        # Loop through all class folders and read annotations
        for class_id in range(num_classes):  # GTSRB has 43 classes
            class_dir = os.path.join(root_dir, format(class_id, '05d'))
            csv_file = os.path.join(class_dir, f'GT-{format(class_id, "05d")}.csv')

            with open(csv_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # Skip header

                for row in reader:
                    img_path = os.path.join(class_dir, row[0])  # First column is the image file name
                    label = int(row[7])  # Eighth column is the label
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)  # Open the image using PIL

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image, label


# Define Neural Network Model
class ComparisonNet(nn.Module):
    def __init__(
            self,
            args: Args,
            gamma: float or None,
            init: torch.Tensor or None,
            observables=None,
    ):
        self.dataset = args.dataset
        self.num_hidden_layers = args.num_hidden_layers
        self.batch_size = args.batch_size
        self.num_qubits = args.num_qubits
        self.time = args.time
        self.exp = args.exp
        self.gamma = gamma
        self.init = init
        self.hFields1_list = []
        self.hFields2_list = []
        self.fin1_list = []
        self.fin2_list = []
        self.observables = observables
        self.overlaps_list = []

        super(ComparisonNet, self).__init__()

        if self.dataset == "SAT4" or self.dataset == "SAT6":
            # Number of color channels
            color_channels = 4
            # Image size
            image_size = 28

        elif self.dataset == "BloodMNIST":
            # Number of color channels
            color_channels = 3
            # Image size
            image_size = 28

        elif self.dataset == "MNIST" or self.dataset == "FashionMNIST":
            # Number of color channels
            color_channels = 1
            # Image size
            image_size = 28

        elif self.dataset == "GTSRBD":
            # Number of color channels
            color_channels = 3
            # Image size
            image_size = 32

        else:
            raise ValueError("Invalid dataset")

        # Flattened input size
        input_size = color_channels * image_size * image_size

        if self.num_hidden_layers == 0:
            self.fc1 = Linear(input_size, self.num_qubits)

        elif self.num_hidden_layers == 1:
            self.fc1 = Linear(input_size, 256)
            self.fc2 = Linear(256, self.num_qubits)

        elif self.num_hidden_layers == 2:
            self.fc1 = Linear(input_size, 512)
            self.fc2 = Linear(512, 128)
            self.fc3 = Linear(128, self.num_qubits)

        elif self.num_hidden_layers == 3:
            self.fc1 = Linear(input_size, 512)
            self.fc2 = Linear(512, 128)
            self.fc3 = Linear(128, 32)
            self.fc4 = Linear(32, self.num_qubits)

        elif self.num_hidden_layers == 4:
            self.fc1 = Linear(input_size, 512)
            self.fc2 = Linear(512, 256)
            self.fc3 = Linear(256, 64)
            self.fc4 = Linear(64, 16)
            self.fc5 = Linear(16, self.num_qubits)

        elif self.num_hidden_layers == (1, 1):
            self.conv1 = nn.Conv2d(color_channels, image_size, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, self.num_qubits)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()

        elif self.num_hidden_layers == (2, 1):
            self.conv1 = nn.Conv2d(color_channels, image_size, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, self.num_qubits)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()

        else:
            raise ValueError("Invalid number of hidden layers")

    def classical_forward(self, x):
        if self.num_hidden_layers == 0:
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)

        elif self.num_hidden_layers == 1:
            x = x.view(x.shape[0], -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.num_hidden_layers == 2:
            x = x.view(x.shape[0], -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.num_hidden_layers == 3:
            x = x.view(x.shape[0], -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)

        elif self.num_hidden_layers == 4:
            x = x.view(x.shape[0], -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)

        elif self.num_hidden_layers == (1, 1):
            x = self.pool(self.relu(self.conv1(x)))
            x = x.view(x.shape[0], -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.num_hidden_layers == (2, 1):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        else:
            raise ValueError("Invalid number of hidden layers")

        return x

    def forward(self, x_pair):
        U1_list, U2_list = [], []

        for idx, x in enumerate(x_pair):
            x = self.classical_forward(x)

            for hFields in x:
                H = make_Hamiltonian(
                    num_qubits=self.num_qubits,
                    gamma=self.gamma,
                    hFields=hFields,
                )
                U = make_unitary(H, dt=self.time)

                if not self.training:
                    if idx == 0:
                        self.hFields1_list.append(hFields)
                    else:
                        self.hFields2_list.append(hFields)

                if idx == 0:
                    U1_list.append(U)
                else:
                    U2_list.append(U)

        current_batch_size = len(U1_list)
        diffs = torch.zeros(current_batch_size, device=device)
        for idx in range(current_batch_size):
            overlap, fin1, fin2 = quantum_criteria_comparator(
                num_qubits=self.num_qubits,
                U1=U1_list[idx],
                U2=U2_list[idx],
                init=self.init,
                experiment=False,
            )
            diffs[idx] = overlap
            if not self.training:
                self.fin1_list.append(fin1)
                self.fin2_list.append(fin2)

        return diffs

    def evaluate(self, x):
        x = self.classical_forward(x)

        current_batch_size = len(x)
        predicted_labels = torch.zeros(current_batch_size, device=device)
        for idx, hFields in enumerate(x):
            H = make_Hamiltonian(
                num_qubits=self.num_qubits,
                gamma=self.gamma,
                hFields=hFields,
            )
            U = make_unitary(H, dt=self.time)

            predicted_label, overlap = quantum_criteria_classifier(
                num_qubits=self.num_qubits,
                U=U,
                observables=self.observables,
                init=self.init,
                # experiment=False,
                experiment=self.exp,
            )
            predicted_labels[idx] = predicted_label
            self.overlaps_list.append(overlap)

        return predicted_labels
