# Quantum Machine Learning Classification Protocol

This repository contains the core implementation for the quantum machine learning classification protocol described in our paper:

**📄 Paper**: [arXiv:2507.13587](https://arxiv.org/abs/2507.13587) - *Quantum Machine Learning Classification Protocol*

## Overview

This project implements a novel quantum machine learning classification protocol that leverages **shadow tomography** techniques for efficient quantum state characterization in machine learning tasks. The framework combines classical neural networks with quantum computing primitives to create hybrid quantum-classical models for image classification.

### Key Features

- **Shadow Tomography Implementation**: Efficient quantum state reconstruction using randomized measurements
- **Hybrid Quantum-Classical Models**: Neural networks that integrate quantum computing components
- **Multi-Dataset Support**: Compatible with medical imaging datasets (MedMNIST family), satellite imagery (SAT), traffic signs (GTSRB), and standard benchmarks (MNIST variants)
- **Quantum Circuit Integration**: Built on PennyLane for quantum computing functionality
- **Scalable Architecture**: Supports various qubit configurations and experimental setups

## Architecture

### Core Components

1. **Shadow Tomography Module** (`shadow_tomography`): Implements efficient quantum state estimation using randomized Pauli measurements
2. **Quantum Neural Networks**: 
   - `ComparisonNet`: Main quantum-classical hybrid model for classification
   - `RegressionNet`: Regression-based quantum model
3. **Dataset Loaders**: Custom PyTorch datasets for quantum ML experiments
4. **Quantum Circuit Components**: Pauli operators, quantum devices, and measurement protocols

### Quantum Computing Features

- **Quantum State Preparation**: Automatic state initialization and normalization
- **Pauli Measurements**: Randomized measurements with X, Y, Z Pauli operators  
- **Quantum Fidelity Estimation**: Shadow-based fidelity calculations between quantum states
- **Hybrid Optimization**: Classical optimization of quantum circuit parameters

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Physics-of-QIT/classification-protocol.git
cd classification-protocol

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# Core dependencies
pip install torch torchvision
pip install pennylane
pip install numpy scipy matplotlib
pip install scikit-learn pandas
pip install medmnist
pip install seaborn tabulate
pip install pillow
```

### Optional Dependencies

For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Directory Setup

The framework expects certain directories to exist:

```bash
mkdir -p data logs dataset src/figures
```

- `data/`: Stores experimental results and model outputs
- `logs/`: Contains training and execution logs  
- `dataset/`: Custom dataset storage
- `src/figures/`: Generated plots and visualizations

## Usage

### Basic Example

```python
import sys
sys.path.append('scripts')

from qml3_library import Args
from qml3_executor import executor

# Set up experiment arguments
args = Args(
    dataset="BloodMNIST",
    idx_run=0,
    num_hidden_layers=2,
    num_classes=8,
    num_qubits=4,
    time=1.0,
    learning_rate=0.001,
    weight_decay=1e-4,
    epochs=50,
    batch_size=32,
    num_train_samples=1000,
    num_test1_samples=200,
    num_test2_samples=200,
    exp=True  # Enable shadow tomography
)

# Run training and testing
train_accuracy, test1_accuracy, test2_accuracy = executor(
    args,
    train=True,
    test1=True, 
    test2=True
)

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test1 Accuracy: {test1_accuracy:.2f}%") 
print(f"Test2 Accuracy: {test2_accuracy:.2f}%")
```

### Shadow Tomography Example

```python
import torch
import sys
sys.path.append('scripts')
from qml3_library import shadow_tomography

# Prepare a quantum state
num_qubits = 3
state = torch.zeros((2**num_qubits, 1), dtype=torch.complex64)
state[0, 0] = 1.0  # |000⟩ state

# Perform shadow tomography
shadows = shadow_tomography(
    num_qubits=num_qubits,
    state=state,
    num_measurements=1000,
    chunk_size=100
)

print(f"Generated {len(shadows)} shadow snapshots")
```

## Supported Datasets

### Medical Imaging (MedMNIST Family)
- **PathMNIST**: Colorectal cancer histology
- **ChestMNIST**: Chest X-ray classification
- **DermaMNIST**: Dermatology lesion classification
- **OCTMNIST**: Optical coherence tomography
- **PneumoniaMNIST**: Pneumonia detection
- **RetinaMNIST**: Retinal disease classification
- **BreastMNIST**: Breast ultrasound classification
- **BloodMNIST**: Blood cell classification
- **TissueMNIST**: Kidney tissue classification
- **OrganAMNIST, OrganCMNIST, OrganSMNIST**: Organ classification variants

### Other Datasets
- **SAT4/SAT6**: Satellite image classification
- **GTSRB**: German Traffic Sign Recognition Benchmark
- **MNIST/FashionMNIST**: Standard benchmarks

## Experimental Framework

The framework supports multiple experimental configurations:

### Training Phases
1. **Training**: Train the quantum-classical hybrid model
2. **Test1**: Evaluate on standard test sets
3. **Test2**: Advanced evaluation with quantum state analysis

### Key Parameters
- `num_qubits`: Number of qubits in quantum circuits
- `num_classes`: Number of classification classes
- `time`: Evolution time for quantum circuits
- `exp`: Experimental mode (enables shadow tomography)
- `gamma`: Scaling parameter for quantum operations

## File Structure

```
classification-protocol/
├── scripts/
│   ├── qml3_library.py      # Core quantum ML functions and classes
│   └── qml3_executor.py     # Main execution framework  
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # GPL-3.0 License
```

## Key Functions

### Shadow Tomography
- `shadow_tomography()`: Main shadow tomography implementation
- `shadow_expectation()`: Expectation value estimation from shadows
- `shadow_fidelity()`: Fidelity estimation between quantum states

### Quantum Operations  
- `quantum_criteria_classifier()`: Quantum-based classification criteria
- `make_hFields()`: Magnetic field generation for quantum evolution
- `density_matrix()`: Density matrix construction from state vectors

### Neural Networks
- `ComparisonNet`: Main hybrid quantum-classical model
- `RegressionNet`: Quantum regression model
- `CustomizedMNIST`: Modified MNIST dataset for quantum experiments

## Performance and Scaling

- **Quantum Simulation**: Runs on classical simulators (CPU/GPU)
- **Scalability**: Supports 2-10 qubits efficiently  
- **Memory Optimization**: Chunked processing for large shadow tomography
- **Parallel Processing**: Batch processing for neural network training

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{quantum_classification_2025,
  title={Quantum Machine Learning Classification Protocol},
  author={[Authors]},
  journal={arXiv preprint arXiv:2507.13587},
  year={2025}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions about this implementation or the associated research, please refer to the paper: https://arxiv.org/abs/2507.13587