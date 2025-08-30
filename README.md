# Quantum Machine Learning Classification Protocol

This repository contains the core implementation for the quantum machine learning classification protocol described in our paper:

**📄 Paper**: [Enhanced image classification via hybridizing quantum dynamics with classical neural networks](https://arxiv.org/abs/2507.13587)

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Physics-of-QIT/classification-protocol.git
cd classification-protocol
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

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Zhou_2025_arxiv,
    title        = {{Enhanced image classification via hybridizing quantum dynamics with classical neural networks}},
    author       = {Zhou, R and Sarkar, Saubhik and Bose, Sougato and Bayat, Abolfazl},
    year         = 2025,
    journal      = {npj Quantum Information},
    doi          = {10.48550/arXiv.2507.13587},
    publisher    = {Nature Portfolio}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
