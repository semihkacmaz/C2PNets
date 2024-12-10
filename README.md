# C2PNets: Machine Learning for Conservative-to-Primitive Variable Recovery in Numerical Relativity

This repository implements neural network architectures for conservative-to-primitive variable recovery in numerical relativity simulations, tailored for neutron star equations of state (EoS) in both tabulated and piecewise polytropic forms. As a precursor study, it evaluates the viability of neural networks as alternatives to traditional root-finding methods. The project also includes TensorRT engine optimization and C++ inference capabilities for performance benchmarking against conventional approaches.

## Key Features

- Multiple basic network architectures optimized for different EoS types:
  - `NNC2PS`: A shallow network optimized for piecewise polytropic EoS
  - `NNC2PL`: A deep architecture extending NNC2PS for enhanced accuracy
  - `NNC2P_Tabulated`: A dedicated network for tabulated EoS handling

- Training pipeline with:
  - Automated data generation for both EoS types
  - Customizable training configurations
  - Physics-informed loss functions ensuring solution physicality
  - Checkpoint management
  - Training visualization

- Inference pipeline with:
  - TorchScript model export
  - TensorRT engine optimization
  - CUDA-accelerated C++ inference

## Installation

```bash
# Clone the repository
git clone https://github.com/semihkacmaz/C2PNets.git
cd C2PNets-main

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training 
```bash
# Piecewise polytropic EoS models
python -m src.training.train --config configs/nnc2ps_config.yaml --model NNC2PS
python -m src.training.train --config configs/nnc2pl_config.yaml --model NNC2PL

# Tabulated EoS model
python -m src.training.train --config configs/nnc2p_tabulated_config.yaml --model NNC2P_Tabulated
```

### TensorRT Engine Generation

``` bash
python -m src.utils.engine_builder --config configs/engine_builder_config.yaml
```

## License

MIT License. See [LICENSE](LICENSE) file for details.
