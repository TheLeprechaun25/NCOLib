# Neural Combinatorial Optimization Library (NCOLib)

## Introduction
The Neural Combinatorial Optimization Library (NCOLib) is an accessible software library designed to simplify the application of neural network models and deep learning algorithms to solve combinatorial optimization problems. It brings the power of NCO closer to non-experts, including industry professionals and junior researchers.

## Features
- **Model Training:** Train *neural constructive* and *neural improvement* models to tackle various optimization tasks.
- **Inference Capabilities:** Deploy trained models to infer and solve optimization problems efficiently.
  
  
## Installation
Ensure that you have Python 3.8 or higher installed, along with pip. NCOLib requires PyTorch, which can be installed using the following command if not already installed:
```bash
pip install torch
```

## Getting Started
### Basic Usage Example
For a hands-on introduction, check out [examples/basic_tour.ipynb](examples/basic_tour.ipynb), which explains the basic usage of the library.

## Roadmap

- Include datasets to be used as an alternative to random generators.
- Add Non-Auto-regressive (heatmap) + search (subclass of Problem).
- Improve basic tour: add comments and convert to a .ipynb.
- Create advanced tour
- Add tests
- Add other GNN architectures: GNN with edges, GCN, GAT, etc.
- Add more problems: scheduling, assignment, etc.
- Add graph-context as the used in POMO decoder.
- Add memory-mechanism to env and model.
- Actor critic and PPO training.
- Add data augmentation (POMO).
  
## Contributing
We welcome contributions from the community! If you are interested in helping to develop NCOLib, whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## License
NCOLib is made available under the MIT License. For more details, see the LICENSE file in the repository.
