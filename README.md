# Neural Combinatorial Optimization Library (NCOLib)

## Introduction
The Neural Combinatorial Optimization Library (NCOLib) is an accessible software library designed to simplify the application of neural network models and deep learning algorithms to solve combinatorial optimization problems. It brings the power of NCO closer to non-experts, including industry professionals and junior researchers.

## Installation
Ensure that you have Python 3.8 or higher installed, along with pip. NCOLib requires PyTorch, which can be installed using the following command if not already installed:
```bash
pip install torch
```

## Features
- **Model Training:** Train *neural constructive* and *neural improvement* models to tackle various optimization tasks.
- **Inference Capabilities:** Deploy trained models to infer and solve optimization problems efficiently.

### NCO Frameworks
- **Neural Constructive:** Construct solutions from scratch, autoregressively.
- **Neural Improvement:** Improve existing solutions with local modifications.

### Supported model types

We use Graph Neural Networks (GNNs) to solve combinatorial optimization problems. Different GNN architectures can be used based on the input features (node-based, edge-based or node- and edge-based) and output actions (node-based or edge-based) at hand:

| Input &darr; Output &rarr; | Node-based | Edge-based |
|----------------------------|------------|------------|
| **Node-based**             | GCN, GT    | EO_GT      |
| **Edge-based**             | EI_GT      | EI_EO_GT   |
| **Node & Edge-based**      | EI_GT      | EI_EO_GT   |

**Key:**
- **GCN:** Graph Convolutional Network. GCNModel class.
- **GT:** Graph Transformer. GTModel class.
- **EI_GT:** Edge-Input Graph Transformer. EdgeInGTModel class.
- **EO_GT:** Edge-Output Graph Transformer. EdgeOutGTModel class.
- **EI_EO_GT:** Edge-Input, Edge-Output Graph Transformer. EdgeInOutGTModel class.

### Supported Deep Learning Training Algorithms
**Reinforcement Learning:**

| RL Algorithm                     | Constructive | Improvement |
|----------------------------------|--------------|-------------|
| **Policy Gradient**              | ✅            | ✅           |
| **Actor Critic**                 |              |             |
| **Proximal Policy Optimization** |              |             |


**Supervised Learning:**
Under development.

## Getting Started
### Basic Usage Example
User needs to define a problem class. The state is where all the problem information is stored, and later used in the model and in the environment to compute the objective value.

The state class has the following fixed attributes:
- state.batch_size 
- state.problem_size 
- state.device 
- state.seed 
- state.node_features 
- state.edge_features 
- state.graph_features 
- state.solutions 
- state.mask
- state.is_complete

While the dynamic, problem-based and user-defined attributes are stored in the **state.data** dictionary. For example: 

```python
state.data.tsp_coordinates = torch.rand((state.batch_size, state.problem_size, 2), device=state.device)
```


For a hands-on introduction, check out [examples/basic_tour.ipynb](examples/basic_tour.ipynb), which explains the basic usage of the library.


## Project Structure
- [nco_lib/](nco_lib/) 
  - [data/](nco_lib/data/)
    - [data_loader.py](nco_lib/data/data_loader.py) - Handles the generation, loading and preprocessing of datasets used.
  - [trainer/](nco_lib/trainer/)
    - [trainer.py](nco_lib/trainer/trainer.py) - Contains the training loop and inference functions.
  - [environment/](nco_lib/environment/)
    - [env.py](nco_lib/environment/env.py) - Defines the main environment, reward and stopping criteria classes.
    - [problem_def.py](nco_lib/environment/problem_def.py) - Contains definitions of the problems that the models are designed to solve.
    - [actions.py](nco_lib/environment/actions.py)- Includes definitions of actions that can be performed within the improvement environment.
  - [models/](nco_lib/models/)
    - [base_layers.py](nco_lib/models/base_layers.py) - Contains definitions for the base layers used in the GNN models.
    - [gcn.py](nco_lib/models/gcn.py) - Defines the Graph Convolutional Network (GCN) model.
    - [graph_transformer.py](nco_lib/models/graph_transformer.py) - Contains the Graph Transformer model definitions.
- [examples/](examples/)
    - [basic_tour.ipynb](examples/basic_tour.ipynb) - A python notebook providing a basic tour of the project’s functionalities.


## Roadmap

- Add other GNN architectures: GIS, GAT, etc.
- Actor critic and PPO training.
- Include datasets to be used as an alternative to random generators.
- Add tests
- Add more problems to tour: scheduling, assignment, etc.
- Add graph-context as the used in POMO decoder.
- Add data augmentation (POMO).
- Add more evaluation metrics beside obj. value: convergence time and stability across different runs
- Add memory-mechanism to env and model.
- Add Non-Auto-regressive (heatmap) + search (subclass of Problem).


## Contributing
We welcome contributions from the community! If you are interested in helping to develop NCOLib, whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## License
NCOLib is made available under the MIT License. For more details, see the LICENSE file in the repository.
