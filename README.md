# Neural Combinatorial Optimization Library (NCOLib)

## Introduction
The Neural Combinatorial Optimization Library (NCOLib) is an accessible software library designed to simplify the application of neural network models and deep learning algorithms to approximately solve combinatorial optimization problems. It brings the power of NCO closer to industry professionals and researchers.

## Use
### 1. Clone the repository
```bash 
git clone https://github.com/TheLeprechaun25/NCOLib.git
cd NCOLib
```

### 2. Install the required package
```bash
pip install torch
```

### 3. Run one of the examples
```bash
python examples/tsp_constructive.py
```

### 4. Apply NCO to your specific problem
In order to apply NCO to your specific problem, you need to define a **problem class**. The state is where all the problem information is stored, and later used in the model and in the environment to compute the objective value.

For a hands-on introduction on how to apply, check out the [examples/](examples/) folder or the [basic_tour.ipynb](examples/basic_tour.ipynb) notebook.


## Features
- **Model Training:** Train deep neural models to tackle combinatorial optimization problems.
- **Inference Capabilities:** Deploy trained models to infer and solve optimization problems efficiently.
- **Experimentation:** Experiment with different models, datasets, and hyperparameters to find the best solution.

### NCO Frameworks
- **Neural Constructive:** Construct solutions from scratch, autoregressively.
- **Neural Improvement:** Improve existing solutions with local modifications.

### Supported model types

We use Graph Neural Networks (GNNs) to solve combinatorial optimization problems. Different GNN architectures can be used based on the input features (node-based, edge-based or node- and edge-based) and output actions (node-based or edge-based) at hand:

| Input &darr; Output &rarr;     | Node-based actions | Edge-based actions |
|--------------------------------|--------------------|--------------------|
| **Node-based features**        | GCN, GT, eGNN      | EO_GT              |
| **Edge-based features**        | EI_GT, eGNN        | EI_EO_GT           |
| **Node & Edge-based features** | EI_GT, eGNN        | EI_EO_GT           |

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

### Done:
- Add data augmentation (POMO).
- Add graph-context via auxiliary or virtual node.
- Add a decoder that uses attention to the graph-context.

### To do:
- Add other GNN architectures: GIS, GAT, etc.
- Actor critic and PPO training.
- Include datasets to be used as an alternative to random generators.
- Add tests
- Add more problems to examples: scheduling, assignment, etc.
- Add more evaluation metrics beside obj. value: convergence time and stability across different runs
- Add memory-mechanism to env and model (MARCO).
- Add Non-Auto-regressive (heatmap) + search (subclass of Problem).


## Contributing
We welcome contributions from the community! If you are interested in helping to develop NCOLib, whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## License
NCOLib is made available under the MIT License. For more details, see the LICENSE file in the repository.
