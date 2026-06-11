# MoQ-NAS: A Framework for Multi-Objective Neural Architecture Search

This repository provides a flexible and extensible framework for Neural Architecture Search (NAS), focusing on multi-objective optimization. It implements and compares Quantum-Inspired Evolutionary Algorithms (Q-NAS and MO-QNAS) against traditional Genetic Algorithms (GA, NSGA-II, NSGA-III).

## Features

- **Multiple Search Algorithms:** Includes implementations for:
  - **Quantum-Inspired NAS (QNAS)** for single-objective optimization.
  - **Multi-Objective QNAS (MO-QNAS).**
  - **Classic Genetic Algorithm (GA).**
  - **NSGA-II and NSGA-III** for multi-objective optimization.
- **Modular Architecture:** A clean, refactored structure that separates algorithms, core components, and utilities.
- **Extensible CNN Library:** A rich set of CNN building blocks, including standard convolutions, residual blocks, and attention mechanisms (SE, CBAM).
- **Flexible Configuration:** Easily configure experiments, search spaces, and network parameters using YAML files.
- **Multi-Objective Optimization:** Optimize for competing objectives simultaneously, such as accuracy, model size, and inference time.
- **Fairness Evaluation:** A post-processing step to evaluate model fairness across different demographic groups, such as skin tone and race.

## Project Structure

The codebase has been refactored into a modular architecture to improve clarity and maintainability.

```
moqnas/
├── algorithms/
│   ├── ga/                   # Classic Genetic Algorithms (GA, NSGA-II, NSGA-III)
│   └── qnas/                 # Quantum-Inspired Algorithms (QNAS, MOQNAS)
│
├── core/
│   ├── cnn/                  # CNN model definitions, trainer, and metrics
│   ├── fairness/             # Fairness evaluation logic and data loaders
│   ├── config.py             # Experiment configuration handler
│   └── evaluation.py         # Population evaluation engine
│
├── dataset_utils/
│   ├── factory.py            # Dataset loading and splitting logic
│   └── transformations.py    # Data augmentation and transforms
│
├── utils/
│   └── helpers.py            # General utility functions
│
├── dataset_configs/
│   └── *.yaml                # YAML files for experiment configuration
│
├── scripts/
│   ├── download_datasets/    # Script to download and prepare datasets like FairFace, WiderFace, Coco 
│   ├── fairness_baseline/    # Evaluate fairness of baseline models (e.g., ResNet, MobileNet)
│   └── readme.md             # Instructions to create person/face datasets for fairness evaluation
│
├── dataset_utils/
│   ├── factory.py            # Dataset loading and splitting logic
│   └── transformations.py    # Data augmentation and transforms
│
├── utils/
│   └── helpers.py            # General utility functions
│
└── ...
```

- `algorithms/`: Contains the core logic for all search algorithms.
- `core/`: Holds shared components essential for any experiment, including the CNN builder/trainer and the evaluation engine.
- `dataset_utils/`: Manages all data loading, preprocessing, and splitting.
- `utils/`: Contains helper functions used across the project.
- `dataset_configs/`: Stores YAML configuration files that define the search space, model parameters, and training settings for experiments.
- `run_*.py`: Executable scripts to launch different types of NAS experiments.

## Fairness Evaluation

The framework includes a **FairnessMetric** module designed to evaluate the performance of trained models across different demographic subgroups. This is treated as a **post-processing** step, meaning it runs on fully trained models to assess their fairness **without influencing the training process itself**.

### How It Works

The fairness evaluation is orchestrated by the `fairness_worker_cuda` function, which performs the following steps for each model architecture in a generation:

1. **Model Loading:** The worker loads a pre-trained model onto a specified CUDA device.
2. **Dataloader Creation:** It creates a special evaluation dataloader for fairness assessment using datasets like **Facet** or **FairFace**.
3. **Inference and Metric Calculation:** The `FairnessMetric` class runs inference on the evaluation dataset and calculates the **True Positive Rate (TPR)** for each demographic group.
4. **Fairness Score Computation:** Based on the per-group TPRs, it computes a final `fairness_score` and other summary metrics.

### Fairness Score Calculation

The primary metric, `fairness_score`, is derived from the per-group TPRs. The key components of this calculation are:

- **Per-Group TPR:** The True Positive Rate is calculated for each demographic group (e.g., for each skin tone in the Facet dataset or each race in the FairFace dataset).
- **Minimum Group TPR (`min_tpr`):** This is the lowest TPR observed across all groups.
- **Sum of Gaps (`spd_sum`):** This value represents the sum of the differences between each group's TPR and the `min_tpr`. A lower `spd_sum` indicates better fairness.
- **Fairness Score:** The final score is calculated as `max(0.0, (beta - spd_sum) / beta)`, where `beta` is a configurable hyperparameter. This score is normalized to a range between **0** and **1**, where **1** represents the best possible fairness.

For the **Facet** dataset, the TPR can be calculated in two ways:

- **hard method:** Assigns each image to a single skin tone class.
- **soft method:** Uses weighted probabilities for each skin tone class.

These fairness metrics are then saved alongside other evaluation results, allowing you to incorporate fairness as a key consideration in your multi-objective NAS experiments.


## Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/MoQ-NAS.git
cd MoQ-NAS
pip install -r requirements.txt
```

### 2. Configuration

All experiments are controlled by configuration files located in the `dataset_configs/` directory. Before running an experiment, you can create or modify a `.yaml` file to define:

- The dataset (`dataset`, `data_path`).  
- The search space (`function_dict`).  
- Algorithm hyperparameters (`max_generations`, `population_size`, etc.).  
- Training parameters (`batch_size`, `max_epochs`, `optimizer`).  


### 3. Running an Experiment

Use one of the root-level scripts to launch an experiment. For example, to run a Multi-Objective QNAS evolution:

```bash
python run_evolution_moqnas.py \
    --config_file configs/your_experiment_config.yaml \
    --experiment_path experiments/my_moqnas_run \
    --log_level INFO
```

- `--config_file`: Specifies the YAML file with the experiment's parameters.  
- `--experiment_path`: Defines the directory where logs, models, and results will be saved.  
- `--log_level`: Sets the verbosity of the log output.  



### 4. Environment Configuration

The following steps are used to configure the environment for the project.

- Miniconda Installation
- Conda Environment Creation
- Package Installation

**Notes**: 
- An NVIDIA GPU is required to run the project. 
- The project is tested on Ubuntu 22.04 LTS with NVIDIA L40S GPU.

#### 4.1 Miniconda Installation

Install Miniconda in the home directory. Refer to the [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) for more information.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

#### 4.2 Conda Environment Creation

```bash
conda create -n moqnas python=3.10
conda activate moqnas
```

#### 4.3 Package Installation

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```

## To-Do
- [ ] Add more search algorithms (e.g., DARTS, ENAS).
- [ ] Improve FP8 mixed-precision training support.
- [ ] Implement additional fairness metrics (e.g., Equal Opportunity, Demographic Parity).
- [ ] Add support for more datasets and tasks (e.g., object detection, segmentation).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
