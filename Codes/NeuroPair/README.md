# NeuroPair: Behavior Prediction from Single-Neuron Activity

This repository implements NeuroPair, a recurrent graph neural network with attention (R-GAT) with per-neuron, behavior-specific attention mechanisms, designed to model and predict behavioral variables from neural recordings in the human claustrum  (CLA) and anterior cingulate cortex (ACC)

This repository contains the implementation of **NeuroPair**, a recurrent graph neural network with neuron-specific attention, used to model and predict behavioral variables from single-neuron recordings in the human claustrum (CLA) and anterior cingulate cortex (ACC). The code accompanies the paper:

> "The human claustrum encodes surprise and uncertainty during adaptive learning", XX et al., *Nature*, 2025.

## Table of Contents

1. [Overview](#overview)
2. [Feature](#features)
3. [Directory Structure](#directory-structure)
4. [Setup & Usage](#setup--usage)
5. (#TODOs)

## Overview

NeuroPair is designed to predict trial-by-trial behaviors from neural population activity, integrating:

- **Recurrent Graph Attention Networks (GAT)** for modeling inter-neuronal connectivity.
- **LSTM modules** for capturing temporal dependencies.
- **Per-neuron, behavior-specific attention** to enhance interpretability.

This framework enables training, validation, and visualization of predicted vs. observed behaviors, supporting reproduction of figures from the associated Nature paper.

## Feature

- Trainable models for individual subjects and brain regions.
- Support for multiple behavioral targets:
  - Prediction errors
  - Safety variance
- Modular code structure for flexibility and reproducibility.
- Integrated plotting and evaluation tools for inspecting model performance.

## Directory Structure

Here's an overview of the repository structure:

```
├── data/                  # Directory to store the dataset
├── model/                  # Model definitions
│   ├── autoregressive.py   # autoregressive model
│   ├── recurrentgat.py     # R-GAT model 
├── nn/                     # Neural network components
│   ├── gat.py              # gat modules
│   ├── lstm.py             # autoregressive modules (e.g., LSTM)
├── viz/                    # Visualization utilities
│   ├── histogram.py        # Histogram generation
│   ├── scatter.py          # Scatter plot generation
├── main.py                 # Main script
├── args.py                 # Hyperparameter and configuration settings
└── requirements.txt        # Requirements (libraries)
```

## Setup & Usage

### 1. Prerequisites

Ensure you have Python installed along with necessary dependencies. Install required packages using:

```bash
pip install -r requirements.txt
```

### 2. Dataset

Place your dataset in the following directory structure:

```
../data/
```

### 3. Run the Main Script

Execute the main script:

```bash
python main.py
```

## Citation

If you use this code, please cite:

> XX et al., "The human claustrum encodes surprise and uncertainty during adaptive learning", *Nature*, 2025.

## Contributing

We welcome contributions!  

## License

This project is licensed under the [MIT License](LICENSE).
