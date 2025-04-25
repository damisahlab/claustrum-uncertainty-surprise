# Behavior Prediction with Autoregressive and Transformer Models

Welcome to the **Behavior Prediction** repository! This project focuses on modeling behavioral prediction using advanced techniques like **Autoregressive Models (e.g., LSTM)** and **Transformers**. 

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Setup & Usage](#setup--usage)
4. [Features](#features)
5. [TODOs](#TODOs)

## Overview

This repository provides implementations of models for behavioral prediction using:
- **Autoregressive approaches** (e.g., LSTM)
- **Transformer architectures**

The models are designed for training, evaluation, and visualization, providing insights into the predictive patterns within the dataset.

## Directory Structure

Here's an overview of the repository structure:

```
├── dataset/claustrum/       # Directory to store the dataset
├── models/                  # Model definitions (LSTM, Transformer, etc.)
│   ├── autoregressive.py    # autoregressive model
│   ├── transformer.py       # Transformer model 
├── nn/                      # Neural network components
│   ├── lstm.py              # autoregressive modules (e.g., LSTM)
│   ├── self_attnetions.py   # Transformer modules (e.g., self-attentions)
├── viz/                     # Visualization utilities
│   ├── scatter_plot.py      # Scatter plot generation (to be implemented)
│   ├── plot_utils.py        # General plotting functions (to be implemented)
├── train.py                 # Training script
├── args.py                  # Hyperparameter and configuration settings
└── output/model/            # Directory for saving trained models
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
../datasets/claustrum/
```

### 3. Run the Training Script
Execute the training script:

```bash
python train.py
```

## Features

### `.models/`
- Contains all model definitions including:
  - **Autoregressive Models (LSTM)**
  - **Transformers**  (TODO)

### `.nn/`
- **`nn/lstm.py`**: Contains the LSTM model implementation.
  - **To-Do**: Verify the correctness of the LSTM model.

### `.viz/`
- Functions for plotting and generating figures (to be developed):
  - Scatter plots
  - Other visualizations
  - **To-Do**: Move plotting functions to `./viz/` directory.

### `train.py`
- Handles training and validation.
- Captures the best-performing model and saves it to:

```
../output/model/
```

  - **To-Do**: 
    - Investigate why the loss is not decreasing (*Assigned to Arman*).
    - Develop the test function.

### `args.py`
- Contains hyperparameters and utility functions.
  - **To-Do**: Refactor utility functions into a dedicated `utils.py` file.

## TODOs

- [ ] Verify the implementation of the LSTM model.
- [ ] Address loss reduction issue during training (*Arman*).
- [ ] Develop scatter plot and other visualization utilities.
- [ ] Move utility functions from `args.py` to `utils.py`.
- [ ] Implement and test the test function.

## Contributing
We welcome contributions!  

## License
This project is licensed under the [MIT License](LICENSE).



