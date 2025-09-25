# Single Neuron Representations of Surprise and Uncertainty in the Human Claustrum

# Table of Contents

- [Overview](#Overview)
- [Abstract](#Abstract)  
- [Requirements](#Requirements)
- [Code and Data Structure](#Code-and-Data-Structure)
- [Installation](#Installation)
- [How to use the code](#How-to-use-the-code)
- [Citation](#Citation)
- [License](#License)

# Overview

This repository contains codes associated with paper *Single Neuron Representations of Surprise and Uncertainty in the Human Claustrum* by XX et al. 

# Abstract

Flexible adaptation in dynamic environments relies on continuous updating of internal generative models of the world. A central question is how such computations are implemented in distributed cortical and subcortical circuits. The claustrum (CLA), with its dense reciprocal connectivity across the neocortex, is anatomically positioned to support such computations.

We recorded single-neuron activity in the human CLA during an aversive learning task, with the anterior cingulate cortex (ACC) and amygdala (AMY) included as controls. CLA and ACC neurons exhibited structured task-related responses, with subpopulations tuned to stimulus onset, outcomes, or both, and region-specific biases toward preferred outcomes. In contrast, AMY neurons showed minimal task modulation. Across trials, CLA neurons were modulated by subjective uncertainty and prediction error, showing patterns distinct from ACC.

Finally, a recurrent graph neural network (R-GNN) incorporating attention mechanisms successfully decoded uncertainty and prediction error on a per-subject basis. These results indicate that the human CLA encodes belief dynamics and may act as an active hub in the brain’s inferential hierarchy, supporting rapid signal propagation and coordination of distributed neural systems for flexible behavior.

# Requirements

## Software Requirements

- **MATLAB (MathWorks, Natick, MA)** for Figures 1–3  
  Required toolboxes:
  
  - Statistics and Machine Learning Toolbox
  
  - Signal Processing Toolbox

- **Python and R** for Figures 4–6

# Code Structure

## Folder structure

1. Codes
   
   - 
   - NeuroPair (Recurrent Graph Neural Network with Attention)
   - utilities (functions necessary for Matlab code)

# Installation

Download or clone the repository

```bash
git clone https://github.com/damisahlab/claustrum-uncertainty-surprise
```

# Usage

The analysis of each figure can be regenerated from that folder using the corresponding code for each figure. 

**Example:** To generate Fig. 1e:
                *HumanCLAandACC_Fig1e()*

# Citation

## Contributing

We welcome contributions!

## License

This project is licensed under the [MIT License](LICENSE).
