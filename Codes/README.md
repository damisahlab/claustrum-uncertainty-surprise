# The human claustrum encodes surprise and uncertainty during adaptive learning

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

This repository contains code for reproducing analyses from the paper *The human claustrum encodes surprise and uncertainty during adaptive learning* by XX et al. 

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

- **Python 3.9 and R version 4.5.0** for Figures 4–6

# Directory Structure

## Directory Structure

Here's an overview of the repository structure:

```
├── NeuroPair/                                   # Recurrent Graph Neural Network with Attention (Python)
├── utilities/                                   # Helper functions for MATLAB analyses
├── HumanCLAandACC_ExtFig8a.py                   # Python script Extended Figure 8a
├── HumanCLAandACC_ExtFig9a_e.py                 # Python script Extended Figure 9a-e
├── HumanCLAandACC_Fig1e.m                       # MATLAB script for Figure 1e
├── HumanCLAandACC_Fig1f.m                       # MATLAB script for Figure 1f
├── HumanCLAandACC_Fig1g.m                       # MATLAB script for Figure 1g
├── HumanCLAandACC_Fig1h.m                       # MATLAB script for Figure 1h
├── HumanCLAandACC_Fig1i.m                       # MATLAB script for Figure 1i
├── HumanCLAandACC_Fig1j.m                       # MATLAB script for Figure 1j
├── HumanCLAandACC_Fig4&5_chord_diagram.R        # R script for Figures 4 & 5 (Chord diagram)
├── HumanCLAandACC_Fig4&5_mutual_information.py  # Python script for Figures 4 & 5 (Mutual information)
├── HumanCLAandACC_Fig4&5_cluster_based_perm.py  # Python script for Figures 4 & 5 (Cluster-based analysis)
├── HumanCLAandACC_Fig4_histogram.py             # Python script for Figure 4 histogram
└── README.md
```

# Installation

Download or clone the repository

```bash
git clone https://github.com/damisahlab/claustrum-uncertainty-surprise
```

# Usage

The analysis of each figure can be regenerated from that folder using the corresponding code for each figure. 

**Example:** To generate Fig. 1e:

                *HumanCLAandACC_Fig1e*

# Citation

If you use this code, please cite:

> XX et al., "The human claustrum encodes surprise and uncertainty during adaptive learning", *Nature*, 2025.

## Contributing

We welcome contributions!

## License

This project is licensed under the [MIT License](LICENSE).
