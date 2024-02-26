# Uncertainty-estimation-for-Head-and-Neck-tumour-Segmentation
This repository contains code and visualizations for comparing various uncertainty estimation methods applied to the auto-segmentation of head and neck tumors, focusing on both Gross Tumor Volume primary (GTV-T) and Nodes (GTV-N).

## Setting Up Your Environment
This study primarily builds upon the original nnUNet framework (version 1), with several modifications to facilitate uncertainty estimation. To replicate our environment and ensure compatibility, please follow the steps below to create and install the necessary components in a new virtual environment.

### Prerequisites
Ensure you have Anaconda or Miniconda installed on your system to manage virtual environments and dependencies.

### Installation Steps
```
conda create -n probseg
conda activate probseg
cd nnUNet_uncertainty
pip install -e .
```


