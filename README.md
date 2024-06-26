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

## Modifications compared to original nnUNet.

1. Add dropout for the middle of UNet.
   
   Files involving:
   - Add new trainer: nnUNetTrainerV2_dropout.py
   - Add new architecture: generic_UNet_center_dropout.py
     
   Change the below dropout parameter in "nnUNetTrainerV2_dropout.py"
```
dropout_op_kwargs = {'p': 0.2, 'inplace': True}
```
2. Add Snapshot checkpoints with learning rate restart.

   Files involving:
   - nnUNetTrainerV2_dropout.py
   - nnUNetTrainerV2.py
   - network_trainer.py

   Set the below snapshot parameters in "nnUNetTrainerV2_dropout.py"
```
        self.restart_lr_after = 1000
        self.restart_multiplications =5
```
By default, the trainer would save once 100 epochs after the training reached "restart_lr_after".

