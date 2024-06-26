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

## Modifications compared to original nnUNet

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
        self.restart_multiplications = 5
```
By default, the trainer would save once 100 epochs after the training reached "restart_lr_after".

3. Add Probabilistic Hierarchical Segmentation (PhiSeg).
   
   Files involving:
   - nnUNetTrainerV2_PhiSeg.py
   - nnUNetTrainerV2_PhiSeg_gamma.py # Loss composition oscillation between reconstruction loss and ELOB loss.
   - nnUNetTrainerV2_PhiSegAdam.py # Using Adam as optimizer.
   - nnUNetTrainerV2_PhiSeg_dc_ce.py # Only optimize on the reconstruction loss (cross entropy + dice loss).
   - PHISeg.py # the architecture for PhiSeg. 
     
If you use PhiSeg, please cite and reference the original paper "Capturing Uncertainty in Medical Image Segmentation":
```
@article{PHiSeg2019Baumgartner,
         author={Baumgartner, Christian F. and Tezcan, Kerem C. and
         Chaitanya, Krishna and H{\"o}tker, Andreas M. and
         Muehlematter, Urs J. and Schawkat, Khoschy and Becker, Anton S. and
         Donati, Olivio and Konukoglu, Ender},
         title={{PHiSeg}: Capturing Uncertainty in Medical Image Segmentation},
         journal={arXiv:1906.04045},
         year={2019},
}
```

4. Add Gaussian noise for Test time augmentation.
   Files involving:
   - neural_network.py (added additional Gaussian noise TTA sample to _internal_maybe_mirror_and_pred_3D())
   - all the involved trainer files (do_gaussian_noise=True)
