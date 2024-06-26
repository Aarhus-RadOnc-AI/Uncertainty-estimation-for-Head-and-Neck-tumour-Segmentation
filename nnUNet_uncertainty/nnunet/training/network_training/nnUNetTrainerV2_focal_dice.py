#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



# from collections import OrderedDict
# from typing import Tuple

# import numpy as np
# import torch
# from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
# from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
# from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
# from nnunet.network_architecture.generic_UNet import Generic_UNet
# from nnunet.network_architecture.initialization import InitWeights_He
# from nnunet.network_architecture.neural_network import SegmentationNetwork
# from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
#     get_patch_size, default_3D_augmentation_params
# from nnunet.training.dataloading.dataset_loading import unpack_dataset
# from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
# from nnunet.utilities.nd_softmax import softmax_helper
# from sklearn.model_selection import KFold
# from torch import nn
# from torch.cuda.amp import autocast
# from nnunet.training.learning_rate.poly_lr import poly_lr
# from batchgenerators.utilities.file_and_folder_operations import *
#from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import DC_and_Focal_loss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_focal_dice(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        
        self.pin_memory = True
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})



