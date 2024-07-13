#!/bin/bash
TASK1_RAW="/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs"

NNUNET_PMAPS="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"
DROP1_PMAPS="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout1__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"
DROP2_PMAPS="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"
DROP3_PMAPS="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout3__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"
DROP5_PMAPS="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout5__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"


NNUNET_UR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1/u_regions/Task901_AUH/imagesTs"
DROP1_UR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout1__nnUNetPlansv2.1/u_regions/Task901_AUH/imagesTs"
DROP2_UR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1/u_regions/Task901_AUH/imagesTs"
DROP3_UR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout3__nnUNetPlansv2.1/u_regions/Task901_AUH/imagesTs"
DROP5_UR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout5__nnUNetPlansv2.1/u_regions/Task901_AUH/imagesTs"


NNUNET_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"
DROP1_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout1__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"
DROP2_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"
DROP3_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout3__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"
DROP5_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout5__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"


NNUNET_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"
DROP1_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout1__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"
DROP2_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"
DROP3_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout3__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"
DROP5_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout5__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"



PH_PMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"
PH_GAMM_PMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs"

PH_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"
PH_GAMM_ER="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/error_region/Task901_AUH/imagesTs"

PH_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"
PH_GAMM_UMAP="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/umaps/Task901_AUH/imagesTs"


########################## error region ########################

# ##PHISEG
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $PH_PMAP/f01234_tta/
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $PH_PMAP/f0_tta/

# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $PH_GAMM_PMAP/f01234_tta/
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $PH_GAMM_PMAP/f0_tta/

# #nnunet
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012456789_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012345678_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012456_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456789_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456_tta
# # #dropout 0.1                  
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP1_PMAPS/f01234_mc10_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP1_PMAPS/f0_mc10_tta
# # #dropout 0.3                  

# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP3_PMAPS/f0_mc10_tta
							  
# # #dropout 0.5                 
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP5_PMAPS/f0_mc10_tta
						
# # #dropout 0.2                  
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0123456789_mc10_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0123456789_mc10_tta_snap
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta_3mm
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta_snap
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_snap
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_testing
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_tta_snap
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc15
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc15_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc5
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc5_tta
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_snap
# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_tta_snap

# python ../src/false_target_detect.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc20







# # external
# ##PHISEG
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $PH_PMAP_EXT/f01234_tta/
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $PH_PMAP_EXT/f0_tta/

# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $PH_GAMM_PMAP_EXT/f01234_tta/
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $PH_GAMM_PMAP_EXT/f0_tta/

# # #nnunet
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $NNUNET_PMAP_EXT/f01234_tta
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $NNUNET_PMAP_EXT/f0_tta
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $NNUNET_PMAP_EXT/f0
						
# # #dropout 0.2                  

# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $DROP2_PMAP_EXT/f0_mc10_tta

python ../src/false_target_detect.py -gt $TASK2_RAW -seg $DROP2_PMAP_EXT/f0_tta_snap 
# python ../src/false_target_detect.py -gt $TASK2_RAW -seg $DROP2_PMAP_EXT/f01234_mc10_tta_snap