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

# CUDA_VISIBLE_DEVICES=1 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 3 4 5 6 -t 901 
# CUDA_VISIBLE_DEVICES=1 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 3 4 5 6 7 8 -t 901 
# CUDA_VISIBLE_DEVICES=2 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1 -tr nnUNetTrainerV2_dropout -f 0 1 2 3 4 -t 901 --snapshots 1100 1200 1300 1400 1500 

# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 -t 901 --disable_tta 
# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 3 4 5 6 -t 901 --disable_tta 
# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 3 4 5 6 7 8 9 -t 901 --disable_tta 
#CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2__nnUNetPlansv2.1 -tr nnUNetTrainerV2 -f 0 1 2 3 4 5 6 7 8  -t 901 --disable_tta 

# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0 -umap $NNUNET_UMAP/f0 -error $NNUNET_ER/f0


########################## genearte uncertainty maps ########################
##PHISEG
# python ../src/uncertainty_map_entropy.py -in_folder $PH_PMAP/f01234_tta/
# python ../src/uncertainty_map_entropy.py -in_folder $PH_GAMM_PMAP/f0_tta/

# python ../src/uncertainty_map_entropy.py -in_folder $PH_PMAP/f0_tta/
# python ../src/uncertainty_map_entropy.py -in_folder $PH_GAMM_PMAP/f01234_tta/

##external
# python ../src/uncertainty_map_entropy.py -in_folder $EXT_PMAPS1/f012456_tta/
# python ../src/uncertainty_map_entropy.py -in_folder $EXT_PMAPS2/f0_mc10_tta/

# #nnunet
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f01234
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f01234_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f01234_tta_3mm
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012456789_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0123456_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012345678_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0123456789_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0123456_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012456_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012_tta
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0_tta

python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012345678
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0123456789
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0123456
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f01234
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f012
# python ../src/uncertainty_map_entropy.py -in_folder $NNUNET_PMAPS/f0
# # #dropout 0.1

# python ../src/uncertainty_map_entropy.py -in_folder $DROP1_PMAPS/f01234_mc10_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP1_PMAPS/f0_mc10_tta
# #dropout 0.3

# python ../src/uncertainty_map_entropy.py -in_folder $DROP3_PMAPS/f0_mc10_tta

# #dropout 0.5
# python ../src/uncertainty_map_entropy.py -in_folder $DROP5_PMAPS/f0_mc10_tta

# #dropout 0.2
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0123456789_mc10_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0123456789_mc10_tta_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f01234_mc10_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f01234_mc10_tta_3mm
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f01234_mc10_tta_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc10
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc10_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc10_testing
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc10_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f01234_tta_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc10_tta_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc15
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc15_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc5
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc5_tta
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_snap
# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_tta_snap

# python ../src/uncertainty_map_entropy.py -in_folder $DROP2_PMAPS/f0_mc20

########################## error region ########################

#EXT
# python ../src/error_region.py -gt $TASK2_RAW -seg $EXT_PMAPS1/f012456_tta
# python ../src/error_region.py -gt $TASK2_RAW -seg $EXT_PMAPS2/f0_mc10_tta


##PHISEG
# python ../src/error_region.py -gt $TASK1_RAW -seg $PH_PMAP/f01234_tta/
# python ../src/error_region.py -gt $TASK1_RAW -seg $PH_PMAP/f0_tta/

# python ../src/error_region.py -gt $TASK1_RAW -seg $PH_GAMM_PMAP/f01234_tta/
# python ../src/error_region.py -gt $TASK1_RAW -seg $PH_GAMM_PMAP/f0_tta/

#nnunet
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234_tta_3mm
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012456789_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012345678_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012456_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456789_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456_tta

python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012345678
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456789
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0123456
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f01234
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f012
# python ../src/error_region.py -gt $TASK1_RAW -seg $NNUNET_PMAPS/f0

# #dropout 0.1                  
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP1_PMAPS/f01234_mc10_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP1_PMAPS/f0_mc10_tta
# #dropout 0.3                  

# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP3_PMAPS/f0_mc10_tta
							  
# #dropout 0.5                 
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP5_PMAPS/f0_mc10_tta
						
# #dropout 0.2                  
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0123456789_mc10_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0123456789_mc10_tta_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta_3mm
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_mc10_tta_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_testing
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc10_tta_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc15
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc15_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc5
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc5_tta
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_tta_snap
# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f01234_tta_snap

# python ../src/error_region.py -gt $TASK1_RAW -seg $DROP2_PMAPS/f0_mc20

########################## genearte uncertainty regions and get FND/FPD ########################
#EXTER
# python ../src/uncertainty_region_class.py -seg $EXT_PMAPS1/f012456_tta -umap $EXT_UMAP1/f012456_tta -error $EXT_ER1/f012456_tta
# python ../src/uncertainty_region_class.py -seg $EXT_PMAPS2/f0_mc10_tta -umap $EXT_UMAP2/f012456_tta -error $EXT_ER2/f012456_tta

# #PHISEG
# python ../src/uncertainty_region_class.py -seg $PH_PMAP/f01234_tta -umap $PH_UMAP/f01234_tta -error $PH_ER/f01234_tta 
# python ../src/uncertainty_region_class.py -seg $PH_PMAP/f0_tta -umap $PH_UMAP/f0_tta -error $PH_ER/f0_tta 

# python ../src/uncertainty_region_class.py -seg $PH_GAMM_PMAP/f01234_tta/ -umap $PH_GAMM_UMAP/f01234_tta -error $PH_GAMM_ER/f01234_tta 
# python ../src/uncertainty_region_class.py -seg $PH_GAMM_PMAP/f0_tta/ -umap $PH_GAMM_UMAP/f0_tta -error $PH_GAMM_ER/f0_tta 
# #nnunet
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f01234 -umap $NNUNET_UMAP/f01234 -error $NNUNET_ER/f01234
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f01234_tta -umap $NNUNET_UMAP/f01234_tta -error $NNUNET_ER/f01234_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012456789_tta -umap $NNUNET_UMAP/f012456789_tta -error $NNUNET_ER/f012456789_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012345678_tta -umap $NNUNET_UMAP/f012345678_tta -error $NNUNET_ER/f012345678_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0123456_tta -umap $NNUNET_UMAP/f0123456_tta -error $NNUNET_ER/f0123456_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012456_tta -umap $NNUNET_UMAP/f012456_tta -error $NNUNET_ER/f012456_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012_tta -umap $NNUNET_UMAP/f012_tta -error $NNUNET_ER/f012_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0_tta -umap $NNUNET_UMAP/f0_tta -error $NNUNET_ER/f0_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0123456789_tta -umap $NNUNET_UMAP/f0123456789_tta -error $NNUNET_ER/f0123456789_tta
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0123456_tta -umap $NNUNET_UMAP/f0123456_tta -error $NNUNET_ER/f0123456_tta

python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012345678 -umap $NNUNET_UMAP/f012345678 -error $NNUNET_ER/f012345678
python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0123456789 -umap $NNUNET_UMAP/f0123456789 -error $NNUNET_ER/f0123456789
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0123456 -umap $NNUNET_UMAP/f0123456 -error $NNUNET_ER/f0123456
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f01234 -umap $NNUNET_UMAP/f01234 -error $NNUNET_ER/f01234
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f012 -umap $NNUNET_UMAP/f012 -error $NNUNET_ER/f012
# python ../src/uncertainty_region_class.py -seg $NNUNET_PMAPS/f0 -umap $NNUNET_UMAP/f0 -error $NNUNET_ER/f0

#dropout 0.1 
# python ../src/uncertainty_region_class.py -seg $DROP1_PMAPS/f01234_mc10_tta -umap $DROP1_UMAP/f01234_mc10_tta -error $DROP1_ER/f01234_mc10_tta 
# python ../src/uncertainty_region_class.py -seg $DROP1_PMAPS/f0_mc10_tta -umap $DROP1_UMAP/f0_mc10_tta -error $DROP1_ER/f0_mc10_tta
# # #dropout 0.3 

# python ../src/uncertainty_region_class.py -seg $DROP3_PMAPS/f0_mc10_tta -umap $DROP3_UMAP/f0_mc10_tta -error $DROP3_ER/f0_mc10_tta 
							 
# # #dropout 0.5 
# python ../src/uncertainty_region_class.py -seg $DROP5_PMAPS/f0_mc10_tta -umap $DROP5_UMAP/f0_mc10_tta -error $DROP5_ER/f0_mc10_tta 
						
# # #dropout 0.2 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0123456789_mc10_tta -umap $DROP2_UMAP/f0123456789_mc10_tta -error $DROP2_ER/f0123456789_mc10_tta 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0123456789_mc10_tta_snap -umap $DROP2_UMAP/f0123456789_mc10_tta_snap -error $DROP2_ER/f0123456789_mc10_tta_snap
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f01234_mc10_tta -umap $DROP2_UMAP/f01234_mc10_tta -error $DROP2_ER/f01234_mc10_tta 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f01234_mc10_tta_snap -umap $DROP2_UMAP/f01234_mc10_tta_snap -error $DROP2_ER/f01234_mc10_tta_snap 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc10 -umap $DROP2_UMAP/f0_mc10 -error $DROP2_ER/f0_mc10 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc10_snap -umap $DROP2_UMAP/f0_mc10_snap -error $DROP2_ER/f0_mc10_snap 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc10_testing -umap $DROP2_UMAP/f0_mc10_testing -error $DROP2_ER/f0_mc10_testing 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc10_tta -umap $DROP2_UMAP/f0_mc10_tta -error $DROP2_ER/f0_mc10_tta 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc10_tta_snap -umap $DROP2_UMAP/f0_mc10_tta_snap -error $DROP2_ER/f0_mc10_tta_snap 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc15 -umap $DROP2_UMAP/f0_mc15 -error $DROP2_ER/f0_mc15 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc15_tta -umap $DROP2_UMAP/f0_mc15_tta -error $DROP2_ER/f0_mc15_tta 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc5 -umap $DROP2_UMAP/f0_mc5 -error $DROP2_ER/f0_mc5 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc5_tta -umap $DROP2_UMAP/f0_mc5_tta -error $DROP2_ER/f0_mc5_tta 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_snap -umap $DROP2_UMAP/f0_snap -error $DROP2_ER/f0_snap 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_tta_snap -umap $DROP2_UMAP/f0_tta_snap -error $DROP2_ER/f0_tta_snap 
# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f01234_tta_snap -umap $DROP2_UMAP/f01234_tta_snap -error $DROP2_ER/f01234_tta_snap 

# python ../src/uncertainty_region_class.py -seg $DROP2_PMAPS/f0_mc20 -umap $DROP2_UMAP/f0_mc20 -error $DROP2_ER/f0_mc20 
