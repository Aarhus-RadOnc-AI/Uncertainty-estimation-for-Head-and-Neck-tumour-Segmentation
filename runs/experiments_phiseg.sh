# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1 -tr nnUNetTrainerV2_PhiSeg -f 0 -t 901 
# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1 -tr nnUNetTrainerV2_PhiSeg -f 0 1 2 3 4 -t 901 

# python ../src/calibration_evaluator.py -ref /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/ -pred /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs/f0_tta/ -l 1 2
# python ../src/calibration_evaluator.py -ref /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/ -pred /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs/f01234_tta/ -l 1 2

#CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1 -tr nnUNetTrainerV2_PhiSeg_gamma -f 0 -t 901 
# CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1 -tr nnUNetTrainerV2_PhiSeg_gamma -f 0 1 2 3 4 -t 901 

CUDA_VISIBLE_DEVICES=0 python ../src/prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1 -tr nnUNetTrainerV2_PhiSeg_gamma -f 0 -t 901 

#python ../src/calibration_evaluator.py -ref /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/ -pred /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs/f0_tta/ -l 1 2

# python ../src/calibration_evaluator.py -ref /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/ -pred /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs/f01234_tta/ -l 1 2

python ../src/calibration_evaluator.py -ref /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/ -pred /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_PhiSeg_gamma__nnUNetPlansv2.1/prob_maps/Task901_AUH/imagesTs/f0_tta/ -l 1 2