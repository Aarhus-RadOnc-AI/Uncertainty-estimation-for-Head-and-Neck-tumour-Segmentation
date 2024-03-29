import argparse
import torch
import os 
import time
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.inference import predict_simple
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import numpy as np
import calibration_evaluator
from pathlib import Path

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def probability_map_prediction():
    """
    uncertainty estimation main function
    including function of segmentation prediciton for masks and softmax probability maps.
    including function of summary of patient level confidence prediciton.
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution '
                             'U-Net. The default is %s. If you are running inference with the cascade and the folder '
                             'pointed to by --lowres_segmentations does not contain the segmentation maps generated by '
                             'the low resolution U-Net then the low resolution segmentation maps will be automatically '
                             'generated. For this case, make sure to set the trainer class here that matches your '
                             '--cascade_trainer_class_name (this part can be ignored if defaults are used).'
                             % default_trainer,
                        required=False,
                        default=default_trainer)

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")


    parser.add_argument("--num_mc_dropout", required=False, default=0, type=int, help=
    "Number of dropout evaluation times with random monte carlo dropout. If set to 0, normal testing was performed") 

    parser.add_argument("--snapshots", nargs="+", required=False, default=[],  help=
    "Number of snapshots models to be predicted from. ")   

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    example: python prob_predictor.py  -i /data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs 
    -o /data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout__nnUNetPlansv2.1 
    -tr nnUNetTrainerV2_dropout -t 901 --num_mc_dropout 5 --disable_tta --snapshots 800 1000
    """
    args = probability_map_prediction()
    task_name = args.task_name
    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)


    source = args.input_folder.split('nnUNet_raw_data/')[1].replace('//', '_')
    out_folder = os.path.join(args.output_folder,  'prob_maps', source)
    if args.folds == 'None':
        init_command = f"nnUNet_predict -i {args.input_folder} -o out_folder -tr {args.trainer_class_name}  -t {args.task_name} -m 3d_fullres -z"
    else:
        init_command = f"nnUNet_predict -i {args.input_folder} -o out_folder -tr {args.trainer_class_name}  -t {args.task_name} -f "+' '.join(args.folds) + " -m 3d_fullres -z"

    task_name = 'f'+''.join(args.folds)
    if args.num_mc_dropout>0:
        task_name+=('_mc'+str(args.num_mc_dropout))
        init_command+=(f' --num_mc_dropout {args.num_mc_dropout}')
    if not args.disable_tta:
        task_name+=('_tta')
    else:
        init_command+=(' --disable_tta')

    if len(args.snapshots)>0:
        task_name+=('_snap')
        init_command+=(f' --snapshots '+ ' '.join(args.snapshots))
    print("prediction tasks : ", task_name)

    out_folder = os.path.join(out_folder, task_name)
    os.makedirs(out_folder, exist_ok=True)

    init_command = init_command.replace('out_folder', out_folder)
    
    print(init_command)

    os.system(init_command) 
#
    #Path(os.path.join(out_folder, 'time_cost'+ str(round(minutes))+'mins')).touch()

    ref_folder = args.input_folder.replace('images', 'labels')
    
    #carlibrate evaluation results
    calibration_evaluator.evaluate_folder(ref_folder, out_folder, [1,2])

    #segmentation evaluation results
    evaluation_command = f'nnUNet_evaluate_folder -ref {ref_folder}  -pred {out_folder} -l 1 2'
    os.system(evaluation_command) 
