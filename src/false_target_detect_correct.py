import matplotlib.pyplot as plt

import numpy as np
from utils import save_json, subfiles, join, maybe_mkdir_p
import os
import numpy as np
import os 
import glob
import json
import SimpleITK as sitk
from sklearn.preprocessing import OneHotEncoder
from multiprocessing.pool import Pool
import json
import argparse
from multiprocessing import Pool
from scipy import ndimage
from skimage import measure

from collections import OrderedDict


def calculate_overlap(mask1, mask2):
    """Calculates the Dice similarity coefficient between two regions."""
    intersection = np.logical_and(mask1, mask2)
    dice = 2 * intersection.sum() / (mask1.sum() + mask2.sum())
    return dice


def find_false_targets(seg_mask, gt_mask):

    "find distictive false target for both GTV-T and GTV-N from segmentation mask and ground truth mask"

    assert seg_mask.shape == gt_mask.shape, "Segmentation mask and ground truth mask have different shapes"

    # Split masks into GTV-T and GTV-N masks
    gt_t_mask = (gt_mask == 1).astype(np.uint8)
    gt_n_mask = (gt_mask == 2).astype(np.uint8)

    seg_t_mask = (seg_mask == 1).astype(np.uint8)
    seg_n_mask = (seg_mask == 2).astype(np.uint8)

    # Find connected targets in GT masks
    gt_t_labels, gt_t_count = measure.label(gt_t_mask, return_num=True)
    gt_n_labels, gt_n_count = measure.label(gt_n_mask, return_num=True)

    # Find connected targets in segmentation masks
    seg_t_labels, seg_t_count = measure.label(seg_t_mask, return_num=True)
    seg_n_labels, seg_n_count = measure.label(seg_n_mask, return_num=True)

    # Calculate false positives in segmentation masks
    false_pos_t_mask = np.zeros_like(seg_t_mask)
    false_pos_n_mask = np.zeros_like(seg_n_mask)

    for i in range(1, seg_t_count+1):
        overlap = False
        for j in range(1, gt_t_count+1):
            gt_t_target = (gt_t_labels == j).astype(np.uint8)
            overlap = overlap or calculate_overlap(gt_t_target, (seg_t_labels == i).astype(np.uint8)) > 0.0
        if not overlap:
            false_pos_t_mask[seg_t_labels == i] = 2

    for i in range(1, seg_n_count+1):
        overlap = False
        for j in range(1, gt_n_count+1):
            gt_n_target = (gt_n_labels == j).astype(np.uint8)
            overlap = overlap or calculate_overlap(gt_n_target, (seg_n_labels == i).astype(np.uint8)) > 0.0
        if not overlap:
            false_pos_n_mask[seg_n_labels == i] = 2

    # Calculate false negatives in GT masks
    false_neg_t_mask = np.zeros_like(gt_t_mask)
    false_neg_n_mask = np.zeros_like(gt_n_mask)

    for i in range(1, gt_t_count+1):
        overlap = False
        for j in range(1, seg_t_count+1):
            seg_t_target = (seg_t_labels == j).astype(np.uint8)
            overlap = overlap or calculate_overlap(seg_t_target, (gt_t_labels == i).astype(np.uint8)) > 0.0
        if not overlap:
            false_neg_t_mask[gt_t_labels == i] = 1

    for i in range(1, gt_n_count+1):
        overlap = False
        for j in range(1, seg_n_count+1):
            seg_n_target = (seg_n_labels == j).astype(np.uint8)
            overlap = overlap or calculate_overlap(seg_n_target, (gt_n_labels == i).astype(np.uint8)) > 0.0
        if not overlap:
            false_neg_n_mask[gt_n_labels == i] = 1

    false_target_t = false_neg_t_mask + false_pos_t_mask
    false_target_n = false_neg_n_mask + false_pos_n_mask
    #print(false_target_t.max() , false_target_n.max() )

    assert false_target_t.max() < 3, "ERROR: FP target and FN overlaps for GTV-T"
    assert false_target_n.max() < 3, "ERROR: FP target and FN overlaps for GTV-N"
    
    return false_target_t, false_target_n

def construct_new_seg_with_uncertianty(segmentation, uncertainty):
    new_seg = np.zeros_like()
    return new_seg

def calculate_detection(false_target, uncertainty_region_mask, exclude_small_targets = False):
    """
    This function calculates the detection rate for false targets given an uncertainty region mask.

    arguments:
    false_target: a binary mask containing the false targets
    uncertainty_region_mask: a binary mask containing the uncertainty region
    The function labels the connected components in the false target mask and calculates the detection rate for each connected component.

    returns:
    detect_count: the number of false targets that intersect with the uncertainty region
    labels[-1]: the total number of false targets
    """
    assert false_target.shape == uncertainty_region_mask.shape, "false_target and uncertainty_region_mask have different shapes"
    ##init detected regions as zeros
    detected_regions = np.zeros_like(false_target)

    labeled, n_labels = ndimage.label(false_target)
    labels = np.arange(1, n_labels+1)

    # Calculate the detection rate for each target
    detection_rate = 0
    detect_count = 0
    small_targets = 0 
    if len(labels) >0 :
        for label in labels:

            # Create a mask for the current target
            target_mask = labeled == label
            if exclude_small_targets:
                if np.sum(target_mask) < 100: ## debug purpose, currently not included in the QA process.
                    small_targets+=1
                else:
                    # Calculate the overlap between the uncertainty region and the target
                    overlap = calculate_overlap(uncertainty_region_mask.flatten(), target_mask.flatten())
                    if overlap > 0:
                        detect_count += 1
                        ## Set the detected target to positve using false targets (originally from ground truth)
                        detected_regions[target_mask] = 1
            else:
                # Calculate the overlap between the uncertainty region and the target
                overlap = calculate_overlap(uncertainty_region_mask.flatten(), target_mask.flatten())
                if overlap > 0:
                    detect_count += 1
                    ## Set the detected target to positve using false targets (originally from ground truth)
                    detected_regions[target_mask] = 1
            
        # Calculate the detection rate for the target
        detection_rate = detect_count / len(labels)
        num_labels = labels[-1] - small_targets

    else:
        # no targets found in the false targets
        detection_rate = np.nan
        num_labels = 0
        detect_count = 0
    #num_labels = num_labels - small_targets
    return detect_count, num_labels, detected_regions


def false_target_detection_and_repair(arguments):
    filename, seg_path, gt_path, uncertainty_region_path, repaied_path , threasholds = arguments 

    results = {}
    patient = filename.split(".nii.gz")[0]
    results['PatientID'] = patient
    seg_img = sitk.ReadImage(join(seg_path,filename))
    seg_mask = sitk.GetArrayFromImage(seg_img)
    gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(join(gt_path,filename)))

    ##init repair mask as the seg mask
    repair_mask = seg_mask


    uregion_path_t = join(uncertainty_region_path,str(threasholds),'GTV-T_union', filename)
    uregion_path_n = join(uncertainty_region_path,str(threasholds),'GTV-N_union', filename)

    print('GTV-T Uncertainty region taking from : ', uregion_path_t)
    print('GTV-N Uncertainty region taking from : ', uregion_path_n)

    uncertainty_region_mask_t = sitk.GetArrayFromImage(sitk.ReadImage(uregion_path_t))
    uncertainty_region_mask_n = sitk.GetArrayFromImage(sitk.ReadImage(uregion_path_n)) 
    
    false_target_t, false_target_n = find_false_targets(seg_mask, gt_mask)

    false_neg_t = (false_target_t == 1).astype(np.uint8)
    false_pos_t = (false_target_t == 2).astype(np.uint8)

    false_neg_n = (false_target_n == 1).astype(np.uint8)
    false_pos_n = (false_target_n == 2).astype(np.uint8)

    detected_num_neg_t, number_fnt_t, detected_regions_t_fn = calculate_detection(false_neg_t, uncertainty_region_mask_t)
    detected_num_pos_t, number_fpt_t, detected_regions_t_fp= calculate_detection(false_pos_t, uncertainty_region_mask_t)
    detected_num_neg_n, number_fnt_n, detected_regions_n_fn = calculate_detection(false_neg_n, uncertainty_region_mask_n)
    detected_num_pos_n, number_fpt_n, detected_regions_n_fp = calculate_detection(false_pos_n, uncertainty_region_mask_n)

    # Set the values of the repaired mask based on the false positive and false negative masks
    repair_mask[detected_regions_t_fp == 1] = 0
    repair_mask[detected_regions_n_fp == 1] = 0
    repair_mask[detected_regions_t_fn == 1] = 1
    repair_mask[detected_regions_n_fn == 1] = 2

    #convert repair mask to sitk images and write image to repair location
    repair_img = sitk.GetImageFromArray(repair_mask)
    repair_img.CopyInformation(seg_img)
    sitk.WriteImage(repair_img, os.path.join(repaied_path, filename))

    results['GTV-T FN detected'] = detected_num_neg_t # GTV=T false negative detection number
    results['GTV-T FP detected'] = detected_num_pos_t # GTV=T false positive detection number
    results['GTV-N FN detected'] = detected_num_neg_n # GTV=N false negative detection number
    results['GTV-N FP detected'] = detected_num_pos_n # GTV=N false negative detection number

    results['GTV-T FN numbers'] = number_fnt_t # GTV=T false negative target numbers
    results['GTV-T FP numbers'] = number_fpt_t # GTV=T false positive target numbers
    results['GTV-N FN numbers'] = number_fnt_n # GTV=N false negative target numbers
    results['GTV-N FP numbers'] = number_fpt_n # GTV=N false negative target numbers

    results['GTV-T FN vol.'] = np.sum(detected_regions_t_fn) # GTV=T false negative target volume
    results['GTV-T FP vol.'] = np.sum(detected_regions_t_fp) # GTV=T false positive target volume
    results['GTV-N FN vol.'] = np.sum(detected_regions_n_fn) # GTV=N false negative target volume
    results['GTV-N FP vol.'] = np.sum(detected_regions_n_fp) # GTV=N false negative target volume

    return  results

def evalution_function_false_target_detect():
    """
    detect false targets using uncertainty map. 

    """
    import argparse
    parser = argparse.ArgumentParser("Compute error region of segmentation predictions.")
    parser.add_argument('-gt', required=True, type=str, help="Folder containing the ground truth segmentations in "
                                                        ".nii.gz format")

    parser.add_argument('-seg', required=True, type=str, help="Folder containing the segmentation masks in "
                                                        ".nii.gz format")

    parser.add_argument('-uregions', required=False, type=str, help="father folder containing the u-region")

    parser.add_argument('-repair', required=False, default =None, type=str, help="Folder to save the repaired segmentation masks in "
                                                        ".nii.gz format")

    parser.add_argument('-th', required=False, type=int, default=7,  help="threshold") 

    args = parser.parse_args()
    if args.repair == None:
        args.repair = args.seg.replace('prob_maps', 'repaired')

    maybe_mkdir_p(args.repair)

    files_in = subfiles(args.seg, suffix=".nii.gz", join=False)

    # multithreading to find false targets, repaire false targets
    arugments = zip(files_in,  [args.seg]*len(files_in), [args.gt]*len(files_in),   [args.uregions]*len(files_in), [args.repair]*len(files_in),  [args.th]*len(files_in))
    p = Pool(128)
    results = p.map(false_target_detection_and_repair, arugments)
    p.close()
    p.join()
    # record false target detection rates
    all_scores = OrderedDict()
    all_scores["all"] = []

    for i in range(len(results)):
        all_scores["all"].append(results[i])
    with open(os.path.join(args.repair, 'false_detection_summary.json'), 'w') as f:
        json.dump(all_scores, f, cls=NpEncoder)

    # evaluate the repaired segmentation socres
    evaluation_command = f'nnUNet_evaluator_overlap -ref {args.gt}  -pred {args.repair} -l 1 2'
    os.system(evaluation_command) 
    print(results)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    """
    detect false targets using uncertainty map. 

    """
    import argparse
    parser = argparse.ArgumentParser("Compute error region of segmentation predictions.")
    parser.add_argument('-gt', required=True, type=str, help="Folder containing the ground truth segmentations in "
                                                        ".nii.gz format")

    parser.add_argument('-seg', required=True, type=str, help="Folder containing the segmentation masks in "
                                                        ".nii.gz format")

    parser.add_argument('-uregions', required=False,  default = None, type=str, help="father folder containing the u-region")

    parser.add_argument('-repair', required=False, default = None, type=str, help="Folder to save the repaired segmentation masks in "
                                                        ".nii.gz format")

    parser.add_argument('-th', required=False, type=int, default=9,  help="threshold") 

    args = parser.parse_args()
    if args.repair == None:
        args.repair = args.seg.replace('prob_maps', 'repaired')

    if args.uregions == None:
        args.uregions = args.seg.replace('prob_maps', 'u_regions')

    maybe_mkdir_p(args.repair)

    files_in = subfiles(args.seg, suffix=".nii.gz", join=False)

    # multithreading to find false targets, repaire false targets
    arugments = zip(files_in,  [args.seg]*len(files_in), [args.gt]*len(files_in),   [args.uregions]*len(files_in), [args.repair]*len(files_in),  [args.th]*len(files_in))
    p = Pool(128)
    results = p.map(false_target_detection_and_repair, arugments)
    p.close()
    p.join()
    # record false target detection rates
    all_scores = OrderedDict()
    all_scores["all"] = []
    all_scores["accumulate"] = {}
    #print(results)

    fp_t_detected = 0
    fp_n_detected = 0
    fn_t_detected = 0
    fn_n_detected = 0

    fp_t_total = 0
    fp_n_total = 0
    fn_t_total = 0
    fn_n_total = 0

    for i in range(len(results)):
        all_scores["all"].append(results[i])
        fn_t_detected +=results[i]['GTV-T FN detected'] 
        fp_t_detected +=results[i]['GTV-T FP detected'] 
        fn_n_detected +=results[i]['GTV-N FN detected'] 
        fp_n_detected +=results[i]['GTV-N FP detected'] 

        fn_t_total +=results[i]['GTV-T FN numbers'] 
        fp_t_total +=results[i]['GTV-T FP numbers'] 
        fn_n_total +=results[i]['GTV-N FN numbers'] 
        fp_n_total +=results[i]['GTV-N FP numbers'] 


    #
    all_scores["accumulate"]['GTV-T FN detected'] = fn_t_detected
    all_scores["accumulate"]['GTV-T FP detected'] = fp_t_detected
    all_scores["accumulate"]['GTV-N FN detected'] = fn_n_detected
    all_scores["accumulate"]['GTV-N FP detected'] = fp_n_detected

    all_scores["accumulate"]['GTV-T FN numbers'] = fn_t_total
    all_scores["accumulate"]['GTV-T FP numbers'] = fp_t_total
    all_scores["accumulate"]['GTV-N FN numbers'] = fn_n_total
    all_scores["accumulate"]['GTV-N FP numbers'] = fp_n_total

    with open(os.path.join(args.repair, 'false_detection_summary.json'), 'w') as f:
        json.dump(all_scores, f, cls=NpEncoder)

    # evaluate the repaired segmentation socres
    evaluation_command = f'nnUNet_evaluate_folder -ref {args.gt}  -pred {args.repair} -l 1 2'
    os.system(evaluation_command) 
    print(len(results))