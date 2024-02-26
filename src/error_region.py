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

def error_region():
    """
    compute error region of segmentation predictions.

    """
    import argparse
    parser = argparse.ArgumentParser("Compute error region of segmentation predictions.")
    parser.add_argument('-gt', required=True, type=str, help="Folder containing the ground truth segmentations in "
                                                        ".nii.gz format")
    parser.add_argument('-seg', required=True, type=str, help="Folder containing the segmentation masks in "
                                                        ".nii.gz format")

    parser.add_argument('-out', required=False, type=str, help="Folder to store the error region masks in "
                                                        ".nii.gz format")
    args = parser.parse_args()

    # Create output folder if not provided
    if args.out is None:
        output_folder = args.seg.replace("prob_maps", "error_region")

    output_folders = [os.path.join(output_folder, "GTV-T"), os.path.join(output_folder, "GTV-N")]

    maybe_mkdir_p(output_folder)
    maybe_mkdir_p(output_folders[0])
    maybe_mkdir_p(output_folders[1])

    files_in = subfiles(args.seg, suffix=".nii.gz", join=False)

    
    #multithreading.
    p = Pool(24)
    p.map(compute_error_region, zip([args.gt]*len(files_in), [args.seg]*len(files_in), 
        [output_folders]*len(files_in), files_in))

    p.close()
    p.join()
    
    
    return 
def define_error(seg_arr_one, gt_arr_one):
    """
    Calculates the error between two arrays by subtracting `gt_arr_one` from `seg_arr_one`.
    If `seg_arr_one` and `gt_arr_one` have different channels, the array with fewer
    channels is padded with zeros before the subtraction is performed.
  
    Parameters:
    seg_arr_one (numpy array): The first array to compare.
    gt_arr_one (numpy array): The second array to compare.
    
    Returns:
    numpy array: The error between the two input arrays.
    """
    

    ##pad zero until segmentation and grount truth have same dimension.
    while seg_arr_one.shape[-1] != gt_arr_one.shape[-1]:
        print("seg - gt dimension disagree:", seg_arr_one.shape[-1], gt_arr_one.shape[-1])
        if gt_arr_one.shape[-1] > seg_arr_one.shape[-1]:
            seg_arr_one = np.concatenate((seg_arr_one, np.zeros_like(seg_arr_one[:, [0]])), axis=-1)
        else:
            gt_arr_one = np.concatenate((gt_arr_one, np.zeros_like(gt_arr_one[:, [0]])), axis=-1)

    if seg_arr_one.shape[-1] == 2: ## if both doesn't have GTV-N
            gt_arr_one = np.concatenate((gt_arr_one, np.zeros_like(gt_arr_one[:, [0]])), axis=-1)
            seg_arr_one = np.concatenate((seg_arr_one, np.zeros_like(seg_arr_one[:, [0]])), axis=-1)

    error = seg_arr_one - gt_arr_one
    error[error > 0] = 2 # false positives
    error[error < 0] = 1 # false negatives

    return error


def compute_error_region(arguments):
    """
    Compute the error region, namely the false negative and false positive regions, of the segmentation predictions.
    Requires the path to a folder containing ground truth segmentations in .nii.gz format and the path to a folder
    containing segmentation masks in .nii.gz format.
    If an output folder is not provided, a new folder for "error_region" will be created in the same directory as the
    segmentation folder.

    Args:
    gt_path: A string containing the path to the folder containing the ground truth segmentations in .nii.gz format.
    seg_path: A string containing the path to the folder containing the segmentation masks in .nii.gz format.
    output_folder: A string containing the path to the folder to store the error region masks in .nii.gz format.
        Default is None.
    """
    import os

    gt_path, seg_path, output_folders, file = arguments


    print("compute_error_region...", file)
    gt = join(gt_path,file)
    seg = join(seg_path, file)

    gt_img = sitk.ReadImage(gt)
    seg_img = sitk.ReadImage(seg)

    gt_arr = sitk.GetArrayFromImage(gt_img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    onehot_encoder = OneHotEncoder(sparse=False)
    gt_arr_one = gt_arr.reshape(gt_arr.shape[0]*gt_arr.shape[1]*gt_arr.shape[2], 1)
    gt_arr_one = onehot_encoder.fit_transform(gt_arr_one)

    seg_arr_one = seg_arr.reshape(seg_arr.shape[0]*seg_arr.shape[1]*seg_arr.shape[2], 1)
    seg_arr_one = onehot_encoder.fit_transform(seg_arr_one)


    error = define_error(seg_arr_one, gt_arr_one) 

    error = error.reshape((gt_arr.shape[0], gt_arr.shape[1], gt_arr.shape[2], error.shape[-1]))

    for c in range(1, error.shape[-1]):
        error_img = sitk.GetImageFromArray(error[:,:,:, c].astype(np.float32))
        error_img.CopyInformation(gt_img)
        sitk.WriteImage( error_img, join(output_folders[c-1], file))
            
    return 

if __name__ == "__main__":
    error_region()




# def entropy(p, normalize = True, combine_channels = True):
#     """
#     Calculates the entropy (uncertainty) of p
#     Args:
#         p (ndarray CxHxWxD ): probability per class
#         normalize (Bool): normalize of uncertainty value. Change base to rescale -H into 0.0 - 1.0
#     Returns:
#         ndarray CxHxWxD 
#     """
#     n_channels  = p.shape[0]
#     H = np.zeros_like(p)

#     if combine_channels:
#         mask = p > 0.000001 # escape for undefined log 0. 
#         h = np.zeros_like(p)
#         if normalize:
#             h[mask] = (np.log(p[mask])/ np.log(n_channels)) # changed base for multiple classes
#         else:
#             h[mask] = np.log(p[mask])

#         H = np.sum(p * h, axis=0)
#     else:
#         #separate channels, each target would have it's separate uncerainty maps
#         for c in range(n_channels):
#             mask = p[c] > 0.0001 # escape for undefined log 0. 

#             h = np.zeros_like(p[c])
#             if normalize:
#                 #print(np.log(p[c][mask].max()), p[c].max())
#                 h[mask] = (np.log(p[c][mask])/ np.log(n_channels))# changed base for multiple classes
#                 H[c] = p[c] * h * n_channels
#                 H[c] = np.clip(H[c], -1, 0)
#             else:
#                 h[mask] = np.log(p[c][mask])
#                 H[c] = p[c] * h * n_channels

#     return -H

# def evaluate_folder(input_folder: str, output_folder: str, save_nii : bool = True, combine_channels: bool = False):
#     """
#     writes a calib_summary.json to output_folder
#     :param input_folder: folder where the softmax probability files are saved. Must be npz/npy files.
#     :param output_folder: folder where the uncertainty maps will be saved.  
#     :return:
#     """
#     files_in = subfiles(input_folder, suffix=".npz", join=False)

#     if output_folder == "":
#         output_folder = input_folder.replace("prob_maps", "umaps")
#         maybe_mkdir_p(output_folder)

#         #output_files= [file.replace("prob_maps", "umaps") for file in files_in] 

#     for i, file in enumerate(files_in):
#         print("loading file: ", join(input_folder, file))
#         arr = np.load(join(input_folder, file), allow_pickle=True)
#         print("unwarpping softmax file: ", file)
#         arr = arr['softmax']
#         print("calculating umap for file: ", file)
#         umap_arr = entropy(arr, normalize = True, combine_channels=combine_channels)
#         np.savez_compressed(join(output_folder, file), umap=umap_arr.astype(np.float16))
#         if save_nii:
#             ref_img = sitk.ReadImage(join(input_folder, file.replace('.npz', '.nii.gz')))
#             if len(umap_arr.shape)>3: # if separate channcel for different targets.
#                 for c in range(umap_arr.shape[0]):
#                     umap_img = sitk.GetImageFromArray(umap_arr[c].astype(np.float32))
#                     umap_img.CopyInformation(ref_img)
#                     postfix = '_'+str(c)+'.nii.gz'
#                     sitk.WriteImage( umap_img, join(output_folder, file.replace('.npz', postfix)))
#             else:
#                 umap_img = sitk.GetImageFromArray(umap_arr.astype(np.float32))
#                 umap_img.CopyInformation(ref_img)
#                 sitk.WriteImage( umap_img, join(output_folder, file.replace('.npz', '.nii.gz')))
#     return

# def uncertainty_region_threasholding(umap_path):
#     uregion_path =  umap_path.replace("umaps", "uregion")

#     for i in range(1,10): # make directories for all 
#         maybe_mkdir_p(os.path.join(uregion_path, str(i)))

#### part 1 for error region