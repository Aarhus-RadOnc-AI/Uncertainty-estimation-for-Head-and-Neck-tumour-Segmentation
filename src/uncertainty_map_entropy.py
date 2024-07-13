import matplotlib.pyplot as plt

import numpy as np
from utils import save_json, subfiles, join, maybe_mkdir_p
import os
import numpy as np
import os 
import glob
import json
import SimpleITK as sitk


def entropy(p, normalize = True, combine_channels = True):
    """
    Calculates the entropy (uncertainty) of p
    Args:
        p (ndarray CxHxWxD ): probability per class
        normalize (Bool): normalize of uncertainty value. Change base to rescale -H into 0.0 - 1.0
    Returns:
        ndarray CxHxWxD 
    """
    n_channels  = p.shape[0]
    H = np.zeros_like(p)

    if combine_channels:
        mask = p > 0.0001 # escape for undefined log 0. 
        h = np.zeros_like(p)
        if normalize:
            h[mask] = (np.log(p[mask])/ np.log(n_channels)) # changed base for multiple classes
        else:
            h[mask] = np.log(p[mask])

        H = np.sum(p * h, axis=0)
    else:
        #separate channels, each target would have it's separate uncerainty maps
        for c in range(n_channels):
            mask = p[c] > 0.0001 # escape for undefined log 0. 

            h = np.zeros_like(p[c])
            if normalize:
                #print(np.log(p[c][mask].max()), p[c].max())
                h[mask] = (np.log(p[c][mask])/ np.log(n_channels))# changed base for multiple classes
                H[c] = p[c] * h * n_channels
                H[c] = np.clip(H[c], -1, 0)
            else:
                h[mask] = np.log(p[c][mask])
                H[c] = p[c] * h * n_channels

    return -H

def evaluate_folder(input_folder: str, output_folder: str, save_nii : bool = True, combine_channels: bool = False):
    """
    writes a calib_summary.json to output_folder
    :param input_folder: folder where the softmax probability files are saved. Must be npz/npy files.
    :param output_folder: folder where the uncertainty maps will be saved.  
    :return:
    """
    files_in = subfiles(input_folder, suffix=".npz", join=False)

    if output_folder == "":
        output_folder = input_folder.replace("prob_maps", "umaps")
        maybe_mkdir_p(output_folder)

        #output_files= [file.replace("prob_maps", "umaps") for file in files_in] 

    for i, file in enumerate(files_in):
        print("loading file: ", join(input_folder, file))
        arr = np.load(join(input_folder, file), allow_pickle=True)
        print("unwarpping softmax file: ", file)
        arr = arr['softmax']
        print("calculating umap for file: ", file)
        umap_arr = entropy(arr, normalize = True, combine_channels=combine_channels)
        np.savez_compressed(join(output_folder, file), umap=umap_arr.astype(np.float16))
        if save_nii:
            ref_img = sitk.ReadImage(join(input_folder, file.replace('.npz', '.nii.gz')))
            if len(umap_arr.shape)>3: # if separate channcel for different targets.
                for c in range(umap_arr.shape[0]):
                    umap_img = sitk.GetImageFromArray(umap_arr[c].astype(np.float32))
                    umap_img.CopyInformation(ref_img)
                    postfix = '_'+str(c)+'.nii.gz'
                    sitk.WriteImage( umap_img, join(output_folder, file.replace('.npz', postfix)))
            else:
                umap_img = sitk.GetImageFromArray(umap_arr.astype(np.float32))
                umap_img.CopyInformation(ref_img)
                sitk.WriteImage( umap_img, join(output_folder, file.replace('.npz', '.nii.gz')))
    return

def calculate_entropy(arguments):
    """
    writes a calib_summary.json to output_folder
    :param input_folder: folder where the softmax probability files are saved. Must be npz/npy files.
    :param output_folder: folder where the uncertainty maps will be saved.  
    :return:
    """
    input_folder, file, output_folder = arguments

    print("loading file: ", join(input_folder, file))
    arr = np.load(join(input_folder, file), allow_pickle=True)
    print("unwarpping softmax file: ", file)
    arr = arr['softmax']
    print("calculating umap for file: ", file)
    umap_arr = entropy(arr, normalize = True, combine_channels=False)
    np.savez_compressed(join(output_folder, file), umap=umap_arr.astype(np.float16))
    ref_img = sitk.ReadImage(join(input_folder, file.replace('.npz', '.nii.gz')))
    if len(umap_arr.shape)>3: # if separate channcel for different targets.
        for c in range(umap_arr.shape[0]):
            umap_img = sitk.GetImageFromArray(umap_arr[c].astype(np.float32))
            umap_img.CopyInformation(ref_img)
            postfix = '_'+str(c)+'.nii.gz'
            sitk.WriteImage( umap_img, join(output_folder, file.replace('.npz', postfix)))

    return

def uncertainty_map_entropy():
    """
    uncertainty estimation main function
    including function of segmentation prediciton for masks and softmax probability maps.
    including function of uncertainty estimation based on entropy maps.
    including function of summary of patient level confidence prediciton.

-
    """
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser("Evaluates the segmentations located in the folder pred. Output of this script is "
                                     "a json file. At the very bottom of the json file is going to be a 'mean' "
                                     "entry with averages metrics across all cases")
    parser.add_argument('-in_folder', required=True, type=str, help="Folder containing the segmentations softmax prob in .npz "
                                                              "format.")
    parser.add_argument('-out_folder', required=False, type=str, default='', help="folder to hold the computed uncertainty "
                                                                "map in .npz and .nii.gz format")

    args = parser.parse_args()

    files_in = subfiles(args.in_folder, suffix=".npz", join=False)

    if args.out_folder == "":
        output_folder = args.in_folder.replace("prob_maps", "umaps")
        maybe_mkdir_p(output_folder)

    input_folders = [args.in_folder] *len(files_in)
    output_folder = [output_folder] *len(files_in)

    #multithreading to compute 10 different thresholds.
    p = Pool(32)
    p.map(calculate_entropy, zip(input_folders, files_in, output_folder))
    p.close()
    p.join()

    return 


if __name__ == "__main__":
    uncertainty_map_entropy()