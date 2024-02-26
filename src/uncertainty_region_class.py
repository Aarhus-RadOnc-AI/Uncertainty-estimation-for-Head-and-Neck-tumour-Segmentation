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



class UncertaintyRegion:
    def __init__(self):
        pass
    def compute_u_region(self, arguments):
        """
        Computes the uncertainty error region between a uncertainty map and a segmentation map and saves the result to an output folder.
        
        The error map is calculated as follows:
            - False positives are assigned a value of 2
            - False negatives are assigned a value of 1
            - Union of False negatives and positives are assigned a value of 3
        Parameters:
        umap_path (str): The path to the directory containing the umap files.
        pmap_path (str): The path to the directory containing the pmap files.
        e_region_path (str): The path to the directory containing the error region files.
        work_path (str): The path to the directory where the output files will be saved.
        
        """
        umap_path, pmap_path, work_path = arguments
        print("...work_path is :", work_path)
        t = int(os.path.basename(work_path))/10

        output_folders = [os.path.join(work_path, "GTV-T"), os.path.join(work_path, "GTV-N")]
        maybe_mkdir_p(output_folders[0])
        maybe_mkdir_p(output_folders[1])


        files_in = subfiles(pmap_path, suffix=".nii.gz", join=False)    

        for file in files_in:
            
            patient = file.split(".nii.gz")[0]
            print(f"......working on patient {patient} with t: {t}")
            
            u_map = np.load(os.path.join(umap_path, patient+'.npz'))['umap']
            
            #gt_img = sitk.ReadImage(join(gt_path, file))
            seg_img = sitk.ReadImage(join(pmap_path, file))
            #err_img = sitk.ReadImage(join(e_region_path, file))

            #gt_arr = sitk.GetArrayFromImage(gt_img)
            seg_arr = sitk.GetArrayFromImage(seg_img)
            u_region_1label = np.transpose((u_map>t).astype(np.int8), (1,2,3,0))

            # no need to change to onehot if ucertainty region load with npz file.
            onehot_encoder = OneHotEncoder(sparse=False)
            seg_arr_onehot = seg_arr.reshape(seg_arr.shape[0]*seg_arr.shape[1]*seg_arr.shape[2], 1)
            seg_arr_onehot = onehot_encoder.fit_transform(seg_arr_onehot)

            u_region_1label_onehot = u_region_1label.reshape(u_region_1label.shape[0]*u_region_1label.shape[1]*u_region_1label.shape[2], u_region_1label.shape[-1])

            #print(f"!!debug u_region_1label shape: {u_region_1label_onehot.shape} ,  {seg_arr_onehot.shape}")

            while u_region_1label_onehot.shape[-1] != seg_arr_onehot.shape[-1]:
                if u_region_1label_onehot.shape[-1] < seg_arr_onehot.shape[-1]:
                    u_region_1label_onehot = np.concatenate((u_region_1label_onehot, np.zeros_like(u_region_1label_onehot[:, [0]])), axis=-1)
                else:
                    seg_arr_onehot = np.concatenate((seg_arr_onehot, np.zeros_like(seg_arr_onehot[:, [0]])), axis=-1)

            #u_region_1label_onehot =  u_region_1label_onehot + 1
            fp = u_region_1label_onehot * seg_arr_onehot
            fp[fp > 0] = 2
            fn = u_region_1label_onehot  - seg_arr_onehot 
            fn[fn < 0] = 0
            error = fn + fp

            u_error = error.reshape((seg_arr.shape[0], seg_arr.shape[1], seg_arr.shape[2], u_region_1label_onehot.shape[-1]))

            for c in range(1, error.shape[-1]):
                u_error_img = sitk.GetImageFromArray(u_error[:,:,:, c].astype(np.float32))
                u_error_img.CopyInformation(seg_img)
                sitk.WriteImage( u_error_img, join(output_folders[c-1], file))

    def save_uncertainty_region_nii(self,arguments):
        file, folder, new_folder = arguments
        img = sitk.ReadImage(join(folder, file))
        arr = sitk.GetArrayFromImage(img)
        arr[arr>0] = 1
        print(f'save_union_uncertainty_region_nii - Patient id: {file}')
        new_img = sitk.GetImageFromArray(arr)
        new_img.CopyInformation(img)

        sitk.WriteImage(new_img, join(new_folder, file), useCompression = True)
        return

    def union_FP_FN_as_one(self, folder, new_folder):
        files_in = subfiles(folder, suffix=".nii.gz", join=False)
        input_folders = [folder] *len(files_in)
        output_folder = [new_folder] *len(files_in)

        #multithreading to compute 
        p = Pool(128)
        p.map(self.save_uncertainty_region_nii, zip(files_in, input_folders, output_folder))
        p.close()
        p.join()

        return 

    def evaluate_uregion(self, error_path, uncertianty_region_path):
        
        results = {}
        for target in ["GTV-T", "GTV-N"]:
            results[target] = {}
            print(f"working on {target}")
            ref_folder = join(error_path, target)
            out_folder = join(uncertianty_region_path, target)
            #evaluation_command = f'nnUNet_evaluate_folder -ref {ref_folder}  -pred {out_folder} -l 1 2'
            print("---working on FND/FPD---")
            evaluation_command = f'nnUNet_evaluator_overlap -ref {ref_folder}  -pred {out_folder} -l 1 2'
            os.system(evaluation_command) 

            with open(join(out_folder, 'summary.json')) as file:
                jsonf_dict = json.load(file)
            results[target]["FND"] = np.round(jsonf_dict["results"]["mean"]["1"]["Dice"], 3)
            results[target]["FPD"] = np.round(jsonf_dict["results"]["mean"]["2"]["Dice"], 3)

            error_union_folder = join(ref_folder+'_union')
            u_error_union_folder = join(out_folder+'_union')
            maybe_mkdir_p(error_union_folder)
            maybe_mkdir_p(u_error_union_folder)

            # union both FP, FN into one uncertainty region.
            self.union_FP_FN_as_one(ref_folder, error_union_folder)
            self.union_FP_FN_as_one(out_folder, u_error_union_folder)

            print("---working on UED---")
            evaluation_command = f'nnUNet_evaluator_overlap -ref {error_union_folder}  -pred {u_error_union_folder} -l 1'
            os.system(evaluation_command) 

            with open(join(u_error_union_folder, 'summary.json')) as file:
                jsonf1_dict = json.load(file)
  
            results[target]["UED"] = np.round(jsonf1_dict["results"]["mean"]["1"]["Dice"], 3)

        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("-Arguments for get uncertainty region with false positive and false "
                                    "negative regions as two labels."
                                    "Both GTV-t/GTV-N were calcuated separated and stored in two folders."
                                    "-Take arguments of error region folder and uncertainty region folder."
                                    "error regions folder should have GTV-T/GTV/N as two separate folders."
                                    "uncertainty regions folder should containing all threshold folders")

    parser.add_argument('-error', required=True, type=str, help="folder containing the error regions")

    parser.add_argument('-u_region', required=False, type=str, help="father folder containing the u-region.")

    parser.add_argument('-umap', required=True, type=str, help="Folder containing the uncertainty map in .npz" 
                                                                "format.")
    parser.add_argument('-seg', required=True, type=str, help="Folder containing segmentation masks in .nii.gz.")


    args = parser.parse_args()

    if args.u_region == None:
        args.u_region = args.umap.replace('umap', 'u_region')


    uncertainty_obj = UncertaintyRegion()

    #multithreading to compute 10 different thresholds.
    p = Pool(10)
    work_paths = []
    for i in range(0, 10):
        work_paths.append(join(args.u_region, str(i)))
    umaps = [args.umap] * 10
    segs = [args.seg] * 10
    p.map(uncertainty_obj.compute_u_region, zip(umaps, segs, work_paths))
    p.close()
    p.join()


    #evaluate thresholds one by one with error region
    targets = ['GTV-T', 'GTV-N']
    dsc_scores = {}
    for i in range(0, 10):
        work_path = join(args.u_region, str(i))
        maybe_mkdir_p(work_path)
        #print("...working on path :", work_path)
        current_scores = uncertainty_obj.evaluate_uregion(args.error,  work_path)
        dsc_scores[i] = current_scores

    with open(join(args.u_region, "uregion_summary.json"), "w") as outfile:
        json.dump(dsc_scores, outfile)
    print(dsc_scores)

    # uncertainty_region()
    # evaluate_overlap_with_error()
    