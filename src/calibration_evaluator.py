

import collections
import inspect
import json
import hashlib
from datetime import datetime
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.metrics as metrics
from sklearn.metrics import log_loss, brier_score_loss
from metrics import *

from scipy import ndimage
from scipy import stats

from utils import save_json, subfiles, join

from collections import OrderedDict

ALL_METRICS = {
    "Expected Calibration Error": ECE, 
    "Maximal Calibration Error": MCE, #not implemented.
    "Negative Log Likelihood": NLL, ##not tested.
    "Brier Score": brier_score
}



class Evaluator:
    """
    Object that holds test probability map and reference segmentations with label information
    and computes a number of calibration metrics on the two. 
    
    'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.

    """

    default_metrics = [
    "Expected Calibration Error",
    #"Brier Score by Label",
    "Brier Score",
    #"Maximal Calibration Error",
    #"Negative Log Likelihood"
    ]

    default_advanced_metrics = [

    ]

    def __init__(self,
                 test=None,
                 reference=None,
                 labels=None,
                 metrics=None,
                 advanced_metrics=None,
                 nan_for_nonexisting=True):

        self.test = None
        self.reference = None
        #self.confusion_matrix = ConfusionMatrix()
        self.labels = None
        self.nan_for_nonexisting = nan_for_nonexisting
        self.result = None

        self.ROI = None


        self.metrics = []
        if metrics is None:
            for m in self.default_metrics:
                self.metrics.append(m)
        else:
            for m in metrics:
                self.metrics.append(m)

        self.advanced_metrics = []
        if advanced_metrics is None:
            for m in self.default_advanced_metrics:
                self.advanced_metrics.append(m)
        else:
            for m in advanced_metrics:
                self.advanced_metrics.append(m)

        self.set_reference(reference)
        self.set_test(test)
        if labels is not None:
            self.set_labels(labels)
        else:
            if test is not None and reference is not None:
                self.construct_labels()

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_ROI(self):
        """
        Set region of interest (ROI) including all targets across all channels. 
        
        """
        roi_test = np.argmax(self.test, axis=1)
        roi_test = roi_test * (roi_test>0.1).astype(int)
        roi_ref = self.reference>0
        # print("np.sum(roi_test), np.sum(roi_ref)", np.sum(roi_test), np.sum(roi_ref))
        # print("shapes", roi_test.shape,roi_ref.shape)

        assert (roi_test.shape==roi_ref.shape), 'ROI of reference and test not match!'
        self.ROI = (roi_test+roi_ref)>0
        print("ROI shapes before", self.ROI.shape, np.sum(self.ROI))
        roi_ref = ndimage.binary_dilation(roi_ref, iterations=4)

        self.ROI = (roi_test+roi_ref)>0
        print("ROI shapes after", self.ROI.shape, np.sum(self.ROI))

        #self.ROI = ndimage.binary_fill_holes(self.ROI)
        #self.ROI = ndimage.binary_dilation(self.ROI, iterations=4)
        #print("ROI shapes", self.ROI.shape, np.sum(self.ROI))

    def set_labels(self, labels):
        """Set the labels.
        :param labels= may be a dictionary (int->str), a set (of ints), a tuple (of ints) or a list (of ints). Labels
        will only have names if you pass a dictionary"""

        if isinstance(labels, dict):
            self.labels = collections.OrderedDict(labels)
        elif isinstance(labels, set):
            self.labels = list(labels)
        elif isinstance(labels, np.ndarray):
            self.labels = [i for i in labels]
        elif isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            raise TypeError("Can only handle dict, list, tuple, set & numpy array, but input is of type {}".format(type(labels)))

    def construct_labels(self):
        """Construct label set from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        elif self.test is None:
            labels = np.unique(self.reference)
        else:
            labels = np.union1d(np.unique(self.test),
                                np.unique(self.reference))
        self.labels = list(map(lambda x: int(x), labels))

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        if isinstance(metrics, set):
            self.metrics = list(metrics)
        elif isinstance(metrics, (list, tuple, np.ndarray)):
            self.metrics = metrics
        else:
            raise TypeError("Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None, normalize=False, **metric_kwargs):
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)

        if reference is not None:
            self.set_reference(reference)

        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")

        if self.labels is None:
            self.construct_labels()
        #self.reference[500:100000]=1 make a fake reference.
        #self.set_ROI()
        self.metrics.sort()

        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: ALL_METRICS[m] for m in self.metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))

        # get results
        self.result = OrderedDict()

        eval_metrics = self.metrics

        if normalize:
            confs = np.max(self.test, axis=1)/np.sum(self.test, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(self.test, axis=1)  # Take only maximum confidence

        preds = np.argmax(self.test, axis=1) # Take maximum confidence as prediction

        # use if label has names.
        # if isinstance(self.labels, dict):

        #     for label, name in self.labels.items():
        #         k = str(name)
        #         self.result[k] = OrderedDict()
                
        #         # if not hasattr(label, "__iter__"):
        #         #     # self.confusion_matrix.set_test(self.test == label)
        #         #     # self.confusion_matrix.set_reference(self.reference == label)
        #         # else:

        #         current_test = 0
        #         current_reference = 0
        #         for l in label:
        #             current_test += (self.test == l)
        #             current_reference += (self.reference == l)
        #             # self.confusion_matrix.set_test(current_test)
        #             # self.confusion_matrix.set_reference(current_reference)
        #         for metric in eval_metrics:
        #             self.result[k][metric] = _funcs[metric](confs, pred, true)

        #else:

        from sklearn.preprocessing import OneHotEncoder
        ## covnert reference mask back to one-hot array.
        onehot_encoder = OneHotEncoder(sparse=False)
        self.reference = self.reference.reshape(len(self.reference), 1)
        self.reference = onehot_encoder.fit_transform(self.reference)

        for i, l in enumerate(self.labels):
            k = str(l)
            print(l," - calculating labeling...")
            self.result[k] = OrderedDict()
            for metric in eval_metrics:
                #if i ==0: 
                self.result[k][metric] = _funcs[metric](conf=confs, pred=preds, 
                    true=self.reference, prob=self.test, label=l)
                print(metric, self.result[k][metric])
                #else:
                    # if metric =="Brier Score by Label":
                    #     self.result[k][metric] = _funcs[metric](conf=confs, pred=preds, 
                    #     true=self.reference, prob=self.test, label=l)
                    # else:
                    #self.result[k][metric] = self.result[str(int(k)-1)][metric]
                        
        #print(self.result)
        return self.result

    def to_dict(self):

        if self.result is None:
            self.evaluate()
        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        a = np.zeros((len(self.labels), len(result_metrics)), dtype=np.float32)

        if isinstance(self.labels, dict):
            for i, label in enumerate(self.labels.keys()):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[self.labels[label]][metric]
        else:
            for i, label in enumerate(self.labels):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[label][metric]

        return a

    def to_pandas(self):
        """Return result as pandas DataFrame."""

        a = self.to_array()

        if isinstance(self.labels, dict):
            labels = list(self.labels.values())
        else:
            labels = self.labels

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        return pd.DataFrame(a, index=labels, columns=result_metrics)


class ComplexEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):

        self.test_arr = None
        self.reference_nifti = None
        super(ComplexEvaluator, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test softmax npy/npz probability map."""
        if test is not None:
            print(test)
            self.test_arr = np.load(test)['softmax'] #Read softmax probs sitk.ReadImage(test) 
            self.test_arr = np.swapaxes(self.test_arr.reshape(self.test_arr.shape[0], -1),0,1) # change to channel last
            super(ComplexEvaluator, self).set_test(self.test_arr)
        else:
            super(ComplexEvaluator, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            super(ComplexEvaluator, self).set_reference(sitk.GetArrayFromImage(self.reference_nifti).flatten())
        else:
            self.reference_nifti = None
            super(ComplexEvaluator, self).set_reference(reference)

    def evaluate(self, test=None, reference=None):

        return super(ComplexEvaluator, self).evaluate(test, reference)


def run_evaluation(args):
    test, ref, evaluator = args
    # evaluate
    evaluator.set_test(test)
    evaluator.set_reference(ref)
    if evaluator.labels is None:
        evaluator.construct_labels()
    current_scores = evaluator.evaluate()
    if type(test) == str:
        current_scores["test"] = test
    if type(ref) == str:
        current_scores["reference"] = ref
    return current_scores

def confidence_interval(n, mean, std, confidence=0.95):
    h = std * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean-h, mean+h

def aggregate_scores(test_ref_pairs,
                     evaluator=ComplexEvaluator,
                     labels=None,
                     nanmean=True,
                     json_output_file=None,
                     json_name="",
                     json_description="",
                     json_task="",
                     num_threads=2,
                     ):
    """
    test = predicted image
    :param test_ref_pairs:
    :param evaluator:
    :param labels: must be a dict of int-> str or a list of int
    :param nanmean:
    :param json_output_file:
    :param json_name:
    :param json_description:
    :param json_task:
    :param metric_kwargs:
    :return:
    """

    if type(evaluator) == type:
        evaluator = evaluator()

    if labels is not None:
        evaluator.set_labels(labels)

    all_scores = OrderedDict()
    all_scores["all"] = []
    all_scores["mean"] = OrderedDict()
    all_scores["std"] = OrderedDict()

    test = [i[0] for i in test_ref_pairs]
    ref = [i[1] for i in test_ref_pairs]
    if num_threads == 1:
        all_res= []
        for test_one, ref_one, evaluator_one in zip(test, ref, [evaluator]*len(ref)):
            all_res.append(run_evaluation([test_one, ref_one, evaluator_one]))
    else:
        p = Pool(num_threads)
        all_res = p.map(run_evaluation, zip(test, ref, [evaluator]*len(ref)))
        p.close()
        p.join()

    for i in range(len(all_res)):
        all_scores["all"].append(all_res[i])

        # append score list for mean
        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["mean"]:
                all_scores["mean"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score not in all_scores["mean"][label]:
                    all_scores["mean"][label][score] = []
                all_scores["mean"][label][score].append(value)

            if label not in all_scores["std"]:
                all_scores["std"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score not in all_scores["std"][label]:
                    all_scores["std"][label][score] = []
                all_scores["std"][label][score].append(value)


    for label in all_scores["mean"]:
        for score in all_scores["mean"][label]:
            if nanmean:
                all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))
            else:
                all_scores["mean"][label][score] = float(np.mean(all_scores["mean"][label][score]))

    for label in all_scores["std"]:
        for score in all_scores["std"][label]:
            if nanmean:
                all_scores["std"][label][score] = float(stats.sem(all_scores["std"][label][score], nan_policy='omit'))
            else:
                all_scores["std"][label][score] = float(stats.sem(all_scores["std"][label][score]))

    # save to file if desired
    # we create a hopefully unique id by hashing the entire output dictionary
    if json_output_file is not None:
        json_dict = OrderedDict()
        json_dict["name"] = json_name
        json_dict["description"] = json_description
        timestamp = datetime.today()
        json_dict["timestamp"] = str(timestamp)
        json_dict["task"] = json_task
        json_dict["results"] = all_scores
        json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
        save_json(json_dict, json_output_file)


    return all_scores


def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, labels: tuple, **metric_kwargs):
    """
    writes a calib_summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False)
    files_pred = subfiles(folder_with_predictions, suffix=".npz", join=False)
    
    assert all([i.replace(".nii.gz", ".npz") in files_pred for i in files_gt]), "files missing in folder_with_predictions"
    assert all([i.replace(".npz",".nii.gz") in files_gt for i in files_pred]), "files missing in folder_with_gts"

    test_ref_pairs = [(join(folder_with_predictions, i), join(folder_with_gts, i.replace(".npz",".nii.gz"))) for i in files_pred]
    res = aggregate_scores(test_ref_pairs, json_output_file=join(folder_with_predictions, "calib_summary.json"),
                           num_threads=40, labels=labels, **metric_kwargs)
    return res

import os
def calibaration_evaluate_folder():
    import argparse
    parser = argparse.ArgumentParser("Evaluates the segmentations located in the folder pred. Output of this script is "
                                     "a json file. At the very bottom of the json file is going to be a 'mean' "
                                     "entry with averages metrics across all cases")
    parser.add_argument('-ref', required=True, type=str, help="Folder containing the reference segmentations in nifti "
                                                              "format.")
    parser.add_argument('-pred', required=True, type=str, help="Folder containing the predicted softmax prob.maps in numpy array (.npz/.npy)"
                                                               "format. File names must match between the folders!")
    parser.add_argument('-l', nargs='+', type=int, required=True, help="List of label IDs (integer values) that should "
                                                                       "be evaluated. Best practice is to use all int "
                                                                       "values present in the dataset, so for example "
                                                                       "for LiTS the labels are 0: background, 1: "
                                                                       "liver, 2: tumor. So this argument "
                                                                       "should be -l 1 2. You can if you want also "
                                                                       "evaluate the background label (0) but in "
                                                                       "this case that would not gie any useful "
                                                                       "information.")
    args = parser.parse_args()

    return evaluate_folder(args.ref, args.pred, args.l)

if __name__ == "__main__":
    calibaration_evaluate_folder()
