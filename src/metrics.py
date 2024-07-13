import numpy as np
import pickle
from sklearn.metrics import log_loss, brier_score_loss
from scipy import ndimage
from scipy import stats


def set_ROI(prob,true, label):
        """
        Set region of interest (ROI) for one channels. 
        
        """
        ROI = None
        roi_test = prob[:, label]
        roi_test = roi_test * (roi_test>0.1).astype(int)
        try:
            roi_ref = true[:, label]>0
        except IndexError:
            return []
            #roi_ref = roi_test
            #if roi_test.sum() < 100: ## if predicted roi_test smaller than 100 mm3 return nan
                

        assert (roi_test.shape==roi_ref.shape), 'ROI of reference and test not match!'
        ROI = (roi_test+roi_ref)>0
        print("ROI shapes before", ROI.shape, np.sum(ROI))
        #roi_ref = ndimage.binary_dilation(roi_ref, iterations=2) # dilate image to enlarge ROI

        #ROI = ndimage.binary_fill_holes(ROI)
        #ROI = ndimage.binary_dilation(ROI, iterations=4)
        #print("ROI shapes", ROI.shape, np.sum(ROI))

        return ROI 

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def ECE(conf, pred, true, prob, label):
    bin_size = 0.1
    ROI = set_ROI(prob, true, label)
    
    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    #prob = np.array(prob[ROI])

    conf = conf[ROI] #np.array(prob[ROI, label])
    pred = np.array(prob[ROI, label]>0.5).astype(int) # s

    try:
        true = np.array(true[ROI,label])
    except IndexError:
        print("no label found for reference label: ", label)
        true = np.zeros(conf.shape)
        #return np.nan


    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
    return ece
        
      
def MCE(conf, pred, true, prob, label):
    bin_size = 0.1
    ROI = set_ROI(prob, true, label)

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf)) 
    return max(cal_errors)


def brier_score(conf, pred, true, prob, label):
    ROI = set_ROI(prob, true, label)

    #y_prob_true = np.array([prob[i, idx] for i, idx in enumerate(true)])

    #y_prob_true = np.max(prob, axis=1)

    prob = np.array(prob[ROI,label])

    #print("!!!!!!!!!!!!!!!!", true.max())
    #true = true[:, label]>0

    try:
        true = np.array(true[ROI,label])
    except IndexError:
        print("no label found for reference label: ", label)
        true = np.zeros(prob.shape)

    #y_true[true>0] = 1
    #y_prob_true =  np.array(1-prob[:, 0])


    try:
        brier_score = brier_score_loss(y_true=true, y_prob=prob)
    except:
       brier_score = np.nan
    return  brier_score


def NLL(conf, pred, true, prob, label):
    ROI = set_ROI(prob, true, label)

    #loss = log_loss(y_true=true[:prob.shape[0]], y_pred=prob, eps=1e-6, normalize=False)
    print("!!!!testing ", prob.shape)
    prob = np.array(prob[ROI,label])
    true = np.array(true[ROI,label])

    # if len(np.unique(true))!= prob.shape[1]:
    #     prob_new = np.zeros((prob.shape[0], 2))
    #     prob_new[:, 0] = prob[:,0]
    #     for n in range(1, prob.shape[1]):
    #         prob_new[:,1] += prob[:,n]

    #     loss = log_loss(y_true=true, y_pred=prob_new, eps=1e-6)

    # else:
    #     loss = log_loss(y_true=true, y_pred=prob, eps=1e-6) # , normalize=False) # set false normalize for sum than mean

    loss = log_loss(y_true=true, y_pred=prob, eps=1e-6) # , normalize=False) # set false normalize for sum than mean
    return loss
