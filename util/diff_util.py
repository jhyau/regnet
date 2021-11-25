import numpy as np
from scipy.signal import find_peaks

pred_over_gt_color = 0 # Mark red the prediction over gt.
gt_over_pred_color = 2 # Mark blue the gt over prediction.

def diff_image(spectrogram, gt_spectrogram):
    """Generate a peak visualization to see how far off they are."""
    h, w = spectrogram.shape
    diff_im = np.ones((h,w,3))
    diff_hue = spectrogram - gt_spectrogram
    diff_hue -= np.min(diff_hue)
    diff_hue /= np.max(diff_hue)
    # clip out values if they exist.
    pred_over_gt_mask = gt_spectrogram < spectrogram 
    gt_over_pred_mask = gt_spectrogram > spectrogram
    # zero out the pred_over_gt mask
    diff_im[pred_over_gt_mask] = 0.
    # color that mask
    diff_im[:,:,pred_over_gt_color][pred_over_gt_mask]= diff_hue[pred_over_gt_mask]
    # zero out the gt_over_pred mask
    diff_im[gt_over_pred_mask] = 0.
    # color that mask
    diff_im[:,:,gt_over_pred_color][gt_over_pred_mask] = diff_hue[gt_over_pred_mask]
    return diff_im
