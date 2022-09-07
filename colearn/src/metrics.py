import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()

def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / (all_positives + 1e-4)

    return recall.mean()

def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / (all_positive_calls + 1e-4)

    return precision.mean()

def specificity(input, target):
    input = (input > 0.5).float()
    input_np = input.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()

    input_np = np.ravel(input_np, order='C')
    target_np = np.ravel(target_np, order='C')

    tn, fp, fn, tp = confusion_matrix(target_np, input_np).ravel()

    spec = tn / (fp + tn)

    return spec

def accuracy(input, target):
    input = (input > 0.5).float()
    input_np = input.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()

    input_np = np.ravel(input_np, order='C')
    target_np = np.ravel(target_np, order='C')

    tn, fp, fn, tp = confusion_matrix(target_np, input_np).ravel()

    spec = (tn+tp) / (fp + fn + tp + tn)

    return spec

def false_negative_positive(input, target):
    input = (input > 0.5).float()
    input_np = input.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()

    input_np = np.ravel(input_np, order='C')
    target_np = np.ravel(target_np, order='C')

    tn, fp, fn, tp = confusion_matrix(target_np, input_np).ravel()

    return fn, fp



"""
autoPET에서 주어진 평가지표에대한 코드들 고친것
"""

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp

def false_pos_pix(gt_array, pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    false_pos = 0
    for idx in range(1, pred_conn_comp.max() + 1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask * gt_array).sum() == 0:
            false_pos = false_pos + comp_mask.sum()
    return false_pos

def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)

    false_neg = 0
    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()

    return false_neg

def dice_score(mask1, mask2):
    # compute foreground Dice coefficient
    overlap = (mask1 * mask2).sum()
    sum = mask1.sum() + mask2.sum()
    dice_score = 2 * overlap / sum
    return dice_score

def compute_metrics(input,target,pixdim):
    # main function

    input = (input > 0.5).float()
    gt_array = target[0,0,:,:,:].cpu().detach().numpy()
    pred_array = input[0,0,:,:,:].cpu().detach().numpy()

    voxel_vol = pixdim[0][1] * pixdim[0][2] * pixdim[0][3] / 1000
    # print(false_neg_pix(gt_array, pred_array))
    # print(false_pos_pix(gt_array, pred_array))


    false_neg_vol = false_neg_pix(gt_array, pred_array) * voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array) * voxel_vol
    dice_sc = dice_score(gt_array, pred_array)


    return dice_sc, false_pos_vol, false_neg_vol