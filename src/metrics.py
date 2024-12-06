'''
   For a fair comparison, we use the same evaluation metrics as ViTVS, with the metrics code sourced from ViTVS.
'''

import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def F1_score(preds, labels, num_classes):
    # Flatten predictions and labels
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    # Calculate the F1 score for each class
    dice_scores = []
    for i in range(num_classes):
        class_preds = (preds_flat == i)
        class_labels = (labels_flat == i)

        # Calculate F1 score for each class
        dice = f1_score(class_labels.cpu().numpy(), class_preds.cpu().numpy(), labels=[True], average='binary')
        dice_scores.append(dice)

    # Return the mean F1 score
    return np.mean(dice_scores)

def dice_score_per_class(preds, labels, num_classes):
    dice_scores = []
    for i in range(num_classes):
        class_preds = (preds == i)
        class_labels = (labels == i)

        # Calculate Dice score for each class
        intersection = np.logical_and(class_labels.cpu().numpy().flatten(), class_preds.cpu().numpy().flatten()).sum()
        union = np.logical_or(class_labels.cpu().numpy().flatten(), class_preds.cpu().numpy().flatten()).sum()
        dice = (2 * intersection) / (union + intersection + 1e-8)  # Add epsilon to avoid division by zero
        dice_scores.append(dice)

    return np.array(dice_scores)


def dice_score(preds, labels, num_classes):
    # Calculate Dice score for each class
    dice_scores = dice_score_per_class(preds, labels, num_classes)

    # Return the mean Dice score
    return np.mean(dice_scores)

def iou_score_per_class(preds, labels, num_classes):
    iou_scores = []
    for i in range(num_classes):
        class_preds = (preds == i)
        class_labels = (labels == i)

        # Calculate IoU score for each class
        intersection = np.logical_and(class_labels.cpu().numpy().flatten(), class_preds.cpu().numpy().flatten()).sum()
        union = np.logical_or(class_labels.cpu().numpy().flatten(), class_preds.cpu().numpy().flatten()).sum()
        iou = (intersection) / (union + 1e-8)  # Add epsilon to avoid division by zero
        iou_scores.append(iou)

    return np.array(iou_scores)


def mean_iou(preds, labels, num_classes):
    # Calculate IoU score for each class
    iou_scores = iou_score_per_class(preds, labels, num_classes)

    # Return the mean IoU
    return np.mean(iou_scores)
'''
def SDR(preds, origin):
    # input origin's STFT
    # output segment STFT's img
    # transfer segment to audio (iSTFT)
    # SDR(i,4) = 10*log(norm(y)/(norm(x-y))); (y = raw audio, x = 经iSTFT得到ix2，x=real(ix2)  (convert_predict_to_audio.m)
'''