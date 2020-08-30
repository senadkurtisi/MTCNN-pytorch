from .globals import *
from .image_utils import *

import numpy as np


def bbox_clean_up(bboxes, threshold):
    ''' Performs clean up on detected
        bboxes by executing NMS with
        given threshold, calibrating them
        and turning them into squares.

    Arguments:
        bboxes (numpy.ndarray): detected bboxes
        threshold (float): NMS threshold
    Returns:
        bboxes (numpy.ndarray): bboxes kept after NMS
    '''
    # Perform NMS on detected bboxes
    keep_ind = nms(bboxes[:, :5], threshold)
    bboxes = bboxes[keep_ind]

    # Calibrate coords of bboxes with respect to offsets
    bboxes = calibrate_bboxes(bboxes[:, :5], bboxes[:, 5:])
    # Turn bboxes into squares
    bboxes = square_up(bboxes)
    bboxes[:, :4] = np.round(bboxes[:, :4])

    return bboxes


def output_clean(threshold, offsets, probs, bboxes=[], landmarks=[]):
    ''' Performs the clean up of outputs of neural net
        by keeping only those which satisfy the
        probability threshold.

    Arguments:
        threshold (float): probability threshold
        offsets (numpy.ndarray): offset output of CNN
        probs (numpy.ndarray): probability output of CNN
        bboxes (numpy.ndarray): detected bboxes
        landmarks (numpy.ndarray): landmarks output of CNN
    Returns:
        to_keep (list): output which satisfy the probability
                        threshold
    '''
    # Detect which elements satisfy the probability threshold
    keep_ind = np.where(probs > threshold)[0]

    offsets = offsets[keep_ind]
    probs = probs[keep_ind]

    to_keep = [offsets, probs]

    if len(bboxes):
        bboxes = bboxes[keep_ind]
        bboxes[:, 4] = probs.copy()
        to_keep.append(bboxes)

    if len(landmarks):
        to_keep.append(landmarks[keep_ind])

    return to_keep


def generate_bboxes(offsets, probs, scale, threshold):
    ''' Generates bounding boxes based on outputs
        of the CNN.

    Arguments:
        offsets (numpy.ndarray): offset output of CNN
        probs (numpy.ndarray): probability output of CNN
        scale (float): scale ratio for input image
    Returns:
        bboxes (numpy.ndarray): generated bboxes
    '''
    STRIDE = 2
    KERNEL = 12

    # indices of boxes where there is probably a face
    keep_ind = np.where(probs > threshold)

    if len(keep_ind[0]) == 0:  # If no sliding window is good enough
        return np.array([])

    # Save only necessary probabilities
    probs = probs[keep_ind[0], keep_ind[1]]

    # Acquire offset for each coordinate (only necessary)
    off_x1, off_y1, off_x2, off_y2 = offsets[0, :, keep_ind[0], keep_ind[1]].T
    offsets = np.vstack([off_x1, off_y1, off_x2, off_y2])

    net_output = np.vstack([probs, offsets]).T

    # PNet is applied to scaled input image so
    # we need to calculate bbox coords with respect
    # to the applied scaling
    bboxes = np.vstack([
        np.round((STRIDE * keep_ind[1] + 1) / scale),
        np.round((STRIDE * keep_ind[0] + 1) / scale),
        np.round((STRIDE * keep_ind[1] + 1 + KERNEL) / scale),
        np.round((STRIDE * keep_ind[0] + 1 + KERNEL) / scale),
    ]).T

    return np.hstack([bboxes, net_output])


def nms(bboxes, threshold, mode='union'):
    ''' Apply NMS to detected bboxes.

    Arguments:
        bboxes (numpy.ndarray): detected bboxes
        threshold (float): NMS threshold
        mode (string): criteria for NMS
    Returns:
        keep_ind (list): list of indices of
                         bboxes kept after NMS
    '''
    # Indices of bboxes we wish to keep after NMS
    keep_ind = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = bboxes[:, :5].T

    global_areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:
        ind = ids[-1]
        keep_ind.append(ind)

        # Top left corner of the intersection
        x1_inter = np.maximum(x1[ind], x1[ids[:-1]])
        y1_inter = np.maximum(y1[ind], y1[ids[:-1]])

        # Bottom right corner of the intersection
        x2_inter = np.minimum(x2[ind], x2[ids[:-1]])
        y2_inter = np.minimum(y2[ind], y2[ids[:-1]])

        # Width and height of the intersection
        w_inter = np.maximum(x2_inter - x1_inter + 1, 0)
        h_inter = np.maximum(y2_inter - y1_inter + 1, 0)

        # Calculating the overlap area
        intersection = w_inter * h_inter
        if mode == 'minimum':
            overlap = intersection / \
                np.minimum(global_areas[ind], global_areas[ids[:-1]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = intersection / \
                (global_areas[ind] + global_areas[ids[:-1]] - intersection)

        # delete all boxes where overlap is too big
        ids = np.delete(ids, np.where(overlap > threshold)[0])
        ids = ids[:-1]

    return keep_ind


def calibrate_bboxes(bboxes, offsets):
    ''' Calibrates edge points of bboxes
        with respect to the offsets.

    Arguments:
        bboxes(numpy.ndarray): bounding boxes which
                               we need to calibrate
        offsets(numpy.ndarray): offsets used for
                                calibration
    Returns:
        bboxes(numpy.ndarray): calibrated bboxes
    '''
    x1, y1, x2, y2 = bboxes[:, : 4].T
    # Width and height of bboxes
    w = np.expand_dims((x2 - x1 + 1), 1)
    h = np.expand_dims((y2 - y1 + 1), 1)

    # Perform the correction of bboxes
    correction = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0: 4] += correction

    return bboxes


def square_up(bboxes):
    ''' Transforms the bounding boxes into squares.

    Arguments:
        bboxes(numpy.ndarray): bounding boxes
    Returns:
        square_boxes(numpy.ndarray): squared bboxes
    '''
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = bboxes[:, : 4].T

    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    longer_side = np.maximum(h, w)

    # Example: width:100, height:70. DIFF = (100-70).
    # X coordinates of both top left and bottom right
    # corners stay the same while y coords shift for DIFF/2.
    # Top left y coord goes up for DIFF/2 while bottom left
    # y coord goes DOWN for DIFF/2. Similar rules apply when
    # the height is larger than width. We get square bbox.
    square_bboxes[:, 0] = x1 + w * 0.5 - longer_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - longer_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + longer_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + longer_side - 1.0

    return square_bboxes
