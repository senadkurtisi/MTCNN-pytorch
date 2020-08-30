from utils.utils import *
from utils.image_utils import prepare_image

import numpy as np


def stage_one(net, img):
    ''' Runs the stage one of the face
        detection process.

    Arguments:
        net (nn.Module): PNet instance
        img (PIL image): input image
    Return:
        bboxes (numpy.ndarray): detected bboxes in
                                the first stage
    '''
    scales = create_scales(img)
    bboxes = []

    for scale in scales:
        # Acquire bounding boxes for current scale
        bboxes_curr = PNetInference(net, img, scale)

        if bboxes_curr is not None:  # One or more bboxes were found
            bboxes.append(bboxes_curr)

    # Combine all bboxes in one matrix
    bboxes = np.vstack(bboxes)
    # Clean up bboxes detected in current stage
    bboxes = bbox_clean_up(bboxes=bboxes, threshold=NMS_THRESHOLDS[0])

    return bboxes


def stage_two(net, bboxes, img):
    ''' Runs the stage two of the face
        detection process.

    Arguments:
        net (nn.Module): RNet instance
        bboxes (numpy.ndarray): detected bboxes
                                in the first stage
        img (PIL image): input image
    Returns:
        bboxes (numpy.ndarray): detected bboxes
                                in the second stage
    '''
    # Acquire image patches based on bbox coords
    img_patches = get_image_patches(bboxes, img, 24)
    # Propagate the image patches through the RNet
    offsets, probs = net(img_patches)

    offsets = offsets.cpu().data.numpy()
    # We only keep the probability that bbxes contain
    # a face. Pretrained PNet has complementary output.
    probs = probs.cpu().data.numpy()[:, 1]

    # Keep only the output for which we are certain that
    # it refers to a face with respect to a probability threshold
    offsets, probs, bboxes = output_clean(
        PROB_THRESHOLDS[1], offsets, probs, bboxes)

    # Clean up bboxes detected in current stage
    bboxes = bbox_clean_up(bboxes=np.hstack(
        [bboxes, offsets]), threshold=NMS_THRESHOLDS[1])

    return bboxes


def stage_three(net, bboxes, img):
    ''' Runs the stage three of the face
        detection process.

    Arguments:
        net (nn.Module): RNet instance
        bboxes (numpy.ndarray): detected bboxes
                                in the second stage
        img (PIL image): input image
    Returns:
        bboxes (numpy.ndarray): detected bboxes
                                in the third stage
    '''
    # Acquire image patches based on bbox coords
    img_patches = get_image_patches(bboxes, img, 48)
    # Propagate the image patches through ONet
    offsets, probs, landmarks = net(img_patches)

    offsets = offsets.cpu().data.numpy()
    # We only keep the probability that bbxes contain
    # a face. Pretrained PNet has complementary output.
    probs = probs.cpu().data.numpy()[:, 1]
    landmarks = landmarks.cpu().data.numpy()

    # Keep only the output for which we are certain that
    # it refers to a face with respect to a probability threshold
    offsets, probs, bboxes, landmarks = output_clean(
        PROB_THRESHOLDS[2], offsets, probs, bboxes, landmarks)

    # Landmark output is given relative to size of the bboxes
    # We need to calculate the absolute position of the landmarks
    width = np.expand_dims((bboxes[:, 2] - bboxes[:, 0]) + 1, 1)
    height = np.expand_dims((bboxes[:, 3] - bboxes[:, 1]) + 1, 1)

    landmarks[:, 0:5] = np.expand_dims(
        bboxes[:, 0], 1) + width * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(
        bboxes[:, 1], 1) + height * landmarks[:, 5:10]

    # Calibrate and clean up residual bboxes
    bboxes = calibrate_bboxes(bboxes, offsets)
    bboxes[:, :4] = np.round(bboxes[:, :4])

    keep_ind = nms(bboxes, NMS_THRESHOLDS[2], 'minimum')
    bboxes, landmarks = bboxes[keep_ind], landmarks[keep_ind]

    return bboxes, landmarks


def PNetInference(net, img, scale):
    ''' Propagates the input image through PNet.
        But beforehad, image is scaled and prepared
        for PyTorch model. Generates bboxes and 
        performs NMS on detected faces.

    Arguments:
        net (nn.Module): PNet instance
        img (PIL image): input image
        scale (float): scale ratio for w&h
    Returns:
        bboxes (numpy.ndarray): detected bboxes (after NMS)
    '''
    # Prepare the image for PNet
    img = prepare_image(img, scale)
    # Propagate the image through PNet
    offsets, probs = output = net(img)

    # Since PNet has complementary output for
    # probabilities we keep only prob. that
    # bbox contains a face
    probs = probs.data.numpy()[0, 1, :, :]
    offsets = offsets.data.numpy()

    # Generate bboxes from the PNet output
    boxes = generate_bboxes(offsets, probs, scale, PROB_THRESHOLDS[0])
    if len(boxes) == 0:
        return None

    return boxes
