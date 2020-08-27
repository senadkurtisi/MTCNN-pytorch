from .globals import *

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import numpy as np

from timeit import default_timer as timer


def prepare_image(img, scale=1.0):
	''' Prepares the image for the first
		stage of the inference process.
		Image is resized, normalized and
		transformed into PyTorch tensor.
	'''
	w, h = img.size
	# Calculate shape of rescaled image
	new_w = int(np.ceil(w*scale))
	new_h = int(np.ceil(h*scale))
	# Resize the image using bilinear interpolation
	img = img.resize((new_w, new_h), Image.BILINEAR)

	# Normalize the input image
	img = np.array(img)
	img = (img-MEAN)/STD
	
	to_torch = transforms.Compose([transforms.ToTensor(),
								   transforms.Lambda(lambda x: x.unsqueeze(0)),
								   transforms.Lambda(lambda x: x.to(device, torch.float))
								])
	img = to_torch(img)
	img.requires_grad = False

	return img	


def create_scales(img):
	''' Creates scale ratios used to scale the
		input image in order to create the image
		pyramid.

	Arguments:
		img(torch.Tensor): input image
	Returns:
		img_scales (list): scale ratios for the
						   image pyramid
	'''
	img_scales = []
	w,h = img.size
	min_dim = min(h, w)

	# Operations bellow scale the image so that
	# min size we WANT detect becomes equal to
	# min size we CAN to detect
	ratio = MIN_DETECTION_SIZE/MIN_FACE_SIZE
	min_dim *= ratio

	power = 0
	while min_dim > MIN_DETECTION_SIZE:
		scale = ratio*(MUL_FACTOR**power)
		img_scales.append(scale)

		min_dim *= MUL_FACTOR
		power += 1

	return img_scales


def generate_bboxes(offsets, probs, scale, threshold):
    STRIDE = 2
    cell_size = 12

    keep_ind = np.where(probs > threshold)
    tx1, ty1, tx2, ty2 = (offsets[0, :4, keep_ind[0], keep_ind[1]]).T

    offsets = np.vstack([tx1, ty1, tx2, ty2])
    probs = np.expand_dims(probs[keep_ind[0], keep_ind[1]],0)


    bboxes = np.vstack([
        np.round((STRIDE*keep_ind[1] + 1.0)/scale),
        np.round((STRIDE*keep_ind[0] + 1.0)/scale),
        np.round((STRIDE*keep_ind[1] + 1.0 + cell_size)/scale),
        np.round((STRIDE*keep_ind[0] + 1.0 + cell_size)/scale),
    ]).T

    return np.hstack([bboxes, offsets.T, probs.T])


def nms(bboxes, threshold):
	''' Performs the classical IoU NMS. '''
	
	keep_ind = []

	x1, y1, x2, y2, probs = [bboxes[:, i] for i in range(5)]
	# Find indices for bbox coordinates sorted by probability
	# that it contains a face in decresing order
	sorted_score_indices = np.argsort(probs)[::-1]

	while len(sorted_score_indices)>0:
		index = sorted_score_indices[0]
		# Since this
		keep_ind.append(index)

		# Calculating the area of current bbox
		h_curr = y2[index]-y1[index]+1
		w_curr = x2[index]-x1[index]+1
		current_area = h_curr*w_curr

		# Find the intersection points of the
		# current bbox with other bboxes
		x1_inter = np.maximum(x1[index], x1[sorted_score_indices[1:]])
		y1_inter = np.maximum(y1[index], y1[sorted_score_indices[1:]])
		x2_inter = np.minimum(x2[index], x2[sorted_score_indices[1:]])
		y2_inter = np.minimum(y2[index], y2[sorted_score_indices[1:]])

		# Calculate IoU - Intersection Over Union
		h_inter = np.maximum((y2_inter-y1_inter+1), 0)
		w_inter = np.maximum((x2_inter-x1_inter+1), 0)
		intersection =  h_inter*w_inter

		remaining_w = x2[sorted_score_indices[1:]]-x1[sorted_score_indices[1:]]+1
		remaining_h = y2[sorted_score_indices[1:]]-y1[sorted_score_indices[1:]]+1
		remaining_areas = remaining_w*remaining_h

		IoU = intersection/(current_area + remaining_areas - intersection)

		# Remove unnecessary bboxes
		to_remove = np.where(IoU>threshold)
		sorted_score_indices = np.delete(sorted_score_indices, to_remove)
		sorted_score_indices = sorted_score_indices[1:]

	return keep_ind


def show_detection(img, bboxes, landmarks=[]):
	draw = ImageDraw.Draw(img)

	for box in bboxes:
		draw.rectangle([(box[0], box[1]), 
						(box[2], box[3])], 
						outline='red')

	img.show()


def calibrate_bboxes(bboxes, offsets):
	''' Calibrates edge points of bboxes
		with respect to the offsets

	Arguments:
		bboxes (numpy.ndarray): bounding boxes which
								we need to calibrate
		offsets (numpy.ndarray): offsets used for 
								 calibration
	Returns:
		bboxes (numpy.ndarray): calibrated bboxes
	'''
	x1, y1, x2, y2 = bboxes.T[:4,:]
	w = np.expand_dims((x2-x1+1),1)
	h = np.expand_dims((y2-y1+1),1)

	correction = np.hstack([w, h, w, h])*offsets
	bboxes[:,0:4] += correction

	return bboxes

def square_up(bboxes):
	''' Transforms the bounding boxes into
		squares.

	Arguments:
		bboxes (numpy.ndarray): bounding boxes
	Returns:
		square_boxes (numpy.ndarray): squared
									  bboxes
	'''
	square_bboxes = np.zeros_like(bboxes)
	x1, y1, x2, y2 = bboxes[:, :4].T

	h = y2 - y1 + 1.0
	w = x2 - x1 + 1.0
	longer_side = np.maximum(h, w)

	square_bboxes[:, 0] = x1 + w*0.5 - longer_side*0.5
	square_bboxes[:, 1] = y1 + h*0.5 - longer_side*0.5
	square_bboxes[:, 2] = square_bboxes[:, 0] + longer_side - 1.0
	square_bboxes[:, 3] = square_bboxes[:, 1] + longer_side - 1.0

	return square_bboxes