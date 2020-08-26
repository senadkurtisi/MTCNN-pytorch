from globals import *

import torch
import torch.nn.functional as F

import cv2
import numpy as np


def load_input():
	''' Loads the input image from the specified path.
		Image is transformed into PyTorch tensor and
		normalized according to the mean and std 
		of the dataset pretrained models were trained.

	Returns:
		img (torch.Tensor): prepared(& loaded) input image
	'''
	# Load the input image
	img = cv2.imread(config.img_loc)
	# CV2 loads image channels in BGR order, we need to correct that
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	print(img.shape)
	# PyTorch expects input to have shape: [batch_size, n_channels, w, h] 
	img = img.swapaxes(0, 2)	# set the channels dimension in front of w & h
	img = torch.Tensor(img)	
	img = img.unsqueeze(0)	# add the batch dimension

	img = (img-MEAN)/STD

	# Save precious memory by not calculating gradients
	img.requires_grad = False

	return img.to(device, torch.float)


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
	_, _, h, w = img.shape
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


def stage_one(net, img):
	scales = create_scales(img)

	for scale in scales:
		output = PNetInference(net, img, scale)


def PNetInference(net, img, scale):
	_, _, h, w = img.shape
	# Calculate shape of rescaled image
	new_w = np.ceil(scale*w).astype(np.int16)
	new_h = np.ceil(h*scale).astype(np.int16)
	img = F.interpolate(img, (new_h, new_w), \
						mode='bilinear', align_corners=True)

	bboxes, probs = net(img)
	# We only keep the probability that bbxes contain
	# a face. Pretrained PNet has complementary output.
	probs = probs[:, 1, :, :]

