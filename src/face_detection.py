from utils.utils import *

import numpy as np
import torch.nn.functional as F

import torchvision.transforms as transforms


def stage_one(net, img):
	''' Runs the stage one of the face
		detection process.
	'''
	scales = create_scales(img)
	bboxes = []

	for scale in scales:
		# Acquire bounding boxes for current scale
		bboxes_curr = PNetInference(net, img, scale)
		if len(bboxes_curr) >1:	# NMS is performed only for 2+ bboxes
			# perform the NMS
			bboxes.append(bboxes_curr)
		elif len(bboxes_curr)==1: # Avoid appending empty lists
			bboxes.append(bboxes_curr[0].T)
		else:
			# print('None', scale)
			pass

	# Combine all bboxes in one matrix
	bboxes = np.vstack(bboxes)
	# Peforms NMS
	keep_ind = nms(bboxes[:, :5], NMS_THRESHOLDS[0])
	bboxes = bboxes[keep_ind]
	# Calibrate acquired bboxes
	bboxes = calibrate_bboxes(bboxes[:,:5], bboxes[:,5:])
	# Turn calibrate bboxes into squares
	bboxes = square_up(bboxes)
	return bboxes


def stage_two(net, bboxes, img):
	''' Runs the stage two of the face
		detection process.
	'''
	img_patches = get_image_patches(bboxes, img, 24)
	offsets, probs = net(img_patches)

	offsets = offsets.detach().data.numpy()
	# We only keep the probability that bbxes contain
	# a face. Pretrained PNet has complementary output.
	probs = probs.detach().data.numpy()[:,1]

	keep_ind = np.where(probs>PROB_THRESHOLDS[1])[0]
	offsets, bboxes = offsets[keep_ind], bboxes[keep_ind]
	bboxes[:, 4] = probs[keep_ind]

	keep_ind = nms(bboxes, NMS_THRESHOLDS[1])
	bboxes = bboxes[keep_ind]
	bboxes = calibrate_bboxes(bboxes, offsets[keep_ind])
	bboxes = square_up(bboxes)
	bboxes[:,:4] = np.round(bboxes[:,:4])

	return bboxes


def stage_three(net, bboxes, img):
	''' Runs the stage three of the face
		detection process.
	'''
	img_patches = get_image_patches(bboxes, img, 48)
	offsets, probs, landmarks = net(img_patches)	

	offsets = offsets.detach().data.numpy()
	# We only keep the probability that bbxes contain
	# a face. Pretrained PNet has complementary output.
	probs = probs.detach().data.numpy()[:,1]

	landmarks = landmarks.detach().data.numpy()

	width = (bboxes[:, 2] - bboxes[:, 0]) + 1
	height = (bboxes[:, 3] - bboxes[:, 1]) + 1

	landmarks[:,:5] = np.expand_dims(bboxes[:, 0],1) \
					+ landmarks[:,:5]*(np.expand_dims(width, 1))
	landmarks[:,5:] = np.expand_dims(bboxes[:, 1],1) \
					+ landmarks[:,5:]*(np.expand_dims(height, 1))

	keep_ind = nms(bboxes, NMS_THRESHOLDS[2], 'minimum')
	bboxes = bboxes[keep_ind]
	bboxes = calibrate_bboxes(bboxes, offsets[keep_ind])
	bboxes = square_up(bboxes)
	bboxes[:,:4] = np.round(bboxes[:,:4])


def PNetInference(net, img, scale):
	# Prepare image for PNet
	img = prepare_image(img, scale)

	# Get the offsets and probabilities by using PNet
	offsets, probs = net(img)
	offsets = offsets.detach().data.numpy()

	# We only keep the probability that bbxes contain
	# a face. Pretrained PNet has complementary output.
	probs = probs.detach().data.numpy()[0, 1, :, :]
	bboxes = generate_bboxes(offsets, probs, scale, PROB_THRESHOLDS[0])

	keep_ind = nms(bboxes, 0.5)
	return bboxes[keep_ind]