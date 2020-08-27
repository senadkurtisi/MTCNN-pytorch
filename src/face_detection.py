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
			keep_ind = nms(bboxes_curr[:, :5], NMS_THRESHOLDS[0])
			bboxes.append(bboxes_curr[keep_ind])
		elif len(bboxes_curr)==1: # Avoid appending empty lists
			bboxes.append(bboxes_curr)

	# Combine all bboxes in one matrix
	bboxes = np.vstack(bboxes)
	# Calibrate acquired bboxes
	bboxes = calibrate_bboxes(bboxes[:,:5], bboxes[:,5:])
	# Turn calibrate bboxes into squares
	bboxes = square_up(bboxes)

	show_detection(img, bboxes)


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

	if len(bboxes)>1:
		keep_ind = nms(bboxes, 0.5)
		return bboxes[keep_ind]
	else:
		return bboxes