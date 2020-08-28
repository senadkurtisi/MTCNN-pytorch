from .globals import *

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import numpy as np

from timeit import default_timer as timer


def prepare_image(img, scale=None):
	''' Prepares the image for the first
		stage of the inference process.
		Image is resized, normalized and
		transformed into PyTorch tensor.
	'''
	if scale:
		w, h = img.size
		# Calculate shape of rescaled image
		new_w = int(np.ceil(w*scale))
		new_h = int(np.ceil(h*scale))
		# Resize the image using bilinear interpolation
		img = img.resize((new_w, new_h), Image.BILINEAR)

	# Normalize the input image
	img = np.array(img)
	img = (img-MEAN)/STD
	img = np.moveaxis(img, 2, 0)
	img = torch.from_numpy(img)
	
	to_torch = transforms.Compose([
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
    # tx1, ty1, tx2, ty2 = [offsets[0, i, keep_ind[0], keep_ind[1]] for i in range(4)]
    tx1, ty1, tx2, ty2 = offsets[0, :4, keep_ind[0], keep_ind[1]].T

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[keep_ind[0], keep_ind[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((STRIDE*keep_ind[1] + 1.0)/scale),
        np.round((STRIDE*keep_ind[0] + 1.0)/scale),
        np.round((STRIDE*keep_ind[1] + 1.0 + cell_size)/scale),
        np.round((STRIDE*keep_ind[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])

    return bounding_boxes.T


def nms(bboxes, threshold, mode='iou'):
	''' Performs the classical IoU NMS. '''
	
	keep_ind = []

	x1, y1, x2, y2, probs = [bboxes[:, i] for i in range(5)]
	# Find indices for bbox coordinates sorted by probability
	# that it contains a face in decresing order
	sorted_score_indices = np.argsort(probs)
	area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)

	while len(sorted_score_indices)>0:
		last = len(sorted_score_indices)-1
		index = sorted_score_indices[last]
		
		keep_ind.append(index)

		# Find the intersection points of the
		# current bbox with other bboxes
		x1_inter = np.maximum(x1[index], x1[sorted_score_indices[:last]])
		y1_inter = np.maximum(y1[index], y1[sorted_score_indices[:last]])

		x2_inter = np.minimum(x2[index], x2[sorted_score_indices[:last]])
		y2_inter = np.minimum(y2[index], y2[sorted_score_indices[:last]])

		# Calculate intersection
		h_inter = np.maximum((y2_inter-y1_inter+1), 0)
		w_inter = np.maximum((x2_inter-x1_inter+1), 0)
		intersection =  h_inter*w_inter

		if mode == 'iou':
			IoU = intersection/(area[index] + area[sorted_score_indices[:last]] - intersection)
			# Remove unnecessary bboxes
			to_remove = np.where(IoU>threshold)[0]
		else:
			overlap = intersection/np.minimum(area[index], area[sorted_score_indices[:last]])
			to_remove = np.where(overlap>threshold)[0]

		# sorted_score_indices = np.delete(sorted_score_indices, np.concatenate([[last], to_remove]))
		sorted_score_indices = np.delete(sorted_score_indices, to_remove)
		sorted_score_indices = sorted_score_indices[:-1]

	return keep_ind


def show_detection(img, bboxes, landmarks=[]):
	draw = ImageDraw.Draw(img)

	for box in bboxes:
		draw.rectangle([(box[0], box[1]), 
						(box[2], box[3])], 
						outline='blue')


	for landmark in landmarks:
		for i in range(5):
			draw.ellipse([(landmark[i]-1, landmark[i+5]-1),
						(landmark[i]+1, landmark[i+5]+1)],
						outline='blue')

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
	x1, y1, x2, y2 = bboxes[:, :4].T
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


def get_image_patches(bboxes, img, patch_size):
	CHANNELS = 3
	bbox_num = len(bboxes)
	w, h = img.size

	patches = np.zeros((bbox_num, CHANNELS, patch_size, patch_size))

	[x, y, ex, ey, dx, dy, edx, edy, h_outlier, w_outlier] = \
										handle_outliers(bboxes, w, h)

	img = np.array(img).astype('uint8')
	good = 0
	bad = 0
	for ind in range(bbox_num):
		patch = np.zeros((h_outlier[ind], w_outlier[ind], CHANNELS)).astype('uint8')
		try:
			patch[dy[ind]:(edy[ind]+1), dx[ind]:(edx[ind]+1)] = \
									img[x[ind]:(ex[ind]+1), y[ind]:(ey[ind]+1)]
			good += 1
		except:
			bad += 1
			h1, h2 = (edy[ind]-dy[ind])+1, (ey[ind]-y[ind])+1
			h_chosen = min(min(h1, h2), h_outlier[ind])
			w1, w2 = (edx[ind]-dx[ind])+1, (ex[ind]-x[ind])+1 
			w_chosen = min(min(w1, w2), w_outlier[ind])

			patch[dy[ind]:(dy[ind]+h_chosen), dx[ind]:(dx[ind]+w_chosen)] = \
				img[y[ind]:(y[ind]+h_chosen), x[ind]:(x[ind]+w_chosen)]

		patch = Image.fromarray(patch)
		patch = patch.resize((patch_size, patch_size), Image.BILINEAR)

		patches[ind] = prepare_image(patch)


	patches = torch.from_numpy(patches)
	patches.requires_grad = False
	return patches.to(device, torch.float)
	# print(f"Total:{bbox_num}, good:{good}, bad:{bad}")



def handle_outliers(bboxes, width, height):
	''' Handles bboxes which reach out of the
		border of the image. Border is proposed
		via width and height of the image.
	'''
	bbox_num = len(bboxes)

	x1, y1, x2, y2 = bboxes[:, :4].T
	w = (x2-x1) + 1
	h = (y2-y1) + 1

	x, ex, y, ey = x1, x2, y1, y2
	edx, edy = w, h
	dx, dy = np.zeros((bbox_num, )), np.zeros((bbox_num, ))

	ind = np.where(ex>width)[0]
	edx[ind] = (width-ex[ind]) + (w[ind])
	ex[ind] = width 

	ind = np.where(x<0)[0]
	dx[ind] = -x[ind]
	x[ind] = 0

	ind = np.where(ey>height)[0]
	edy[ind] = (height-ey[ind]) + (h[ind])
	ey[ind] = height

	ind = np.where(y<0)[0]
	dy[ind] = -y[ind]
	y[ind] = 0

	corrected_coords = [x, y, ex, ey, dx, dy, edx, edy, h, w]
	corrected_coords = [coord.astype('int32') for coord in corrected_coords]

	return corrected_coords