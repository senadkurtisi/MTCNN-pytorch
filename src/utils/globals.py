from argparse import ArgumentParser
import torch
from numpy import sqrt

# Device on which we port model and images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Object which contains all necessary parameters
parser = ArgumentParser()
parser.add_argument("--img_loc", type=str, default='imgs/girls.jpg',
                    help='image in which we detect faces')
# Locations of the files which contain pretrained weights
parser.add_argument("--p_weights_loc", type=str,
                    default='pretrained_weights/pnet.npy',
                    help='PNet weights location')
parser.add_argument("--r_weights_loc", type=str,
                    default='pretrained_weights/rnet.npy',
                    help='RNet weights location')
parser.add_argument("--o_weights_loc", type=str,
                    default='pretrained_weights/onet.npy',
                    help='ONet weights location')
# Object used for accessing mentioned parameters
config = parser.parse_args()

# Minimum face size we WANT to detect
MIN_FACE_SIZE = 15.0
# Minimum size we CAN detect
MIN_DETECTION_SIZE = 12.0
# Factor used for calculating input image scales
MUL_FACTOR = sqrt(0.5)

# Probability thresholds for sliding windows
PROB_THRESHOLDS = [0.8, 0.85, 0.9]
# Thresholds used for overlap in NMS
NMS_THRESHOLDS = [0.5, 0.5, 0.5]

# Mean and std of the dataset on which pretrained
# models were trained
MEAN = 127.5
STD = 1 / 0.0078125
