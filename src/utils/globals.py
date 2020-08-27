from argparse import ArgumentParser
import torch
from numpy import sqrt

# Device on which we port model and images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Object which contains all necessary parameters
parser = ArgumentParser()
parser.add_argument("--img_loc", type=str, default='imgs/office5.jpg',
                    help='image in which we detect faces')
parser.add_argument("--p_weights_loc", type=str,
                    default='pretrained_weights/pnet.npy')
parser.add_argument("--r_weights_loc", type=str,
                    default='pretrained_weights/rnet.npy')
parser.add_argument("--o_weights_loc", type=str,
                    default='pretrained_weights/onet.npy')
config = parser.parse_args()


MIN_FACE_SIZE = 15.0
MIN_DETECTION_SIZE = 12.0
MUL_FACTOR = 0.707

PROB_THRESHOLDS = [0.6, 0.7, 0.8]
NMS_THRESHOLDS = [0.7, 0.7, 0.7]

MEAN = 127.5
STD = 1/0.0078125
