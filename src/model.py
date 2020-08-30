''' Models for all three stages of the detection process '''

from utils.globals import *

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # This approach was taken from the repo
        # this project was inspired from. If we
        # don't include the next line in our
        # pipeline MTCNN performs badly.
        # The repo:
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):
    ''' Peforms the first stage of the
        detection process. Outputs probabilities
        that and sliding windows contain
        a face and an offset for the bbox.
    '''

    def __init__(self):

        super(PNet, self).__init__()

        self.kernels = [10, 16, 32]
        self.kernel_size = 3
        self.pool = (2, 2)

        self.prob_num = 2
        self.offset_num = 4

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernels[0], self.kernel_size, 1)),
            ('prelu1', nn.PReLU(self.kernels[0])),
            ('pool1', nn.MaxPool2d(*self.pool, ceil_mode=True)),

            ('conv2', nn.Conv2d(
                self.kernels[0], self.kernels[1], self.kernel_size, 1)),
            ('prelu2', nn.PReLU(self.kernels[1])),

            ('conv3', nn.Conv2d(
                self.kernels[1], self.kernels[2], self.kernel_size, 1)),
            ('prelu3', nn.PReLU(self.kernels[2]))
        ]))

        self.conv4_1 = nn.Conv2d(self.kernels[2], self.prob_num, 1, 1)
        self.conv4_2 = nn.Conv2d(self.kernels[2], self.offset_num, 1, 1)

        self.load_pretrained()

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.p_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = torch.FloatTensor(pretrained_weights[name]).to(device)

    def forward(self, input):
        x = self.features(input)

        # Probability that a sliding window contains a face
        probs = F.softmax(self.conv4_1(x), dim=1)
        # Offset for that sliding window's bbox
        offsets = self.conv4_2(x)

        return offsets, probs


class RNet(nn.Module):
    ''' Peforms the second stage of the
        detection process. Outputs probabilities
        that and sliding windows contain
        a face and an offset for the bbox.
    '''

    def __init__(self):

        super(RNet, self).__init__()

        self.kernels = [28, 48, 64, 128]
        self.kernel_size = [3, 3, 2]
        self.pool = (3, 2)
        self.fc_size = 576

        self.prob_num = 2
        self.offset_num = 4

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernels[0], self.kernel_size[0], 1)),
            ('prelu1', nn.PReLU(self.kernels[0])),
            ('pool1', nn.MaxPool2d(*self.pool, ceil_mode=True)),

            ('conv2', nn.Conv2d(
                self.kernels[0], self.kernels[1], self.kernel_size[1], 1)),
            ('prelu2', nn.PReLU(self.kernels[1])),
            ('pool2', nn.MaxPool2d(*self.pool, ceil_mode=True)),

            ('conv3', nn.Conv2d(
                self.kernels[1], self.kernels[2], self.kernel_size[2], 1)),
            ('prelu3', nn.PReLU(self.kernels[2])),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(self.fc_size, self.kernels[3])),
            ('prelu4', nn.PReLU(self.kernels[3]))
        ]))

        self.conv5_1 = nn.Linear(self.kernels[3], self.prob_num)
        self.conv5_2 = nn.Linear(self.kernels[3], self.offset_num)

        self.load_pretrained()

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.r_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = torch.FloatTensor(pretrained_weights[name]).to(device)

    def forward(self, input):
        x = self.features(input)

        # Probability that a sliding window contains a face
        probs = F.softmax(self.conv5_1(x), dim=1)
        # Offset for that sliding window's bbox
        offsets = self.conv5_2(x)

        return offsets, probs


class ONet(nn.Module):
    ''' Peforms the third stage of the
        detection process. Outputs probabilities
        that sliding windows contain a face, 
        an offset for the according bboxes
        and 5 landmark points (5x&5y coords)
        for these bboxes.
    '''

    def __init__(self):

        super(ONet, self).__init__()

        self.kernels = [32, 64, 64, 128, 256]
        self.kernel_size = [3, 3, 3, 2]
        self.pool = [(3, 2), (3, 2), (2, 2)]
        self.fc_size = 1152
        self.dropout_p = 0.25

        self.prob_num = 2
        self.offset_num = 4
        self.landmark_num = 10

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernels[0], self.kernel_size[0], 1)),
            ('prelu1', nn.PReLU(self.kernels[0])),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(
                self.kernels[0], self.kernels[1], self.kernel_size[1], 1)),
            ('prelu2', nn.PReLU(self.kernels[1])),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(
                self.kernels[1], self.kernels[2], self.kernel_size[2], 1)),
            ('prelu3', nn.PReLU(self.kernels[2])),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(
                self.kernels[0], self.kernels[3], self.kernel_size[3], 1)),
            ('prelu4', nn.PReLU(self.kernels[3])),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(self.fc_size, self.kernels[4])),
            ('drop5', nn.Dropout(self.dropout_p)),
            ('prelu5', nn.PReLU(self.kernels[4])),
        ]))

        self.conv6_1 = nn.Linear(self.kernels[4], self.prob_num)
        self.conv6_2 = nn.Linear(self.kernels[4], self.offset_num)
        self.conv6_3 = nn.Linear(self.kernels[4], self.landmark_num)

        self.load_pretrained()

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.o_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = torch.FloatTensor(pretrained_weights[name]).to(device)

    def forward(self, input):
        x = self.features(input)

        # Probability that a sliding window contains a face
        probs = F.softmax(self.conv6_1(x), dim=1)
        # Offsets for these sliding window's bboxes
        offsets = self.conv6_2(x)
        # Landmarks for the bboxes
        landmarks = self.conv6_3(x)

        return offsets, probs, landmarks
