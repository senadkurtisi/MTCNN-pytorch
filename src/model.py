''' Models for all three stages of the inference process '''

from utils.globals import *

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from collections import OrderedDict

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()

        return x.view(x.shape[0], -1)


class PNet(nn.Module):
    ''' CNN which represents the first stage of the 
        inference process for bounding boxes. This
        model outputs estimated coordinates for numerous
        bounding boxes as one return value, while the 
        second one is probability that thos bounding boxes
        contain a face. Coordinates represent x and y values
        for top left and bottom right corner of bbox.
    '''
    def __init__(self):
        super(PNet, self).__init__()

        self.kernel_size = 3
        self.kernel_num = [10, 16, 32]
        self.prob_num = 2
        self.bboxes_values = 4

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernel_num[0], self.kernel_size, bias=True)),
            ('prelu1', nn.PReLU(3, self.kernel_num[0])),
            ('pool1', nn.MaxPool2d(2, 2)),

            ('conv2', nn.Conv2d(self.kernel_num[0],
                      self.kernel_num[1], self.kernel_size, bias=True)),
            ('prelu2', nn.PReLU(self.kernel_num[1])),

            ('conv3', nn.Conv2d(self.kernel_num[1],
                      self.kernel_num[2], self.kernel_size, bias=True)),
            ('prelu3', nn.PReLU(self.kernel_num[2])),
        ]))

        self.conv4_1 = nn.Conv2d(self.kernel_num[2], self.bboxes_values, 3)
        self.conv4_2 = nn.Conv2d(self.kernel_num[2], self.prob_num, 3)

        self.load_pretrained()

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.p_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = Tensor(pretrained_weights[name], device = device)


    def forward(self, input):
        x = self.features(input)

        # Compute the parameters necessary for bboxes 
        bboxes = self.conv4_2(x)
        # Compute probabilities that boxes contain face
        probs = F.softmax(self.conv4_1(x), dim=1)

        return bboxes, probs



class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.kernel_size = [3, 3, 2]
        self.kernel_num = [28, 48, 64]
        self.pool_size = (2,2)
        self.fc_size = 128

        self.prob_num = 2
        self.bboxes_values = 4

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernel_num[0], self.kernel_size[0], bias=True)),
            ('prelu1', nn.PReLU(3, self.kernel_num[0])),
            ('pool1', nn.MaxPool2d(*self.pool_size)),

            ('conv2', nn.Conv2d(self.kernel_num[0],
                      self.kernel_num[1], self.kernel_size[1], bias=True)),
            ('prelu2', nn.PReLU(self.kernel_num[1])),
            ('pool2', nn.MaxPool2d(*self.pool_size)),

            ('conv3', nn.Conv2d(self.kernel_num[1],
                      self.kernel_num[2], self.kernel_size[2], bias=True)),
            ('prelu3', nn.PReLU(self.kernel_num[2])),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(self.kernel_num[2], self.fc_size, bias=True)),
            ('prelu4', nn.PReLU(self.fc_size)),
        ]))

        self.conv5_1 = nn.Linear(self.fc_size, self.bboxes_values, bias=True)
        self.conv5_2 = nn.Linear(self.fc_size, self.prob_num, bias=True)

        self.load_pretrained()
        

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.r_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = Tensor(pretrained_weights[name], device = device)


    def forward(self, input):
        x = self.features(input)

        # Compute the parameters necessary for bboxes 
        bboxes = self.conv5_2(x)
        # Compute probabilities that boxes contain face
        probs = F.softmax(self.conv5_1(x), dim=1)

        return bboxes, probs



class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.kernel_size = [3, 3, 2, 2]
        self.kernel_num = [32, 64, 64, 128]
        self.pool_size = [(2,2)]*3
        self.fc_size = 256

        self.prob_num = 2
        self.bboxes_values = 4
        self.landmark_values = 10

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernel_num[0], self.kernel_size[0], bias=True)),
            ('prelu1', nn.PReLU(3, self.kernel_num[0])),
            ('pool1', nn.MaxPool2d(*self.pool_size[0])),

            ('conv2', nn.Conv2d(self.kernel_num[0],
                      self.kernel_num[1], self.kernel_size[1], bias=True)),
            ('prelu2', nn.PReLU(self.kernel_num[1])),
            ('pool2', nn.MaxPool2d(*self.pool_size[1])),

            ('conv3', nn.Conv2d(self.kernel_num[1],
                      self.kernel_num[2], self.kernel_size[2], bias=True)),
            ('prelu3', nn.PReLU(self.kernel_num[2])),
            ('pool3', nn.MaxPool2d(*self.pool_size[2])),

            ('conv4', nn.Conv2d(self.kernel_num[2],
                      self.kernel_num[3], self.kernel_size[3], bias=True)),
            ('prelu4', nn.PReLU(self.kernel_num[3])),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(self.kernel_num[3], self.fc_size, bias=True)),
            ('prelu5', nn.PReLU(self.fc_size)),
        ]))

        self.conv6_1 = nn.Linear(self.fc_size, self.bboxes_values, bias=True)
        self.conv6_2 = nn.Linear(self.fc_size, self.prob_num, bias=True)
        self.conv6_3 = nn.Linear(self.fc_size, self.landmark_values, bias=True)

        self.load_pretrained()
        

    def load_pretrained(self):
        ''' Loads pretrained weights for PNet model. '''
        pretrained_weights = np.load(config.o_weights_loc)[()]

        for name, param in self.named_parameters():
            param.data = Tensor(pretrained_weights[name], device = device)


    def forward(self, input):
        x = self.features(input)

        # Compute the parameters necessary for bboxes 
        bboxes = self.conv6_1(x)
        # Compute probabilities that boxes contain face
        probs = F.softmax(self.conv6_2(x), dim=1)
        # Compute the x and y values for landmark points
        landmarks = self.conv6_3(x)

        return bboxes, probs, landmarks

