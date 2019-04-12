from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import data_utils
import decoderWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F

#Discriminator for GAN training
class Discriminator(nn.Module):

    def __init__(self,
                one_hot,
                hidden_size,
                batch_size,
                num_layers,
                number_of_actions,
                ):

        super(Discriminator, self).__init__()
        # TODO: not sure whether one hot can affect the result, added here and require further
        # experiments to find out.
        self.HUMAN_SIZE = 54
        self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ## The time stamps which exclude from calculation
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, 1)
        
    def forward(self, s, a, h=None): # s: seq * batch * input_size, a: seq * batch * input_size
        p, hidden = self.gru(s, h)
        p = torch.cat([p , a], 2)
        prob = torch.sigmoid(self.dense2(F.relu(self.dense1(p))))
        return prob

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
