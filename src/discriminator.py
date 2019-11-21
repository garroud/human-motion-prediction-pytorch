from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import data_utils
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
                use_GNN=False,
                wgan=True,
                ):

        super(Discriminator, self).__init__()
        # TODO: not sure whether one hot can affect the result, added here and require further experiments to find out.
        self.HUMAN_SIZE = 54 if not use_GNN else 63
        self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ## The time stamps which exclude from calculation
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_size + self.HUMAN_SIZE, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, 1)
        self.wgan = wgan
    def forward(self, s, a, h=None): # s: seq * batch * input_size, a: seq * batch * input_size
        p, hidden = self.gru(s, h)
        p = torch.cat([p , a], 2)
        if self.wgan:
            prob = self.dense3(F.relu(self.dense2(F.relu(self.dense1(p)))))
        else:
            prob = torch.sigmoid(self.dense3(F.relu(self.dense2(F.relu(self.dense1(p))))))
        return prob

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        
#
# # GNN discriminator for GAN  training
# class GNNDiscriminator(nn.Module):
#     def __init__(self,
#                 source_seq_len,
#                 edge_hidden_dim,
#                 edge_out_dim,
#                 node_hidden_dim,
#                 node_out_dim,
#                 rec_encode,
#                 send_encode,
#                 number_of_actions,
#                 batch_size,
#                 num_passing,
#                 one_hot=False,
#                 weight_share=False,
#                 do_prob=0,
#                 dtype=torch.float32,
#                 wgan=True
#                 ):
#         super(GNNDiscriminator, self).__init__()
#         self.input_size = 3 if not one_hot else 3 + number_of_actions
#         self.JOINT_DIM = (source_seq_len - 1) * 3
#         self.encoder_input_size = self.JOINT_DIM + number_of_actions if one_hot else self.JOINT_DIM
#         self.batch_size = batch_size
#         self.encode = GNNEncoder(
#             self.encoder_input_size,
#             node_hidden_dim,
#             node_out_dim,
#             edge_hidden_dim,
#             edge_out_dim,
#             num_passing,
#             rec_encode,
#             send_encode,
#             weight_share,
#             do_prob,
#         )
#         self.model = GNNDecoder(
#             self.input_size,
#             node_out_dim,
#             edge_hidden_dim,
#             edge_out_dim,
#             rec_encode,
#             send_encode,
#             do_prob,
#             dtype
#         )
#         self.mlp1 = MLP(input_size, node_hidden_dim, node_out_dim, do_prob)
#         self.dense1 = nn.Linear(node_out_dim+self.input_size, node_hidden_dim)
#         self.dense2 = nn.Linear(node_hidden_dim, node_hidden_dim // 2)
#         self.dense3 = nn.Linear(node_hidden_dim // 2, 1)
#
#     def forward(self, encoder_inputs, s, a):
#         node_hidden = self.encode(encoder_inputs)
#         p, hidden = self.model(s,node_hidden)
#         p = torch.cat([node_hidden,p], dim=0)
#         p = torch.cat([p, a], dim=3)
#         if wgan:
#             prob = self.dense3(F.relu(self.dense2(F.relu(self.dense1(p)))))
#         else:
#             prob = torch.sigmoid(self.dense3(F.relu(self.dense2(F.relu(self.dense1(p))))))
#         return prob
