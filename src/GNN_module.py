"""A GNN encoder and decider model for human motion prediction in pytorch"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import data_utils
import decoderWrapper
from helper import *
import torch
import torch.nn as nn
import torch.functional as F

# two basic operation for node and edge conversion
def node2edge(x, rec_encode, send_encode):
    recs = torch.matmul(rec_encode, x)
    sends = torch.matmul(send_encode, x)
    edges = torch.cat([send, recs], dim=-1)
    return edges

def edge2node(x, rec_encode, send_encode):
    incomming = torch.matmul(rec_encode.t(), x)
    return incomming / rec_encode.t().sum(1).expend(-1,rec_encode.size(0))

class GNNModel(nn.Module):
    def __init__(self,
                source_seq_len,
                target_seq_len,
                rnn_size,
                batch_size,
                number_of_actions,
                device,
                one_hot=False,
                residual_velocities=False,
                stochastic=False,
                dtype=torch.float32):
        super(GNNModel, self).__init__()
        self.NUM_JOINT = 32
        self.JOINT_DIM = 3 * source_seq_len
        self.input_size= self.JOINT_DIM + number_of_actions if one_hot else self.JOINT_DIM

        print("One hot is ", one_hot)
        print("Input size is %d" % self.input_size)

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.stochastic = stochastic

        self.encoder = GNNEncoder(
            self.input_size,
            self.edge_out_dim,
        )

class GNNEncoder(nn.Module):
    def __init__(self,
                input_size,
                edge_hidden_dim,
                edge_out_dim,
                node_hidden_dim,
                node_out_dim,
                num_passing,
                rec_encode,
                send_encode,
                weight_share=True,
                do_prob=0):

        super(GNNModel, self).__init__()
        self.input_size = input_size
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.do_prob = do_prob
        self.num_passing = num_passing
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        print("Dim of hidden node state is ", self.node_hidden_dim)
        print("Dim of hidden edge state is ", self.edge_hidden_dim)

        # transform original node feature to node embedding
        self.mlp1 = MLP(self.input_size, self.node_hidden_dim, self.node_out_dim)

        if weight_share:
            mlp_struct = [
                MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim, do_prob=do_prob),
                MLP(self.edge_out_dim, self.node_hidden_dim, self.node_out_dim, do_prob=do_prob)
            ]
            self.passing_list = [mlp_struct] * self.num_passing
        else:
            self.passing_list = [
                [MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim,   do_prob=do_prob),
                MLP(self.edge_out_dim, self.node_hidden_dim, self.node_out_dim, do_prob=do_prob)]
                for _ in range(num_passing)
            ]
    def forward(self, x):
        x = self.mlp1(x)
        for pass in range(self.num_passing):
            x = node2edge(x, self.rec_encode, self.send_encode)
            x = self.passing_list[pass][0](x)
            x = edge2node(x, self.rec_encode, self.send_encode)
            x = self.passing_list[pass][1](x)
        return x

class GNNDecoder(nn.Module):
    def __init__(self,
                input_size,
                node_hidden_size,
                node_out_size,
                edge_hidden_size,
                edge_out_size,
                rec_encode,
                send_encode,
                stochastic,
                one_hot,
                do_prob=0,
                dtype=torch.float32
                ):
        self.node_hidden_size = node_hiden_size
        self.node_out_size = node_out_size
        self.edge_hidden_size = edge_hidden_size
        self.edge_out_size = edge_out_size
        self.dtype=dtype

        #transfer from node embedding to edge embedding
        self.node2edge = MLP(self.node_hidden_size * 2, self.edge_hidden_size, self.edge_out_size, do_prob=do_prob)
        self.input_size = input_size + self.edge_out_size
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        # weight for GNN
        self.hidden_r = nn.Linear(self.node_hidden_size, self.node_hidden_size, bias=False)
        self.hidden_i = nn.Linear(self.node_hidden_size, self.node_hidden_size, bias=False)
        self.hidden_n = nn.Linear(self.node_hidden_size, self.node_hidden_size, bias=False)

        self.input_r = nn.Linear(self.input_size, self.node_hidden_size, bias=True)
        self.input_i = nn.Linear(self.input_size, self.node_hidden_size, bias=True)
        self.input_n = nn.Linear(self.input_size, self.node_hidden_size, bias=True)

    # This is the one single step GRU
    def one_step_forward(self, node_hidden, inputs):
        edges_input = node2edge(node_hidden, self.rec_encode, self.send_encode)
        edge_output = serlf.node2edge(edges_input)
        msg = edge2node(edge_output, self.rec_encode, self.send_encode)
        node_input = torch.cat(msg, inputs, dim=-1)
        # Perform GRU for each joint parrallelly
        r = F.sigmoid(self.input_r(node_input) + self.hidden_r(node_hidden))
        i = F.sigmoid(self.input_z(node_input) + self.hidden_i(node_hidden))
        n = FF.tanh(self.input_n(node_input) + r * self.hidden_n(node_hidden))
        hidden = (1.0 - i) * n + i * node_hidden
        return hidden

    def forward(self, input, hidden):
        output = torch.zeros(input.shape[0], input.shape[1], input.shape[2],self.node_hidden_size, requires_grad=False, dtype=self.dtype)
        if input.is_cuda:
            output.cuda()
        # do the recurrent nn
        for i in range(input.shape[0]):
            hidden = self.one_step_forward(hidden, input[i,:,:,:], self.rec_encode, self.send_encode)
            output[i,:,:,:] = hidden
        # the second output is useless, just to compatible to standard rnn
        return output, output

# Two layer MLP module
class MLP(nn.Module):
    def __init__(self, n_in, n_out, do_prob=0):
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hidden, n_out),
            nn.ReLU()
        )
        #TODO: verify whether to add bn
        self.init_weights()

    def init_weights(self):
        for m in self.modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight_data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.model(x)
        return x
