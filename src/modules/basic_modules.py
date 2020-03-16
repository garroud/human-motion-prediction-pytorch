from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
# from human_motion.helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


# two basic operation for node and edge conversion
def node2edge(x, rec_encode, send_encode):
    recs = torch.matmul(rec_encode, x)
    sends = torch.matmul(send_encode, x)
    # edges = torch.matmul(recs.unsqueeze(3), sends.unsqueeze(2)).view(recs.size(0), recs.size(1), -1)
    edges = torch.cat([sends, recs], dim=-1)
    return edges

def edge2node(x, rec_encode, send_encode, edge_weight):
    x = x * edge_weight
    incomming = torch.matmul(rec_encode.t(), x)
    weight_norm = torch.matmul(rec_encode.t(), edge_weight)
    return incomming / weight_norm


class GNNEncoder(nn.Module):
    def __init__(self,
                node_hidden_dim,
                node_out_dim,
                edge_hidden_dim,
                edge_out_dim,
                rec_encode,
                send_encode,
                num_passing=1,
                weight_share=False,
                do_prob=0):

        super(GNNEncoder, self).__init__()
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.do_prob = do_prob
        self.num_passing = num_passing
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        self.weight_share = weight_share

        # instead of using prob, output weight for further usage
        self.edge_decode = nn.Sequential(
            nn.Linear(self.edge_out_dim, 1),
        )

        print("Dim of hidden node state is ", self.node_hidden_dim)
        print("Dim of hidden edge state is ", self.edge_hidden_dim)

        if num_passing > 0:
            passing_list = [
                [MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim,   do_prob=do_prob),
                MLP(self.edge_out_dim, self.node_hidden_dim, self.node_out_dim, do_prob=do_prob)]
                for _ in range(num_passing)
            ]
            self.passing_list = nn.ModuleList(list(chain(*passing_list)))

    # residual version
    def forward(self, node_feature, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(node_feature.shape[0], self.rec_encode.shape[0], 1).to(node_feature.device)

        node_skip = node_feature
        edge_skip = None
        for passing in range(self.num_passing):
            idx = passing // 2
            edge_feature = node2edge(node_feature, self.rec_encode, self.send_encode)
            edge_feature = self.passing_list[idx*2](edge_feature)
            if not edge_skip is None:
                edge_feature = edge_skip + edge_feature
            edge_skip = edge_feature
            # node_feature = edge2node(edge_feature, self.rec_encode, self.send_encode, self.edge_decode(edge_feature))
            node_feature = edge2node(edge_feature, self.rec_encode, self.send_encode, edge_weight)
            node_feature = self.passing_list[idx*2+1](node_feature)
            if not node_feature is None:
                node_feature = node_skip + node_feature
            node_skip = node_feature
        return node_feature, self.edge_decode(edge_feature)

# equivalent to an RNN with graph
class RGNN(nn.Module):
    def __init__(self,
                input_size,
                node_hidden_dim,
                node_out_dim,
                edge_hidden_dim,
                edge_out_dim,
                rec_encode,
                send_encode,
                joint_dim=21,
                num_passing=1,
                do_prob=0,
                dtype=torch.float32
                ):
        super(RGNN, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.dtype=dtype
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        #transfer from node embedding to edge embedding
        self.GNN = GNNEncoder(
            # self.node_out_dim,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            num_passing,
            weight_share=False,
            do_prob=do_prob
        )
        # self.input_size = input_size + self.edge_out_dim
        self.JOINT_DIM = joint_dim
        self.input_size = input_size

        # each joint has a gru if not sharing weight
        self.grus = nn.ModuleList([nn.GRU(input_size=self.input_size,hidden_size=self.node_out_dim) for _ in range(self.JOINT_DIM)])

    # This is the one single step GRU for sharing weight
    # def one_step_forward(self, node_hidden, inputs):
    #     # node_input = inputs.clone()
    #     node_input = self.GNN(inputs.view([1]+node_input.shape))
    #     # Perform GRU for each joint parrallelly
    #     r = torch.sigmoid(self.input_r(node_input) +self.hidden_r(node_hidden))
    #     z = torch.sigmoid(self.input_i(node_input) + self.hidden_i(node_hidden))
    #     n = torch.tanh(self.input_n(node_input) +  r * self.hidden_n(node_hidden))
    #     node_hidden = (1.0 - z) * node_hidden + z * n
    #     # node_hidden = self.GNN(node_hidden)
    #     return node_hidden

    # This is a single step for not sharing weight
    def one_step_forward(self, node_hidden, inputs):
        # inputs = self.GNN(inputs)
        inputs = inputs.unsqueeze(0).permute(2,0,1,3)
        node_hidden = node_hidden.permute(1,0,2).contiguous()

        hidden = []
        for i in range(self.JOINT_DIM):
            hidden.append(self.grus[i](inputs[i],node_hidden[i:i+1])[1])
        hidden = torch.cat(hidden, dim=0).permute(1,0,2).contiguous()
        # hidden = self.GNN(hidden)
        return hidden

    def forward(self, input, hidden):
        outputs = []
        input = input.clone()
        if (len(input.shape) == 3):
            input.unsqueeze_(0)
        if hidden is None:
            hidden = torch.zeros(
                input.shape[1], input.shape[2], self.node_out_dim
            ).to(input.device)
        # do the recurrent nn
        for i, inp in enumerate(input):
            hidden = self.one_step_forward(hidden, inp)
            outputs.append(hidden.unsqueeze(0))
        # the second output is useless, just to compatible to standard rnn
        outputs = torch.cat(outputs,dim=0).to(input.device)
        # return outputs, hidden
        return outputs, hidden

# Two layer MLP module
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, do_prob=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout= nn.Dropout(do_prob)

        # self.init_weights()

    def batch_norm(self, inputs):
        orign_shape = inputs.shape
        x = inputs.view(-1, orign_shape[-1])
        x = self.bn(x)
        return x.view(orign_shape)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)
