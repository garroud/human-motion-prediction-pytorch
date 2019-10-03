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

from itertools import chain

# two basic operation for node and edge conversion
def node2edge(x, rec_encode, send_encode):
    recs = torch.matmul(rec_encode, x)
    sends = torch.matmul(send_encode, x)
    edges = torch.cat([sends, recs], dim=-1)
    return edges

def edge2node(x, rec_encode, send_encode):
    incomming = torch.matmul(rec_encode.t(), x)
    return incomming / rec_encode.sum(0).view(-1,1).expand(incomming.size())

class GNNModel(nn.Module):
    def __init__(self,
        source_seq_len,
        target_seq_len,
        edge_hidden_dim,
        edge_out_dim,
        node_hidden_dim,
        node_out_dim,
        num_passing,
        rec_encode,
        send_encode,
        batch_size,
        number_of_actions,
        device,
        weight_share=True,
        do_prob=0.0,
        one_hot=False,
        residual_velocities=False,
        stochastic=False,
        dtype=torch.float32
    ):
        super(GNNModel, self).__init__()
        self.NUM_JOINT = 32
        self.JOINT_DIM = 3 * source_seq_len
        self.encoder_input_size= self.JOINT_DIM + number_of_actions if one_hot else  self.JOINT_DIM
        self.decoder_input_size = 3 + number_of_actions if one_hot else 3

        print("One hot is ", one_hot)
        print("Encoder input size is %d" % self.encoder_input_size)
        print("decoder input size is %d" % self.decoder_input_size)

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.node_hidden_dim = edge_hidden_dim
        self.node_out_dim = node_out_dim
        self.batch_size = batch_size
        self.stochastic = stochastic
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        self.dtype=dtype

        # declare the encoder, it outputs (batch_size * NUM_JOINT * node_hidden_dim)
        self.encoder = GNNEncoder(
            self.encoder_input_size,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            num_passing,
            self.rec_encode,
            self.send_encode,
            weight_share,
            do_prob,
        )

        core_decoder = GNNDecoder(
            self.decoder_input_size,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            do_prob,
            self.dtype,
        )

        if not stochastic:
            self.decoder = decoderWrapper.DecoderWrapper(
                core_decoder,
                self.node_out_dim,
                # 3d coordinate or 3d angle,
                3,
                self.target_seq_len,
                residual_velocities,
                device,
                dtype
            )
        else:
            self.decoder = decoderWrapper.StochasticDecoderWrapper(
                core_decoder,
                self.node_out_dim,
                3,
                self.target_seq_len,
                residual_velocities,
                device,
                dtype
            )
        self.loss = nll_gauss

    # encoder input should be batch_size * num_joint * input_size
    # decoder input should be target_seq_len * batch_size * num_joint * (3 + one_hot)
    def forward(self, encoder_input, decoder_input):
        node_hidden = self.encoder(encoder_input)
        if not self.stochastic:
            output, state = self.decoder(decoder_input, node_hidden)
            return output, state
        else:
            means, stds, samples, states = self.decoder(decoder_input, node_hidden)
            return means, stds, samples, states


class GNNEncoder(nn.Module):
    def __init__(self,
                input_size,
                node_hidden_dim,
                node_out_dim,
                edge_hidden_dim,
                edge_out_dim,
                num_passing,
                rec_encode,
                send_encode,
                weight_share=True,
                do_prob=0):

        super(GNNEncoder, self).__init__()
        self.input_size = input_size
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.do_prob = do_prob
        self.num_passing = num_passing
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        self.weight_share = weight_share

        print("Dim of hidden node state is ", self.node_hidden_dim)
        print("Dim of hidden edge state is ", self.edge_hidden_dim)

        # transform original node feature to node embedding
        self.mlp1 = MLP(self.input_size, self.node_hidden_dim, self.node_out_dim)
        if weight_share:
            self.passing_list = nn.ModuleList([
                MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim, do_prob=do_prob),
                MLP(self.edge_out_dim, self.node_hidden_dim, self.node_out_dim, do_prob=do_prob)
            ])
        else:
            self.passing_list = [
                [MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim,   do_prob=do_prob),
                MLP(self.edge_out_dim, self.node_hidden_dim, self.node_out_dim, do_prob=do_prob)]
                for _ in range(num_passing)
            ]
            self.passing_list = nn.ModuleList(list(chain(self.passing_list)))

    def forward(self, x):
        x = self.mlp1(x)
        for passing in range(self.num_passing):
            idx = 0 if self.weight_share else passing // 2
            x = node2edge(x, self.rec_encode, self.send_encode)
            x = self.passing_list[idx](x)
            x = edge2node(x, self.rec_encode, self.send_encode)
            x = self.passing_list[idx+1](x)
        return x

class GNNDecoder(nn.Module):
    def __init__(self,
                input_size,
                node_out_dim,
                edge_hidden_dim,
                edge_out_dim,
                rec_encode,
                send_encode,
                do_prob=0,
                dtype=torch.float32
                ):
        super(GNNDecoder, self).__init__()
        self.node_out_dim = node_out_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.dtype=dtype

        #transfer from node embedding to edge embedding
        self.node2edge = MLP(self.node_out_dim * 2, self.edge_hidden_dim, self.edge_out_dim, do_prob=do_prob)
        self.input_size = input_size + self.edge_out_dim
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        # weight for GNN
        self.hidden_r = nn.Linear(self.node_out_dim, self.node_out_dim, bias=False)
        self.hidden_i = nn.Linear(self.node_out_dim, self.node_out_dim, bias=False)
        self.hidden_n = nn.Linear(self.node_out_dim, self.node_out_dim, bias=False)

        self.input_r = nn.Linear(self.input_size, self.node_out_dim, bias=True)
        self.input_i = nn.Linear(self.input_size, self.node_out_dim, bias=True)
        self.input_n = nn.Linear(self.input_size, self.node_out_dim, bias=True)

    # This is the one single step GRU
    def one_step_forward(self, node_hidden, inputs):
        edges_input = node2edge(node_hidden, self.rec_encode, self.send_encode)
        edge_output = self.node2edge(edges_input)
        msg = edge2node(edge_output, self.rec_encode, self.send_encode)
        node_input = torch.cat([msg, inputs], dim=-1)
        # Perform GRU for each joint parrallelly
        r = torch.sigmoid(self.input_r(node_input) + self.hidden_r(node_hidden))
        i = torch.sigmoid(self.input_i(node_input) + self.hidden_i(node_hidden))
        n = torch.tanh(self.input_n(node_input) + r * self.hidden_n(node_hidden))
        hidden = (1.0 - i) * n + i * node_hidden
        return hidden, hidden

    def forward(self, input, hidden):
        outputs = []
        if (len(input.shape) == 3):
            input = input.view(
                list(chain(
                    [1],
                    input.shape,
                ))
            )
        # do the recurrent nn
        for i, inp in enumerate(input):
            hidden, state = self.one_step_forward(hidden, inp)
            outputs.append(hidden.view(list(chain(
                [1],
                hidden.shape,
            ))))
        # the second output is useless, just to compatible to standard rnn
        outputs = torch.cat(outputs,dim=0)
        if input.is_cuda:
            outputs.cuda()
        return outputs, state

# Two layer MLP module
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, do_prob=0):
        super(MLP, self).__init__()
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.model(x)
        return x

#do some test here
if __name__ == "__main__":
    encoder_input = torch.zeros([8, 32, 150],requires_grad=False, dtype=torch.float32).cuda()
    encoder_input.random_()
    decoder_input = torch.zeros([25, 8, 32, 3], requires_grad=False, dtype=torch.float32).cuda()
    decoder_input.random_()
    #test against a fully connected graph
    off_diag = np.ones([32,32]) - np.eye(32)
    rec_encode = np.array(encode_onehot(np.where(off_diag)[1]),dtype=np.float32)
    send_encode = np.array(encode_onehot(np.where(off_diag)[0]),dtype=np.float32)
    rec_encode = torch.FloatTensor(rec_encode).cuda()
    send_encode = torch.FloatTensor(send_encode).cuda()

    print(rec_encode.shape)
    print(send_encode.shape)

    gnn_model = GNNModel(
        source_seq_len=50,
        target_seq_len=25,
        edge_hidden_dim=512,
        edge_out_dim=512,
        node_hidden_dim=512,
        node_out_dim=512,
        num_passing=2,
        rec_encode=rec_encode,
        send_encode=send_encode,
        batch_size=128,
        number_of_actions=15,
        device='cuda',
        residual_velocities=True,
        stochastic=True
    )

    gnn_model = gnn_model.to('cuda')

    means, logstds, samples, states = gnn_model(encoder_input, decoder_input)
    print(means.shape)
    gnn_model.backward()
