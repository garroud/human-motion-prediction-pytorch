from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
from modules.basic_modules import *
from modules.decoderWrapper import StochasticDecoderWrapper2
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class SimpleEdgeInferModule(nn.Module):
    def __init__(self,
        encoder_input_size,
        output_size,
        node_hidden_dim,
        node_out_dim,
        edge_hidden_dim,
        edge_out_dim,
        rec_encode,
        send_encode,
        num_passing=1,
        weight_share=False,
        do_prob=0,
        edge_update=1):

        super(SimpleEdgeInferModule, self).__init__()

        self.encoder_input_size = encoder_input_size
        self.output_size = output_size
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        self.edge_update = edge_update
        self.mlp = MLP(
            self.encoder_input_size,
            self.node_hidden_dim,
            self.node_out_dim,
        )

        self.encoder = GNNEncoder(
            # self.node_out_dim,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            num_passing,
            weight_share=weight_share,
            do_prob=do_prob
        )

        self.node_decode = nn.Linear(self.node_out_dim, self.output_size)

        self.decoder = GNNEncoder(
            # self.node_out_dim,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            1,
            weight_share=weight_share,
            do_prob=do_prob
        )

    def forward(self,x,edge_weight=None):

        x = self.mlp(x)
        x, edge_weight = self.encoder(x, edge_weight)
        node_decode, _ = self.decoder(x, edge_weight)
        return self.node_decode(node_decode), edge_weight


class EdgeInferModule(nn.Module):
    def __init__(self,
        encoder_input_size,
        decoder_input_size,
        output_size,
        target_seq_len,
        node_hidden_dim,
        node_out_dim,
        edge_hidden_dim,
        edge_out_dim,
        rec_encode,
        send_encode,
        joint_dim,
        device,
        num_passing=1,
        weight_share=False,
        do_prob=0,
        dtype=torch.float32):

        super(EdgeInferModule, self).__init__()

        self.encoder_input_size = encoder_input_size
        self.output_size = output_size
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.rec_encode = rec_encode
        self.send_encode = send_encode

        self.mlp = MLP(
            self.encoder_input_size,
            self.node_hidden_dim,
            self.node_out_dim,
        )

        self.encoder = GNNEncoder(
            # self.node_out_dim,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            num_passing,
            weight_share=weight_share,
            do_prob=do_prob
        )

        core_decoder = RGNN(
            decoder_input_size,
            self.node_hidden_dim,
            self.node_out_dim,
            self.edge_hidden_dim,
            self.edge_out_dim,
            self.rec_encode,
            self.send_encode,
            joint_dim,
            1,
            do_prob,
            dtype,
        )

        self.decoder = StochasticDecoderWrapper2(
                core_decoder,
                self.node_out_dim,
                output_size,
                target_seq_len,
                True,
                device,
                self.node_hidden_dim,
                self.node_out_dim,
                self.edge_hidden_dim,
                self.node_out_dim,
                self.rec_encode,
                self.send_encode,
                1,
                do_prob,
                None,
                dtype,
            )

    def forward(self,x,decoder_input,edge_weight=None):

        x = self.mlp(x)
        x, edge_weight = self.encoder(x, edge_weight)
        means,_,_,_,edge_weight= self.decoder(decoder_input, x,edge_weight)


        return means, edge_weight

#do some test here
if __name__ == "__main__":
    # encoder_input = torch.zeros([49, 8, 21,18],requires_grad=False, dtype=torch.float32).cuda()
    # encoder_input = torch.zeros([8, 21, 50 * 3 + 15],requires_grad=False, dtype=torch.float32).cuda()
    # encoder_input.random_()
    # decoder_input = torch.zeros([24, 8, 21, 18], requires_grad=False, dtype=torch.float32).cuda()
    # decoder_input.random_()
    # #test against a fully connected graph
    off_diag = np.ones([5,5]) - np.eye(5)
    rec_encode = np.array(encode_onehot(np.where(off_diag)[1]),dtype=np.float32)
    send_encode = np.array(encode_onehot(np.where(off_diag)[0]),dtype=np.float32)
    rec_encode = torch.FloatTensor(rec_encode).cuda()
    send_encode = torch.FloatTensor(send_encode).cuda()

    x = torch.zeros(128,5,48*4).cuda()
    decoder_input = torch.zeros(10,128,5,4).cuda()
    model = EdgeInferModule(
        48*4,
        4,
        4,
        10,
        128,
        128,
        128,
        128,
        rec_encode,
        send_encode,
        joint_dim=5,
        device='cuda',
        num_passing=2,
    ).cuda()
    n, e = model(x,decoder_input)
    print(n.shape)
    print(e.shape)
