from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import torch
import torch.nn as nn
from helper import *
from gnn_module import GNNEncoder
class DecoderWrapper(nn.Module):
    def __init__(self,
                 cell,
                 rnn_size,
                 output_size,
                 target_seq_len,
                 residual_velocities,
                 device,
                 mask=None,
                 dtype=torch.float32):
        super(DecoderWrapper, self).__init__()
        self._cell = cell
        self.rnn_size = rnn_size
        self.output_size = output_size
        self.target_seq_len = target_seq_len
        self.residual = residual_velocities
        self.dtype= dtype
        self.device = device
        self.linear = nn.Linear(self.rnn_size, self.output_size)
        #Initial the linear op
        torch.nn.init.uniform_(self.linear.weight, -0.04 , 0.04)
        self.mask = mask
    def forward(self,input,state):

        def loop_function(prev, i):
            return prev

        outputs = []
        prev = None
        for i, inp in enumerate(input):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            # is it useful?
            inp = inp.clone()

            temp, state = self._cell(inp, state)
            # ugly hack here
            if len(inp.shape) == 3:
                next_frame = self.linear(temp) + inp[:,:,:self.output_size] if self.residual else self.linear(temp)
                inp[:,:self.output_size] = next_frame
                outputs.append(next_frame.view(1,input.shape[1], -1))
            else:
                next_frame = self.linear(temp) + inp[:,:,:,:self.output_size] if self.residual else self.linear(temp)
                if not self.mask is None:
                    next_frame = next_frame * self.mask
                outputs.append(next_frame.view(1, input.shape[1], input.shape[2], -1))
                inp[:,:,:self.output_size] = next_frame
            if loop_function is not None:
                prev = inp

        outputs = torch.cat(outputs, dim=0)
        return outputs, state

# Mean and std are seq * batch * 54 , sample is seq * batch * input_size
class StochasticDecoderWrapper(nn.Module):
    def __init__(self,
                 cell,
                 rnn_size,
                 # inter_dim,
                 output_size,
                 target_seq_len,
                 residual_velocities,
                 device,
                 mask=None,
                 dtype=torch.float32):
        super(StochasticDecoderWrapper, self).__init__()
        self._cell = cell
        self.rnn_size = rnn_size
        # self.inter_dim = inter_dim
        self.output_size = output_size
        self.residual = residual_velocities
        self.target_seq_len = target_seq_len
        self.device = device
        self.mask = mask
        self.dtype = dtype
        self.mean = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size // 2),
            nn.ReLU(),
            nn.Linear(self.rnn_size // 2, self.output_size)
        )
        # instead model std, model log(std) instead
        self.logstd = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size // 2),
            nn.ReLU(),
            nn.Linear(self.rnn_size // 2, self.output_size)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)


    def forward(self, input, state):

        def loop_function(prev, i):
            return prev

        means  = []
        logstds = []
        samples = []

        prev = None
        for i, inp in enumerate(input):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()
            # _, state = self._cell(inp, state)
            logstd = self.logstd(state)
            logstd = torch.clamp(logstd, -10., 10.)

            if len(input.shape) == 3:
                mean = self.mean(state) + inp[:,:self.output_size] if self.residual else self.mean(state)
                sample = reparam_sample_gauss(mean, logstd)
                # sample = reparam_sample_gauss(mean, logstd) + inp[:,:self.output_size] if self.residual else reparam_sample_gauss(mean, logstd)
                means.append(mean.view(1, input.shape[1], -1))
                logstds.append(logstd.view(1, input.shape[1], -1))
                samples.append(sample.view(1, input.shape[1], -1))
                inp[:,:output_size] = mean
            else:
                tmp = self.mean(state)
                mean = tmp + inp[:,:,:self.output_size] if self.residual else tmp
                sample = reparam_sample_gauss(mean, logstd)
                # sample = tmp + inp[:,:,:self.output_size] if self.residual else tmp
                if not self.mask is None:
                    mean = mean * self.mask
                    sample = sample * self.mask
                means.append(mean.view(1, input.shape[1], -1))
                logstds.append(logstd.view(1, input.shape[1], -1))
                samples.append(sample.view(1, input.shape[1], -1))
                inp[:,:,:self.output_size] = mean
            _, state = self._cell(inp, state)
            if loop_function is not None:
                prev = inp

        output_mean = torch.cat(means, dim=0)
        output_logstd = torch.cat(logstds, dim=0)
        output_sample = torch.cat(samples, dim=0)

        return output_mean, output_logstd, output_sample, state


# Mean and std are seq * batch * 54 , sample is seq * batch * input_size
class StochasticDecoderWrapper2(nn.Module):
    def __init__(self,
                 cell,
                 rnn_size,
                 # inter_dim,
                 output_size,
                 target_seq_len,
                 residual_velocities,
                 device,
                 node_hidden_dim,
                 node_out_dim,
                 edge_hidden_dim,
                 edge_out_dim,
                 rec_encode,
                 send_encode,
                 num_passing,
                 do_prob=0,
                 mask=None,
                 dtype=torch.float32):
        super(StochasticDecoderWrapper2, self).__init__()
        self._cell = cell
        self.rnn_size = rnn_size
        # self.inter_dim = inter_dim
        self.output_size = output_size
        self.residual = residual_velocities
        self.target_seq_len = target_seq_len
        self.device = device
        self.mask = mask
        self.dtype = dtype
        self.rec_encode = rec_encode
        self.send_encode = send_encode
        self.mean = nn.Sequential(
            nn.Linear(node_out_dim, node_out_dim // 2),
            nn.ReLU(),
            nn.Linear(node_out_dim // 2, self.output_size)
        )
        # instead model std, model log(std) instead
        self.logstd = nn.Sequential(
            nn.Linear(node_out_dim, node_out_dim//2),
            nn.ReLU(),
            nn.Linear(node_out_dim // 2, self.output_size)
        )
        self.GNN = GNNEncoder(
            # self.node_out_dim,
            node_hidden_dim,
            node_out_dim,
            edge_hidden_dim,
            edge_out_dim,
            rec_encode,
            send_encode,
            num_passing,
            weight_share=False,
            do_prob=do_prob
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)

    def forward(self, input, state, edge_weight):

        def loop_function(prev, i):
            return prev

        means  = []
        logstds = []
        samples = []
        edge_weights = []

        prev = None
        for i, inp in enumerate(input):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()
            edge_weights.append(edge_weight.view(1, input.shape[1],-1))
            # _, state = self._cell(inp, state)
            gnn_out, edge_weight = self.GNN(state, edge_weight)
            logstd = self.logstd(gnn_out)
            logstd = torch.clamp(logstd, -10., 10.)

            if len(input.shape) == 3:
                mean = self.mean(gnn_out) + inp[:,:self.output_size] if self.residual else self.mean(gnn_out)
                sample = reparam_sample_gauss(mean, logstd)
                # sample = reparam_sample_gauss(mean, logstd) + inp[:,:self.output_size] if self.residual else reparam_sample_gauss(mean, logstd)
                means.append(mean.view(1, input.shape[1], -1))
                logstds.append(logstd.view(1, input.shape[1], -1))
                samples.append(sample.view(1, input.shape[1], -1))
                inp[:,:output_size] = mean
            else:
                tmp = self.mean(gnn_out)
                mean = tmp + inp[:,:,:self.output_size] if self.residual else tmp
                sample = reparam_sample_gauss(mean, logstd)
                # sample = tmp + inp[:,:,:self.output_size] if self.residual else tmp
                if not self.mask is None:
                    mean = mean * self.mask
                    sample = sample * self.mask
                means.append(mean.view(1, input.shape[1], -1))
                logstds.append(logstd.view(1, input.shape[1], -1))
                samples.append(sample.view(1, input.shape[1], -1))
                inp[:,:,:self.output_size] = mean
            _, state = self._cell(inp, state)
            if loop_function is not None:
                prev = inp

        output_mean = torch.cat(means, dim=0)
        output_logstd = torch.cat(logstds, dim=0)
        output_sample = torch.cat(samples, dim=0)
        output_edgeweight = torch.cat(edge_weights, dim=0)
        return output_mean, output_logstd, output_sample, state, output_edgeweight
