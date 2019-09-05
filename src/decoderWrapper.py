from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import torch
import torch.nn as nn
from helper import *

class DecoderWrapper(nn.Module):
    def __init__(self,
                 cell,
                 rnn_size,
                 output_size,
                 target_seq_len,
                 residual_velocities,
                 device,
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

    def forward(self,input,state):
        output = torch.zeros(self.target_seq_len,input.shape[1], input.shape[2] ,requires_grad=False, dtype=self.dtype).to(self.device)
        for i in xrange(self.target_seq_len):
            temp, state = self._cell(input, state)
            new_frame = input.clone()
            new_frame[:,:,:self.output_size] = self.linear(temp) + input[:,:,:self.output_size] if self.residual else self.linear(temp)
            output[i] = new_frame
        return output, state

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
                 dtype=torch.float32):
        super(StochasticDecoderWrapper, self).__init__()
        self._cell = cell
        self.rnn_size = rnn_size
        # self.inter_dim = inter_dim
        self.output_size = output_size
        self.residual = residual_velocities
        self.target_seq_len = target_seq_len
        self.device = device
        self.dtype = dtype
        # self.init_dec = nn.Sequential(
        #                      nn.Linear(self.rnn_size, self.inter_dim),
        #                      nn.ReLU())
        # self.init_mean = nn.Linear(self.rnn_size, self.output_size)
        # self.dec = nn.Sequential(
        #                      nn.Linear(self.rnn_size, self.inter_dim),
        #                      nn.ReLU())
        self.mean = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.output_size)
        )
        # instead model std, model log(std) instead
        self.logstd = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)


    def forward(self, input, state):
        output_mean = torch.zeros(self.target_seq_len,input.shape[1], self.output_size ,requires_grad=False, dtype=self.dtype).to(self.device)
        output_logstd = torch.zeros(self.target_seq_len,input.shape[1], self.output_size ,requires_grad=False, dtype=self.dtype).to(self.device)
        output_sample = torch.zeros(self.target_seq_len,input.shape[1], input.shape[2] ,requires_grad=False, dtype=self.dtype).to(self.device)
        last_mean = input[0,:,:self.output_size]
        for i in xrange(self.target_seq_len):
            temp, state = self._cell(input, state)
            mean = self.mean(temp)
            logstd = self.logstd(temp)
            next_frame = input.clone()
            next_frame[:,:,:self.output_size] = reparam_sample_gauss(mean, logstd) + input[:,:,:self.output_size] if self.residual else reparam_sample_gauss(mean, logstd)
            # output_mean[i] = mean + last_mean if self.residual else mean
            output_mean[i] = mean
            output_logstd[i] = logstd
            last_mean = last_mean + mean if self.residual else mean
            output_sample[i] = next_frame
            # using the stochastic sample as the input of next stage
            # input = next_frame[i]
            # use mean as the input of the next stage
            input = input.clone()
            input[:,:,:self.output_size] = last_mean
        return output_mean, output_logstd, output_sample, state
