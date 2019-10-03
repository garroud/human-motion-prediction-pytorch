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

        def loop_function(prev, i):
            return prev

        outputs = []
        prev = None
        for i, inp in enumerate(input):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            # is it useful?
            inp = inp.detach()

            temp, state = self._cell(inp, state)
            # ugly hack here
            if len(inp.shape) == 3:
                next_frame = self.linear(temp) + inp[:,:,:self.output_size] if self.residual else self.linear(temp)
                inp[:,:self.output_size] = next_frame
                outputs.append(next_frame.view(1,input.shape[1], -1))
            else:
                next_frame = self.linear(temp) + inp[:,:,:,:self.output_size] if self.residual else self.linear(temp)
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

        def loop_function(prev, i):
            return prev

        means  = []
        logstds = []
        samples = []
        prev = None
        
        for i, inp in enumerate(input):
            inp_mean = inp
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()

            temp, state = self._cell(inp, state)
            logstd = self.logstd(temp)

            if len(input.shape) == 3:
                mean = self.mean(temp) + inp[:,:self.output_size] if self.residual else self.mean(temp)
                sample = reparam_sample_gauss(mean, logstd)
                means.append(mean.view(1, input.shape[1], input.shape[2]))
                logstds.append(logstd.view(1, input.shape[1], input.shape[2]))
                samples.append(sample.view(1, input.shape[1], input.shape[2]))
                inp[:,:output_size] = samples
            else:
                mean = self.mean(temp) + inp[:,:,:self.output_size] if self.residual else self.mean(temp)
                sample = reparam_sample_gauss(mean, logstd)
                means.append(mean.view(1, input.shape[1], input.shape[2],-1))
                logstds.append(logstd.view(1, input.shape[1], input.shape[2],-1))
                samples.append(sample.view(1, input.shape[1], input.shape[2],-1))
                inp[:,:,:self.output_size] = sample
            if loop_function is not None:
                prev = inp

        output_mean = torch.cat(means, dim=0)
        output_logstd = torch.cat(logstds, dim=0)
        output_sample = torch.cat(samples, dim=0)

        return output_mean, output_logstd, output_sample, state
