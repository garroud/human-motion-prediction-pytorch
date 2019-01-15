from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import torch
import torch.nn as nn

class DecoderWrapper(nn.Module):
    def __init__(self,
                 cell,
                 linear,
                 target_seq_len,
                 residual_velocities,
                 dtype=torch.float32):
        super(DecoderWrapper, self).__init__()
        self._cell = cell
        self._linear = linear
        self.target_seq_len = target_seq_len
        self.residual = residual_velocities
        self.dtype= dtype

    def forward(self,input,state):
        output = torch.zeros(self.target_seq_len, input.shape[1], input.shape[2] ,requires_grad=False, dtype=self.dtype).cuda()
        for i in xrange(self.target_seq_len):
            input, state = self._cell(input, state)
            input = self._linear(input) + input if self.residual else self._linear(input)
            output[i] = input
        return output, state
