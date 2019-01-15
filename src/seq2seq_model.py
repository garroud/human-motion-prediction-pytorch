"""Seq-to-Seq model for human motion prediction in pytorch"""

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

from tensorboardX import SummaryWriter

class Seq2SeqModel(nn.Module):

    def __init__(self,
                architecture,
                source_seq_len,
                target_seq_len,
                rnn_size,
                num_layers,
                max_gradient_norm,
                batch_size,
                learning_rate,
                learning_rate_decay_factor,
                summaries_dir,
                loss_to_use,
                number_of_actions,
                one_hot=True,
                residual_velocities=False,
                dtype=torch.float32):
        """Create the model.

        Args:
        architecture: [basic, tied] whether to tie the decoder and decoder.
        source_seq_len: lenght of the input sequence.
        target_seq_len: lenght of the target sequence.
        rnn_size: number of units in the rnn.
        num_layers: number of rnns to stack.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.
        summaries_dir: where to log progress for tensorboard.
        loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
        number_of_actions: number of classes we have.
        one_hot: whether to use one_hot encoding during train/test (sup models).
        residual_velocities: whether to use a residual connection that models velocities.
        dtype: the data type to use to store internal variables.
        """
        super(Seq2SeqModel, self).__init__()
        self.HUMAN_SIZE = 54
        self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

        print("One hot is ", one_hot)
        print("Input size is %d" % self.input_size)

        self.train_writer = SummaryWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
        self.test_writer = SummaryWriter(os.path.normpath(os.path.join(summaries_dir, 'test')))

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.global_step = 0

        # === Create the RNN that will keep the state ===
        print('rnn_size= {0}'.format(rnn_size))
        self.encoder = nn.GRU(input_size=self.input_size,hidden_size=self.rnn_size,num_layers=num_layers)
        if architecture == "tied":
            self.decoder = self.encoder
        elif architecture == "basic_rnn_seq2seq":
            self.decoder = nn.GRU(input_size=self.input_size,hidden_size=self.rnn_size,num_layers=num_layers)
        else:
            raise(ValueError, "Unknown architecture: %s " % architecture )
        self.linear = nn.Linear(self.rnn_size, self.input_size)
        #Initial the linear op
        torch.nn.init.uniform_(self.linear.weight, -0.04 , 0.04)
        self.decoder = decoderWrapper.DecoderWrapper(self.decoder, self.linear, target_seq_len, residual_velocities,dtype)

        self.loss = nn.MSELoss(reduction='mean')


    def forward(self, encoder_input, decoder_input):
        # h0 = torch.zeros(self.num_layers, self.batch_size, self.rnn_size).cuda()
        _ , interState = self.encoder(encoder_input,None)
        last_frame = decoder_input[0,:,:].view(1,-1,self.input_size)
        output, state = self.decoder(last_frame,interState)
        return output, state

    def get_batch( self, data, actions ):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys    = list(data.keys())
        chosen_keys = np.random.choice( len(all_keys), self.batch_size )

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
        decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in xrange( self.batch_size ):

          the_key = all_keys[ chosen_keys[i] ]

          # Get the number of frames
          n, _ = data[ the_key ].shape

          # Sample somewherein the middle
          idx = np.random.randint( 16, n-total_frames )

          # Select the data around the sampled points
          data_sel = data[ the_key ][idx:idx+total_frames ,:]

          # Add the data
          encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
          decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
          decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn( self, data, action ):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState( SEED )

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
        T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        return idx

    def get_batch_srnn(self, data, action ):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                  "posing", "purchases", "sitting", "sittingdown", "smoking",
                  "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if not action in actions:
          raise ValueError("Unrecognized action {0}".format(action))

        frames = {}
        frames[ action ] = self.find_indices_srnn( data, action )

        batch_size = 8 # we always evaluate 8 seeds
        subject    = 5 # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

        encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
        decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
        decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in xrange( batch_size ):

          _, subsequence, idx = seeds[i]
          idx = idx + 50

          data_sel = data[ (subject, action, subsequence, 'even') ]

          data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

          encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
          decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
          decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


        return encoder_inputs, decoder_inputs, decoder_outputs
