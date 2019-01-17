"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import argparse

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin

import data_utils
import seq2seq_model

import torch

parser = argparse.ArgumentParser(description="Human Motion Model")
#Learning specification
parser.add_argument('--learning_rate', default=0.005, type=float, metavar='N', help='Initial learning rate.')
parser.add_argument('--learning_rate_decay_factor', default=0.95, type=float, metavar='N', help='Learning rate is multiplied by this much.')
parser.add_argument('--learning_rate_step', default=10000, type=int, metavar="N",help="Every this many steps, do decay.")
parser.add_argument('--max_gradient_norm', default=5, type=int, metavar='N', help="Clip gradients to this norm")
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='Batch size for each iteration')
parser.add_argument('--iterations', default=int(1e5), type=int, metavar='N', help="Iterations to train for")
#Architecture
parser.add_argument('--architecture', default='tied', type=str, metavar='S', help="Seq2Seq model architecture for use: [basic, tied]")
parser.add_argument("--size", default=1024, type=int, metavar='N', help="Size of each model layer.")
parser.add_argument("--num_layers", default=1, type=int, metavar='N', help="Number of layers in the model.")
parser.add_argument("--seq_length_in", default=50, type=int, metavar='N', help="Number of frames to feed into the encoder. 25 fps")
parser.add_argument("--seq_length_out", default=10, type=int, metavar='N', help="Number of frames that the decoder has to predict. 25fps")
parser.add_argument("--omit_one_hot", action='store_true', help="Whether to remove one-hot encoding from the data")
parser.add_argument("--residual_velocities", action='store_true', help="Add a residual connection that effectively models velocities")
parser.add_argument("--data_dir", default=os.path.normpath("./data/h3.6m/dataset"), type=str, metavar='S', help="Data directory")
parser.add_argument("--train_dir", default=os.path.normpath("./experiments/"), type=str, metavar='S', help="Training directory.")

parser.add_argument("--action",default="all", type=str, metavar='S', help="The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
parser.add_argument("--loss_to_use",default="sampling_based", metavar='S', help="The type of loss to use, supervised or sampling_based")

parser.add_argument("--test_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--save_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--show_every", default=100, type=int, metavar='N', help="How often to show error during training.")
parser.add_argument("--sample", action='store_true' ,help="Set to True for sampling.")
parser.add_argument("--use_cpu", action='store_true', help="Whether to use the CPU")
parser.add_argument("--load", default=0, type=int, metavar='N', help="Try to load a previous checkpoint.")

FLAGS = parser.parse_args()

train_dir = os.path.normpath(os.path.join( FLAGS.train_dir, FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  FLAGS.architecture,
  FLAGS.loss_to_use,
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel'))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

device = torch.device('cpu' if FLAGS.use_cpu else 'cuda')

transform = lambda x: torch.tensor(x, dtype=torch.float32).permute(1,0,2).to(device)

def create_model(actions, sampling=False):

    model = seq2seq_model.Seq2SeqModel(
        FLAGS.architecture,
        FLAGS.seq_length_in if not sampling else 50,
        FLAGS.seq_length_out if not sampling else 100,
        FLAGS.size, # hidden layer size
        FLAGS.num_layers,
        FLAGS.batch_size,
        summaries_dir,
        FLAGS.loss_to_use if not sampling else "sampling_based",
        len( actions ),
        device,
        not FLAGS.omit_one_hot,
        FLAGS.residual_velocities,
        dtype=torch.float32)
    #initalize a new model
    if FLAGS.load <= 0:
        print("Creating model with fresh parameters.")
        #TODO: Initial parameter here
        return model
    #Load model from iteration
    if os.path.isfile(os.path.join(train_dir, 'checkpoint-{0}.pt'.format(FLAGS.load))):
        model.load_state_dict(torch.load(os.path.join(train_dir, 'checkpoint-{0}.pt'.format(FLAGS.load))))
    else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))

    #TODO: Load a pretarined model
    return model

def train():

    actions = define_actions(FLAGS.action)
    number_of_actions = len(actions)
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
        actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(actions,sampling=False)
    model.to(device)
    print("Model created")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot )



    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    lr = FLAGS.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in xrange(FLAGS.iterations):

        start_time = time.time()
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set, not FLAGS.omit_one_hot )

        model.train()
        output, _ = model(transform(encoder_inputs), transform(decoder_inputs))
        optimizer.zero_grad()

        step_loss = model.loss(output, transform(decoder_outputs))
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),FLAGS.max_gradient_norm)
        optimizer.step()

        if current_step % FLAGS.show_every == 0:
            print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

        step_time += (time.time() - start_time) / FLAGS.test_every
        loss += step_loss / FLAGS.test_every
        current_step += 1

        ## step decay ##
        if current_step % FLAGS.learning_rate_step == 0:
            lr *= FLAGS.learning_rate_decay_factor
            for g in optimizer.param_groups:
                g['lr'] = lr

        ## Validation step ##
        if current_step % FLAGS.test_every == 0:
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set, not FLAGS.omit_one_hot )
            ##TODO: a forward pass
            model.eval()
            output, _ = model(transform(encoder_inputs), transform(decoder_inputs))
            step_loss = model.loss(output,transform(decoder_outputs))
            val_loss = step_loss
            print()
            print("{0: <16} |".format("milliseconds"), end="")
            for ms in [80, 160, 320, 400, 560, 1000]:
                print(" {0:5d} |".format(ms), end="")
            print()

            # === Validation with srnn's seeds ===
            for action in actions:
                # Evaluate the model on the test batches
                encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)
                srnn_poses, _ = model(transform(encoder_inputs), transform(decoder_inputs))
                srnn_loss = model.loss(srnn_poses,transform(decoder_outputs))

                # Denormalize the output
                srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(),
                  data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

                # Save the errors here
                mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

                # Training is done in exponential map, but the error is reported in
                # Euler angles, as in previous work.
                # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
                N_SEQUENCE_TEST = 8
                for i in np.arange(N_SEQUENCE_TEST):
                  eulerchannels_pred = srnn_pred_expmap[i]

                  # Convert from exponential map to Euler angles
                  for j in np.arange( eulerchannels_pred.shape[0] ):
                    for k in np.arange(3,97,3):
                      eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                        data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

                  # The global translation (first 3 entries) and global rotation
                  # (next 3 entries) are also not considered in the error, so the_key
                  # are set to zero.
                  # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                  gt_i=np.copy(srnn_gts_euler[action][i])
                  gt_i[:,0:6] = 0
                  # Now compute the l2 error. The following is numpy port of the error
                  # function provided by Ashesh Jain (in matlab), available at
                  # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                  idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]

                  euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                  euc_error = np.sum(euc_error, 1)
                  euc_error = np.sqrt( euc_error )
                  mean_errors[i,:] = euc_error

                # This is simply the mean error over the N_SEQUENCE_TEST examples
                mean_mean_errors = np.mean( mean_errors, 0 )

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <16} |".format(action), end="")
                for ms in [1,3,7,9,13,24]:
                  if FLAGS.seq_length_out >= ms+1:
                    print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
                  else:
                    print("   n/a |", end="")
                print()

            print()
            print("============================\n"
                  "Global step:         %d\n"
                  "Learning rate:       %.4f\n"
                  "Step-time (ms):     %.4f\n"
                  "Train loss avg:      %.4f\n"
                  "--------------------------\n"
                  "Val loss:            %.4f\n"
                  "srnn loss:           %.4f\n"
                  "============================" % (current_step,
                  lr, step_time*1000, loss,
                  val_loss, srnn_loss))
            print()

            previous_losses.append(loss)

            # Save the model
            if current_step % FLAGS.save_every == 0:
              print( "Saving the model..." ); start_time = time.time()
              torch.save(model.state_dict(), os.path.normpath(os.path.join(train_dir, 'checkpoint-{0}.pt'.format(current_step))))
              print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )

            # Reset global time and loss
            step_time, loss = 0, 0

            sys.stdout.flush()

def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def sample():
  """Sample predictions for srnn's seeds"""

  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions( FLAGS.action )
  print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
  sampling = True
  model = create_model(actions, sampling=True)
  model.eval().to(device)
  print("Model created")

  # Load all the data
  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

  # === Read and denormalize the gt with srnn's seeds, as we'll need them
  # many times for evaluation in Euler Angles ===
  srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                            data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False )
  srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                            data_std, dim_to_ignore, not FLAGS.omit_one_hot )

  # Clean and create a new h5 file of samples
  SAMPLES_FNAME = 'samples.h5'
  try:
    os.remove( SAMPLES_FNAME )
  except OSError:
    pass

  for action in actions:

      #Make prediction with srnn's seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)
      srnn_poses, _ = model(transform(encoder_inputs), transform(decoder_inputs))
      srnn_loss = model.loss(srnn_poses, transform(decoder_outputs))
      srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(), data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)

      # Save the samples
      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        for i in np.arange(8):
          # Save conditioning ground truth
          node_name = 'expmap/gt/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
          # Save prediction
          node_name = 'expmap/preds/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,97,3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )

  return

def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


if __name__ == "__main__":
      if FLAGS.sample:
        sample()
      else:
        train()
