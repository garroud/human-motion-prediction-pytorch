"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os
import random
import sys
import time
import h5py
import argparse

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
from human_motion.helper import *
from human_motion.data_utils import _some_variables
import human_motion.data_utils as data_utils
import modules.seq2seq_model
import modules.human_motion_models
import modules.discriminator
import torch
from torch import autograd
from statistics import mean
from tensorboardX import SummaryWriter
from datasets.human_motion_dataset import HumanDataset

parser = argparse.ArgumentParser(description="Human Motion Model")

# specification for GNN model
parser.add_argument('--node_hidden_dim', default=256, type=int, metavar='N', help="in message passing, the hidden dim of node")
parser.add_argument('--node_output_dim', default=256, type=int, metavar='N', help="in message passing, the output dim of node")
parser.add_argument('--edge_hidden_dim', default=256, type=int, metavar='N', help="in message passing, the hidden dim of edge")
parser.add_argument('--edge_output_dim', default=256, type=int, metavar='N', help="in message passing, the output dim of node")
parser.add_argument('--num_passing', default=1, type=int, metavar='N', help='number of message passing performed in GNN encoder')
parser.add_argument('--do_prob', default=0, type=float, metavar='N', help='dropout prob for message passing')
parser.add_argument('--use_GNN', action='store_true', help='train with GNN')
parser.add_argument('--message_sharing', action='store_true', help='message passing with weight sharing')
#Learning specification
parser.add_argument('--learning_rate', default=0.05, type=float, metavar='N', help='Initial learning rate.')
parser.add_argument('--learning_rate_decay_factor', default=0.95, type=float, metavar='N', help='Learning rate is multiplied by this much.')
parser.add_argument('--learning_rate_step', default=10000, type=int, metavar="N",help="Every this many steps, do decay.")
parser.add_argument('--max_gradient_norm', default=20, type=int, metavar='N', help="Clip gradients to this norm")
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='Batch size for each iteration')
parser.add_argument('--iterations', default=int(1e5), type=int, metavar='N', help="Iterations to train for")
#Architecture
parser.add_argument('--architecture', default='tied', type=str, metavar='S', help="Seq2Seq model architecture for use: [basic, tied]")
parser.add_argument("--size", default=1024, type=int, metavar='N', help="Size of each model layer.")
parser.add_argument("--num_layers", default=1, type=int, metavar='N', help="Number of layers in the model.")
parser.add_argument("--seq_length_in", default=50, type=int, metavar='N', help="Number of frames to feed into the encoder. 25 fps")
parser.add_argument("--seq_length_out", default=25, type=int, metavar='N', help="Number of frames that the decoder has to predict. 25fps")
parser.add_argument("--omit_one_hot", action='store_true', help="Whether to remove one-hot encoding from the data")
parser.add_argument("--residual_velocities", action='store_true', help="Add a residual connection that effectively models velocities")
parser.add_argument("--data_dir", default=os.path.normpath("./data/h3.6m/dataset"), type=str, metavar='S', help="Data directory")
parser.add_argument("--train_dir", default=os.path.normpath("./experiments_edge/"), type=str, metavar='S', help="Training directory.")

parser.add_argument("--action",default="all", type=str, metavar='S', help="The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
parser.add_argument("--loss_to_use",default="sampling_based", metavar='S', help="The type of loss to use, supervised or sampling_based")

parser.add_argument("--test_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--save_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--show_every", default=100, type=int, metavar='N', help="How often to show error during training.")
parser.add_argument("--sample", action='store_true' ,help="Set to True for sampling.")
parser.add_argument("--test", action='store_true' ,help="Set to True for model testing.")
parser.add_argument("--use_cpu", action='store_true', help="Whether to use the CPU")
parser.add_argument("--load", default=0, type=int, metavar='N', help="Try to load a previous checkpoint")

FLAGS = parser.parse_args()

train_dir = os.path.normpath(os.path.join( FLAGS.train_dir,
  'gnn' if FLAGS.use_GNN else 'normal',
  FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  FLAGS.architecture,
  FLAGS.loss_to_use,
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel'))

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

summaries_dir = os.path.normpath(os.path.join( train_dir, "log-", str(datetime.now()))) # Directory for TB summaries

writer = SummaryWriter(summaries_dir)
device = torch.device('cpu' if FLAGS.use_cpu else 'cuda')

#mask for the node with DOF = 1
NUM_JOINT = 21
COOR_DIM = 3
mask = np.ones([21,3])
node_dof_1 = [2,4,6,8,15,19]
mask[node_dof_1, :] = [1.,0.,0.]
mask = torch.tensor(mask,dtype=torch.float32).to(device)
index_to_cal = mask.view(-1)
index_to_cal = [i for i in range(len(index_to_cal)) if index_to_cal[i] == 1]
# encoding the incoming and outcoming nodes
rec_encode = []
send_encode = []
# re-indexed node, only 21 node should be considered in the graph
# parent = [-1,0,1,2,3,0,5,6,7,0,9,10,11,10,13,14,15,10,17,18,19]
# adj_matrix = np.zeros([21, 21])
# for node, parent in enumerate(parent):
#     if parent != -1:
#         adj_matrix[node, parent] = 1
#         adj_matrix[parent, node] = 1
# rec_encode = np.array(encode_onehot(np.where(adj_matrix)[1]),dtype=np.float32)
# send_encode = np.array(encode_onehot(np.where(adj_matrix)[0]),dtype=np.float32)
off_diag = np.ones([21,21]) - np.eye(21)
rec_encode = np.array(encode_onehot(np.where(off_diag)[1]),dtype=np.float32)
send_encode = np.array(encode_onehot(np.where(off_diag)[0]),dtype=np.float32)
rec_encode = torch.tensor(rec_encode, dtype=torch.float32).to(device)
send_encode = torch.tensor(send_encode, dtype=torch.float32).to(device)


def create_model(actions, sampling=False):
    policy_net = gnn_module.GNNModel3(
        FLAGS.seq_length_in if not sampling else 50,
        FLAGS.seq_length_out if not sampling else 25,
        FLAGS.edge_hidden_dim,
        FLAGS.edge_output_dim,
        FLAGS.node_hidden_dim,
        FLAGS.node_hidden_dim,
        FLAGS.num_passing,
        rec_encode,
        send_encode,
        FLAGS.batch_size,
        len( actions ),
        device,
        FLAGS.message_sharing,
        FLAGS.do_prob,
        not FLAGS.omit_one_hot,
        FLAGS.residual_velocities,
        mask=mask,
        dtype=torch.float32
    )

    #initalize a new model
    if FLAGS.load <= 0:
        print("Creating model with fresh parameters.")
        #TODO: Initial parameter here
        return policy_net

    #Load model from iteration
    if os.path.isfile(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(FLAGS.load))):
        policy_net.load_state_dict(torch.load(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(FLAGS.load))))
    elif FLAGS.load > 0:
        raise ValueError("Asked to load policy checkpoint {0}, but it does not seem to exist".format(FLAGS.load))

    return policy_net

dataset = HumanDataset(
    action=FLAGS.action,
    seq_length_in=FLAGS.seq_length_in,
    seq_length_out=FLAGS.seq_length_out,
    encoder_input_size=FLAGS.seq_length_in*COOR_DIM,
    decoder_input_size=COOR_DIM,
    decoder_output_size=COOR_DIM*NUM_JOINT,
    num_joint=NUM_JOINT,
    data_dir=FLAGS.data_dir,
    one_hot=not FLAGS.omit_one_hot,
    use_GNN=FLAGS.use_GNN,
    device=device,
)

def train(use_GNN=False):
    policy_net = create_model(dataset.actions, False)
    policy_net.to(device)
    print("Model created.")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = dataset.get_srnn_gts(to_euler=True)
    srnn_gts_expmap = dataset.get_srnn_gts(to_euler=False)

    ############################################################################
    ################# network training here##################################
    ############################################################################
    step_time, loss, val_loss, best_loss = 0.0, 0.0, 0.0, 0.0
    # current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    current_step = 0

    previous_losses = []

    step_time, loss = 0, 0
    lr = FLAGS.learning_rate
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in xrange(FLAGS.iterations):

        start_time = time.time()
        encoder_inputs, decoder_inputs, decoder_outputs = dataset.get_batch_train(FLAGS.batch_size)
        policy_net.train()
        with autograd.detect_anomaly():
            pred_nodes, _ = policy_net(encoder_inputs,decoder_inputs)
            optimizer.zero_grad()
            step_loss = policy_net.loss(pred_nodes, decoder_outputs, index_to_cal)
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(),FLAGS.max_gradient_norm)
            optimizer.step()
        writer.add_scalar('train/nll_gauss', step_loss, current_step)
        if current_step % FLAGS.show_every == 0:
            print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

        loss += step_loss / FLAGS.test_every

        ## Validation step ##
        if current_step % FLAGS.test_every == 0:
            print()
            print("{0: <16} |".format("milliseconds"), end="")
            for ms in [40, 80, 160, 320, 400, 560, 1000]:
                print(" {0:5d} |".format(ms), end="")
            print()

            errors = np.zeros(FLAGS.seq_length_out)
            policy_net.eval()
            # === Validation with srnn's seeds ===
            for action in dataset.actions:
                # Evaluate the model on the test batches
                encoder_inputs, decoder_inputs, decoder_outputs = dataset.get_batch_srnn(action)
                srnn_poses, _ = policy_net(encoder_inputs, decoder_inputs)
                # Denormalize the output
                srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(),dataset.data_mean, dataset.data_std, dataset.dim_to_ignore, dataset.actions, dataset.one_hot)
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
                    eulerchannels_pred[:,0:6] = 0
                    # Now compute the l2 error. The following is numpy port of the error
                    # function provided by Ashesh Jain (in matlab), available at
                    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                    # idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]
                    idx_to_use = np.where(gt_i[0] != 0)[0]
                    # idx_to_use = np.where(eulerchannels_pred[0] != 0)[0]
                    euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                    euc_error = np.sum(euc_error, 1)
                    euc_error = np.sqrt( euc_error )
                    mean_errors[i,:] = euc_error

                # This is simply the mean error over the N_SEQUENCE_TEST examples
                mean_mean_errors = np.mean( mean_errors, 0 )
                errors = errors + mean_mean_errors
                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <16} |".format(action), end="")
                for ms in [0,1,3,7,9,13,24]:
                    if FLAGS.seq_length_out >= ms+1:
                        print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
                    else:
                        print("   n/a |", end="")
                print()
                for s in [0,1,3,7,9,13,24]:
                    if FLAGS.seq_length_out >= s+1:
                        writer.add_scalar('val/srnn_error_avr_{}'.format((s+1)*40),
                        errors[s]/len(dataset.actions),current_step)


                # validation and save the best model
                encoder_inputs, decoder_inputs, decoder_outputs = dataset.get_batch_validation()
                policy_net.eval()
                pred_nodes, _ = policy_net(encoder_inputs, decoder_inputs)
                target = decoder_outputs
                # step_loss = nn.MSELoss(reduction='mean')(means, target)
                step_loss = policy_net.loss(pred_nodes,target,index_to_cal)
                val_loss = step_loss
                writer.add_scalar('val/nll_gauss',val_loss, current_step)
                # Save the best model
                if best_loss == 0 or best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(policy_net.state_dict(), os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt')))

        # Save the model periodically
        if current_step % FLAGS.save_every == 0:
            print( "Saving the model..." ); start_time = time.time()
            torch.save(policy_net.state_dict(), os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(current_step))))
            print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000))

        current_step += 1
        # Reset global time and loss
        step_time, loss = 0, 0

# hacky produce result for debuging
def sample_test(model_name=None):
    """Sample predictions for srnn's seeds"""
    sampling = True

    model = create_model(dataset.actions, sampling)
    # if model_name is None:
    #     load_model_name = "pretrain-policy-checkpoint-best.pt"
    # else:
    #     load_model_name = model_name
    # model.load_state_dict(torch.load(os.path.join(train_dir, load_model_name)))
    # model.load_state_dict(torch.load(cust_model))
    model.eval().to(device)
    print("Model created")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = dataset.get_srnn_gts(to_euler=False)
    srnn_gts_euler = dataset.get_srnn_gts(to_euler=True)

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'

    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    print()
    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
        print(" {0:5d} |".format(ms), end="")
    print()

    for action in dataset.actions:

        #Make prediction with srnn's seeds
        encoder_inputs, decoder_inputs, decoder_outputs = dataset.get_batch_srnn(action)
        srnn_poses,edge_weight = model(encoder_inputs, decoder_inputs)
        edge_weight = edge_weight.permute(1,0,2).cpu().detach().numpy()
        srnn_poses = srnn_poses.view(srnn_poses.shape[0], srnn_poses.shape[1], -1)
        srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(), dataset.data_mean, dataset.data_std, dataset.dim_to_ignore, dataset.actions, dataset.one_hot)
        # Save the samples
        with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
          for i in np.arange(8):
            # Save conditioning ground truth
            node_name = 'expmap/gt/{1}_{0}'.format(i, action)
            hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
            # Save prediction
            node_name = 'expmap/preds/{1}_{0}'.format(i, action)
            hf.create_dataset( node_name, data=srnn_pred_expmap[i] )
            node_name = 'expmap/edge_weight/{1}_{0}'.format(i, action)
            hf.create_dataset(node_name, data=edge_weight[i])
        # Compute and save the errors here
        mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

        for i in np.arange(8):

          eulerchannels_pred = srnn_pred_expmap[i]
          for j in np.arange( eulerchannels_pred.shape[0] ):
            for k in np.arange(3,97,3):
              eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

          srnn_gts_euler[action][i][:,0:6] = 0

          # eulerchannels_pred[:,:] = eulerchannels_pred[0]
          # Pick only the dimensions with sufficient standard deviation. Others are ignored.
          idx_to_use = np.where( np.std( srnn_gts_euler[action][i], 0 ) > 1e-4 )[0]
          # print(idx_to_use)
          euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
          euc_error = np.sum(euc_error, 1)
          euc_error = np.sqrt( euc_error )
          mean_errors[i,:] = euc_error

        mean_mean_errors = np.mean( mean_errors, 0 )

        # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <16} |".format(action), end="")
        for ms in [1,3,7,9,13,24]:
          if FLAGS.seq_length_out >= ms+1:
            print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
          else:
            print("   n/a |", end="")
        print()

        # print( action )
        # print( ','.join(map(str, mean_mean_errors.tolist() )) )

        with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
          node_name = 'mean_{0}_error'.format( action )
          hf.create_dataset( node_name, data=mean_mean_errors )

    return


if __name__ == "__main__":
    if FLAGS.sample:
        sample_test()
        # val_visual(FLAGS.use_GNN)
    else:
        train()
