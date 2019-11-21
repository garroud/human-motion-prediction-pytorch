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
from helper import *
from data_utils import _some_variables
import data_utils
import seq2seq_model
import gnn_module
import discriminator
import torch
from torch import autograd
from statistics import mean
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Human Motion Model")

# specification for GNN model
parser.add_argument('--node_hidden_dim', default=256, type=int, metavar='N', help="in message passing, the hidden dim of node")
parser.add_argument('--node_output_dim', default=256, type=int, metavar='N', help="in message passing, the output dim of node")
parser.add_argument('--edge_hidden_dim', default=256, type=int, metavar='N', help="in message passing, the hidden dim of edge")
parser.add_argument('--edge_output_dim', default=256, type=int, metavar='N', help="in message passing, the output dim of node")
parser.add_argument('--num_passing', default=1, type=int, metavar='N', help='number of message passing performed in GNN encoder')
parser.add_argument('--do_prob', default=0, type=float, metavar='N', help='dropout prob for message passing')
parser.add_argument('--use_coordinate', action='store_true', help='train with coordinate in GNN')
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
parser.add_argument("--train_dir", default=os.path.normpath("./experiments/"), type=str, metavar='S', help="Training directory.")

parser.add_argument("--action",default="all", type=str, metavar='S', help="The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
parser.add_argument("--loss_to_use",default="sampling_based", metavar='S', help="The type of loss to use, supervised or sampling_based")

parser.add_argument("--test_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--save_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--show_every", default=100, type=int, metavar='N', help="How often to show error during training.")
parser.add_argument("--sample", action='store_true' ,help="Set to True for sampling.")
parser.add_argument("--test", action='store_true' ,help="Set to True for model testing.")
parser.add_argument("--use_cpu", action='store_true', help="Whether to use the CPU")
parser.add_argument("--load", default=0, type=int, metavar='N', help="Try to load a previous checkpoint")
###params for IRL training###
parser.add_argument("--irl_training", action="store_true", help="set to use IRL training scheme")
parser.add_argument("--train_discrim_iter", default=2000, type=int, metavar="N", help="Pretrain iterations for train discriminator.")
parser.add_argument("--train_GAN_iter", default=20000, type=int, metavar="N", help="training iterations for the GAN training.")
parser.add_argument("--policy_lr", default = 1e-3, type=float,  metavar="N", help= "learning rate of adversarial training of policy net")
parser.add_argument("--discrim_lr", default = 1e-3, type=float, metavar="N", help= "learning rate of adversarial training of descrim net")
parser.add_argument("--discrim_hidden_size", default=1024, type=int, metavar='N', help= "hidden size of discriminator net")
parser.add_argument("--discrim_num_layers", default=1, type=int, metavar="N", help="number of layers in the discriminator")
parser.add_argument("--discrim_load", default=-1, type=int, metavar="N", help= "load pretrained model to discriminator")
parser.add_argument("--stochastic", action="store_true", help="use stochastic training approach")
parser.add_argument("--skip_pretrain_policy", action="store_true", help="skip pretrain process of policy net")
FLAGS = parser.parse_args()

train_dir = os.path.normpath(os.path.join( FLAGS.train_dir,
  'gnn' if FLAGS.use_GNN else 'normal',
  'coor' if FLAGS.use_coordinate else 'expmap',
  FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  FLAGS.architecture,
  FLAGS.loss_to_use,
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel',
  'irl_training' if FLAGS.irl_training else 'normal training',
  'stochastic_sampling' if FLAGS.stochastic else 'no sampling'))

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

writer = SummaryWriter(summaries_dir)
device = torch.device('cpu' if FLAGS.use_cpu else 'cuda')

#mask for the node with DOF = 1
mask = np.ones([21,3])
node_dof_1 = [2,4,6,8,15,19]
mask[node_dof_1, :] = [1.,0.,0.]
mask = torch.tensor(mask,dtype=torch.float32).to(device)
index_to_cal = mask.view(-1)
index_to_cal = [i for i in range(len(index_to_cal)) if index_to_cal[i] == 1]
# useless now
def create_model(actions, sampling=False, use_GNN=False):

    if use_GNN:
        # load systematic meta data
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
            # True if FLAGS.architecture == 'tied' else False,
            FLAGS.message_sharing,
            FLAGS.do_prob,
            not FLAGS.omit_one_hot,
            FLAGS.residual_velocities,
            stochastic=FLAGS.stochastic,
            mask=mask,
            dtype=torch.float32
        )
        #
        # discrim_net = discriminator.GNNDiscriminator(
        #     FLAGS.seq_length_in if not sampling else 50
        #     FLAGS.edge_hidden_dim,
        #     FLAGS.edge_output_dim,
        #     FLAGS.node_hidden_dim,
        #     FLAGS.node_output_dim,
        #     rec_encode,
        #     send_encode,
        #     len(actions),
        #     FLAGS.batch_size,
        #     not FLAGS.omit_one_hot,
        #     FLAGS.do_prob,
        #     torch.float32,
        # )
    else:
        policy_net = seq2seq_model.Seq2SeqModel(
            FLAGS.architecture,
            FLAGS.seq_length_in if not sampling else 50,
            FLAGS.seq_length_out if not sampling else 25,
            FLAGS.size, # hidden layer size
            FLAGS.num_layers,
            FLAGS.batch_size,
            summaries_dir,
            FLAGS.loss_to_use if not sampling else "sampling_based",
            len( actions ),
            device,
            not FLAGS.omit_one_hot,
            FLAGS.residual_velocities,
            stochastic=FLAGS.stochastic,
            dtype=torch.float32
        )

    discrim_net = discriminator.Discriminator(
            not FLAGS.omit_one_hot,
            FLAGS.discrim_hidden_size,
            FLAGS.batch_size,
            FLAGS.discrim_num_layers,
            len( actions ),
            FLAGS.use_GNN,
            False,
        )

    #initalize a new model
    if FLAGS.load <= 0 and FLAGS.discrim_load <=0:
        print("Creating model with fresh parameters.")
        #TODO: Initial parameter here
        return policy_net, discrim_net

    #Load model from iteration
    if os.path.isfile(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(FLAGS.load))):
        policy_net.load_state_dict(torch.load(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(FLAGS.load))))
    elif FLAGS.load > 0:
        raise ValueError("Asked to load policy checkpoint {0}, but it does not seem to exist".format(FLAGS.load))

    if os.path.isfile(os.path.join(train_dir, 'discrim-checkpoint-{0}.pt'.format(FLAGS.discrim_load))):
        discrim_net.load_state_dict(torch.load(os.path.join(train_dir, 'discrim-checkpoint-{0}.pt'.format(FLAGS.load))))
    elif FLAGS.discrim_load > 0:
        raise ValueError("Asked to load discrim checkpoint {0}, but it does not seem to exist".format(FLAGS.discrim_load))

    return policy_net, discrim_net

#TODO: train with the GAIL framework.
def train_IRL(use_GNN=False):
    actions = define_actions(FLAGS.action)
    number_of_actions = len(actions)

    if use_GNN:
        print("Creating GNN model.")
    else:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    #There should be two net for the task
    policy_net, discrim_net = create_model(actions, False, use_GNN)
    policy_net.to(device)
    discrim_net.to(device)
    print("Model created.")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
            actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot, use_GNN)
    srnn_gts_euler = get_srnn_gts( actions, policy_net, test_set, data_mean,
                          data_std, dim_to_ignore, not FLAGS.omit_one_hot,
                          to_euler=not FLAGS.use_coordinate)


    ############################################################################
    ################# Pretrain policy network ##################################
    ############################################################################
    if FLAGS.skip_pretrain_policy:
        if not os.path.isfile(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt'.format(FLAGS.load))):
            raise ValueError("the best policy checkpoint does not seem to exist".format(FLAGS.load))
    else:
        step_time, loss, val_loss, best_loss = 0.0, 0.0, 0.0, 0.0
        # current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        current_step = 0

        previous_losses = []

        step_time, loss = 0, 0
        lr = FLAGS.learning_rate
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr,weight_decay=1e-4)
        for i in xrange(FLAGS.iterations):

            start_time = time.time()
            encoder_inputs, decoder_inputs, decoder_outputs,_,_ = policy_net.get_batch(train_set, actions)
            policy_net.train()
            with autograd.detect_anomaly():
                means,logstds, samples , _ = policy_net(encoder_inputs,decoder_inputs)
                optimizer.zero_grad()
                target = decoder_outputs.view(decoder_outputs.size(0),decoder_outputs.size(1), -1, 3)
                if i < 20000:
                    step_loss = nn.MSELoss(reduction='mean')(means, target)
                else:
                    step_loss = policy_net.loss(means, logstds, target)
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(),FLAGS.max_gradient_norm)
                optimizer.step()
            writer.add_scalar('train/nll_gauss', step_loss * 63./51,i)
            if current_step % FLAGS.show_every == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss * 63./51))

            step_time += (time.time() - start_time) / FLAGS.test_every
            loss += step_loss / FLAGS.test_every
            current_step += 1

            # step decay ##
            # if current_step % FLAGS.learning_rate_step == 0:
            #     lr *= FLAGS.learning_rate_decay_factor
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

            ## Validation step ##
            if current_step % FLAGS.test_every == 0:
                mse_losses = []
                val_losses = []
                for _ in range(1):
                    encoder_inputs, decoder_inputs, decoder_outputs,_,_= policy_net.get_batch(test_set, actions,validation=True)
                    policy_net.eval()
                    means, logstds, output, _= policy_net(encoder_inputs, decoder_inputs)
                    target = decoder_outputs.view(decoder_outputs.size(0),decoder_outputs.size(1), -1, 3)
                    if i < 20000:
                        step_loss = nn.MSELoss(reduction='mean')(means, target)
                    else:
                        step_loss = policy_net.loss(means, logstds, target)
                    mse_loss = nn.MSELoss(reduction='mean')(means.view(-1),decoder_outputs.view(-1)).item()
                    val_loss = step_loss
                    val_losses.append(val_loss)
                    mse_losses.append(mse_loss)
                # val_loss = mean(val_losses)
                # mse_loss = mean(mse_losses)
                print("traing generator, iter {0:04d}: val loss: {1:.4f} MSE Loss : {2:.4f}".format(current_step, val_loss * 63./51, mse_loss * 63./51))
                writer.add_scalar('val/nll_gauss',val_loss* 63./51, i)
                writer.add_scalar('val/mse_loss', mse_loss * 63./51, i)
                # Save the best model
                if best_loss == 0 or best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(policy_net.state_dict(), os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt')))

            # Save the model periodically
            if current_step % FLAGS.save_every == 0:
                print( "Saving the model..." ); start_time = time.time()
                torch.save(policy_net.state_dict(), os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-{0}.pt'.format(current_step))))
                print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000))

            # Reset global time and loss
            step_time, loss = 0, 0
            sys.stdout.flush()

    print ("policy net pretrain is done.")

    ############################################################################
    #########################   Start Adeversial Training ######################
    ############################################################################

    # load the best pretained policy
    policy_state_dict = torch.load(os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt')))
    policy_net.load_state_dict(policy_state_dict)
    print("Load the best model for policy net")

    # optimizer
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=FLAGS.policy_lr)
    # optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=FLAGS.policy_lr)
    # optimizer_discrim = torch.optim.SGD(discrim_net.parameters(), lr=FLAGS.discrim_lr)
    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=FLAGS.discrim_lr)
    schedular_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy,step_size=30000, gamma=0.2)
    schedular_discrim = torch.optim.lr_scheduler.StepLR(optimizer_discrim,step_size=30000, gamma=0.2)
    discrim_criterion = nn.BCELoss()
    discrim_criterion.to(device)
    discrim_net.train()
    policy_net.train()

    ###pretrain discriminator
    for i in range(FLAGS.train_discrim_iter):
        encoder_inputs, decoder_inputs, decoder_outputs,encoder_inputs_ori,decoder_inputs_ori = policy_net.get_batch(train_set, actions)
        _,_ , predict_seq , _ = policy_net(encoder_inputs, decoder_inputs)
        predict_seq = predict_seq.view(predict_seq.shape[0], predict_seq.shape[1],-1)
        expert_state, expert_action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, decoder_outputs)
        state, action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, predict_seq)
        pre_mod_p, pre_exp_p = update_discrim(1.0, discrim_net, optimizer_discrim, discrim_criterion, expert_state, expert_action, state, action, device, FLAGS.seq_length_in)

        if (i+1) % FLAGS.show_every == 0:
            print("train discriminator: iter ", (i+1), ' exp: ', pre_exp_p.item(), ' mod: ', pre_mod_p.item())
        # if pre_mod_p < 0.1:
        #     break

    # Save pretrain discriminator model
    torch.save(discrim_net.state_dict(), os.path.normpath(os.path.join(train_dir, 'pretrain-discrim-checkpoint.pt')))

    print ("Discrim net pretrain is done.")

    #####################################################################
    ########################### GAN training ############################
    #####################################################################
    exp_p = []
    mod_p = []
    pre_mod_p = 1
    for i_iter in range(FLAGS.train_GAN_iter):

        # ts0 = time.time()
        encoder_inputs, decoder_inputs, decoder_outputs,encoder_inputs_ori,decoder_inputs_ori = policy_net.get_batch(train_set, actions)
        _,_ , predict_seq , _ = policy_net(encoder_inputs, decoder_inputs)
        predict_seq = predict_seq.view(predict_seq.shape[0], predict_seq.shape[1],-1)
        expert_state, expert_action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, decoder_outputs)
        state, action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, predict_seq)
        # ts1 = time.time()

        t0 = time.time()
        if pre_mod_p > 0.3:
            train = True
        else:
            train = False
        # train = False
        pre_mod_p, pre_exp_p = update_discrim(1.0, discrim_net, optimizer_discrim, discrim_criterion, expert_state, expert_action, state, action, device, FLAGS.seq_length_in, train)

        pre_mod_p = pre_mod_p.item()
        pre_exp_p = pre_exp_p.item()

        exp_p.append(pre_exp_p)
        mod_p.append(pre_mod_p)

        #update policy network
        if pre_mod_p < 0.7:
            # _,_ , predict_seq , _ = policy_net(encoder_inputs, decoder_inputs)
            # predict_seq = predict_seq.view(predict_seq.shape[0], predict_seq.shape[1],-1)
            # state, action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, predict_seq)
           pre_mod_p = update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, state, action,FLAGS.seq_length_in,10.0, device)
        writer.add_scalar('gan/pre_exp_p', pre_exp_p, i_iter)
        writer.add_scalar('gan/pre_mod_p', pre_mod_p, i_iter)
        # writer.add_scalar('gan/local_mod_p',local_mod_p, i_iter)
        # schedular_policy.step()
        # schedular_discrim.step()

        # t1 = time.time()
        if (i_iter + 1) % FLAGS.show_every == 0:
        # if (i_iter + 1) % 1 == 0:
            print("train GAN: iter ", (i_iter+1), ' exp: ', pre_exp_p, ' mod: ', pre_mod_p)

        if (i_iter + 1) % FLAGS.save_every == 0:
            torch.save(policy_net.state_dict(),os.path.normpath(os.path.join(train_dir, 'policy-checkpoint-{0}.pt'.format(i_iter + 1))))
            torch.save(discrim_net.state_dict(),os.path.normpath(os.path.join(train_dir, 'discrim-checkpoint-{0}.pt'.format(i_iter + 1))))

def train_IRL_WGAN(use_GNN=False):
    actions = define_actions(FLAGS.action)
    number_of_actions = len(actions)

    if use_GNN:
        print("Creating GNN model.")
    else:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    #There should be two net for the task
    policy_net, discrim_net = create_model(actions, False, use_GNN)
    policy_net.to(device)
    discrim_net.to(device)
    print("Model created.")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
            actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot, use_GNN)
    srnn_gts_euler = get_srnn_gts( actions, policy_net, test_set, data_mean,
                          data_std, dim_to_ignore, not FLAGS.omit_one_hot,
                          to_euler=True)


    ############################################################################
    ################# Pretrain policy network ##################################
    ############################################################################
    if FLAGS.skip_pretrain_policy:
        if not os.path.isfile(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt'.format(FLAGS.load))):
            raise ValueError("the best policy checkpoint does not seem to exist".format(FLAGS.load))
    else:
        step_time, loss, val_loss, best_loss = 0.0, 0.0, 0.0, 0.0
        # current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        current_step = 0

        previous_losses = []

        step_time, loss = 0, 0
        lr = FLAGS.learning_rate
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr)
        for _ in xrange(FLAGS.iterations):

            start_time = time.time()
            encoder_inputs, decoder_inputs, decoder_outputs,_,_= policy_net.get_batch(train_set, actions)
            # encoder_inputs, decoder_inputs, decoder_outputs= policy_net.get_batch_srnn(test_set, 'sittingdown')
            policy_net.train()
            with autograd.detect_anomaly():
                means,logstds, samples , _,_ = policy_net(encoder_inputs,decoder_inputs)
                optimizer.zero_grad()
                target = decoder_outputs
                # step_loss = nn.MSELoss(reduction='mean')(means, target)
                step_loss = policy_net.loss(means, logstds, target,index_to_cal)
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(),FLAGS.max_gradient_norm)
                optimizer.step()
            writer.add_scalar('train/nll_gauss', step_loss,current_step)
            if current_step % FLAGS.show_every == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / FLAGS.test_every
            loss += step_loss / FLAGS.test_every

            # step decay ##
            # if current_step % FLAGS.learning_rate_step == 0:
            #     lr *= FLAGS.learning_rate_decay_factor
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

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
                for action in actions:
                    # Evaluate the model on the test batches
                    encoder_inputs, decoder_inputs, decoder_outputs = policy_net.get_batch_srnn(test_set, action)
                    srnn_poses, _,_,_,_ = policy_net(encoder_inputs, decoder_inputs)
                    # Denormalize the output
                    srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(),
                      data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)
                    srnn_gt_expmap = data_utils.revert_output_format(decoder_outputs.cpu().detach().numpy(),data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)
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
                        errors[s]/len(actions),current_step)


                # validation and save the best model
                encoder_inputs, decoder_inputs, decoder_outputs,_,_= policy_net.get_batch(test_set, actions,validation=True)
                policy_net.eval()
                means, logstds, output, _,_= policy_net(encoder_inputs, decoder_inputs)
                target = decoder_outputs
                # step_loss = nn.MSELoss(reduction='mean')(means, target)
                step_loss = policy_net.loss(means, logstds, target,index_to_cal)
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
            sys.stdout.flush()

    print ("policy net pretrain is done.")

    ############################################################################
    #########################   Start Adeversial Training ######################
    ############################################################################

    # load the best pretained policy
    policy_state_dict = torch.load(os.path.normpath(os.path.join(train_dir, 'pretrain-policy-checkpoint-best.pt')))
    policy_net.load_state_dict(policy_state_dict)
    print("Load the best model for policy net")

    # optimizer
    optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=FLAGS.policy_lr)
    optimizer_discrim = torch.optim.SGD(discrim_net.parameters(), lr=FLAGS.discrim_lr)
    schedular_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy,step_size=30000, gamma=0.2)
    schedular_discrim = torch.optim.lr_scheduler.StepLR(optimizer_discrim,step_size=30000, gamma=0.2)
    discrim_net.train()
    policy_net.train()
    critic_time = 5
    clip_value = 0.02
    #####################################################################
    ########################### GAN training ############################
    #####################################################################
    for param in discrim_net.parameters():
        param.data.clamp_(-clip_value, clip_value)
    for i_iter in range(FLAGS.train_GAN_iter):
        # ts0 = time.time()
        discrim_loss = 0.
        for _ in range(critic_time):
            encoder_inputs, decoder_inputs, decoder_outputs,encoder_inputs_ori,decoder_inputs_ori = policy_net.get_batch(train_set, actions)
            _,_ , predict_seq , _ = policy_net(encoder_inputs, decoder_inputs)
            predict_seq = predict_seq.view(predict_seq.shape[0], predict_seq.shape[1],-1)
            expert_state, expert_action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, decoder_outputs)
            state, action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, predict_seq)
            loss = update_discrim_WGAN(discrim_net, optimizer_discrim, expert_state, expert_action, state, action, FLAGS.seq_length_in,c=clip_value)
            discrim_loss += loss
        discrim_loss /= critic_time

        # update generator
        encoder_inputs, decoder_inputs, decoder_outputs,encoder_inputs_ori,decoder_inputs_ori = policy_net.get_batch(train_set, actions)
        _,_ , predict_seq , _ = policy_net(encoder_inputs, decoder_inputs)
        predict_seq = predict_seq.view(predict_seq.shape[0], predict_seq.shape[1],-1)
        state, action = get_state_action(encoder_inputs_ori, decoder_inputs_ori, predict_seq)
        generator_loss = update_policy_WGAN(policy_net, optimizer_policy, discrim_net, state, action,FLAGS.seq_length_in,10.0)
        writer.add_scalar('gan/discrim_loss', discrim_loss, i_iter)
        writer.add_scalar('gan/generator_loss', generator_loss, i_iter)
        schedular_policy.step()
        schedular_discrim.step()

        # t1 = time.time()
        if (i_iter + 1) % FLAGS.show_every == 0:
        # if (i_iter + 1) % 1 == 0:
            print("train GAN: iter ", (i_iter+1), ' discrim:', discrim_loss , ' gen: ', generator_loss)

        if (i_iter + 1) % FLAGS.save_every == 0:
            torch.save(policy_net.state_dict(),os.path.normpath(os.path.join(train_dir, 'policy-checkpoint-{0}.pt'.format(i_iter + 1))))
            torch.save(discrim_net.state_dict(),os.path.normpath(os.path.join(train_dir, 'discrim-checkpoint-{0}.pt'.format(i_iter + 1))))

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
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set, actions )

        model.train()
        output, _ = model(encoder_inputs, decoder_inputs)
        optimizer.zero_grad()
        with autograd.detect_anomaly():
            step_loss = model.loss(output[:,:,:model.HUMAN_SIZE], decoder_outputs[:,:,:model.HUMAN_SIZE])
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
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set, actions)
            ##TODO: a forward pass
            model.eval()
            output, _ = model(encoder_inputs, decoder_inputs)
            step_loss = model.loss(output[:,:,:model.HUMAN_SIZE],decoder_outputs[:,:,:model.HUMAN_SIZE])
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
                srnn_poses, _ = model(encoder_inputs, decoder_inputs)
                srnn_loss = model.loss(srnn_poses[:,:,:model.HUMAN_SIZE],decoder_outputs[:,:,:model.HUMAN_SIZE])
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

def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True):
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
    _, _, srnn_expmap = model.get_batch_srnn(test_set, action)
    srnn_expmap = srnn_expmap.permute(1,0,2).cpu().detach().numpy()
    # expmap -> rotmat -> euler
    for i in np.arange(srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3]))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler

def sample(use_GNN):
  """Sample predictions for srnn's seeds"""

  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions( FLAGS.action )
  print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
  sampling = True
  model, _ = create_model(actions, use_GNN=use_GNN, sampling=True)
  model.eval().to(device)
  print("Model created")

  # Load all the data
  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
          actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot, use_GNN)

  # === Read and denormalize the gt with srnn's seeds, as we'll need them
  # many times for evaluation in Euler Angles ===
  srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                            data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False)
  srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                            data_std, dim_to_ignore, not FLAGS.omit_one_hot
                            )

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

  for action in actions:

      #Make prediction with srnn's seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)
      if not FLAGS.stochastic:
          srnn_poses, _ = model(encoder_inputs, decoder_inputs)
      else:
          srnn_poses,_,_,_ = model(encoder_inputs, decoder_inputs)
      srnn_poses = srnn_poses.view(srnn_poses.shape[0], srnn_poses.shape[1], -1)
      srnn_loss = nn.MSELoss(reduction='mean')(srnn_poses[:,:,:model.HUMAN_SIZE], decoder_outputs[:,:,:model.HUMAN_SIZE])
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

        srnn_gts_euler[action][i][:,0:6] = 0

        # eulerchannels_pred[:,:] = eulerchannels_pred[0]
        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( srnn_gts_euler[action][i], 0 ) > 1e-4 )[0]
        idx_to_use2 = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]
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

def val_visual(use_GNN):
    """Sample predictions for srnn's seeds"""

    FLAGS.load = 0

    actions = define_actions( FLAGS.action )
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    sampling = True
    model, _ = create_model(actions, use_GNN=use_GNN, sampling=True)

    load_model_name = "pretrain-policy-checkpoint-best.pt"
    model.load_state_dict(torch.load(os.path.join(train_dir, load_model_name)))
    model.eval().to(device)
    print("Model created")
    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
            actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot, use_GNN)

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False)
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot
                              )

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples_val_visual.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    encoder_inputs, decoder_inputs, decoder_outputs, _ ,_ = model.get_batch(test_set, actions, validation=True)
    if not FLAGS.stochastic:
        srnn_poses, _ = model(encoder_inputs, decoder_inputs)
    else:
        srnn_poses,_,_,_ = model(encoder_inputs, decoder_inputs)
    srnn_poses = srnn_poses.view(srnn_poses.shape[0], srnn_poses.shape[1], -1)
    srnn_loss = nn.MSELoss(reduction='mean')(srnn_poses[:,:,:model.HUMAN_SIZE], decoder_outputs[:,:,:model.HUMAN_SIZE])
    srnn_pred_expmap = data_utils.revert_output_format(srnn_poses.cpu().detach().numpy(), data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)
    srnn_gt_expmap = data_utils.revert_output_format(decoder_outputs.cpu().detach().numpy(), data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)

    # Save the samples
    with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        for i in np.arange(15):
            # Save conditioning ground truth
            node_name = 'expmap/gt/{1}_{0}'.format(i, "validation")
            hf.create_dataset( node_name, data=srnn_gt_expmap[i] )
            # Save prediction
            node_name = 'expmap/preds/{1}_{0}'.format(i, "validation")
            hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

# hacky produce result for debuging
def sample_test(use_GNN,test_val=False):
    """Sample predictions for srnn's seeds"""

    FLAGS.load = 0

    actions = define_actions( FLAGS.action )
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    sampling = True
    model, _ = create_model(actions, use_GNN=use_GNN, sampling=True)

    load_model_name = "pretrain-policy-checkpoint-best.pt"
    # cust_model = 'models/pretrain-policy-checkpoint-best-deterministic.pt'
    model.load_state_dict(torch.load(os.path.join(train_dir, load_model_name)))
    # model.load_state_dict(torch.load(cust_model))
    model.eval().to(device)
    print("Model created")
    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
            actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot, use_GNN)

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False)
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot
                              )

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

    for action in actions:

        #Make prediction with srnn's seeds
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)
        if not FLAGS.stochastic:
            srnn_poses, _ = model(encoder_inputs, decoder_inputs)
        else:
            srnn_poses,_,_,_,edge_weight = model(encoder_inputs, decoder_inputs)
            edge_weight = edge_weight.permute(1,0,2).cpu().detach().numpy()
        srnn_poses = srnn_poses.view(srnn_poses.shape[0], srnn_poses.shape[1], -1)
        srnn_loss = nn.MSELoss(reduction='mean')(srnn_poses[:,:,:model.HUMAN_SIZE], decoder_outputs[:,:,:model.HUMAN_SIZE])
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


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot,
                   use_GNN=False):
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
  # train_subject_ids = [1]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train, use_GNN)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot)
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot)
  print("done reading data.")
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


if __name__ == "__main__":
    if FLAGS.test:
        sample_test(FLAGS.use_GNN)
        # val_visual(FLAGS.use_GNN)
    elif FLAGS.sample:
        sample(FLAGS.use_GNN)
    elif FLAGS.irl_training:
        train_IRL_WGAN(FLAGS.use_GNN)
    else:
        train()
