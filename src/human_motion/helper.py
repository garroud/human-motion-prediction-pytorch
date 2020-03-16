from torch import nn
import torch
import numpy as np
from torch import distributions
from itertools import chain
import random
# update the discriminator
def update_discrim(dis_times, discrim_net, optimizer_discrim, discrim_criterion, expert_state, expert_action, state, action, device, start_idx, train=True):
    expert_state = expert_state.detach()
    expert_action = expert_action.detach()
    state = state.detach()
    action = action.detach()
    g_o_ave = 0.0
    e_o_ave = 0.0
    discrim_ave = 0.0
    discrim_net.train()
    for _ in range(int(dis_times)):
        g_o = discrim_net(state, action)[start_idx:,:,:]
        e_o = discrim_net(expert_state, expert_action)[start_idx:,:,:]
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        g_o = g_o.clamp(0.0,1.0)
        e_o = e_o.clamp(0.0,1.0)

        discrim_loss = discrim_criterion(g_o, torch.zeros(g_o.shape).to(device)) + \
               discrim_criterion(e_o, torch.ones(e_o.shape).to(device))
        discrim_ave += discrim_loss.item()
        if train:
            optimizer_discrim.zero_grad()

            discrim_loss.backward()
            # torch.nn.utils.clip_grad_norm_(discrim_net.parameters(), 20.0)
            optimizer_discrim.step()

    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times, discrim_ave / dis_times

def update_discrim_WGAN(discrim_net, optimizer_discrim, expert_state, expert_action, state, action, start_idx, c=0.01):
    expert_state = expert_state.detach()
    expert_action = expert_action.detach()
    state = state.detach()
    action = action.detach()
    g_o = discrim_net(state, action)[start_idx:,:,:]
    e_o = discrim_net(expert_state, expert_action)[start_idx:,:,:]
    optimizer_discrim.zero_grad()
    discrim_loss = torch.mean((e_o - g_o).view(-1))
    discrim_loss.backward()
    optimizer_discrim.step()
    for param in discrim_net.parameters():
        param.data.clamp_(-c,c)
    return discrim_loss.item()


#update policy network in GAN training
# TODO: added addtional loss, maybe not compatible with original code
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, state, action, start_idx, clip_grad_norm, device,addition_loss=None):
    policy_net.train()
    g_o = discrim_net(state, action)[start_idx:, :, :].clamp(0.0,1.0)
    optimizer_policy.zero_grad()
    policy_loss = discrim_criterion(g_o, torch.ones(g_o.shape).to(device))
    if not addition_loss is None:
        policy_loss += addition_loss
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad_norm)
    optimizer_policy.step()

    return g_o.cpu().data.mean().item()

#update policy network in GAN training
def update_policy_WGAN(policy_net, optimizer_policy, discrim_net, state, action, start_idx, clip_grad_norm):
    g_o = discrim_net(state, action)[start_idx:, :, :].clamp(0.,1.)
    optimizer_policy.zero_grad()
    policy_loss = torch.mean(g_o.view(-1))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad_norm)
    optimizer_policy.step()

    return policy_loss.item()

# get the state and action for training, the shape is seq * batch * dim
def get_state_action(encoder_inputs, decoder_inputs, decoder_outputs):
    try:
        state = torch.cat([encoder_inputs, decoder_inputs], 0)
        action = torch.cat([
            encoder_inputs[1:,:,:decoder_outputs.shape[2]],
            decoder_inputs[0:1,:,:decoder_outputs.shape[2]],
            decoder_outputs],
            dim=0)
    except:
        print ("Error to get actiona and state")
        print (encoder_inputs.shape)
        print (decoder_inputs.shape)
        print (decoder_outputs.shape)
        exit()
    return state, action

# calculate the negative log likihood of a normal distribution sequence,
def nll_gauss(mean, logstd, x):
    pi = torch.FloatTensor([np.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    nll_element = (x - mean).pow(2) * torch.exp(-1.0 * logstd) + 2.0*logstd + torch.log(2.0*pi)
    # nll_element = torch.abs(x - mean) * torch.exp(-1.0 * logstd) + logstd
    # print("max {0:.4f} , min {1:.4f}".format(torch.max(torch.abs(x - mean)).item(), torch.min(torch.abs(x - mean)).item()))
    return torch.sum(nll_element) / (x.size(0) * x.size(1))
    # return torch.mean(nll_element)

#Sampling a sequence to perform reparametrization trick
def reparam_sample_gauss(mean, logstd):
    eps = torch.FloatTensor(logstd.size()).normal_()
    # eps = distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0])).sample(logstd.size())
    if mean.is_cuda:
        eps = eps.cuda()
    res = eps.view(logstd.size()).mul(torch.exp(logstd)).add_(mean)
    # print(res.size())
    return res

# Given var and sampled result, get the mean
def reverse_sample_gauss(logstd, sample):
    eps = torch.FloatTensor(logstd.size()).normal_()
    if var.is_cuda:
        eps = eps.cuda()
    return sample.sub_(eps.mul(torch.exp(logstd)))

# encode to a one hot matrix
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
