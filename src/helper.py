from torch import nn
import torch
import numpy as np
from torch import distributions

# update the discriminator
def update_discrim(dis_times, discrim_net, optimizer_discrim, discrim_criterion, expert_state, expert_action, state, action, device, start_idx, train=True):
    expert_state = expert_state.detach()
    expert_action = expert_action.detach()
    state = state.detach()
    action = action.detach()
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        g_o = discrim_net(state, action)[start_idx:,:,:]
        e_o = discrim_net(expert_state, expert_action)[start_idx:,:,:]
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()

        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, torch.zeros(g_o.shape).to(device)) + \
                           discrim_criterion(e_o, torch.ones(e_o.shape).to(device))
            discrim_loss.backward()
            optimizer_discrim.step()

    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times

#update policy network in GAN training
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, state, action, start_idx, clip_grad_norm, device):
    g_o = discrim_net(state, action)[start_idx:, :, :]
    optimizer_policy.zero_grad()
    policy_loss = discrim_criterion(g_o, torch.ones(g_o.shape).to(device))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad_norm)
    optimizer_policy.step()

    return g_o.cpu().data.mean()

# get the state and action for training, the shape is seq * batch * dim
def get_state_action(encoder_inputs, decoder_inputs, decoder_outputs):
    try:
        whole_seq = torch.cat([encoder_inputs, decoder_inputs[0:1, : , :], decoder_outputs[:,:,:]], 0)
    except:
        print ("Error to get actiona and state")
        print (encoder_inputs.shape)
        print (decoder_inputs.shape)
        print (decoder_outputs.shape)
        exit()
    state = whole_seq[:-1, :, :]
    action = whole_seq[1:, :, :]
    return state, action

# calculate the negative log likihood of a normal distribution sequence,
def nll_gauss(mean, logstd, x):
    pi = torch.FloatTensor([np.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    # nll_element = (x - mean).pow(2) * torch.exp(-1.0 * logstd) + 2.0*logstd + torch.log(2.0*pi)
    nll_element = torch.abs(x - mean) * torch.exp(-1.0 * logstd) + logstd
    # print("max {0:.4f} , min {1:.4f}".format(torch.max(torch.abs(x - mean)).item(), torch.min(torch.abs(x - mean)).item()))
    # return 0.5 * torch.sum(nll_element)
    return torch.mean(nll_element)

#Sampling a sequence to perform reparametrization trick
def reparam_sample_gauss(mean, logstd):
    # eps = torch.FloatTensor(logstd.size()).normal_()
    eps = distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0])).sample(logstd.size())
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
