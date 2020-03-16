from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

from nri.utils import *
from modules.nri_models import EdgeInferModule
from human_motion.helper import get_state_action, update_discrim, update_policy, nll_gauss

from modules.discriminator import PhysicsDiscriminator
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='exp_nri',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction_steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=50,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--data_dir', type=str, default='data/spring/',help='data path for loading')
parser.add_argument('--edge_update', type=int, default=1, help='update # of train per edge weight')
parser.add_argument('--max_gradient_norm',type=float, default=6.0, help='gradient clip')
parser.add_argument('--train_gan',action='store_true',default='false',help='use gan to train the model')
parser.add_argument('--discrim_epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--discrim_hidden_size', type=int, default=128,
                    help='hidden unit size in discriminator.')
parser.add_argument('--discrim_num_layers', type=int, default=1,
                    help='Number of layers in discrim\'s rnn.')
parser.add_argument('--gan_lr', type=float, default=1e-4,
                    help='generator learning rate in GAN')
parser.add_argument('--discrim_lr', type=float, default=5e-4,
                    help='discriminator learning rate')
parser.add_argument('--gan_epochs', type=int, default=50,
                    help='number of epochs to gan training')
parser.add_argument('--ignore_pretrain', action='store_true', help='escape pretrain phase')
parser.add_argument('--load_folder', type=str, default=None, help='load pretrain model folder')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    pretrain_file = os.path.join(save_folder, 'pretrain.pt')
    discrim_file = os.path.join(save_folder, 'discrim.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, args.data_dir, args.suffix)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rec_encode = np.array(encode_onehot(np.where(off_diag)[1]),dtype=np.float32)
send_encode = np.array(encode_onehot(np.where(off_diag)[0]),dtype=np.float32)
rec_encode = torch.FloatTensor(rec_encode)
send_encode = torch.FloatTensor(send_encode)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    rec_encode = rec_encode.cuda()
    send_encode = send_encode.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

model = EdgeInferModule(
    (args.timesteps - args.prediction_steps)* 4,
    4,
    4,
    args.prediction_steps,
    128,
    128,
    128,
    128,
    rec_encode,
    send_encode,
    joint_dim=args.num_atoms,
    device='cuda' if args.no_cuda else 'cpu',
    tau=args.temp,
    num_passing=2,
    do_prob=0.2,
)

discriminator = PhysicsDiscriminator(
    4 * args.num_atoms,
    args.discrim_hidden_size,
    args.batch_size,
    args.discrim_num_layers,
)

# if args.load_folder:
#     encoder_file = os.path.join(args.load_folder, 'pretrain.pt')
#     model.load_state_dict(torch.load(encoder_file))

    # args.save_folder = False


optimizer = optim.Adam(model.parameters(),lr=args.lr)
optimizer_gan = optim.Adam(model.parameters(),lr=args.gan_lr, betas=(0.9, 0.999))
# optimizer_gan = optim.SGD(model.parameters(),lr=args.gan_lr, momentum=0.9)
# optimizer_discrim = optim.Adam(discriminator.parameters(),lr=args.discrim_lr,  betas=(0.9, 0.999))
optimizer_discrim = optim.SGD(discriminator.parameters(),lr=args.discrim_lr,  momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer_discrim, step_size=args.lr_decay,gamma=args.gamma)
discrim_l = nn.BCELoss(reduction="sum")

if args.cuda:
    model.cuda()
    discriminator.cuda()
    discrim_l.cuda()

discrim_criterion = lambda x,y : discrim_l(x,y) / args.batch_size

def eval(sample=False):
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    model.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()

        assert(data.size(2) >= args.timesteps)
        input_length = args.timesteps - args.prediction_steps
        input_raw = data[:,:,:input_length,:]
        input_data = input_raw.view(data.shape[0], data.shape[1], -1)
        input_state = input_raw.permute(2,0,1,3).contiguous()

        target = data[:, :, input_length:args.timesteps, :].permute(2,0,1,3).contiguous()
        # useless decoder input, only for compatible purposes

        decoder_input = torch.cat([input_state[-1:,:,:,:],target[:-1,:,:,:]], dim=0)

        mean,logstd, node_sample,_, edge_logits, edge_sample = model(input_data, decoder_input, randomize=sample)
        if sample:
            output = node_sample
            edges_output = edge_sample
        else:
            output = mean
            edges_output = torch.sigmoid(edge_logits)
        # edges_output = torch.sigmoid(edges_output[0].unsqueeze(2))
        edges_output = edges_output[0].unsqueeze(2)
        target = target.view(target.size(0), target.size(1), -1)
        loss_nll = nll_gauss(mean, logstd, target)
        prob = torch.cat([1.0-edges_output, edges_output], dim=-1)

        acc = edge_accuracy(prob, relations)

        nll_val.append(loss_nll.item())
        acc_val.append(acc)
        mse_val.append(F.mse_loss(output, target).item())

    print('mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)))
    log.flush()
    return np.mean(nll_val)

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    model.train()
    scheduler.step()
    for batch_idx, (data, relations) in enumerate(train_loader):
        loss = torch.tensor(0.0)
        if args.cuda:
            data, relations = data.cuda(), relations.float().cuda()
            loss = loss.cuda()
        assert(data.size(2) >= args.timesteps)
        input_length = args.timesteps - args.prediction_steps
        input_raw = data[:,:,:input_length,:]
        input_data = input_raw.view(data.shape[0], data.shape[1], -1)
        input_state = input_raw.permute(2,0,1,3).contiguous()
        target = data[:, :, input_length:args.timesteps, :].permute(2,0,1,3).contiguous()
        # useless decoder input, only for compatible purposes

        decoder_input = torch.cat([input_state[-1:,:,:,:],target[:-1,:,:,:]], dim=0)

        target = target.view(target.size(0), target.size(1), -1)
        output, logstd, _, _, edges, _ = model(input_data, decoder_input)
        edges_output = torch.sigmoid(edges.unsqueeze(3))
        # edges_output = edges.unsqueeze(3)
        target = target.view(target.size(0), target.size(1), -1)

        loss_nll = nll_gauss(output, logstd, target)

        prob = torch.cat([1.0-edges_output, edges_output], dim=-1)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                                 args.edge_types)

        # supervise_loss = nn.BCELoss()
        # supervise_loss(edges_output, relations)
        # vars = torch.var(edges.permute(1,2,0).contiguous(), dim=-1,keepdim=False)
        loss = loss_nll - loss_kl
        # loss = loss_nll + torch.sum(torch.log(edges_output + 1e-7)) / (target.size(0) * target.size(1))
        # loss = loss_nll
        optimizer.zero_grad()
        # loss = loss_nll
        acc = edge_accuracy(prob[0], relations)
        acc_train.append(acc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_gradient_norm)
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss.item())
        kl_train.append(loss_kl.item())

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'time: {:.4f}s'.format(time.time() - t))
    nll_val = eval(sample=False)
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(model.state_dict(), encoder_file)
        torch.save(model.state_dict(), pretrain_file)
        print('Best model so far, saving...')

    return nll_val


def train_discrim(discrim_epoch):
    model.eval()
    discriminator.train()

    pre_mod_ps = []
    pre_exp_ps = []
    discrim_losses = []
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        assert(data.size(2) >= args.timesteps)
        input_length = args.timesteps - args.prediction_steps

        input_raw = data[:,:,:input_length,:]
        input_data = input_raw.view(data.shape[0], data.shape[1], -1)
        input_state = input_raw.permute(2,0,1,3).contiguous()
        optimizer_discrim.zero_grad()

        target = data[:, :, input_length:args.timesteps, :].permute(2,0,1,3).contiguous()
        # useless decoder input, only for compatible purposes

        decoder_input = torch.cat([input_state[-1:,:,:,:],target[:-1,:,:,:]], dim=0)

        target = target.view(target.size(0), target.size(1), -1)
        _, _, output, _, _ , _= model(input_data, decoder_input, randomize=True)

        input_state = input_state.view(input_state.size(0),input_state.size(1),-1)
        decoder_input = decoder_input.view(decoder_input.size(0), decoder_input.size(1),-1)

        state, action = get_state_action(input_state[:-1,:,:], decoder_input,output)
        ex_state, ex_action = get_state_action(input_state[:-1,:,:], decoder_input, target)

        pre_mod_p, pre_exp_p, discrim_loss = update_discrim(1.0, discriminator, optimizer_discrim, discrim_criterion, ex_state, ex_action, state, action,args.device, input_length)

        pre_mod_ps.append(pre_mod_p.item())
        pre_exp_ps.append(pre_exp_p.item())
        discrim_losses.append(discrim_loss)

    print('discrim Epoch: {:04d}'.format(discrim_epoch),
          'pre_mod_p: {:.10f}'.format(np.mean(pre_mod_ps)),
          'pre_exp_p: {:.10f}'.format(np.mean(pre_exp_ps)),
          'discrim_loss: {:.10f}'.format(np.mean(discrim_losses)))
    torch.save(discriminator.state_dict(), discrim_file)
    return 0
    # return eval(True)


def train_gan(gan_epoch):
    model.train()
    discriminator.train()
    pre_mod_ps = []
    pre_exp_ps = []
    discrim_losses = []
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        assert(data.size(2) >= args.timesteps)
        input_length = args.timesteps - args.prediction_steps

        input_raw = data[:,:,:input_length,:]
        input_data = input_raw.view(data.shape[0], data.shape[1], -1)
        input_state = input_raw.permute(2,0,1,3).contiguous()
        optimizer_discrim.zero_grad()

        target = data[:, :, input_length:args.timesteps, :].permute(2,0,1,3).contiguous()
        # useless decoder input, only for compatible purposes

        decoder_input = torch.cat([input_state[-1:,:,:,:],target[:-1,:,:,:]], dim=0)

        target = target.view(target.size(0), target.size(1), -1)
        mean, logstd, output, edge_logits,_, edges_output = model(input_data, decoder_input)
        loss_nll = nll_gauss(output, logstd, target)
        edges_output = edges_output.unsqueeze(3)
        prob = torch.cat([1.0-edges_output, edges_output], dim=-1)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                                 args.edge_types)
        add_loss = loss_nll * 0.2 + torch.sum(torch.abs(edge_logits)) / args.batch_size - loss_kl
        input_state = input_state.view(input_state.size(0),input_state.size(1),-1)
        decoder_input = decoder_input.view(decoder_input.size(0), decoder_input.size(1),-1)

        state, action = get_state_action(input_state[:-1,:,:], decoder_input,output)
        ex_state, ex_action = get_state_action(input_state[:-1,:,:], decoder_input, target)

        # local_mod_p = update_policy(model, optimizer_gan, discriminator, discrim_criterion, state, action, input_length, 20.0, args.device,addition_loss=add_loss)

        pre_mod_p, pre_exp_p, discrim_loss = update_discrim(1.0, discriminator, optimizer_discrim, discrim_criterion, ex_state, ex_action, state, action,args.device, input_length, train=True)
        pre_mod_ps.append(pre_mod_p.item())
        pre_exp_ps.append(pre_exp_p.item())
        discrim_losses.append(add_loss.item())
        if pre_mod_p < 0.7:
            local_mod_p = update_policy(model, optimizer_gan, discriminator, discrim_criterion, state, action, input_length, 20.0, args.device,addition_loss=add_loss)
        # print(pre_mod_p)

    print('gan Epoch: {:04d}'.format(gan_epoch),
          'pre_mod_p: {:.10f}'.format(np.mean(pre_mod_ps)),
          'pre_exp_p: {:.10f}'.format(np.mean(pre_exp_ps)),
          'vae_loss: {:.10f}'.format(np.mean(discrim_losses)))
    torch.save(model.state_dict(), encoder_file)
    torch.save(discriminator.state_dict(), discrim_file)
    return eval(True)


def test(sample):
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0

    model.eval()
    model.load_state_dict(torch.load(encoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        assert(data.size(2) >= args.timesteps)
        input_length = args.timesteps - args.prediction_steps
        input_raw = data[:,:,:input_length,:]
        input_data = input_raw.view(data.shape[0], data.shape[1], -1)
        input_state = input_raw.permute(2,0,1,3).contiguous()
        optimizer_discrim.zero_grad()

        target = data[:, :, input_length:args.timesteps, :].permute(2,0,1,3).contiguous()
        # useless decoder input, only for compatible purposes

        decoder_input = torch.cat([input_state[-1:,:,:,:],target[:-1,:,:,:]], dim=0)

        mean,logstd, node_sample,_, edge_logits, edge_sample = model(input_data, decoder_input, randomize=sample)
        if sample:
            output = node_sample
            edges_output = edge_sample
        else:
            output = mean
            edges_output = torch.sigmoid(edge_logits)

        edges_output = edges_output[0].unsqueeze(2)
        target = target.view(target.size(0), target.size(1), -1)
        loss_nll = nll_gaussian(output, target, args.var)

        prob = torch.cat([1.0-edges_output, edges_output], dim=-1)

        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy(prob, relations)
        acc_test.append(acc)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())

        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

    # mean_mse = tot_mse / counter
    # mse_str = '['
    # for mse_step in mean_mse[:-1]:
    #     mse_str += " {:.12f} ,".format(mse_step)
    # mse_str += " {:.12f} ".format(mean_mse[-1])
    # mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    # print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        # print('MSE: {}'.format(mse_str), file=log)
        log.flush()

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
if not args.ignore_pretrain:
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if args.save_folder:
                torch.save(model.state_dict(), encoder_file)
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    test(False)
else:
    print("ignore pretrain.")

# if not args.load_folder is None:
#     model.load_state_dict(torch.load(os.path.join(args.load_folder, 'pretrain.pt')))
    # discriminator.load_state_dict(torch.load(os.path.join(args.load_folder, 'discrim_pretrain.pt')))
    # args.save_folder = False

if args.train_gan:
    # # discrim pre-train
    # for epoch in range(args.discrim_epochs):
    #     pre_mod_p = train_discrim(epoch)
    # print("discriminator pretrained")
    # # #gan training
    # for epoch in range(args.gan_epochs):
    #     local_mod_p = train_gan(epoch)
    for epoch in range(args.gan_epochs):
        if epoch % 3 == 2:
            local_mod_p = train_gan(epoch)
        else:
            pre_mod_p = train_discrim(epoch)
        scheduler.step()
test(True)
if log is not None:
    print(save_folder)
    log.close()
