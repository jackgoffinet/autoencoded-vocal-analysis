from __future__ import print_function, division
"""
MWE for DLGM

"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from os import listdir
import argparse

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from usv_dataset import get_partition, setup_data_loaders
from rankonenormal import RankOneNormal


z_dim = 100

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 4, 4)
        self.fc1 = nn.Linear(4*26*26, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, z_dim)
        self.fc32 = nn.Linear(256, z_dim)
        self.fc33 = nn.Linear(256, z_dim)

    def forward(self, v):
        v = v.view(-1, 1, 32, 32)
        z = F.relu(self.conv1(v))
        z = F.relu(self.conv2(z))
        z = z.squeeze(1).reshape(-1,4*26*26)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        log_d = self.fc31(z)
        u = self.fc32(z)
        mu = self.fc33(z)
        return mu, log_d, u


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.g_1 = nn.Linear(z_dim, z_dim, bias=False)
        self.g_2 = nn.Linear(z_dim, z_dim, bias=False)
        self.g_3 = nn.Linear(z_dim, z_dim, bias=False)
        self.t_2 = nn.Linear(z_dim, z_dim)
        self.t_1 = nn.Linear(z_dim, z_dim)
        self.t_01 = nn.Linear(z_dim, 256)
        self.t_02 = nn.Linear(256, 4*26*26)
        self.convt1 = nn.ConvTranspose2d(4, 32, 4, padding=0)
        self.convt2 = nn.ConvTranspose2d(32, 1, 4, padding=0)

    def forward(self, xis):
        xi_1, xi_2, xi_3 = xis
        h_3 = self.g_3(xi_3)
        h_2 = F.relu(self.t_2(h_3)) + self.g_2(xi_2)
        h_1 = F.relu(self.t_1(h_2)) + self.g_1(xi_1)
        v = F.relu(self.t_01(h_1))
        v = self.t_02(v)
        v = v.view(-1, 4, 26, 26)
        v = F.relu(self.convt1(v))
        v = self.convt2(v)
        v = v.view(-1, 1024)
        v = torch.sigmoid(v) # Changed 10/31/18
        return v

# define a PyTorch module for the VAE
class VAE(nn.Module):
    """Variational Auto-Encoder"""

    def __init__(self):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cuda()


    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module <decoder> with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.iarange("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            mu = x.new_zeros((x.shape[0],z_dim))
            log_d = x.new_zeros((x.shape[0],z_dim))
            u = x.new_zeros((x.shape[0],z_dim))
            db = RankOneNormal(mu, log_d, u)
            xi_1 = pyro.sample("latent_1", db)
            xi_2 = pyro.sample("latent_2", db)
            xi_3 = pyro.sample("latent_3", db)
            loc_img = self.decoder.forward((xi_1, xi_2, xi_3))
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).independent(1), obs=x.reshape(-1, 1024))
            # return the loc so we can visualize it later
            return loc_img


    # define the guide (i.e. variational distribution)
    def guide(self, v):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.iarange("data", v.shape[0]):
            mu, log_d, u = self.encoder.forward(v)
            db = RankOneNormal(mu, log_d, u)
            pyro.sample("latent_1", db)
            pyro.sample("latent_2", db)
            pyro.sample("latent_3", db)


    # define a helper function for reconstructing images
    def reconstruct_img(self, v):
        # Encode image <v>.
        mu, log_d, u = self.encoder.forward(v)
        db = RankOneNormal(mu, log_d, u)
        xi_1 = pyro.sample("latent_1", db)
        xi_2 = pyro.sample("latent_2", db)
        xi_3 = pyro.sample("latent_3", db)
        loc_img = self.decoder.forward((xi_1, xi_2, xi_3))
        return loc_img


    def visualize(self, test_loader, filename='out.pdf'):
        f, axarr = plt.subplots(2,5)
        for i, temp in enumerate(test_loader):
            if i == 5:
                break
            x = temp['image'].cuda()
            im_num = 0
            axarr[0,i].imshow(x[im_num].detach().cpu().numpy())
            x = x.view(-1, 1024)
            reconstructed = self.reconstruct_img(x)
            axarr[1,i].imshow(reconstructed[im_num].detach().cpu().numpy().reshape((32,32)))
            axarr[0,i].axis('off')
            axarr[1,i].axis('off')
        plt.savefig(filename)
        plt.close('all')


def main(args):
    # clear param store
    pyro.clear_param_store()

    # set up data loaders
    dirs = ['data/fd/' + i for i in listdir('data/fd/') if 'fd' in i]
    # dirs += ['data/opto/' + i for i in listdir('data/opto/') if 'opto' in i]
    split=0.8
    checkpoint = None
    if args.load_state > 0:
        filename = str(args.load_state).zfill(2) + '/checkpoint.tar'
        checkpoint = torch.load(filename)
        partition = checkpoint['partition']
    else:
        partition = get_partition(dirs, split)
    train_loader, test_loader = setup_data_loaders(partition, batch_size=args.batch_size)

    # set up VAE
    vae = VAE()
    if checkpoint is not None:
        vae.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        vae.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # set up the optimizer
    optimizer = Adam({"lr": args.learning_rate})
    if checkpoint is not None:
        optimizer.set_state(checkpoint['optimizer_state'])

    # set up the inference algorithm
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    # set up the loss dictionaries
    if checkpoint is not None:
        train_elbo = checkpoint['train_elbo']
        test_elbo = checkpoint['test_elbo']
        start_epoch = checkpoint['epoch']
    else:
        train_elbo = {}
        test_elbo = {}
        start_epoch = 0

    print("num train samples: ", len(train_loader)*args.batch_size)
    print("num batches: ", len(train_loader))
    # training loop
    for epoch in range(start_epoch, args.num_epochs, 1):
        epoch_loss = 0.

        for i, data in enumerate(train_loader):
            x = data['image'].cuda().view(-1,1024)
            batch_loss = svi.step(x)
            epoch_loss += batch_loss


        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        print("[epoch %03d]  average train loss: %.4f" % (epoch, total_epoch_loss_train))

        # Every some number of epochs, do a bunch of stuff.
        if epoch % args.test_frequency == 0:
            print('testing')
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, temp in enumerate(test_loader):
                x = temp['image']
                x = x.cuda()
                x = x.view(-1,1024)
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

            # make a loss plot
            if epoch > 0:
                x_vals = range(0,epoch+1,args.test_frequency)
                plt.plot(x_vals, [test_elbo[x] for x in x_vals], label='test')
                x_vals = range(0,epoch+1,1)
                plt.plot(x_vals, [train_elbo[x] for x in x_vals], label='train')
                plt.legend()
                # plt.ylim(ymin=, ymax=-260)
                plt.title("Loss")
                plt.savefig(str(args.save_state).zfill(2) + '/loss.pdf')
                plt.close('all')

            # visualize round trip
            filename = str(args.save_state).zfill(2) + "/vis_test_"+str(epoch).zfill(3)+".pdf"
            vae.visualize(test_loader, filename=filename)
            filename = str(args.save_state).zfill(2) + "/vis_train_"+str(epoch).zfill(3)+".pdf"
            vae.visualize(train_loader, filename=filename)

            # save state: epoch, losses, partition
            if args.save_state > 0:
                filename = str(args.save_state).zfill(2) + '/checkpoint.tar'
                state = {
                    'train_elbo': train_elbo,
                    'test_elbo': test_elbo,
                    'epoch': epoch,
                    'encoder_state_dict': vae.encoder.state_dict(),
                    'decoder_state_dict': vae.decoder.state_dict(),
                    'optimizer_state': optimizer.get_state(),
                    'partition': partition,
                }
                torch.save(state, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-ls', '--load-state', default=0, type=int, help='directory number of saved state')
    parser.add_argument('-ss', '--save-state', default=0, type=int, help='directory number of saved state')
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('-tf', '--test-frequency', default=10, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    # parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    args = parser.parse_args()

    model = main(args)
