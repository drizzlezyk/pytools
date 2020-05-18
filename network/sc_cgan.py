import argparse
import os
import numpy as np
import scanpy as sc
import math

from anndata import AnnData
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch

import pre

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, data_size,1)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, out_size):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.encode = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
        )
        self.decode = nn.Sequential(
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, out_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        f = self.encode(gen_input)
        img = self.decode(f)
        # img = img.view(img.size(0), )
        return img, f


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + out_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator





base_path = '/Users/zhongyuanke/data/'
file1 = base_path + 'merge_result/cd14_cd34_b_cells.h5ad'
file2 = base_path + 'merge_result/293t_jurkat.h5ad'
file_scvi = base_path + 'scvi/scvi_batch_example.h5ad'
file_count = base_path + 'merge_result/293t_jurkat_batch.csv'

adata = pre.read_sc_data(file2)
sc.pp.filter_genes(adata, min_cells=100)
out_size = adata.shape[1]
# sc.pp.filter_cells(adata, min_genes=1000)
x = adata.X.A
# y = list(map(int, adata.obs['batch']))
y = pre.get_label_by_count(file_count)
print(len(y))
# y = pd.factorize(y)[0]

# y = np.zeros((adata.shape[0]), dtype=np.long)


x = torch.tensor(x)
y = torch.tensor(y)
torch_dataset = torch.utils.data.TensorDataset(x, y)
print(x.shape)
sample_size = adata.shape[0]

generator = Generator(out_size)
discriminator = Discriminator()
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataloader = torch.utils.data.DataLoader(
    dataset=torch_dataset,
    batch_size=opt.batch_size,
    shuffle=True,)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)


# ----------
#  Training
# ----------
#
# for epoch in range(opt.n_epochs):
#     for i, (imgs, labels) in enumerate(dataloader):
#         batch_size = imgs.shape[0]
#
#         # Adversarial ground truths
#         valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#         fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
#
#         # Configure input
#         real_imgs = Variable(imgs.type(FloatTensor))
#         labels = Variable(labels.type(LongTensor))
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise and labels as generator input
#         z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
#         gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
#
#         # Generate a batch of images
#         gen_imgs = generator(z, gen_labels)[0]
#
#         # Loss measures generator's ability to fool the discriminator
#         validity = discriminator(gen_imgs, gen_labels)
#         g_loss = adversarial_loss(validity, valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#
#         optimizer_D.zero_grad()
#
#         # Loss for fake images
#         validity_fake = discriminator(gen_imgs.detach(), gen_labels)
#         d_fake_loss = adversarial_loss(validity_fake, fake)
#
#         # Loss for real images
#         validity_real = discriminator(real_imgs, labels)
#         d_real_loss = adversarial_loss(validity_real, valid)
#
#
#
#         # Total discriminator loss
#         d_loss = (d_real_loss + d_fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#         )
#
#         batches_done = epoch * len(dataloader) + i
#     torch.save(generator, 'generator.pkl')
#     torch.save(discriminator, 'discrminator.pkl')

g = torch.load('generator.pkl')
d = torch.load('discrminator.pkl')



z = Variable(FloatTensor(np.random.normal(0, 1, (sample_size, opt.latent_dim))))
# Get labels ranging from 0 to n_classes for n rows
# labels = np.array([num for _ in range(n_row) for num in range(n_row)])
# labels = Variable(LongTensor(labels))

gen_imgs = generator(z, y)[0]
feature = generator(z, y)[1]

gen_imgs = gen_imgs.detach().numpy()
feature = feature.detach().numpy()
gen_imgs = np.array(gen_imgs)
print(gen_imgs)
adata = AnnData(X=gen_imgs)
adata.obsm['mid'] = feature

adata.write(base_path+'gan/293t_jurkat_gan_200.h5ad')

