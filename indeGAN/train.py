import argparse
import os 
import math
import numpy as np
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import torch.optim as optim
from torchinfo import summary

from model import Generator, Discriminator

from util import crop_to_subregions, sliding_window_labels, compute_gradient_penalty, show_result, CustomDataset, CustomCropTransform


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='./', help='directory contrains the data and outputs')
  parser.add_argument('--epochs', type=int, default=160, help='training epoch number')
  parser.add_argument('--sub_res', type=int, default=128, help='The resolution of final output image')
  parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
  parser.add_argument('--full_res', type=int, default=512, help='The resolution of dataset image')
  parser.add_argument('--window_size', type=int, default=2, help='The window size of training')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
  parser.add_argument('--latent_size', type=int, default=128, help='ngf')
  parser.add_argument('--ngf', type=int, default=512, help='ngf')
  parser.add_argument('--ndf', type=int, default=512, help='ndf')

  opt = parser.parse_args()
  n_sub = int((opt.full_res / opt.sub_res) ** 2)
  n_sub_on_axis = int((opt.full_res / opt.sub_res))
  label_size = n_sub
  G_lr = 1e-4
  D_lr = 4e-4
  
  
  sliding_indices = sliding_window_labels(n_sub_on_axis, opt.window_size, 1)
  print(sliding_indices)
  
  G = Generator(opt.ngf, opt.latent_size, label_size, opt.sub_res)
  D = Discriminator(opt.ndf, label_size, opt.sub_res)
  
  G.weight_init(mean=0.0, std=0.02)
  D.weight_init(mean=0.0, std=0.02)

  G.cuda()
  D.cuda()
  
  G_optimizer = optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.9))
  D_optimizer = optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.9))

  G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=100, gamma=0.5)
  D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimizer, step_size=100, gamma=0.5)

  classification_loss = torch.nn.CrossEntropyLoss()

  summary(G, input_size=[(1, opt.latent_size, 1, 1), (1, 1)])
  summary(D, input_size=[(1, 3, opt.sub_res, opt.sub_res)])

  
  
  transform = CustomCropTransform(grid_size=n_sub_on_axis, resize_res=opt.full_res)
  transformed_dataset = CustomDataset(datasets.ImageFolder(os.path.join(f'../GAN/data/celeba_hq')), transform=transform)

  train_loader = DataLoader(
      transformed_dataset,
      batch_size=opt.batch_size,  # Since each subregion is a single data point
      shuffle=True,
      num_workers=12,
      pin_memory=True,
      drop_last=True
  )

  
  for epoch in range(opt.epochs):
    with tqdm(train_loader, unit="batch", ncols=110, miniters=0) as tepoch:
      tepoch.set_description(f"EPOCH [{epoch+1}/{opt.epochs}]")
      i = 0
      
      for x, y in tepoch:
        x, y = x.cuda(), y.cuda()
        
        batches_done = epoch * len(train_loader) + i
        i = i + 1
        
        #subregions = crop_to_subregions(x, opt.sub_res)
      
        fakes = []
        noise = torch.randn(x.size(0), opt.latent_size, 1, 1)
        noise = Variable(noise.cuda())
        
        D.zero_grad()
        fake = G(noise, y)
        fakes.append(fake)
        
        fake_D_out, _ = D(fake.detach())
        real_D_out, real_class_out = D(x)
        
        D_class_loss = classification_loss(real_class_out, y.long().view(-1))
        
        D_loss = fake_D_out.mean() - real_D_out.mean() + D_class_loss
        #D_loss = -torch.mean(D(xs, single_y)) + torch.mean(D(fake.detach(), single_y))
        gradient_penalty = compute_gradient_penalty(D, x, fake.detach(), y)
        lambda_gp = 10
        D_loss = D_loss + lambda_gp * gradient_penalty
        
        D_loss.backward()
        D_optimizer.step()
        
        
        ###  G Training Phase  ###
        G.zero_grad()
        fake_out, fake_class_out = D(fake)
        
        D_class_loss = classification_loss(fake_class_out, y.long().view(-1))
        G_loss = - fake_out.mean() 
        G_train_loss = G_loss + D_class_loss
        
        G_train_loss.backward()
        G_optimizer.step()
        
        
        # current_window = sliding_indices[i % len(sliding_indices)]
        # labels = torch.tensor(current_window).view(1, -1, 1).cuda()
        # print(labels.size())
        # noise = torch.randn(1, opt.latent_size).repeat(1, labels.size(0), 1, 1).cuda()
        # print(labels, noise.size())
        # fake = G(noise, y)
        
        # print(fake.size())
        #subregions = crop_to_subregions(x, opt.sub_res, indices_of_interest=current_window)
        #single_y = np.array([[current_window[idx]]] * opt.batch_size)
        #single_y = torch.from_numpy(single_y).cuda()
          
        
        
        
        # sub = []
        # #for j in range(n_sub): # opt.winsize ** 2
        # for j in range(opt.window_size ** 2): # opt.winsize ** 2
        #   start_j = j * opt.batch_size
        #   sub.append(fake[start_j:start_j+opt.batch_size])


        # # (NSUB, B, C, H, W)
        # f_rows = torch.zeros(opt.window_size, opt.batch_size, 3, opt.sub_res, opt.sub_res * opt.window_size).cuda()
        # f_cols = torch.zeros(opt.window_size, opt.batch_size, 3, opt.sub_res * opt.window_size, opt.sub_res).cuda()

        # for j in range(opt.window_size):
        #     start_j = j * opt.window_size
        #     f_rows[j] = torch.cat(sub[start_j: start_j+opt.window_size], dim=3).cuda()
        #     f_cols[j] = torch.cat(sub[j::opt.window_size], dim=2).cuda()


        # boundary_thickness = 1

        # diff_rows = F.mse_loss(f_rows[:-1, :, :, -boundary_thickness: :], f_rows[1:, :, :, :boundary_thickness, :])
        # diff_cols = F.mse_loss(f_cols[:-1, :, :, :, -boundary_thickness:], f_cols[1:, :, :, :, :boundary_thickness])
        # B_loss = diff_rows + diff_cols

        # # if epoch + 1 >= 3:
        # #   B_lambda = 15.5
        # # else:
        # #   B_lambda = 1
        
        # B_lambda = 50 * np.log10(0.2 * epoch + 7)

        

        
        if batches_done % 500 == 0:
          fixed_p ='Fixed_results/' + 'celeb' + str(batches_done) + '.png'
          show_result(G, show=False,  path=fixed_p, nb=16, latent_size=opt.latent_size, label_size=label_size, sub_size=opt.sub_res)
        
    G_scheduler.step()
    D_scheduler.step()
        
        
        


if __name__ == '__main__':
    main()