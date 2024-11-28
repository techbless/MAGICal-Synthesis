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

from model import Generator, Discriminator, FeatureExtractor

from util import CustomDataset, CustomCropTransform
from util import sliding_window_labels, compute_gradient_penalty, show_result

import GPUtil

def add_unit(mem: float) -> str:
    if mem > 1024:
        mem = round(mem / 1024, 2)
        mem = f"{mem}GiB"
    else:
        mem = round(mem, 2)
        mem = f"{mem}MiB"
    return mem

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='./', help='directory contrains the data and outputs')
  parser.add_argument('--epochs', type=int, default=160, help='training epoch number')
  parser.add_argument('--sub_res', type=int, default=64, help='The resolution of final output image')
  parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
  parser.add_argument('--full_res', type=int, default=256, help='The resolution of dataset image')
  parser.add_argument('--window_size', type=int, default=3, help='The window size of training')
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
  
  root = opt.root
  data_dir = root + 'dataset/'
  check_point_dir = root + 'check_points/'
  weight_dir = root+ 'weight/'
  if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
  if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
  
  
  sliding_indices = sliding_window_labels(n_sub_on_axis, opt.window_size, 1)
  print(sliding_indices)
  
  G = Generator(opt.ngf, opt.latent_size, label_size, opt.sub_res)
  D = Discriminator(opt.ndf, label_size, opt.sub_res)
  #feature_extractor = FeatureExtractor()
  
  G.weight_init(mean=0.0, std=0.02)
  D.weight_init(mean=0.0, std=0.02)

  G.cuda()
  D.cuda()
  #feature_extractor.cuda()
  
  G_optimizer = optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.9))
  D_optimizer = optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.9))

  
  classification_loss = torch.nn.CrossEntropyLoss()

  summary(G, input_size=[(1, opt.latent_size, 1, 1), (1, 1)])
  summary(D, input_size=[(1, 3, opt.sub_res, opt.sub_res)])
  
  if opt.resume != 0:
    check_point = torch.load(check_point_dir+'check_point_epoch_%i.pth' % opt.resume)
    G.load_state_dict(check_point['G_net'])
    D.load_state_dict(check_point['D_net'])
    G_optimizer.load_state_dict(check_point['G_optimizer'])
    D_optimizer.load_state_dict(check_point['D_optimizer'])

  G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=100, gamma=0.5)
  D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimizer, step_size=100, gamma=0.5)

  
  transform = CustomCropTransform(grid_size=n_sub_on_axis, resize_res=opt.full_res)
  #transformed_dataset = CustomDataset(datasets.ImageFolder(os.path.join(f'../../GAN/data/FFHQ512')), transform=transform)
  transformed_dataset = CustomDataset(datasets.ImageFolder(os.path.join(f'../../GAN/data/FFHQ512')), transform=transform)

  train_loader = DataLoader(
      transformed_dataset,
      batch_size=opt.batch_size,  # Since each subregion is a single data point
      shuffle=True,
      num_workers=12,
      pin_memory=True,
      drop_last=True
  )

  
  for epoch in range(1 + opt.resume, opt.epochs):
    with tqdm(train_loader, unit="batch", ncols=110, miniters=0) as tepoch:
      tepoch.set_description(f"EPOCH [{epoch+1}/{opt.epochs}]")
      i = 0
      
      for x, y in tepoch:
        x, y = x.cuda(), y.cuda()
        
        batches_done = epoch * len(train_loader) + i
        i = i + 1
        
      
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
        gradient_penalty = compute_gradient_penalty(D, x, fake.detach(), y)
        lambda_gp = 10
        D_loss = D_loss + lambda_gp * gradient_penalty
        
        D_loss.backward()
        D_optimizer.step()
        
        
        # x = None
        # torch.cuda.empty_cache()
        
        ###  G Training Phase  ###
        G.zero_grad()
        fake_out, fake_class_out = D(fake)
        
        D_class_loss = classification_loss(fake_class_out, y.long().view(-1))
        G_loss = - fake_out.mean() 
        G_train_loss = G_loss + D_class_loss
        
        G_train_loss.backward()
        G_optimizer.step()
        
        # fake = None
        # torch.cuda.empty_cache()
        
        ### Boundary Training Phase ###
        G.zero_grad()
        
        current_window = sliding_indices[i % len(sliding_indices)]
        labels = torch.tensor(current_window).view(1, -1, 1).cuda()
        
        noise = torch.randn(opt.latent_size).repeat(labels.size(1), 1).view(-1, opt.latent_size, 1 , 1).cuda()
        
        fake = G(noise, labels)
        fake_out, fake_class_out = D(fake)
        D_class_loss = classification_loss(fake_class_out, labels.long().view(-1))
        
                     
        b = 1
        sub = []
        for j in range(opt.window_size ** 2): # opt.winsize ** 2
          sub.append(fake[j])


        # (NSUB, B, C, H, W)
        #f_rows = torch.zeros(opt.window_size, b, 3, opt.sub_res, opt.sub_res * opt.window_size).cuda()
        #f_cols = torch.zeros(opt.window_size, b, 3, opt.sub_res * opt.window_size, opt.sub_res).cuda()
        f_rows = torch.tensor([]).cuda()
        f_cols = torch.tensor([]).cuda()
        
        for j in range(opt.window_size):
            start_j = j * opt.window_size
            r = torch.cat(sub[start_j: start_j+opt.window_size], dim=2)
            c = torch.cat(sub[j::opt.window_size], dim=1)
            # r = feature_extractor(r)
            # c = feature_extractor(c)
            r = r.view(1, 1, r.size(0), r.size(1), r.size(2))
            c = c.view(1, 1, c.size(0), c.size(1), c.size(2))
            
            f_rows = torch.cat([f_rows, r], dim=0)
            f_cols = torch.cat([f_cols, c], dim=0)
            
        boundary_thickness = 1

        diff_rows = F.mse_loss(f_rows[:-1, :, :, -boundary_thickness: :], f_rows[1:, :, :, :boundary_thickness, :])
        diff_cols = F.mse_loss(f_cols[:-1, :, :, :, -boundary_thickness:], f_cols[1:, :, :, :, :boundary_thickness])
        B_loss = diff_rows + diff_cols

        # if epoch + 1 >= 3:
        #   B_lambda = 15.5
        # else:
        #   B_lambda = 1
        
        B_lambda = 205

        G_loss = - fake_out.mean() 
        G_train_loss = G_loss + D_class_loss + (B_lambda * B_loss)
        
        G_train_loss.backward()
        G_optimizer.step()
        
        # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        
        # print(torch.cuda.memory_summary())

        # if batches_done % 1000 == 0:
        #   for gpu in GPUtil.getGPUs():
        #     gpu_util = f"{gpu.load}%"
        #     mem_total = add_unit(gpu.memoryTotal)
        #     mem_used = add_unit(gpu.memoryUsed)
        #     mem_used_percent = f"{round(gpu.memoryUtil * 100, 2)}%"
        #     print(f"ID: {gpu.id}, Util: {gpu_util}, Memory: {mem_used} / {mem_total} ({mem_used_percent})")
        
        if batches_done % 2000 == 0:
          fixed_p ='Fixed_results/' + 'celeb' + str(batches_done) + '.png'
          fixed_original_p = 'Fixed_original_results/' + 'celeb' + str(batches_done) + '.png'
          show_result(G, show=False,  path=fixed_p, nb=16, latent_size=opt.latent_size, label_size=label_size, sub_size=opt.sub_res)
          show_result(G, show=False,  path=fixed_original_p, nb=1, latent_size=opt.latent_size, label_size=label_size, sub_size=opt.sub_res)
        
    show_result(G, show=False,  path='epochs/out_' + str(epoch) + '.png', nb=16, latent_size=opt.latent_size, label_size=label_size, sub_size=opt.sub_res)
          
    check_point = {
      'G_net' : G.state_dict(), 
      'G_optimizer' : G_optimizer.state_dict(),
      'D_net' : D.state_dict(),
      'D_optimizer' : D_optimizer.state_dict(),
    }
    
    with torch.no_grad():
      G.eval()
      D.eval()
      torch.save(check_point, check_point_dir + 'check_point_epoch_%d.pth' % (epoch))
      G.train()
      D.train()

    G_scheduler.step()
    D_scheduler.step()
        
        
        


if __name__ == '__main__':
    main()