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

from model import Generator, Discriminator

def crop_to_subregions(image, sub_size):
    _, _, H, W = image.shape  # Assuming image is a tensor of shape [B, C, H, W]
    assert H % sub_size == 0 and W % sub_size == 0, "Image dimensions must be divisible by sub_size"

    subregions = []

    # Calculate number of subregions in each dimension
    n_sub_h = H // sub_size
    n_sub_w = W // sub_size

    for i in range(n_sub_h):
        for j in range(n_sub_w):
            start_i = i * sub_size
            start_j = j * sub_size
            subregion = image[:, :, start_i:start_i + sub_size, start_j:start_j + sub_size]
            subregions.append(subregion)

    return subregions


# fixed noise & label
def show_result(G, num_epoch, show = False, save = False, path = 'result.png', nb = 16, latent_size = 128, label_size=4, sub_size = 64):
    img_size = sub_size * int(math.sqrt(label_size))

    if nb > 1:
        size_figure_grid = int(nb/4)
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(15, 15), constrained_layout=True)
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    else:
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    G.eval()

    z = torch.randn(nb, latent_size)
    z = z.repeat(label_size, 1)
    z = Variable(z.cuda())

    y = np.concatenate([
        [[i]] * nb for i in range(label_size)
    ])
    y = torch.from_numpy(y).cuda()

    G_result = G(z, y)
    G_result = G_result * 0.5 + 0.5

    sub = []
    for i in range(label_size):
        start_i = i * nb
        sub.append(G_result[start_i:start_i+nb])

    n_sub_h = img_size // sub_size
    n_sub_w = img_size // sub_size

    rows = []
    for i in range(n_sub_w):
        start_i = i * n_sub_w
        rows.append(sub[start_i: start_i+n_sub_w])


    full_images = []
    for i in range(n_sub_h):
        full_images.append(torch.cat(rows[i], dim=3))

    full_images = torch.cat(full_images, dim=2)


    if nb > 1:
        for k in range(4*4):
            i = k // 4
            j = k % 4
            ax[i, j].cla()
            full_image = full_images[k]
            full_image = full_image.permute(1, 2, 0).cpu().data.numpy()
            full_image = (full_image * 255).astype(np.uint8)
            ax[i, j].imshow(full_image)
        plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()
    else:
        full_image = full_images[0].permute(1, 2, 0).cpu().data.numpy()
        full_image = (full_image * 255).astype(np.uint8)
        ax.imshow(full_image)
        fig.savefig(path, dpi=full_image.shape[0]) 
        plt.close(fig)

    G.train()
    
def sliding_window_labels(grid_size, window_size, step):
    labels = np.arange(grid_size**2).reshape((grid_size, grid_size))
    sub_regions = []

    for i in range(0, grid_size - window_size + 1, step):
        for j in range(0, grid_size - window_size + 1, step):
            window = labels[i:i+window_size, j:j+window_size]
            sub_regions.append(window.flatten())

    return sub_regions

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='./', help='directory contrains the data and outputs')
  parser.add_argument('--epochs', type=int, default=120, help='training epoch number')
  parser.add_argument('--out_res', type=int, default=128, help='The resolution of final output image')
  parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
  parser.add_argument('--cuda', action='store_true', help='Using GPU to train')
  parser.add_argument('--img_size', type=int, default=512, help='The resolution of dataset image')
  parser.add_argument('--win_size', type=int, default=2, help='The window size of training')

  opt = parser.parse_args()

  n_sub = int((opt.img_size / opt.out_res) ** 2)
  
  label_size = n_sub
  n_sub_on_axis = int(math.sqrt(label_size))
                      
  sliding_indices = sliding_window_labels(n_sub_on_axis, opt.win_size, 1)
  
  print(sliding_indices)

  root = opt.root
  data_dir = root + 'dataset/'
  check_point_dir = root + 'check_points/'
  output_dir = root + 'output/'
  weight_dir = root+ 'weight/'
  if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

  ## The schedule contains [num of epoches for starting each size][batch size for each size][num of epoches for the transition phase]
  # schedule = [[0, 2, 3, 4, 5, 6, 7], [4, 4, 4, 32, 4, 2, 1], [0, 1, 1, 1, 1, 1, 5]]
  schedule = [[0, 15, 30, 45, 60, 75, 90], [4, 4, 4, 4, 2, 2, 1], [0, 5, 5, 5, 5, 5, 5]]
  batch_size = schedule[1][0]
  growing = schedule[2][0]
  epochs = opt.epochs
  latent_size = 384 # the dimension of latent vector for conditional will be (latent_size) / 3
  out_res = opt.out_res
  G_lr = 1e-4
  D_lr = 1e-4 * 4
  lambd = 10

  device = torch.device('cuda:0' if (torch.cuda.is_available() and opt.cuda)  else 'cpu')

  transform = transforms.Compose([
        transforms.Resize(out_res),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

  D_net = Discriminator(latent_size, label_size, out_res).to(device)
  G_net = Generator(latent_size, label_size, out_res).to(device)

  fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
  D_optimizer = optim.Adam(D_net.parameters(), lr=D_lr, betas=(0, 0.99))
  G_optimizer = optim.Adam(G_net.parameters(), lr=G_lr, betas=(0, 0.99))

  classification_loss = torch.nn.CrossEntropyLoss()


  D_running_loss = 0.0
  G_running_loss = 0.0
  iter_num = 0

  D_epoch_losses = []
  G_epoch_losses = []

  if torch.cuda.device_count() > 1:
    print('Using ', torch.cuda.device_count(), 'GPUs')
    D_net = nn.DataParallel(D_net)
    G_net = nn.DataParallel(G_net)

  if opt.resume != 0:
    check_point = torch.load(check_point_dir+'check_point_epoch_%i.pth' % opt.resume)
    fixed_noise = check_point['fixed_noise']
    G_net.load_state_dict(check_point['G_net'])
    D_net.load_state_dict(check_point['D_net'])
    G_optimizer.load_state_dict(check_point['G_optimizer'])
    D_optimizer.load_state_dict(check_point['D_optimizer'])
    G_epoch_losses = check_point['G_epoch_losses']
    D_epoch_losses = check_point['D_epoch_losses']
    G_net.depth = check_point['depth']
    D_net.depth = check_point['depth']
    G_net.alpha = check_point['alpha']
    D_net.alpha = check_point['alpha']

  def find_index(arr, value):
    index = -1
    for i in range(len(arr)):
        if arr[i] < value:
            index = i
        else:
            break
    return index

  try:
    if opt.resume != 0:
      #c = next(x[0] for x in enumerate(schedule[0]) if x[1]>opt.resume)
      c = find_index(schedule[0], opt.resume)
    else:
      c = 0

    batch_size = schedule[1][c]
    growing = schedule[2][c]
    #dataset = datasets.ImageFolder(data_dir, transform=transform)
    #dataset = datasets.CelebA(data_dir, split='all', transform=transform)
    dataset = datasets.ImageFolder(os.path.join(f'../GAN/data/celeba_hq'), transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    tot_iter_num = (len(dataset)/batch_size)
    G_net.fade_iters = (1-G_net.alpha)/(schedule[0][c+1]-opt.resume)/(2*tot_iter_num)
    D_net.fade_iters = (1-D_net.alpha)/(schedule[0][c+1]-opt.resume)/(2*tot_iter_num)


  except Exception as e:
    print(e)
    print('Fully Grown\n')
    c = -1
    batch_size = schedule[1][c]
    growing = schedule[2][c]

    #dataset = datasets.CelebA(data_dir, split='all', transform=transform, download=False)
    dataset = datasets.ImageFolder(os.path.join(f'../GAN/data/celeba_hq'), transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    tot_iter_num = (len(dataset)/batch_size)
    print(schedule[0][c], opt.resume)

    if G_net.alpha < 1:
      G_net.fade_iters = (1-G_net.alpha)/(opt.epochs-opt.resume)/(2*tot_iter_num)
      D_net.fade_iters = (1-D_net.alpha)/(opt.epochs-opt.resume)/(2*tot_iter_num)


  size = 2**(G_net.depth+1)
  print("Output Resolution: %d x %d" % (size, size))
  print("Batch Size: %d" % (batch_size))
  print("Growing Phase %d" % (growing))
  print("Depth of Nets: %d" % (G_net.depth))

  y_real = torch.ones(batch_size * n_sub, 1).cuda()
  y_fake = torch.zeros(batch_size * n_sub, 1).cuda()
  
  

  
  for epoch in range(1+opt.resume, opt.epochs+1):
    G_net.train()
    D_epoch_loss = 0.0
    G_epoch_loss = 0.0
    if epoch in schedule[0]:

      if (2 **(G_net.depth +1) < out_res):
        c = schedule[0].index(epoch)
        batch_size = schedule[1][c]
        growing = schedule[2][c]
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        tot_iter_num = tot_iter_num = (len(dataset)/batch_size)
        G_net.growing_net(growing*tot_iter_num)
        D_net.growing_net(growing*tot_iter_num)
        size = 2**(G_net.depth+1)
        print("Output Resolution: %d x %d" % (size, size))
        print("Batch Size: %d" % (batch_size))
        print("Growing Phase %d" % (growing))
        print("Depth of Nets: %d" % (G_net.depth))

    # y = np.concatenate([[[i]] * batch_size for i in range(n_sub)])
    # y = torch.from_numpy(y).cuda()

    
    

    print("epoch: %i/%i" % (int(epoch), int(epochs)))
    databar = tqdm(data_loader, unit="batch", ncols=120, miniters=0)

    for i, x in enumerate(databar):
      batches_done = (epoch - 1) * len(data_loader) + i

      x = F.interpolate(x[0], size=size*int(math.sqrt(label_size))).to(device)


      subregions = crop_to_subregions(x, size)
      
      fakes = []
      noise = torch.randn(x.size(0), latent_size)
      noise = Variable(noise.cuda())
      
      y = torch.empty(0).cuda()
      
      #for idx, xs in enumerate(subregions):
      current_window = sliding_indices[i % len(sliding_indices)]
      for idx in current_window:
        xs = subregions[idx]
        #xs = torch.cat(subregions, dim=0)
        #single_y = np.concatenate([[[i]] * 1 for i in range(n_sub)])
        single_y = np.array([[idx]] * batch_size)
        single_y = torch.from_numpy(single_y).cuda()
        
        y = torch.cat((y, single_y), dim=0)
        
    
        xs = xs.view(-1, 3, size, size)
        xs = Variable(xs.cuda())
        D_net.zero_grad()

        # noise = torch.randn(samples.size(0), latent_size, 1, 1, device=device)

        fake = G_net(noise, single_y)
        
        fakes.append(fake)

        fake_out, _ = D_net(fake.detach())
        real_out, real_class_out = D_net(xs)
        d_class_loss = classification_loss(real_class_out, single_y.long().view(-1))

      

        ## Gradient Penalty

        eps = torch.rand(xs.size(0), 1, 1, 1, device=device)
        eps = eps.expand_as(xs)
        x_hat = eps * xs + (1 - eps) * fake.detach()
        x_hat.requires_grad = True
        px_hat, _ = D_net(x_hat)
        grad = torch.autograd.grad(
                      outputs = px_hat.sum(),
                      inputs = x_hat, 
                      create_graph=True
                      )[0]
        grad_norm = grad.view(xs.size(0), -1).norm(2, dim=1)
        gradient_penalty = lambd * ((grad_norm  - 1)**2).mean()

        ###########

        D_loss = fake_out.mean() - real_out.mean() + gradient_penalty + d_class_loss

        D_loss.backward()
        D_optimizer.step()

      ##  update G

      fake = torch.cat(fakes, dim=0)
      G_net.zero_grad()
      # noise = torch.randn(x.size(0), latent_size)
      # noise = noise.repeat(label_size, 1)
      # noise = Variable(noise.cuda())
      # fake = G_net(noise, y)

      sub = []
      #for j in range(n_sub): # opt.winsize ** 2
      for j in range(opt.win_size ** 2): # opt.winsize ** 2
          start_j = j * batch_size
          sub.append(fake[start_j:start_j+batch_size])

      n_sub_h = opt.win_size # opt.win_size
      n_sub_w = opt.win_size # opt.win_size
      
      # n_sub_h = int(math.sqrt(label_size)) # opt.win_size
      # n_sub_w = int(math.sqrt(label_size)) # opt.win_size


      # (NSUB, B, C, H, W)
      f_rows = torch.zeros(n_sub_w, batch_size, 3, size, size*n_sub_w).cuda()
      f_cols = torch.zeros(n_sub_h, batch_size, 3, size*n_sub_h, size).cuda()

      for j in range(n_sub_w):
          start_j = j * n_sub_w
          f_rows[j] = torch.cat(sub[start_j: start_j+n_sub_w], dim=3).cuda()
          f_cols[j] = torch.cat(sub[j::n_sub_h], dim=2).cuda()


      boundary_thickness = 1

      diff_rows = F.mse_loss(f_rows[:-1, :, :, -boundary_thickness: :], f_rows[1:, :, :, :boundary_thickness, :])
      diff_cols = F.mse_loss(f_cols[:-1, :, :, :, -boundary_thickness:], f_cols[1:, :, :, :, :boundary_thickness])
      B_loss = diff_rows + diff_cols

      B_lambda = 15.5 * np.log10(G_net.depth + .5)

      fake_out, fake_class_out = D_net(fake)
      
      d_class_loss = classification_loss(fake_class_out, y.long().view(-1)) # BCE

      G_loss = - fake_out.mean() 
      G_train_loss = G_loss + B_lambda * B_loss + d_class_loss


      

      G_train_loss.backward()
      G_optimizer.step()

      ##############

      D_running_loss += D_loss.item()
      G_running_loss += G_loss.item()

      iter_num += 1


      if i % 5000== 0:
        D_running_loss /= iter_num
        G_running_loss /= iter_num
        #print('iteration : %d, gp: %.2f' % (i, gradient_penalty))
        #databar.set_description('D_loss: %.3f   G_loss: %.3f' % (D_running_loss ,G_running_loss))
        databar.set_postfix({"G_loss": G_running_loss, "D_loss": D_running_loss, "B_loss": B_loss.item()})
                   
        iter_num = 0
        D_running_loss = 0.0
        G_running_loss = 0.0

      if batches_done % 2000 == 0:
        show_result(G_net, (epoch+1), save=True, path='output/img' + str(batches_done) + '.png', nb=16, latent_size=latent_size, label_size=label_size, sub_size=size)
        #show_result(G_net, (epoch+1), save=True, path=original_fixed_p, nb=1)

      
    D_epoch_losses.append(D_epoch_loss/tot_iter_num)
    G_epoch_losses.append(G_epoch_loss/tot_iter_num)


    check_point = {'G_net' : G_net.state_dict(), 
          'G_optimizer' : G_optimizer.state_dict(),
          'D_net' : D_net.state_dict(),
          'D_optimizer' : D_optimizer.state_dict(),
          'D_epoch_losses' : D_epoch_losses,
          'G_epoch_losses' : G_epoch_losses,
          'fixed_noise': fixed_noise,
          'depth': G_net.depth,
          'alpha':G_net.alpha
          }

    with torch.no_grad():
      G_net.eval()
      torch.save(check_point, check_point_dir + 'check_point_epoch_%d.pth' % (epoch))
      torch.save(G_net.state_dict(), weight_dir + 'G_weight_epoch_%d.pth' %(epoch))
      
      # out_imgs = G_net(fixed_noise)
      # out_grid = make_grid(out_imgs, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**G_net.depth))).permute(1,2,0)
      # plt.imshow(out_grid.cpu())
      # plt.savefig(output_dir + 'size_%i_epoch_%d' %(size ,epoch))




if __name__ == '__main__':
    main()