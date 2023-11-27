import os
import math
import numpy as np
import itertools
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def compute_gradient_penalty(D, real_samples, fake_samples, y):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, y)
    fake = torch.ones(d_interpolates.size()).cuda()

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

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


def sliding_window_labels(grid_size, window_size, step):
    labels = np.arange(grid_size**2).reshape((grid_size, grid_size))
    sub_regions = []

    for i in range(0, grid_size - window_size + 1, step):
        for j in range(0, grid_size - window_size + 1, step):
            window = labels[i:i+window_size, j:j+window_size]
            sub_regions.append(window.flatten())

    return sub_regions


def show_result(G, show = False, path = 'result.png', nb = 16, latent_size = 128, label_size=4, sub_size = 64):
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