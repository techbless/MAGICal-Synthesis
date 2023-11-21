import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from utils import EqualizedLR_Conv2d, Pixel_norm, Minibatch_std


  

class FromRGB(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
    self.relu = nn.LeakyReLU(0.2)
    
  def forward(self, x):
    x = self.conv(x)
    return self.relu(x)

class ToRGB(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=(1, 1))
  
  def forward(self, x):

    return self.conv(x)

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Linear(num_classes, num_features)
        self.beta_embed = nn.Linear(num_classes, num_features)

    def forward(self, x, y):
        out = self.bn(x)
        y = y.long()
        # Assuming y is a batch of class labels, convert to one-hot encoding
        y_one_hot = F.one_hot(y, num_classes=self.gamma_embed.in_features).float()
        
        gamma = self.gamma_embed(y_one_hot).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(y_one_hot).view(-1, self.num_features, 1, 1)

        return out * gamma + beta

    # def forward(self, x, y):
    #     out = self.bn(x)
    #     y = y.float()
    #     gamma = self.gamma_embed(y).view(-1, self.num_features, 1, 1)
    #     beta = self.beta_embed(y).view(-1, self.num_features, 1, 1)
    #     return out * gamma + beta

class G_Block(nn.Module):
  def __init__(self, in_ch, out_ch, initial_block=False):
    super().__init__()
    if initial_block:
      self.upsample = None
      self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
    else:
      self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
      self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.relu = nn.LeakyReLU(0.2)
    self.pixelwisenorm = Pixel_norm()
    nn.init.normal_(self.conv1.weight)
    nn.init.normal_(self.conv2.weight)
    nn.init.zeros_(self.conv1.bias)
    nn.init.zeros_(self.conv2.bias)

  def forward(self, x):
    if self.upsample is not None:
      x = self.upsample(x)

    # x = self.conv1(x*scale1)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pixelwisenorm(x)
    # x = self.conv2(x*scale2)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pixelwisenorm(x)
    return x


class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, labels_size, initial_block=False):
        super().__init__()

        if initial_block:
            self.minibatchstd = Minibatch_std()
            conv1_in_ch = in_ch + 1  # Adding 1 to account for minibatch std feature
        else:
            self.minibatchstd = None
            conv1_in_ch = in_ch
            self.cbn = ConditionalBatchNorm2d(out_ch, labels_size)  # Conditional Batch Norm for non-initial blocks

        # Convolutional layers with Equalized Learning Rate
        self.conv1 = EqualizedLR_Conv2d(conv1_in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        kernel_size_conv2 = (4, 4) if initial_block else (3, 3)
        self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=kernel_size_conv2, stride=(1, 1), padding=0 if initial_block else (1, 1))

        # Output layer for the initial block
        if initial_block:
            self.outlayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_ch, 1)
            )
        else:
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, y=None):
        if self.minibatchstd is not None:
            x = self.minibatchstd(x)
        
        x = self.conv1(x)
        x = self.relu(x)

        if self.minibatchstd is None and y is not None:
            x = self.cbn(x, y)  # Apply CBN only in non-initial blocks

        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)

        return x




class Generator(nn.Module):
  def __init__(self, latent_size, label_size, out_res):
    super().__init__()
    self.latent_size = latent_size
    self.label_size = label_size
    self.depth = 1
    self.alpha = 1
    self.fade_iters = 0
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    self.current_net = nn.ModuleList([G_Block(latent_size*2, latent_size, initial_block=True)])
    self.toRGBs = nn.ModuleList([ToRGB(latent_size, 3)])
    self.embed = nn.Embedding(label_size, latent_size)

    # __add_layers(out_res)
    for d in range(2, int(np.log2(out_res))):
      if d < 6:
        ## low res blocks 8x8, 16x16, 32x32 with 512 channels
        in_ch, out_ch = 512, 512
      else:
        ## from 64x64(5th block), the number of channels halved for each block
        in_ch, out_ch = int(512 / 2**(d - 6)), int(512 / 2**(d - 5))
      self.current_net.append(G_Block(in_ch, out_ch))
      self.toRGBs.append(ToRGB(out_ch, 3))


  def forward(self, z, label):
    label = label.long()
    z = z.view(-1, self.latent_size, 1, 1)
    embedded_label = self.embed(label).view(-1, self.latent_size, 1, 1)
    
    x = torch.cat((z, embedded_label), dim=1)


    for block in self.current_net[:self.depth-1]:
      x = block(x)
    out = self.current_net[self.depth-1](x)
    x_rgb = self.toRGBs[self.depth-1](out)
    if self.alpha < 1:
      x_old = self.upsample(x)
      old_rgb = self.toRGBs[self.depth-2](x_old)
      x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb

      self.alpha += self.fade_iters

    return x_rgb
    

  def growing_net(self, num_iters):
    
    self.fade_iters = 1/num_iters
    self.alpha = 1/num_iters

    self.depth += 1

class Discriminator(nn.Module):
    def __init__(self, latent_size, label_size, out_res):
        super().__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.current_net = nn.ModuleList([D_Block(latent_size, latent_size, label_size, initial_block=True)])
        self.fromRGBs = nn.ModuleList([FromRGB(3, latent_size)])
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                in_ch, out_ch = 512, 512
            else:
                in_ch, out_ch = int(512 / 2**(d - 5)), int(512 / 2**(d - 6))
            self.current_net.append(D_Block(in_ch, out_ch, label_size))
            self.fromRGBs.append(FromRGB(3, in_ch))
  
    def forward(self, x_rgb, y):
        x = self.fromRGBs[self.depth-1](x_rgb)

        x = self.current_net[self.depth-1](x, y)
        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRGBs[self.depth-2](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.fade_iters

        for block in reversed(self.current_net[:self.depth-1]):
            x = block(x, y)

        return x
    
    def growing_net(self, num_iters):
        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters
        self.depth += 1





