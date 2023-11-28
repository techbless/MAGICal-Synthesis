import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

import numpy as np


class DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, initial=False):
        super(DResidualBlock, self).__init__()
        self.initial = initial
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        if initial:
          self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False)
        else:
          self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
         

          self.use_shortcut = stride != 1 or in_channels != out_channels
          if self.use_shortcut:
              self.shortcut = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
              )
          else:
              self.shortcut = nn.Identity()

    def forward(self, x):
        if self.initial:
          out = self.conv1(x)
          out = self.lrelu(out)
        else:
          out = self.conv1(x)
          #out = self.lrelu(out)

          shortcut = self.shortcut(x)
          out += shortcut
          out = self.lrelu(out)

        return out


class GResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, initial=False):
        super(GResidualBlock, self).__init__()
        self.initial = initial
        
        self.relu = nn.ReLU(inplace=True)
        
        if initial:
          self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride=stride, kernel_size=4, bias=False)
        else:
          self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)

          self.use_shortcut = stride != 1 or in_channels != out_channels
          if self.use_shortcut:
              self.shortcut = nn.Sequential(
                  nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=1, bias=False),
              )
          else:
              self.shortcut = nn.Identity()

    def forward(self, x):
      if self.initial:
        out = self.conv1(x)
        out = self.relu(out)
      else:
        out = self.conv1(x)
        #out = self.relu(out)

        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)

      return out
      
      
class Generator(nn.Module):
  # initializers
  def __init__(self, ngf=512, latent_size=384, label_size=128, sub_res=128):
    super(Generator, self).__init__()
    self.ngf = ngf
    self.latent_size = latent_size
    self.label_size = label_size
    
    self.embed = nn.Embedding(label_size, label_size)
    
    self.blocks = nn.ModuleList([GResidualBlock(latent_size + label_size, ngf, stride=1, initial=True)])
    print(latent_size + label_size, ngf)
    
    for d in range(1, int(np.log2(sub_res)) - 1):
      if d < 3:
        in_ch, out_ch = ngf, ngf
      else:
        in_ch, out_ch = int(ngf / 2**(d-3)), int(ngf / 2**(d-2))
      self.blocks.append(GResidualBlock(in_ch, out_ch, stride=2))
      print(in_ch, out_ch)

    d = int(np.log2(sub_res)) - 2
    print(int(ngf / 2**(d-2)), 3)
    self.out_layer = nn.Sequential(
      nn.ConvTranspose2d(int(ngf / 2**(d-2)), 3, 1, stride=1, bias=False),
      nn.Tanh()
    )
    

  # weight_init
  def weight_init(self, mean, std):
      for m in self._modules:
          normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, z, label):
      label = label.long()
      embedded_label = self.embed(label).view(-1, self.label_size, 1, 1)
      x = torch.cat((z, embedded_label), dim=1)

      #x = self.fc1(x).view(-1, self.ngf, 4, 4)
      for block in self.blocks:
        x = block(x)
      
      x = self.out_layer(x)

      return x


#(X - F + 2P) / S +1
class Discriminator(nn.Module):
  def __init__(self, ndf=512, label_size = 128, sub_res=128):
    super(Discriminator, self).__init__()
    self.ndf = ndf
    self.sub_res = sub_res
    self.label_size = label_size
    
    # Embedding which outputs a vector of img_size
    self.embed = nn.Embedding(self.label_size, sub_res * sub_res)
    
    self.blocks = nn.ModuleList([DResidualBlock(ndf, ndf, stride=1, initial=True)])
    print(ndf, ndf)
    for d in range(2, int(np.log2(sub_res)) - 1):
      if d < 3:
        in_ch, out_ch = ndf, ndf
      else:
        in_ch, out_ch = int(ndf / 2**(d - 2)), int(ndf / 2**(d - 3))
      print(in_ch, out_ch)
      self.blocks.append(DResidualBlock(in_ch, out_ch, stride=2))
    
    d = int(np.log2(sub_res)) - 2
    print(3, int(ndf / 2**(d - 2)))
    self.blocks.append(DResidualBlock(3, int(ndf / 2**(d - 2)), stride=2))

    self.out_layer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(ndf, 1)
    )
    
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(ndf, self.label_size)
    )

    # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

    # forward method
  def forward(self, x):
    # label = label.long()
    # embedded_label = self.embed(label).view(-1, 1, self.sub_res, self.sub_res)
    
    # x = torch.cat([x, embedded_label], dim=1)

    for block in reversed(self.blocks):
      x = block(x)
    validity_output = self.out_layer(x)
    class_output = self.classifier(x)

    return validity_output, class_output
      

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    #m.bias.data.zero_()
