import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils


class DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        #out = self.lrelu(out)

        shortcut = self.shortcut(x)
        out += shortcut
        out = self.lrelu(out)

        return out


class GResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(GResidualBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        #out = self.relu(out)

        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)

        return out
      
      
class Generator(nn.Module):
  # initializers
  def __init__(self, ngf=512, latent_size=384, label_size=128):
    super(Generator, self).__init__()
    self.ngf = ngf
    self.latent_size = latent_size
    self.label_size = label_size
    
    self.embed = nn.Embedding(label_size, label_size)
    self.fc1 = nn.Linear(latent_size + label_size, 4*4*ngf*8)  # latent vector
    self.res_block2 = GResidualBlock(ngf*8, ngf*4, stride=2)
    self.res_block3 = GResidualBlock(ngf*4, ngf*4, stride=2)
    self.res_block4 = GResidualBlock(ngf*4, ngf*2, stride=2)
    self.res_block5 = GResidualBlock(ngf*2, ngf*2, stride=2)
    self.res_block6 = GResidualBlock(ngf*2, ngf, stride=2)
    self.res_block7 = GResidualBlock(ngf, ngf, stride=2)

    self.deconv8 = nn.Sequential(
      nn.ConvTranspose2d(ngf*2, 3, 4, stride=2, padding=1, bias=False),
      nn.Tanh()
    )

  # weight_init
  def weight_init(self, mean, std):
      for m in self._modules:
          normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, z, label):
      label = label.long()
      embedded_label = self.embed(label).view(-1, self.label_size)
      x = torch.cat((z, embedded_label), dim=1)

      x = self.fc1(x).view(-1, self.ngf*8, 4, 4)
      x = self.res_block2(x)
      x = self.res_block3(x)
      x = self.res_block4(x)
      x = self.res_block5(x)
      #x = self.res_block6(x)
      #x = self.res_block7(x)
      x = self.deconv8(x)

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

    self.res_block1 = DResidualBlock(3 + 1, ndf, stride=2) # 3(RGB) + 1(Condition)
    self.res_block2 = DResidualBlock(ndf, ndf, stride=2)
    self.res_block3 = DResidualBlock(ndf, ndf*2, stride=2)
    self.res_block4 = DResidualBlock(ndf*2, ndf*2, stride=2)
    self.res_block5 = DResidualBlock(ndf*2, ndf*4, stride=2)
    #self.res_block6 = DResidualBlock(ndf*4, ndf*4, stride=2)
    #self.res_block7 = DResidualBlock(ndf*4, ndf*8, stride=2)
    #self.res_block8 = DResidualBlock(ndf*8, ndf*4, stride=1)


    self.fc9 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(4 * 4 * ndf * 4, 1)
    )

    # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

    # forward method
  def forward(self, x, label):
    label = label.long()
    embedded_label = self.embed(label).view(-1, 1, self.sub_res, self.sub_res)
    

    x = torch.cat([x, embedded_label], dim=1)

    x = self.res_block1(x)
    x = self.res_block2(x)
    x = self.res_block3(x)
    x = self.res_block4(x)
    x = self.res_block5(x)
    #x = self.res_block6(x)
    #x = self.res_block7(x)
    #x = self.res_block8(x)
    x = self.fc9(x)

    return x
      

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    #m.bias.data.zero_()
