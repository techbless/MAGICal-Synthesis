import torch
import torch.nn as nn
from model import Generator
from util import show_result


epoch = 137
root = './'
check_point_dir = root + 'check_points/'

ngf = 512
latent_size = 128
label_size = 16
sub_res = 64
G = Generator(ngf, latent_size, label_size, sub_res)

check_point = torch.load(check_point_dir+'check_point_epoch_%i.pth' % epoch)
G.load_state_dict(check_point['G_net'])
G.cuda()


for i in range(0, 2200):
  fixed_original_p = './output/' + 'eval_' + str(i) + '.png'
  show_result(G, show=False,  path=fixed_original_p, nb=1, latent_size=latent_size, label_size=label_size, sub_size=sub_res)