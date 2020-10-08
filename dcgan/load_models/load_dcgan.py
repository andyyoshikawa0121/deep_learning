# matplotlib inline
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import slackweb

cuda = torch.cuda.is_available()

ngpu = 1

if cuda:
    print("cuda is available")
else:
    print("cuda is not available")
    exit()
torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

cudnn.benchmark = True

nc = 3
nz = 100
ngf = 64
ndf = 64
npic = 5000
num_epoch = 40
gan_type = "dcgan"
func_type = "normal"
dataroot = (
    f"E:/GenerateImages/{gan_type}/{func_type}/cifar-10_epoch_{num_epoch}/images/"
)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) * 4 * 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) * 8 * 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) * 16 * 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) * 32 * 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)
model_path = Path("../models/dcgan_normal_cifar-10_40.pth")
netG.load_state_dict(torch.load(model_path))
fixed_noise = torch.randn(npic, nz, 1, 1, device=device)
img_list = []

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
img_list.append(utils.make_grid(fake, padding=2, normalize=True, nrow=64))

# plt.subplot(1, 1, 1)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
# plt.show()

for index, pic in enumerate(fake):
    utils.save_image(pic, f"{dataroot}generate_image_{index}.png", normalize=True)
