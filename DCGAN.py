import warnings
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
'c065c0e2593b8b161a2d7873e42418bf6a21106c')
data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)

batch_size = 256
transformer = torchvision.transforms.Compose([
torchvision.transforms.Resize((64, 64)),torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
pokemon, batch_size=batch_size,
shuffle=True, num_workers=d2l.get_dataloader_workers())

warnings.filterwarnings('ignore')

class G_block(nn.Module):
  def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
  padding=1, **kwargs):

    super(G_block, self).__init__(**kwargs)
    self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
    kernel_size, strides, padding, bias=False)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.activation = nn.ReLU()
  def forward(self, X):
    return self.activation(self.batch_norm(self.conv2d_trans(X)))

x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)

n_G = 64
net_G = nn.Sequential(G_block(in_channels=100, out_channels=n_G*8,
strides=1, padding=0),
G_block(in_channels=n_G*8, out_channels=n_G*4), 
G_block(in_channels=n_G*4, out_channels=n_G*2), 
G_block(in_channels=n_G*2, out_channels=n_G), 
nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
kernel_size=4, stride=2, padding=1, bias=False),
nn.Tanh()) 

x = torch.zeros((1, 100, 1, 1))

class D_block(nn.Module):
  def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
  padding=1, alpha=0.2, **kwargs):
    super(D_block, self).__init__(**kwargs)
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
    strides, padding, bias=False)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.activation = nn.LeakyReLU(alpha, inplace=True)
  def forward(self, X):
    return self.activation(self.batch_norm(self.conv2d(X)))

x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)

n_D = 64
net_D = nn.Sequential(
  D_block(n_D), # Output: (64, 32, 32)
  D_block(in_channels=n_D, out_channels=n_D*2), # Output: (64 * 2, 16, 16)
  D_block(in_channels=n_D*2, out_channels=n_D*4), # Output: (64 * 4, 8, 8)
  D_block(in_channels=n_D*4, out_channels=n_D*8), # Output: (64 * 8, 4, 4)
  nn.Conv2d(in_channels=n_D*8, out_channels=1,
  kernel_size=4, bias=False)) # Output: (1, 1, 1)

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
device=d2l.try_gpu()):
  loss = nn.BCEWithLogitsLoss(reduction='sum')
  for w in net_D.parameters():
    nn.init.normal_(w, 0, 0.02)
  for w in net_G.parameters():
    nn.init.normal_(w, 0, 0.02)
  net_D, net_G = net_D.to(device), net_G.to(device)
  trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
  trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
  trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
  animator = d2l.Animator(xlabel='epoch', ylabel='loss',
      xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
      legend=['discriminator', 'generator'])
  animator.fig.subplots_adjust(hspace=0.3)
  for epoch in range(1, num_epochs + 1):
    timer = d2l.Timer()
    metric = d2l.Accumulator(3) 
    for X, _ in data_iter:
        batch_size = X.shape[0]
        Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
        X, Z = X.to(device), Z.to(device)
        metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
            d2l.update_G(Z, net_D, net_G, loss, trainer_G),
            batch_size)
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
         [torch.cat([
            fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
                  for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
    f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')

latent_dim, lr, num_epochs = 100, 0.005, 20

train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
