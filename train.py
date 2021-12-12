import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from torchvision import transforms
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from helpers import linear_warmup

from model import VAE

torch.set_default_tensor_type("torch.cuda.FloatTensor")

class Hyperparams(dict):
    def __getattr__(self, attr):
        try: return self[attr]
        except KeyError: return None

    def __setattr__(self, attr, value):
        self[attr] = value

H = Hyperparams()
H.grad_clip = 200.0
H.skip_threshold = 400.0
H.width = 384
H.lr = 0.0002
H.zdim = 16
H.wd = 0.01
H.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21" # "layer string"
H.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3" # "layer string"
H.warmup_iters = 100
H.adam_beta1 = 0.9
H.adam_beta2 = 0.9
H.dataset = 'cifar10'
H.n_batch = 16
H.image_size = 32
H.image_channels = 3
H.bottleneck_multiple = 0.2
H.no_bias_above = 0.01
H.num_mixtures = 10
H.k = 0

vae = VAE(H)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=H.n_batch, shuffle=True)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

optimizer = AdamW(vae.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for epoch in range(256):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.permute([0,2,3,1]).cuda()
        labels = labels.cuda()

        nll, kl = vae(inputs, labels)
        loss = nll + kl
        vae.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
        distortion_nans = torch.isnan(nll).sum()
        rate_nans = torch.isnan(kl).sum()
        nans = dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1)

        # only update if no rank has a nan and if the grad norm is below a specific threshold
        if nans['distortion_nans'] == 0 and nans['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
            optimizer.step()
        scheduler.step()

        if i % 1000 == 1:
            print("Epoch:", epoch)
            print(f"Loss: {loss} (nll: {nll}, kl: {kl})")
            _, ax = plt.subplots(1,11,figsize=(10,12))
            repeated_input = inputs[0:1].repeat(10,1,1,1)
            different_labels = torch.LongTensor(torch.arange(10).cpu()).cuda()
            recs = vae.reconstruct(repeated_input, different_labels)
            ax[0].imshow((inputs[0] / 2 + 0.5).cpu().numpy())
            for l in range(10):
                ax[1+l].imshow(recs[l])
                ax[1+l].set_title(classes[l])
            plt.tight_layout()
            plt.show()
    print(f"Epoch {epoch} done")
    if epoch % 10 == 0:
        torch.save(vae.state_dict(), f"vdcvae-epoch{epoch}.pt")
print('Finished Training')

"""
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_samples = 5
_, ax = plt.subplots(num_samples, 10, figsize=(num_samples*5,15))
for i in range(10):
    img = torch.rand(num_samples,32,32,3)
    label = torch.LongTensor([i]).cuda()
    recs = vae.reconstruct(img, label, k=0)
    for j in range(num_samples):
        if j == 0: ax[j,i].set_title(classes[i])
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        ax[j,i].imshow(recs[j])
"""
