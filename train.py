import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from torchvision import transforms
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from helpers import linear_warmup
from hyperparams import get_default_hyperparams
from model import VAE

torch.set_default_tensor_type("torch.cuda.FloatTensor")

H = get_default_hyperparams()
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
        torch.save(vae.state_dict(), f"weights-epoch{epoch}.pt")
print('Finished Training')
