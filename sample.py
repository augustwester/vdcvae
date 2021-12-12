import torch
import matplotlib.pyplot as plt
from model import VAE
from hyperparams import get_default_hyperparams

H = get_default_hyperparams()
vae = VAE(H)
vae.load_state_dict(torch.load("weights.pt"))

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_samples = 5
_, ax = plt.subplots(num_samples, 10, figsize=(num_samples*5,15))
for i in range(10):
    img = torch.rand(num_samples,32,32,3)
    label = torch.LongTensor([i]).cuda()
    recs = vae.reconstruct(img, label)
    for j in range(num_samples):
        if j == 0: ax[j,i].set_title(classes[i])
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        ax[j,i].imshow(recs[j])
plt.show()
