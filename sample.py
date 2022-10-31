import torch
import matplotlib.pyplot as plt
from model import VAE
from hyperparams import get_default_hyperparams

H = get_default_hyperparams()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(H).to(device)
vae.load_state_dict(torch.load("weights.pt", map_location=device))

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_samples = 5
_, ax = plt.subplots(num_samples, 10, figsize=(num_samples*5,15))
for i in range(10):
    img = torch.rand(num_samples,32,32,3).to(device)
    label = torch.LongTensor([i]).to(device)
    recs = vae.reconstruct(img, label, k=0)
    for j in range(num_samples):
        if j == 0: ax[j,i].set_title(classes[i])
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        ax[j,i].imshow(recs[j])
plt.show()
