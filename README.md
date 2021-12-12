# Conditional Very Deep VAE

After reading [Rewon Child's paper on very deep variational autoencoders (VDVAE)](https://arxiv.org/abs/2011.10650), I felt inspired to see if I could make their model **conditional**. A conditional VAE is similar to a regular VAE but includes a label during the forward pass. This allows the model to pick up on the correlation between a label and its corresponding image features, which ultimately allows for more fine-grained control over the model's generated samples. For instance, once a conditional VAE has been trained on the CIFAR-10 dataset, you can specify that you want "horse" or "airplane" samples from the model's learned distribution.

The VDVAE uses a hierarchical "top-down" structure inspired by the work of Sønderby et al. on [ladder variational autoencoders](https://arxiv.org/abs/1602.02282). In short, this model first computes a "bottom-up" pass of the input image to generate features. Then, a "top-down" pass samples multiple latent variables at multiple resolutions conditioned on the features extracted in the "bottom-up" pass. For much more on this, I suggest you read [the paper](https://arxiv.org/abs/2011.10650), which also includes the following helpful diagram:

In order to make the model conditional, we augment the architecture slightly by adding an embedding layer at the top of the decoder (the right part of the diagram). This embedding layer takes a one-hot encoded vector (representing the image's class) as input, which is then filtered down throughout the decoder in the "top-down" pass. When the model has been trained, we sample values by inputting into the decoder a one-hot vector representing the desired class, after which an image resembling this class should (hopefully) be sampled from the learned distribution.
