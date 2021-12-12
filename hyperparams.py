class Hyperparams(dict):
    def __getattr__(self, attr):
        try: return self[attr]
        except KeyError: return None

    def __setattr__(self, attr, value):
        self[attr] = value

def get_default_hyperparams():
    H = Hyperparams()
    H.grad_clip = 200.0
    H.skip_threshold = 400.0
    H.width = 384
    H.lr = 0.0002
    H.zdim = 16
    H.wd = 0.01
    H.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
    H.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
    H.warmup_iters = 100
    H.adam_beta1 = 0.9
    H.adam_beta2 = 0.9
    H.dataset = "cifar10"
    H.n_batch = 16
    H.image_size = 32
    H.image_channels = 3
    H.bottleneck_multiple = 0.2
    H.no_bias_above = 0.01
    H.num_mixtures = 10
    H.k = 0

    return H
