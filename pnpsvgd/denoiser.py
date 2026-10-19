import torch
from math import sqrt
import numpy as np
import torch


def apply_DRUNET(model, img, mu, sigma, shape, edge_max=512, verb=False):
    """
    Applying DRUNet to current image

    model: DRUNet denoiser model
    img: image
    mu: noise level factor for DRUNet
    sigma: noise level factor for DRUNet
    shape: shape of image dimension
    edge_max: padding size for image before inputing into DRUNet (Must be in 2^n)
    verb: verbosity
    """
    # Shifting and Scaling
    if verb: print(f'img, min{img.min()}, max{img.max()}')

    batch_size, _ = img.size()
    img = img.unsqueeze(1)
    img_min = torch.min(img, -1, keepdim=True)[0]
    numerator = img - img_min
    numerator_max = torch.max(numerator, -1, keepdim=True)[0]
    scaled_img = (numerator / numerator_max).view(batch_size, 1, shape[0], shape[1])

    if verb: print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')

    # cropping only: dim 2, 3
    extra_left, extra_right = 0, edge_max - shape[1]
    extra_top, extra_bottom = 0, edge_max - shape[0]
    scaled_img = torch.nn.functional.pad(scaled_img,
                                         (extra_left, extra_right, extra_top, extra_bottom),
                                         mode='constant', value=0)

    # Create sigma map
    gamma = sqrt(sigma * mu)
    sigma_map = torch.tensor(gamma, dtype=torch.float).cuda().repeat(batch_size, 1, edge_max, edge_max)

    # concatenate at the channel
    scaled_img = torch.cat((scaled_img, sigma_map), dim=1)

    with torch.no_grad():
        denoise_img = model(scaled_img.cuda().half())
    denoise_img = denoise_img[..., :shape[0], :shape[1]].reshape(batch_size, 1, -1)
    denoise_img = denoise_img * numerator_max + img_min

    if verb: print(f'denoise_img, min{denoise_img.min()}, max{denoise_img.max()}')

    return denoise_img.squeeze().detach().cpu()


def apply_DRUNET_optimization(model, img, mu, sigma, edge_max=512, verb=False):
    """
    Applying DRUNet to current image for optimization procedure

    model: DRUNet denoiser model
    img: image
    mu: noise level factor for DRUNet
    sigma: noise level factor for DRUNet
    edge_max: padding size for image before inputing into DRUNet (Must be in 2^n)
    verb: verbosity
    """
    # Shifting and Scaling
    if verb: print(f'img, min{img.min()}, max{img.max()}')

    img_min = img.min()
    numerator = img - img_min
    scaled_img = numerator / numerator.max()

    if verb: print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')

    # cropping only: dim 2, 3
    extra_left, extra_right = 0, edge_max - scaled_img.shape[1]
    extra_top, extra_bottom = 0, edge_max - scaled_img.shape[0]
    scaled_img = np.pad(scaled_img,
                        ((extra_top, extra_bottom), (extra_left, extra_right)),
                        mode='constant', constant_values=0)

    # Create sigma map
    gamma = sqrt(sigma * mu)
    sigma_map = torch.tensor(gamma, dtype=torch.float).repeat(1, 1, scaled_img.shape[0], scaled_img.shape[1])

    # concatenate at the channel
    scaled_img = torch.cat((torch.from_numpy(scaled_img).unsqueeze(0).unsqueeze(0), sigma_map), dim=1)

    with torch.no_grad():
        denoise_img = model(scaled_img.cuda().half())
    denoise_img = denoise_img[0, 0, :img.shape[0], :img.shape[1]]
    denoise_img = denoise_img * (numerator.max()) + img_min

    if verb: print(f'denoise_img, min{denoise_img.min()}, max{denoise_img.max()}')

    return denoise_img.cpu().detach().numpy()
