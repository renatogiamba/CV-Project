import numpy
import random
import torch
import torch.cuda
import torch.nn
import torch.nn.functional
import torchmetrics
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from typing import *


def set_seed(seed: int) -> None:
    """
    Set the randomized algorithms as reproducible.
    Parameters:
    ===========
    seed (int): An integer used to initialize the internal state of randomized 
        algorithms.
    """

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def content_loss_fn(
        gen_features: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(gen_features, features)



def adversarial_loss_fn(
        gen_validities: torch.Tensor, validities: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(gen_validities, validities)
    


def psnr_fn(gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torchmetrics.functional.peak_signal_noise_ratio(gen_imgs, imgs)


def ssim_fn(gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torchmetrics.functional.structural_similarity_index_measure(gen_imgs, imgs)


def normalize(imgs: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    return torchvision.transforms.functional.normalize(
        imgs, mean, std)
