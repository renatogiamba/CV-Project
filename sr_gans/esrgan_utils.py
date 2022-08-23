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


def pixel_loss_fn(gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(gen_imgs, imgs)


def content_loss_fn(
        gen_features: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(gen_features, features)


def adversarial_loss_fn(
        gen_validities: torch.Tensor, validities: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        gen_validities, validities)


def psnr_fn(gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torchmetrics.functional.peak_signal_noise_ratio(gen_imgs, imgs)


def ssim_fn(gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torchmetrics.functional.structural_similarity_index_measure(gen_imgs, imgs)


def normalize(imgs: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    return torchvision.transforms.functional.normalize(
        imgs, mean, std)


def esrgan_interpolation(
        PSNR_checkpoint_filename: str,
        GAN_checkpoint_filename: str,
        alpha: float, device: torch.device) -> None:
    PSNR_ckpt = torch.load(PSNR_checkpoint_filename, map_location=device)
    GAN_ckpt = torch.load(GAN_checkpoint_filename, map_location=device)
    interp_ckpt = torch.load(GAN_checkpoint_filename, map_location=device)

    for param_name in PSNR_ckpt["gen_model"]:
        interp_ckpt["gen_model"][param_name] = (1. - alpha) * \
            PSNR_ckpt["gen_model"][param_name] + alpha * \
            GAN_ckpt["gen_model"][param_name]

    torch.save(interp_ckpt, f"./models/ESRGAN_interp_{alpha:.2f}.ckpt")
