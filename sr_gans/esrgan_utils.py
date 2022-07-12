import numpy
import random
import torch
import torch.cuda
import torch.nn
import torch.nn.functional


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


def pixel_loss_fn(
        gen_imgs: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(gen_imgs, imgs)


def content_loss_fn(
        gen_features: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(gen_features, features)


def adversarial_loss_fn(
        gen_validities: torch.Tensor, validities: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        gen_validities, validities)
