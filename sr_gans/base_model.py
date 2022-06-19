import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
from typing import *


def adversarial_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(predictions, targets)


def content_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(predictions, targets)


class BaseSRGAN(torch.nn.Module):
    """
    Pytorch subclass for handling the base functionalities of the SRGAN models. It is
    a sort of interface that the final models HAVE TO implement.
    """

    def __init__(
            self, gen: torch.nn.Module, disc: torch.nn.Module) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        gen (torch.nn.Module): The Pytorch generator model for this GAN.

        disc (torch.nn.Module): The Pytorch discriminator model for this GAN.
        """

        super(BaseSRGAN, self).__init__()

        # store the generator and the discriminator models
        self.gen = gen
        self.disc = disc

    def fit_generator_step(
            self, lr_imgs: torch.Tensor, hr_imgs: torch.Tensor,
            reals: torch.Tensor, fakes: torch.Tensor,
            gen_optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """
        Perform a step in the generator training process on a single batch.

        This function HAS TO be overriden by the subclasses that implement this
        interface.

        TODO: Update this docs and method.
        """

        raise NotImplementedError(
            "You HAVE TO implement the fit_generator_step() method.")

    def fit_discriminator_step(
            self, lr_imgs: torch.Tensor, hr_imgs: torch.Tensor,
            reals: torch.Tensor, fakes: torch.Tensor,
            gen_optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """
        Perform a step in the discriminator training process on a single batch.

        This function HAS TO be overriden by the subclasses that implement this
        interface.

        TODO: Update this docs and method.
        """

        raise NotImplementedError(
            "You HAVE TO implement the fit_discriminator_step() method.")

    # NOT COMPLETED YET!!
    def fit(
            self,
            train_dl: torch.utils.data.DataLoader,
            #valid_dl: torch.utils.data.DataLoader,
            num_epochs: int,
            gen_optimizer: torch.optim.Optimizer,
            disc_optimizer: torch.optim.Optimizer) -> None:
        for epoch in range(num_epochs):
            for batch in train_dl:
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]
                batch_size = lr_imgs.shape[0]
                out_shape = self.disc.out_shape

                reals = torch.ones(
                    (batch_size, *out_shape), dtype=torch.float)
                fakes = torch.zeros(
                    (batch_size, *out_shape), dtype=torch.float)

                gen_step_outs = self.fit_generator_step(
                    lr_imgs, hr_imgs, reals, fakes, gen_optimizer)
                disc_step_outs = self.fit_discriminator_step(
                    lr_imgs, hr_imgs, reals, fakes, disc_optimizer)

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        hr_imgs = self.gen(lr_imgs)
        return hr_imgs
