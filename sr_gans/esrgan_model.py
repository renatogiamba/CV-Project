import collections
import torch
import torch.jit
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import torchmetrics
import torchmetrics.functional
import torchvision
import torchvision.models
from typing import *

from .base_model import (
    BaseSRGAN
)
from .esrgan_utils import (
    adversarial_loss_fn, content_loss_fn, pixel_loss_fn
)


class Conv2dK3P1Act(torch.nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, stride: int,
            negative_slope: float = 0.01) -> None:
        super(Conv2dK3P1Act, self).__init__()
        self.body = torch.nn.Sequential(collections.OrderedDict([
            ("conv", torch.nn.Conv2d(in_channels,
             out_channels, 3, stride=stride, padding=1)),
            ("act", torch.nn.LeakyReLU(negative_slope=negative_slope))
        ]))

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        new_img_features = self.body(img_features)
        return new_img_features


class Conv2dK3P1NormAct(torch.nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, stride: int,
            negative_slope: float = 0.01) -> None:
        super(Conv2dK3P1NormAct, self).__init__()
        self.body = torch.nn.Sequential(collections.OrderedDict([
            ("conv", torch.nn.Conv2d(in_channels,
             out_channels, 3, stride=stride, padding=1)),
            ("norm", torch.nn.BatchNorm2d(out_channels)),
            ("act", torch.nn.LeakyReLU(negative_slope=negative_slope))
        ]))

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        new_img_features = self.body(img_features)
        return new_img_features


class Residual(torch.nn.Module):
    def __init__(self, in_channels: int, scale: float) -> None:
        super(Residual, self).__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.conv1 = Conv2dK3P1Act(in_channels, in_channels, 1)
        self.conv2 = Conv2dK3P1Act(2 * in_channels, in_channels, 1)
        self.conv3 = Conv2dK3P1Act(3 * in_channels, in_channels, 1)
        self.conv4 = Conv2dK3P1Act(4 * in_channels, in_channels, 1)
        self.conv_final = torch.nn.Conv2d(
            5 * in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        new_img_features1 = self.conv1(img_features)
        new_img_features1 = torch.cat([
            img_features, new_img_features1], dim=1)
        new_img_features2 = self.conv2(new_img_features1)
        new_img_features2 = torch.cat([
            new_img_features1, new_img_features2], dim=1)
        new_img_features3 = self.conv3(new_img_features2)
        new_img_features3 = torch.cat([
            new_img_features2, new_img_features3], dim=1)
        new_img_features4 = self.conv4(new_img_features3)
        new_img_features4 = torch.cat([
            new_img_features3, new_img_features4], dim=1)
        new_img_features_final = self.conv_final(new_img_features4)
        new_img_features = new_img_features_final * self.scale + img_features
        return new_img_features


class ResidualInResidual(torch.nn.Module):
    def __init__(self, in_channels: int, scale: float) -> None:
        super(ResidualInResidual, self).__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.body = torch.nn.Sequential(collections.OrderedDict([
            (f"res{i + 1}", Residual(in_channels, scale))
            for i in range(3)
        ]))

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        new_img_features_final = self.body(img_features)
        new_img_features = new_img_features_final * self.scale + img_features
        return new_img_features


class Upscale(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(Upscale, self).__init__()
        self.body = torch.nn.Sequential(collections.OrderedDict([
            ("conv", torch.nn.Conv2d(in_channels,
             4 * in_channels, 3, stride=1, padding=1)),
            ("act", torch.nn.LeakyReLU()),
            ("pxshf", torch.nn.PixelShuffle(upscale_factor=2))
        ]))

    def forward(self, img_features: torch.Tensor) -> None:
        new_img_features = self.body(img_features)
        return new_img_features


class Generator(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, res_scale: float) -> None:
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1)
        self.res_body = torch.nn.Sequential(collections.OrderedDict([
            (f"resres{i + 1}", ResidualInResidual(out_channels, res_scale))
            for i in range(16)
        ]))
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1)
        self.up_body = torch.nn.Sequential(collections.OrderedDict([
            (f"up{i + 1}", Upscale(out_channels))
            for i in range(2)
        ]))
        self.recon_body = torch.nn.Sequential(collections.OrderedDict([
            ("conv1", torch.nn.Conv2d(out_channels,
             out_channels, 3, stride=1, padding=1)),
            ("act", torch.nn.LeakyReLU()),
            ("conv_final", torch.nn.Conv2d(out_channels,
             in_channels, 3, stride=1, padding=1))
        ]))

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        features1 = self.conv1(lr_imgs)
        res_features = self.res_body(features1)
        features2 = self.conv2(res_features)
        features = features2 + features1
        upscaled_features = self.up_body(features)
        hr_imgs = self.recon_body(upscaled_features)
        return hr_imgs


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels: int, img_size: int) -> None:
        super(Discriminator, self).__init__()
        out_size = int(img_size // 16)

        self.feature_body = torch.nn.Sequential(collections.OrderedDict([
            ("conv1", Conv2dK3P1Act(in_channels, 64, 1, negative_slope=0.2)),
            ("conv2", Conv2dK3P1NormAct(64, 64, 2, negative_slope=0.2)),
            ("conv3", Conv2dK3P1NormAct(64, 128, 1, negative_slope=0.2)),
            ("conv4", Conv2dK3P1NormAct(128, 128, 2, negative_slope=0.2)),
            ("conv5", Conv2dK3P1NormAct(128, 256, 1, negative_slope=0.2)),
            ("conv6", Conv2dK3P1NormAct(256, 256, 2, negative_slope=0.2)),
            ("conv7", Conv2dK3P1NormAct(256, 512, 1, negative_slope=0.2)),
            ("conv8", Conv2dK3P1NormAct(512, 512, 2, negative_slope=0.2))
        ]))
        self.classifier_body = torch.nn.Sequential(collections.OrderedDict([
            ("lin1", torch.nn.Linear(512 * out_size * out_size, 1_024)),
            ("act", torch.nn.LeakyReLU()),
            ("lin_class", torch.nn.Linear(1_024, 1))
        ]))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch_size = imgs.shape[0]

        features = self.feature_body(imgs)
        validities = self.classifier_body(features.reshape(batch_size, -1))
        return validities


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=False)
        vgg_model.load_state_dict(torch.load("./models/vgg19-dcbb9e9d.pth"))

        self.vgg_feature_body = torch.nn.Sequential(
            *list(vgg_model.features.children())[:35]
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        features = self.vgg_feature_body(imgs)
        return features


class ESRGAN(BaseSRGAN):
    def __init__(
            self, device: torch.device, in_channels: int, out_channels: int,
            img_size: int, res_scale: float,
            gen_optimizer_init: Callable,
            gen_optimizer_args: Dict[str, Any],
            disc_optimizer_init: Callable,
            disc_optimizer_args: Dict[str, Any],
            patience_metrics: Dict[str, float]) -> None:
        super(ESRGAN, self).__init__(
            device,
            Generator(in_channels, out_channels, res_scale),
            Discriminator(in_channels, img_size),
            torch.jit.load(
                "./models/esrgan_feature_extractor_jit.pt", map_location=device),
            gen_optimizer_init, gen_optimizer_args,
            disc_optimizer_init, disc_optimizer_args,
            patience_metrics)

    def train_generator_start(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_imgs = inputs["lr_imgs"]
        hr_imgs = inputs["hr_imgs"]

        gen_hr_imgs = self.gen_model(lr_imgs)
        with torch.no_grad():
            features = self.feature_model(hr_imgs)
        gen_features = self.feature_model(gen_hr_imgs)
        with torch.no_grad():
            pred_reals = self.disc_model(hr_imgs)
        pred_fakes = self.disc_model(gen_hr_imgs)
        return {
            **inputs,
            "gen_hr_imgs": gen_hr_imgs,
            "features": features,
            "gen_features": gen_features,
            "pred_reals": pred_reals,
            "pred_fakes": pred_fakes
        }

    def train_discriminator_start(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hr_imgs = inputs["hr_imgs"]
        gen_hr_imgs = inputs["gen_hr_imgs"]

        pred_reals = self.disc_model(hr_imgs)
        pred_fakes = self.disc_model(gen_hr_imgs)
        return {
            **inputs,
            "pred_reals": pred_reals,
            "pred_fakes": pred_fakes
        }

    @torch.no_grad()
    def validation_start(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_imgs = inputs["lr_imgs"]
        hr_imgs = inputs["hr_imgs"]

        gen_hr_imgs = self.gen_model(lr_imgs)
        return {
            **inputs,
            "gen_hr_imgs": gen_hr_imgs
        }

    def train_generator_step(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hr_imgs = inputs["hr_imgs"]
        gen_hr_imgs = inputs["gen_hr_imgs"]
        features = inputs["features"]
        gen_features = inputs["gen_features"]
        reals = inputs["reals"]
        pred_fakes = inputs["pred_fakes"]
        pred_reals = inputs["pred_reals"]

        pixel_loss = pixel_loss_fn(gen_hr_imgs, hr_imgs)
        content_loss = content_loss_fn(gen_features, features)
        adversarial_loss = adversarial_loss_fn(
            pred_fakes - pred_reals.mean(dim=0, keepdim=True), reals)
        loss = (pixel_loss + content_loss + adversarial_loss) / 3.
        if loss.item() > 1000.0:
            loss = 1.e-2 * pixel_loss + content_loss + 5.e-2 * adversarial_loss
        return {
            "loss": loss,
            "pixel_loss": pixel_loss,
            "content_loss": content_loss,
            "adversarial_loss": adversarial_loss
        }

    def train_discriminator_step(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reals = inputs["reals"]
        fakes = inputs["fakes"]
        pred_fakes = inputs["pred_fakes"]
        pred_reals = inputs["pred_reals"]

        adversarial_real_loss = adversarial_loss_fn(
            pred_reals - pred_fakes.mean(dim=0, keepdim=True), reals)
        adversarial_fake_loss = adversarial_loss_fn(
            pred_fakes - pred_reals.mean(dim=0, keepdim=True), fakes)
        loss = (adversarial_real_loss + adversarial_fake_loss) / 2.
        return {
            "loss": loss,
            "adversarial_real_loss": adversarial_real_loss,
            "adversarial_fake_loss": adversarial_fake_loss
        }

    @torch.no_grad()
    def validation_step(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hr_imgs = inputs["hr_imgs"]
        gen_hr_imgs = inputs["gen_hr_imgs"]

        psnr = torchmetrics.functional.peak_signal_noise_ratio(
            torch.sigmoid(gen_hr_imgs), hr_imgs)
        ssim = torchmetrics.functional.structural_similarity_index_measure(
            torch.sigmoid(gen_hr_imgs), hr_imgs)
        return {
            "psnr": psnr,
            "ssim": ssim
        }
    
    def patience_step(self, metrics: Dict[str, float]) -> int:
        score = 0

        if metrics["psnr"] > self.last_patience_metrics["psnr"]:
            score += 1
            self.last_patience_metrics["psnr"] = metrics["psnr"]
        elif metrics["psnr"] < self.last_patience_metrics["psnr"]:
            score -= 1
        
        if metrics["ssim"] > self.last_patience_metrics["ssim"]:
            score += 1
            self.last_patience_metrics["ssim"] = metrics["ssim"]
        if metrics["ssim"] < self.last_patience_metrics["ssim"]:
            score -= 1
        
        return score
    
    def warmup(
            self,
            train_dl: torch.utils.data.DataLoader,
            num_epochs: int, loss_type: str, lr: Optional[float] = None,
            checkpoint_filename: Optional[str] = None) -> None:
        """
        Perform the training procedure based only on a single loss from the generator.

        Parameters:
        ===========
        train_dl (torch.utils.data.DataLoader): The Pytorch data loader for the 
            training dataset.
        
        num_epochs (int): Maximum number of iterations on the training dataset.

        loss_type (str): Name of the generator loss to train on.
        
        lr (Optional[float]): A new learning rate for the optimizers. Set to 'None' 
            if you don't want to change it.

        checkpoint_filename (Optional[str]): The path to a checkpoint file to load. Set 
            to 'None' if you want to mantain the current state.
        """

        # if requested, load a checkpoint
        if checkpoint_filename is not None:
            print("[Loading the best checkpoint...]")
            self.load_checkpoint(checkpoint_filename)
            if lr is not None:
                for param_group in self.gen_optimizer.param_groups:
                    param_group["lr"] = lr
                for param_group in self.disc_optimizer.param_groups:
                    param_group["lr"] = lr
            print("[Best checkpoint loaded.]\n")

        # prepare auxiliary variables
        num_train_batches = len(train_dl)

        for epoch in range(num_epochs):

            # TRAINING PROCEDURE

            # set models in training mode
            self.gen_model.train()
            self.disc_model.train()

            # iterate over the training dataset
            for i, batch in enumerate(train_dl):
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]
                reals = torch.ones(
                    (lr_imgs.shape[0], 1), dtype=torch.float, device=self.device)
                fakes = torch.zeros(
                    (lr_imgs.shape[0], 1), dtype=torch.float, device=self.device)

                # reset the optimizer of the generator model
                self.gen_optimizer.zero_grad()

                # compute the needed inputs for the generator model
                gen_inputs = self.train_generator_start({
                    "lr_imgs": lr_imgs,
                    "hr_imgs": hr_imgs,
                    "reals": reals,
                    "fakes": fakes
                })

                # compute the losses for the generator model
                gen_losses = self.train_generator_step(gen_inputs)

                # perform the backpropagation pass of the generator model
                gen_loss = gen_losses[loss_type]
                gen_loss.backward()

                # update the weigths of the generator model
                self.gen_optimizer.step()

                # reset the optimizer of the discriminator model
                self.disc_optimizer.zero_grad()

                # compute the needed inputs for the discriminator model
                disc_inputs = self.train_discriminator_start({
                    "lr_imgs": lr_imgs,
                    "hr_imgs": hr_imgs,
                    "reals": reals,
                    "fakes": fakes,
                    "gen_hr_imgs": gen_inputs["gen_hr_imgs"].detach()
                })

                # compute the losses for the discriminator model
                disc_losses = self.train_discriminator_step(disc_inputs)

                # perform the backpropagation pass of the discriminator model
                disc_loss = disc_losses["loss"]
                disc_loss.backward()

                # update the weigths of the discriminator model
                self.disc_optimizer.step()

                # log results
                print(
                    f"Train Warmup on {loss_type} [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_train_batches:3d}]")
                print(
                    f"Generator [{loss_type} {gen_losses[loss_type].item():.2f}]")
                print("Discriminator ", end="")
                for loss_name, loss in disc_losses.items():
                    print(f"[{loss_name} {loss.item():.2f}] ", end="")
                print("\n")

            print("[Saving the checkpoint...]")
            self.save_checkpoint(
                checkpoint_filename if checkpoint_filename is not None
                else "esrgan.ckpt")
            print("[Checkpoint saved.]\n")
