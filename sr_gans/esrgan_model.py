import collections
import torch
import torch.jit
import torch.nn
import torch.nn.init
import torch.optim
import torch.utils
import torch.utils.data
import torchmetrics
import torchmetrics.functional
import torchvision
import torchvision.models
from typing import *

from .esrgan_utils import (
    adversarial_loss_fn,
    content_loss_fn,
    normalize,
    pixel_loss_fn,
    psnr_fn,
    ssim_fn
)


class Residual(torch.nn.Module):
    """
    Pytorch subclass for hadling the base residual block of the 
    ESRGAN generator.
    """

    def __init__(self, in_channels: int, grow_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the feature maps.
        grow_channels (int): The number of channels to progressively add to the 
            feature maps after each convolution.
        """

        super(Residual, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels, grow_channels, 3, stride=1, padding=1)
        self.act1 = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = torch.nn.Conv2d(
            in_channels + grow_channels, grow_channels, 3, stride=1, padding=1)
        self.act2 = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = torch.nn.Conv2d(
            in_channels + 2 * grow_channels, grow_channels, 3, stride=1, padding=1)
        self.act3 = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = torch.nn.Conv2d(
            in_channels + 3 * grow_channels, grow_channels, 3, stride=1, padding=1)
        self.act4 = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv5 = torch.nn.Conv2d(
            in_channels + 4 * grow_channels, in_channels, 3, stride=1, padding=1)

        # initialize the weights for faster convergence
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1.weight.data *= 0.1
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2.weight.data *= 0.1
        torch.nn.init.zeros_(self.conv2.bias)

        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.conv3.weight.data *= 0.1
        torch.nn.init.zeros_(self.conv3.bias)

        torch.nn.init.kaiming_normal_(self.conv4.weight)
        self.conv4.weight.data *= 0.1
        torch.nn.init.zeros_(self.conv4.bias)

        torch.nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5.weight.data *= 0.1
        torch.nn.init.zeros_(self.conv5.bias)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        img_features (torch.Tensor): The input batch of feature maps.

        Returns:
        ========
        (torch.Tensor): The output batch of feature maps.
        """

        img_features1 = self.conv1(img_features)
        img_features1 = self.act1(img_features1)
        img_features2 = self.conv2(
            torch.cat([
                img_features, img_features1
            ], dim=1))
        img_features2 = self.act2(img_features2)
        img_features3 = self.conv3(
            torch.cat([
                img_features, img_features1, img_features2
            ], dim=1))
        img_features3 = self.act3(img_features3)
        img_features4 = self.conv4(
            torch.cat([
                img_features, img_features1, img_features2, img_features3
            ], dim=1))
        img_features4 = self.act4(img_features4)
        img_features5 = self.conv5(
            torch.cat([
                img_features, img_features1, img_features2,
                img_features3, img_features4
            ], dim=1))
        new_img_features = img_features5 * 0.2 + img_features

        return new_img_features


class ResidualInResidual(torch.nn.Module):
    """
    Pytorch subclass for hadling the intermediate residual in residual block of the 
    ESRGAN generator.
    """

    def __init__(self, in_channels: int, grow_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the feature maps.
        grow_channels (int): The number of channels to progressively add to the 
            feature maps after each convolution.
        """

        super(ResidualInResidual, self).__init__()

        self.res1 = Residual(in_channels, grow_channels)
        self.res2 = Residual(in_channels, grow_channels)
        self.res3 = Residual(in_channels, grow_channels)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        img_features (torch.Tensor): The input batch of feature maps.

        Returns:
        ========
        (torch.Tensor): The output batch of feature maps.
        """

        res_img_features = self.res1(img_features)

        # add some noise to the features to benefit from stochastic variation
        if self.training:
          noise1 = torch.randn_like(res_img_features)
          res_img_features = res_img_features + 0.1 * noise1

        res_img_features = self.res2(res_img_features)

        # add some noise to the features to benefit from stochastic variation
        if self.training:
          noise2 = torch.randn_like(res_img_features)
          res_img_features = res_img_features + 0.1 * noise2

        res_img_features = self.res3(res_img_features)

        # add some noise to the features to benefit from stochastic variation
        if self.training:
          noise3 = torch.randn_like(res_img_features)
          res_img_features = res_img_features + 0.1 * noise3

        new_img_features = res_img_features * 0.2 + img_features

        return new_img_features


class Upscale(torch.nn.Module):
    """
    Pytorch subclass for hadling the upscale block of the ESRGAN generator.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the feature maps.
        """

        super(Upscale, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, 3, stride=1, padding=1)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        img_features (torch.Tensor): The input batch of feature maps.

        Returns:
        ========
        (torch.Tensor): The output batch of upscaled feature maps.
        """

        new_img_features = self.conv(
            torch.nn.functional.interpolate(
                img_features, scale_factor=2., mode="nearest"))
        new_img_features = self.act(new_img_features)
        return new_img_features


class FirstConv2dActNormStage(torch.nn.Module):
    """
    Pytorch subclass for hadling the first convolutional stage of the 
    ESRGAN discriminator.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the feature maps.
        out_channels (int): The number of output channels of the feature maps.
        """

        super(FirstConv2dActNormStage, self).__init__()

        self.body = torch.nn.Sequential(collections.OrderedDict([
            ("conv1", torch.nn.Conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=True)),
            ("act1", torch.nn.LeakyReLU(negative_slope=0.2)),
            ("conv2", torch.nn.Conv2d(
                out_channels, out_channels, 4, stride=2, padding=1, bias=False)),
            ("norm2", torch.nn.BatchNorm2d(out_channels)),
            ("act2", torch.nn.LeakyReLU(negative_slope=0.2))
        ]))

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        img_features (torch.Tensor): The input batch of feature maps.

        Returns:
        ========
        (torch.Tensor): The output batch of feature maps.
        """

        new_img_features = self.body(img_features)

        return new_img_features


class Conv2dActNormStage(torch.nn.Module):
    """
    Pytorch subclass for hadling the convolutional stage of the
    ESRGAN discriminator.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the feature maps.
        out_channels (int): The number of output channels of the feature maps.
        """

        super(Conv2dActNormStage, self).__init__()

        self.body = torch.nn.Sequential(collections.OrderedDict([
            ("conv1", torch.nn.Conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=False)),
            ("norm1", torch.nn.BatchNorm2d(out_channels)),
            ("act1", torch.nn.LeakyReLU(negative_slope=0.2)),
            ("conv2", torch.nn.Conv2d(
                out_channels, out_channels, 4, stride=2, padding=1, bias=False)),
            ("norm2", torch.nn.BatchNorm2d(out_channels)),
            ("act2", torch.nn.LeakyReLU(negative_slope=0.2))
        ]))

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        img_features (torch.Tensor): The input batch of feature maps.

        Returns:
        ========
        (torch.Tensor): The output batch of feature maps.
        """

        new_img_features = self.body(img_features)

        return new_img_features


class Generator(torch.nn.Module):
    """
    Pytorch subclass for hadling the the ESRGAN generator network.
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the LR images.
        out_channels (int): The number of output channels of the HR images.
        num_blocks (int): The number of residual in residual blocks.
        """

        super(Generator, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.res_body = torch.nn.Sequential(collections.OrderedDict([
            (f"resres{i + 1}", ResidualInResidual(64, 32))
            for i in range(num_blocks)
        ]))
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.up_body = torch.nn.Sequential(collections.OrderedDict([
            (f"up1", Upscale(64)),
            (f"up2", Upscale(64))
        ]))
        self.reconstr_body = torch.nn.Sequential(collections.OrderedDict([
            ("conv1", torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            ("act", torch.nn.LeakyReLU(negative_slope=0.2)),
            ("conv2", torch.nn.Conv2d(64, out_channels, 3, stride=1, padding=1))
        ]))

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        lr_imgs (torch.Tensor): The input batch of LR images.

        Returns:
        ========
        (torch.Tensor): The output batch of HR images.
        """

        features1 = self.conv1(lr_imgs)
        res_features = self.res_body(features1)
        features2 = self.conv2(res_features)
        features = features2 + features1
        upscaled_features = self.up_body(features)
        hr_imgs = self.reconstr_body(upscaled_features)

        return hr_imgs


class Discriminator(torch.nn.Module):
    """
    Pytorch subclass for hadling the the ESRGAN discriminator network.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        in_channels (int): The number of input channels of the HR images.
        """

        super(Discriminator, self).__init__()

        self.feature_body = torch.nn.Sequential(collections.OrderedDict([
            ("conv_stage1", FirstConv2dActNormStage(in_channels, 64)),
            ("conv_stage2", Conv2dActNormStage(64, 128)),
            ("conv_stage3", Conv2dActNormStage(128, 256)),
            ("conv_stage4", Conv2dActNormStage(256, 512)),
            ("conv_stage5", Conv2dActNormStage(512, 512)),
        ]))
        self.classifier_body = torch.nn.Sequential(collections.OrderedDict([
            ("lin1", torch.nn.Linear(512 * 4 * 4, 100)),
            ("act", torch.nn.LeakyReLU(negative_slope=0.2)),
            ("lin2", torch.nn.Linear(100, 1))
        ]))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        imgs (torch.Tensor): The input batch of images.

        Returns:
        ========
        (torch.Tensor): The output batch of validities.
        """

        batch_size = imgs.shape[0]

        features = self.feature_body(imgs)
        features = features.view(batch_size, -1)
        validities = self.classifier_body(features)
        
        return validities


class FeatureExtractor(torch.nn.Module):
    """
    Pytorch subclass for hadling the the ESRGAN feature extractor.
    """

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        super(FeatureExtractor, self).__init__()

        # load a pretrained VGG19 network on ImageNet
        vgg_model = torchvision.models.vgg19()
        vgg_model.load_state_dict(torch.load("./models/vgg19-dcbb9e9d.pth"))

        # extract the desidered layer/layers
        self.vgg_feature_body = torch.nn.Sequential(
            *list(vgg_model.features.children())[:35]
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        imgs (torch.Tensor): The input batch of images.

        Returns:
        ========
        (torch.Tensor): The output batch of feature maps.
        """

        features = self.vgg_feature_body(imgs)

        return features


class ESRGAN(torch.nn.Module):
    """
    Pytorch subclass for hadling the the ESRGAN network.
    """

    def __init__(
            self, device: torch.device, in_channels: int, out_channels: int,
            num_blocks: int) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        device (torch.device): The Pytorch device to perform the computations on.
        in_channels (int): The number of input channels of the LR and HR images.
        out_channels (int): The number of output channels of the HR images.
        num_blocks (int): The number of residual in residual blocks for the generator.
        """

        super(ESRGAN, self).__init__()

        # store the device
        self.device = device

        # build the generator and the discriminator networks
        self.gen_model = Generator(in_channels, out_channels, num_blocks)
        self.disc_model = Discriminator(in_channels)

        # load an optimezed version of the feature extractor network
        self.feature_model = torch.jit.load(
            "./models/esrgan_feature_extractor_jit.pt", map_location=device)

        # initially set the optimizers to None.
        self.gen_optimizer = None
        self.disc_optimizer = None

    def _adjust_optimizer(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        """
        Reset the right parameters in an optimizer that has been swapped between
        CPU and GPU or that has been loaded from a checkpoint.

        Parameters:
        ===========
        optimizer (torch.optim.Optimizer): The optimizer to reset.
        lr (float): A new learning rate for the optimizer.
        """

        # flag needed if we are working on a GPU
        capturable = self.device == torch.device("cuda")

        # set parameters
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            param_group["capturable"] = capturable

    def _configure_warmup_optimizer(
            self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer for the pretraining phase.

        Parameters:
        ===========
        model (torch.nn.Module): The model whose weights will be optimized.

        Returns:
        ========
        (torch.optim.Optimizer): The desired optimizer.
        """

        return torch.optim.Adam(
            model.parameters(), lr=2.e-4, betas=(0.9, 0.999),
            capturable=True)

    def _configure_fit_optimizer(
            self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer for the training phase.

        Parameters:
        ===========
        model (torch.nn.Module): The model whose weights will be optimized.

        Returns:
        ========
        (torch.optim.Optimizer): The desired optimizer.
        """

        return torch.optim.Adam(
            model.parameters(), lr=1.e-4, betas=(0.9, 0.999),
            capturable=True)

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters:
        ===========
        lr_imgs (torch.Tensor): The input batch of LR images.

        Returns:
        ========
        (torch.Tensor): The output batch of HR images.
        """

        hr_imgs = self.gen_model(lr_imgs)
        return hr_imgs

    def save_PSNR_checkpoint(
            self, epoch: int, psnr: float, ssim: float) -> None:
        """
        Save a checkpoint of the network for the pretraining phase.

        Parameters:
        ===========
        epoch (int): The epoch the model stopped the pretraining in.
        psnr (float): The current psnr in the epoch.
        ssim (float): The current ssim in the epoch.
        """

        # prepare the checkpoint
        ckpt = {
            "gen_model": self.gen_model.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict()
        }

        # save the checkpoint
        torch.save(
            ckpt, f"./models/ESRGAN_PSNR-epoch:{epoch}-"
            f"psnr:{psnr:.2f}-ssim:{ssim:.2f}.ckpt")

    def save_GAN_checkpoint(
            self, epoch: int, psnr: float, ssim: float) -> None:
        """
        Save a checkpoint of the network for the training phase.

        Parameters:
        ===========
        epoch (int): The epoch the model stopped the training in.
        psnr (float): The current psnr in the epoch.
        ssim (float): The current ssim in the epoch.
        """

        # prepare the checkpoint
        ckpt = {
            "gen_model": self.gen_model.state_dict(),
            "disc_model": self.disc_model.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict()
        }

        # save the checkpoint
        torch.save(
            ckpt, f"./models/ESRGAN_GAN-epoch:{epoch}-"
            f"psnr:{psnr:.2f}-ssim:{ssim:.2f}.ckpt")

    def load_PSNR_checkpoint(
            self, epoch: int, psnr: float, ssim: float) -> None:
        """
        Load a checkpoint of the network for the pretraining phase.

        Parameters:
        ===========
        epoch (int): The epoch the model will resume the pretraining at.
        psnr (float): The current psnr in the epoch.
        ssim (float): The current ssim in the epoch.
        """

        # prepare the checkpoint
        ckpt = torch.load(
            f"./models/ESRGAN_PSNR-epoch:{epoch}-"
            f"psnr:{psnr:.2f}-ssim:{ssim:.2f}.ckpt", map_location=self.device)
        
        # load the checkpoint
        self.gen_model.load_state_dict(ckpt["gen_model"])
        self.gen_optimizer.load_state_dict(ckpt["gen_optimizer"])

    def load_PSNR_checkpoint_for_GAN(self, checkpoint_filename: str) -> None:
        """
        Load a checkpoint of the network for the pretraining phase and convert it
        for the training phase.

        Parameters:
        ===========
        checkpoint_filename (str): Name of the file where the checkpoint is saved.
        """

        # prepare the checkpoint
        ckpt = torch.load(checkpoint_filename, map_location=self.device)

        # load the checkpoint
        self.gen_model.load_state_dict(ckpt["gen_model"])

    def load_PSNR_checkpoint_for_test(self, checkpoint_filename: str) -> None:
        """
        Load a checkpoint of the network for the pretraining phase and convert it
        for the testing phase.

        Parameters:
        ===========
        checkpoint_filename (str): Name of the file where the checkpoint is saved.
        """

        # prepare the checkpoint
        ckpt = torch.load(checkpoint_filename, map_location=self.device)

        # load the checkpoint
        self.gen_model.load_state_dict(ckpt["gen_model"])

    def load_GAN_checkpoint(
            self, epoch: int, psnr: float, ssim: float) -> None:
        """
        Load a checkpoint of the network for the training phase.

        Parameters:
        ===========
        epoch (int): The epoch the model will resume the training at.
        psnr (float): The current psnr in the epoch.
        ssim (float): The current ssim in the epoch.
        """

        # prepare the checkpoint
        ckpt = torch.load(
            f"./models/ESRGAN_GAN-epoch:{epoch}-"
            f"psnr:{psnr:.2f}-ssim:{ssim:.2f}.ckpt", map_location=self.device)
        
        # load the checkpoint
        self.gen_model.load_state_dict(ckpt["gen_model"])
        self.disc_model.load_state_dict(ckpt["disc_model"])
        self.gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
        self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])

    def load_GAN_checkpoint_for_test(self, checkpoint_filename: str) -> None:
        """
        Load a checkpoint of the network for the training phase and convert it
        for the testing phase.

        Parameters:
        ===========
        checkpoint_filename (str): Name of the file where the checkpoint is saved.
        """

        # prepare the checkpoint
        ckpt = torch.load(checkpoint_filename, map_location=self.device)

        # load the checkpoint
        self.gen_model.load_state_dict(ckpt["gen_model"])
        self.disc_model.load_state_dict(ckpt["disc_model"])

    def fit(
            self,
            train_dl: torch.utils.data.DataLoader,
            val_dl: torch.utils.data.DataLoader,
            PSNR_checkpoint_filename: str,
            num_epochs: int, start_epoch: int,
            psnr: float, ssim: float, lr: float) -> None:
        """
        Perform the training and the validation phases.

        Parameters:
        ===========
        train_dl (torch.utils.data.DataLoader): Pytorch data loader the the 
            training dataset.
        val_dl (torch.utils.data.DataLoader): Pytorch data loader the the 
            validation dataset.
        PSNR_checkpoint_filename (str): Name of the file where a checkpoint for the 
            pretraining phase is saved.
        num_epochs (int): Number of epochs to train.
        start_epoch (int): The epoch to restart the training in.
        psnr (float): The current psnr when the model stopped training.
        ssim (float): The current ssim when the model stopped training.
        lr (float): An eventually new learning rate.
        """

        # build the optimizer for the generator
        self.gen_optimizer = self._configure_fit_optimizer(self.gen_model)

        # build the optimizer for the discriminator
        self.disc_optimizer = self._configure_fit_optimizer(self.disc_model)

        # load the checkpoint for pretraining
        print("[Loading checkpoint...]")
        self.load_PSNR_checkpoint_for_GAN(PSNR_checkpoint_filename)
        print("[Checkpoint loaded.]\n")

        # if there is a checkpoint for training, load it
        if start_epoch > 0:
            print("[Loading checkpoint...]")
            self.load_GAN_checkpoint(start_epoch, psnr, ssim)
            print("[Checkpoint loaded.]\n")
        
        # reset the right parameters for the optimizers
        self._adjust_optimizer(self.gen_optimizer, lr)
        self._adjust_optimizer(self.disc_optimizer, lr)

        # store useful data
        num_train_batches = len(train_dl)
        num_val_batches = len(val_dl)
        best_psnr = 0. if start_epoch == 0 else psnr
        best_ssim = 0. if start_epoch == 0 else ssim

        # iterate over the number of epochs
        for epoch in range(start_epoch, num_epochs):

            # set both the models in training mode
            self.gen_model.train()
            self.disc_model.train()

            # iterate over the batches of the training dataset
            for i, batch in enumerate(train_dl):
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]
                reals = torch.ones(
                    (lr_imgs.shape[0], 1), dtype=torch.float, device=self.device)
                fakes = torch.zeros(
                    (lr_imgs.shape[0], 1), dtype=torch.float, device=self.device)

                # freeze the discriminator
                self.disc_model.requires_grad_(requires_grad=False)

                # zero out the gradients for the generator weights
                self.gen_optimizer.zero_grad()

                # perform the generator forward pass
                gen_hr_imgs = self.gen_model(lr_imgs)

                with torch.no_grad():
                    # perform the feature extractor forward pass for HR images
                    features = self.feature_model(normalize(
                        hr_imgs,
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]))
                
                # perform the feature extractor forward pass for generated HR images
                gen_features = self.feature_model(normalize(
                    gen_hr_imgs,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]))

                with torch.no_grad():
                    # perform the discriminator forward pass for the HR images
                    pred_reals = self.disc_model(hr_imgs)
                
                # perform the discriminator forward pass for the generated HR images
                pred_fakes = self.disc_model(gen_hr_imgs)

                # compute the intermediate losses and the generator loss
                pixel_loss = pixel_loss_fn(gen_hr_imgs, hr_imgs)
                content_loss = content_loss_fn(gen_features, features)
                adversarial_real_loss = adversarial_loss_fn(
                    pred_reals - pred_fakes.mean(dim=0, keepdim=True), fakes) * 0.5
                adversarial_fake_loss = adversarial_loss_fn(
                    pred_fakes - pred_reals.mean(dim=0, keepdim=True), reals) * 0.5
                adversarial_loss = adversarial_real_loss + adversarial_fake_loss
                gen_loss = 1.e-2 * pixel_loss + \
                    content_loss + \
                    5.e-3 * adversarial_loss
                
                # perform the generator backward pass
                gen_loss.backward()

                # update the weights of the generator
                self.gen_optimizer.step()

                # unfreeze the discriminator
                self.disc_model.requires_grad_(requires_grad=True)

                # freeze the generator
                self.gen_model.requires_grad_(requires_grad=False)

                # zero out the gradients for the discriminator weights
                self.disc_optimizer.zero_grad()

                # perform the discriminator forward pass for the HR images
                pred_reals = self.disc_model(hr_imgs)

                # perform the discriminator forward pass for the generated HR images
                pred_fakes = self.disc_model(gen_hr_imgs.detach())

                # compute the first discriminator loss
                adversarial_real_loss = adversarial_loss_fn(
                    pred_reals - pred_fakes.mean(dim=0, keepdim=True), reals) * 0.5
                
                # perform the first discriminator backward pass
                adversarial_real_loss.backward(retain_graph=True)

                # compute the second discriminator loss
                adversarial_fake_loss = adversarial_loss_fn(
                    pred_fakes - pred_reals.mean(dim=0, keepdim=True), fakes) * 0.5
                
                # perform the second discriminator backward pass
                adversarial_fake_loss.backward()

                # compute the discriminator loss
                disc_loss = adversarial_real_loss + adversarial_fake_loss

                # update the weights of the discriminator
                self.disc_optimizer.step()

                # unfreeze the generator
                self.gen_model.requires_grad_(requires_grad=True)

                print(
                    f"Train [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_train_batches:3d}]")
                print(
                    f"Generator [loss {gen_loss.item():.2f}] "
                    f"[pixel_loss {pixel_loss.item():.2f}] "
                    f"[content_loss {content_loss.item():.2f}] "
                    f"[adversarial_loss {adversarial_loss.item():.2f}]")
                print(
                    f"Discriminator [loss {disc_loss.item():.2f}] "
                    f"[adversarial_real_loss {adversarial_real_loss.item():.2f}] "
                    f"[adversarial_fake_loss {adversarial_fake_loss.item():.2f}]\n")

            # store validation metrics
            psnrs = list()
            ssims = list()

            # set both the models in evaluation mode
            self.gen_model.eval()
            self.disc_model.eval()

            # iterate over the batches of the validation dataset
            for i, batch in enumerate(val_dl):
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]

                with torch.no_grad():
                    # perform the generator forward pass
                    gen_hr_imgs = self.gen_model(lr_imgs)
                
                # compute the validation metrics
                psnr = psnr_fn(gen_hr_imgs, hr_imgs).item()
                ssim = ssim_fn(gen_hr_imgs, hr_imgs).item()
                psnrs.append(psnr)
                ssims.append(ssim)

                print(
                    f"Validation [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_val_batches:3d}]")
                print(
                    f"GAN [psnr {psnr:.2f}] "
                    f"[ssim {ssim:.2f}]\n")

            # compute the average of the validation metrics
            psnrs = torch.tensor(psnrs, dtype=torch.float, device=self.device)
            psnr = torch.mean(psnrs).item()
            ssims = torch.tensor(ssims, dtype=torch.float, device=self.device)
            ssim = torch.mean(ssims).item()

            print(
                f"Validation [average]")
            print(
                f"GAN [psnr {psnr:.2f}] "
                f"[ssim {ssim:.2f}]\n")

            # eventually save the best checkpoint
            if psnr > best_psnr and ssim > best_ssim:
                print("[Saving checkpoint...]")
                self.save_GAN_checkpoint(epoch + 1, psnr, ssim)
                best_psnr = psnr
                best_ssim = ssim
                print("[Checkpoint saved.]\n")
            if epoch == num_epochs - 1:
                print("[Saving checkpoint...]")
                self.save_GAN_checkpoint(epoch + 1, psnr, ssim)
                print("[Checkpoint saved.]\n")

    def warmup(
            self,
            train_dl: torch.utils.data.DataLoader,
            val_dl: torch.utils.data.DataLoader,
            num_epochs: int, start_epoch: int,
            psnr: float, ssim: float, lr: float) -> None:
        """
        Perform the pretraining and the validation phases.

        Parameters:
        ===========
        train_dl (torch.utils.data.DataLoader): Pytorch data loader the the 
            training dataset.
        val_dl (torch.utils.data.DataLoader): Pytorch data loader the the 
            validation dataset.
        num_epochs (int): Number of epochs to pretrain.
        start_epoch (int): The epoch to restart the pretraining in.
        psnr (float): The current psnr when the model stopped pretraining.
        ssim (float): The current ssim when the model stopped pretraining.
        lr (float): An eventually new learning rate.
        """

        # build the optimizer for the generator
        self.gen_optimizer = self._configure_warmup_optimizer(self.gen_model)

        # if there is a checkpoint for pretraining, load it
        if start_epoch > 0:
            print("[Loading checkpoint...]")
            self.load_PSNR_checkpoint(start_epoch, psnr, ssim)
            print("[Checkpoint loaded.]\n")
        
        # reset the right parameters for the optimizer
        self._adjust_optimizer(self.gen_optimizer, lr)

        # store useful data
        num_train_batches = len(train_dl)
        num_val_batches = len(val_dl)
        best_psnr = 0. if start_epoch == 0 else psnr
        best_ssim = 0. if start_epoch == 0 else ssim

        # iterate over the number of epochs
        for epoch in range(start_epoch, num_epochs):

            # set the generator model in training mode
            self.gen_model.train()

            # iterate over the batches of the training dataset
            for i, batch in enumerate(train_dl):
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]

                # zero out the gradients for the generator weights
                self.gen_optimizer.zero_grad()

                # perform the generator forward pass
                gen_hr_imgs = self.gen_model(lr_imgs)

                # compute the generator loss
                gen_loss = pixel_loss_fn(gen_hr_imgs, hr_imgs)

                # perform the generator backward pass
                gen_loss.backward()

                # update the weights of the generator
                self.gen_optimizer.step()

                print(
                    f"Train Warmup [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_train_batches:3d}]")
                print(
                    f"Generator [pixel_loss {gen_loss.item():.2f}]\n")

            # store validation metrics
            psnrs = list()
            ssims = list()

            # set the generator model in evaluation mode
            self.gen_model.eval()

            # iterate over the batches of the validation dataset
            for i, batch in enumerate(val_dl):
                lr_imgs = batch["lr_imgs"]
                hr_imgs = batch["hr_imgs"]

                with torch.no_grad():
                    # perform the generator forward pass
                    gen_hr_imgs = self.gen_model(lr_imgs)
                
                # compute the validation metrics
                psnr = psnr_fn(gen_hr_imgs, hr_imgs).item()
                ssim = ssim_fn(gen_hr_imgs, hr_imgs).item()
                psnrs.append(psnr)
                ssims.append(ssim)

                print(
                    f"Validation [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_val_batches:3d}]")
                print(
                    f"GAN [psnr {psnr:.2f}] "
                    f"[ssim {ssim:.2f}]\n")

            # compute the average of the validation metrics
            psnrs = torch.tensor(psnrs, dtype=torch.float, device=self.device)
            psnr = torch.mean(psnrs).item()
            ssims = torch.tensor(ssims, dtype=torch.float, device=self.device)
            ssim = torch.mean(ssims).item()

            print(
                f"Validation [average]")
            print(
                f"GAN [psnr {psnr:.2f}] "
                f"[ssim {ssim:.2f}]\n")

            # eventually save the best checkpoint
            if psnr > best_psnr and ssim > best_ssim:
                print("[Saving checkpoint...]")
                self.save_PSNR_checkpoint(epoch + 1, psnr, ssim)
                best_psnr = psnr
                best_ssim = ssim
                print("[Checkpoint saved.]\n")
            if epoch == num_epochs - 1:
                print("[Saving checkpoint...]")
                self.save_PSNR_checkpoint(epoch + 1, psnr, ssim)
                print("[Checkpoint saved.]\n")

    @torch.no_grad()
    def test(
            self,
            test_dl: torch.utils.data.DataLoader,
            is_GAN_ckpt: bool, checkpoint_filename: str) -> None:
        """
        Perform the testing phase.

        Parameters:
        ===========
        test_dl (torch.utils.data.DataLoader): Pytorch data loader the the 
            test dataset.
        is_GAN_ckpt: (bool): Boolean flag indicating whether the checkpoint is for 
            training or not.
        checkpoint_filename (str): Name of the file where a checkpoint for the 
            pretraining or the training phase is saved.
        """

        # load the checkpoint
        print("[Loading checkpoint...]")
        if is_GAN_ckpt:
            self.load_GAN_checkpoint_for_test(checkpoint_filename)
        else:
            self.load_PSNR_checkpoint_for_test(checkpoint_filename)
        print("[Checkpoint loaded.]\n")

        # store useful data
        num_test_batches = len(test_dl)

        # store test metrics
        psnrs = list()
        ssims = list()

        # set both the models in evaluation mode
        self.gen_model.eval()
        self.disc_model.eval()

        # iterate over the batches of the test dataset
        for i, batch in enumerate(test_dl):
            lr_imgs = batch["lr_imgs"]
            hr_imgs = batch["hr_imgs"]

            # perform the generator forward pass
            gen_hr_imgs = self.gen_model(lr_imgs)

            # compute the test metrics
            psnr = psnr_fn(gen_hr_imgs, hr_imgs).item()
            psnrs.append(psnr)
            ssim = ssim_fn(gen_hr_imgs, hr_imgs).item()
            ssims.append(ssim)

            print(
                f"Test [batch {i + 1:3d}/{num_test_batches:3d}]")
            print(
                f"GAN [psnr {psnr:.2f}] "
                f"[ssim {ssim:.2f}]\n")

        # compute the average of the test metrics
        psnrs = torch.tensor(psnrs, dtype=torch.float, device=self.device)
        psnr = torch.mean(psnrs).item()
        ssims = torch.tensor(ssims, dtype=torch.float, device=self.device)
        ssim = torch.mean(ssims).item()

        print(
            f"Test [average]")
        print(
            f"GAN [psnr {psnr:.2f}] "
            f"[ssim {ssim:.2f}]\n")
