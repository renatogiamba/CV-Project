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
import numpy as np

from .srgan_utils import (
    adversarial_loss_fn,
    content_loss_fn,
    normalize,
    psnr_fn,
    ssim_fn
)

class ResidualBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the Generator fundamental part: Residual Block
    """

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_features, 0.8)
        self.relu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(in_features, 0.8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out+x


class UpscaleBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the upscale part of Generator
    """

    def __init__(self, in_features):
        super(UpscaleBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_features, in_features*4, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_features*4)
        self.ps = torch.nn.PixelShuffle(upscale_factor=2)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ps(out)
        out = self.relu(out)
        return out


class GeneratorRN(torch.nn.Module):
    """
    Pytorch class for handling the SRGAN Generator network
    """

    def __init__(self, in_channels:int, out_channels:int, n_residual_blocks:int):
        super(Generator, self).__init__()

        # First conv layer
        self.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.relu = torch.nn.PReLU()

        # Features residual block
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = torch.nn.Sequential(*res_blocks)

        # Second conv layer
        self.conv2 = torch.nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64, 0.8)

        # Upscale block
        upsampling = []
        for _ in range(2):
            upsampling.append(UpscaleBlock(64))
        self.upsampling = torch.nn.Sequential(*upsampling)

        # Output layer
        self.conv3 = torch.nn.Conv2d(
            64, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out2 = self.bn1(out2)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out


class Discriminator(torch.nn.Module):
    """
    Pytorch class for handling the SRGAN Discriminator network
    """

    def __init__(self, in_channels:int) -> None:
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            # First Conv layer:input size. (3) x 96 x 96
            torch.nn.Conv2d(in_channels, 64, kernel_size=3,padding=1, stride=1),
            torch.nn.LeakyReLU(0.2, True),
            # First discriminator block:state size. (64) x 48 x 48
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            # Second discriminator block:state size. (128) x 24 x 24
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            # Third discriminator block:state size. (256) x 12 x 12
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            # Last Conv layer:state size. (512) x 6 x 6
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.classifier = torch.nn.Conv2d(
            512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.model(x)
        out = self.classifier(out)
        return out


class FeatureExtractor(torch.nn.Module):
    """
    Pytorch subclass for hadling the the SRGAN feature extractor.
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = torchvision.models.vgg19()
        vgg19_model.load_state_dict(torch.load("./models/vgg19-dcbb9e9d.pth"))

        self.vgg_loss = torch.nn.Sequential(
            *list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.vgg_loss(img)
        

class SRGAN(torch.nn.Module):
    """
    Pytorch subclass for hadling the the SRGAN network.
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

        super(SRGAN, self).__init__()

        # store the device
        self.device = device

        # build the generator and the discriminator networks
        self.gen_model = GeneratorRN(in_channels, out_channels, num_blocks)
        self.disc_model = Discriminator(in_channels)

        # load the feature extractor network
        self.feature_model = FeatureExtractor()

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
            ckpt, f"./models/SRGAN_GAN-epoch_{epoch}-"
            f"psnr_{psnr:.2f}-ssim_{ssim:.2f}.ckpt")
   
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
            f"./models/SRGAN_GAN-epoch_{epoch}-"
            f"psnr_{psnr:.2f}-ssim_{ssim:.2f}.ckpt", map_location=self.device)
        
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
                
                lr_imgs = torch.autograd.Variable(batch["lr_imgs"].type(torch.cuda.FloatTensor)).to(self.device)
                hr_imgs = torch.autograd.Variable(batch["hr_imgs"].type(torch.cuda.FloatTensor)).to(self.device)
                patch_h, patch_w = int(96 / 2 ** 4), int(96 / 2 ** 4)
                shape=(1, patch_h, patch_w)
                reals = torch.autograd.Variable(
                    torch.cuda.FloatTensor(np.ones((lr_imgs.size(0), *shape))),requires_grad=False)
                fakes = torch.autograd.Variable(
                    torch.cuda.FloatTensor(np.ones((lr_imgs.size(0), *shape))),requires_grad=False)
                
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
                loss_content = content_loss_fn(gen_features, features)
                loss_GAN = adversarial_loss_fn(pred_fakes,reals)

                loss_gen = loss_content + 1e-3 * loss_GAN

                # perform the generator backward pass
                loss_gen.backward()

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

                # compute the intermediate losses and the discriminator loss
                adversarial_real_loss = adversarial_loss_fn(pred_reals, reals) * 0.5

                # perform the first discriminator backward pass
                adversarial_real_loss.backward(retain_graph=True)

                adversarial_fake_loss = adversarial_loss_fn(pred_fakes, fakes) * 0.5

                # perform the second discriminator backward pass
                adversarial_fake_loss.backward()

                loss_disc = adversarial_real_loss + adversarial_fake_loss

                # update the weights of the discriminator
                self.disc_optimizer.step()

                # unfreeze the generator
                self.gen_model.requires_grad_(requires_grad=True)

                print(
                    f"Train [epoch {epoch + 1:3d}/{num_epochs:3d}] "
                    f"[batch {i + 1:3d}/{num_train_batches:3d}]")
                print(
                    f"Generator [loss {loss_gen.item():.2f}] "
                    f"[content_loss {loss_content.item():.2f}] "
                    f"[adversarial_loss {loss_GAN.item():.2f}]")
                print(
                    f"Discriminator [loss {loss_disc.item():.2f}] "
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
            print(
                f"Please insert a checkpoint"
            )
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
