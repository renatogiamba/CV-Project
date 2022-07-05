import collections
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.models


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
        super(Conv2dK3P1Act, self).__init__()
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
        new_img_features1 = torch.cat([
            img_features, self.conv1(img_features)], dim=1)
        new_img_features2 = torch.cat([
            new_img_features1, self.conv2(new_img_features1)], dim=1)
        new_img_features3 = torch.cat([
            new_img_features2, self.conv3(new_img_features2)], dim=1)
        new_img_features4 = torch.cat([
            new_img_features3, self.conv4(new_img_features3)], dim=1)
        new_img_features_final = torch.cat([
            new_img_features4, self.conv_final(new_img_features4)], dim=1)
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
        vgg_model = torchvision.models.vgg19(
            pretrained=True, model_dir="./models")

        self.vgg_feature_body = torch.nn.Sequential(
            *list(vgg_model.features.children())[:35]
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        features = self.vgg_feature_body(imgs)
        return features


class ESRGAN(torch.nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            img_size: int, res_scale: float,
            gen_optimizer: torch.optim.Optimizer,
            disc_optimizer: torch.optim.Optimizer) -> None:
        self.gen_model = Generator(in_channels, out_channels, res_scale)
        self.disc_model = Discriminator(in_channels, img_size)
        self.feature_model = FeatureExtractor().requires_grad_(requires_grad=False)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
    
    def save_checkpoint(self, filename: str) -> None:
        ckpt = {
            "gen_model": self.gen_model.state_dict(),
            "disc_model": self.disc_model.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict()
        }
        torch.save(ckpt, f"./models/{filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        ckpt = torch.load(f"./models/{filename}")
        self.gen_model.load_state_dict(ckpt["gen_model"])
        self.disc_model.load_state_dict(ckpt["disc_model"])
        self.gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
        self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])

        #LRs for optimizers?

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        hr_imgs = self.gen_model(lr_imgs)
        return hr_imgs

    def fit(
            self, train_dl: torch.utils.data.DataLoader, num_epochs: int) -> None:
        pass
