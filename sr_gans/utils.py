import torch
import torch.nn as nn
import os
import config
import numpy as np
import PIL
from PIL import Image
from torchvision.utils import save_image
from torchvision.models import vgg19


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
      tensors[:, c].mul_(config.std[c]).add_(config.mean[c])
    return torch.clamp(tensors, 0, 255)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "/content/drive/MyDrive/CV_Project/saved_models/"+filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = PIL.Image.open("/content/data/Set14/" + file).convert("RGB")
        with torch.no_grad():
            upscaled_img = gen(
                torch.autograd.Variable(config.test_transform(image))
                .unsqueeze(0)
                .to("cuda")
            )
        save_image(denormalize(upscaled_img), f"saved/{file}")
    gen.train()

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg_loss = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.vgg_loss(img)