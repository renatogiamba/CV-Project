import torch
import torchvision
import numpy as np
import PIL
from PIL import Image

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 8
HIGH_RES = 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

test_transform = torchvision.transforms.Compose(
                [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
                ]
                )

lr_transforms = torchvision.transforms.Compose(
                [
                torchvision.transforms.Resize((HIGH_RES // 4, HIGH_RES // 4), PIL.Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
                ]
                )
hr_transforms = torchvision.transforms.Compose(
                [
                torchvision.transforms.Resize((HIGH_RES, HIGH_RES), PIL.Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
                ]
                )