import torch
import torch.cuda
import torch.jit
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms

import sr_gans


if __name__ == "__main__":
    BATCH_SIZE = 16
    IMG_SIZE = 96
    GEN_CHANNELS = 64
    RES_SCALE = 0.2
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NUM_EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr_gans.set_seed(0xDEADBEEF)

    lr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE // 4)
    )
    lr_transforms = torch.jit.script(lr_transforms.eval())
    lr_transforms = torch.jit.freeze(lr_transforms)
    hr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE)
    )
    hr_transforms = torch.jit.script(hr_transforms.eval())
    hr_transforms = torch.jit.freeze(hr_transforms)

    div2k_train_ds = sr_gans.DIV2KDataset(True, lr_transforms, hr_transforms)
    div2k_train_dl = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sr_gans.DataCollator())
    div2k_val_ds = sr_gans.DIV2KDataset(False, lr_transforms, hr_transforms)
    div2k_val_dl = torch.utils.data.DataLoader(
        div2k_val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())

    model = sr_gans.ESRGAN(
        3, GEN_CHANNELS, IMG_SIZE, RES_SCALE,
        torch.optim.Adam, {"lr": 0.002, "betas": (0.9, 0.999)},
        torch.optim.Adam, {"lr": 0.002, "betas": (0.9, 0.999)})
    model.fit(div2k_train_dl, div2k_val_dl, NUM_EPOCHS, 5)
