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
    IMG_SIZE = 128
    GEN_CHANNELS = 64
    RES_SCALE = 0.2
    #MEAN = [0.485, 0.456, 0.406]
    MEAN = [0.4884, 0.4459, 0.3967]
    #STD = [0.229, 0.224, 0.225]
    STD = [0.2831, 0.2649, 0.2747]
    NUM_EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr_gans.set_seed(0xDEADBEEF)

    lr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE // 4),
        torchvision.transforms.Normalize(MEAN, STD)
    ).to(device=DEVICE)
    lr_transforms = torch.jit.script(lr_transforms.eval())
    lr_transforms = torch.jit.freeze(lr_transforms)
    hr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE),
        torchvision.transforms.Normalize(MEAN, STD)
    ).to(device=DEVICE)
    hr_transforms = torch.jit.script(hr_transforms.eval())
    hr_transforms = torch.jit.freeze(hr_transforms)

    div2k_train_ds = sr_gans.DIV2KDataset(
        True, DEVICE, lr_transforms, hr_transforms)
    div2k_train_dl = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sr_gans.DataCollator())
    div2k_val_ds = sr_gans.DIV2KDataset(
        False, DEVICE, lr_transforms, hr_transforms)
    div2k_val_dl = torch.utils.data.DataLoader(
        div2k_val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())

    model = sr_gans.ESRGAN(
        DEVICE, 3, GEN_CHANNELS, IMG_SIZE, RES_SCALE,
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        {"psnr": -1000., "ssim": 0.}).to(device=DEVICE)
    model.fit(div2k_train_dl, div2k_val_dl, NUM_EPOCHS, 5)
