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
    NUM_WARMUP_EPOCHS = 5
    NUM_EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr_gans.set_seed(0xDEADBEEF)

    lr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE // 4)
    ).to(device=DEVICE)
    lr_transforms = torch.jit.script(lr_transforms.eval())
    lr_transforms = torch.jit.freeze(lr_transforms)
    hr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(IMG_SIZE)
    ).to(device=DEVICE)
    hr_transforms = torch.jit.script(hr_transforms.eval())
    hr_transforms = torch.jit.freeze(hr_transforms)

    set14_ds = sr_gans.Set14Dataset(DEVICE, lr_transforms, hr_transforms)
    set14_dl = torch.utils.data.DataLoader(
        set14_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())

    model = sr_gans.ESRGAN(
        DEVICE, 3, GEN_CHANNELS, IMG_SIZE, RES_SCALE,
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        {"psnr": -1000., "ssim": -1000.}).to(device=DEVICE)
    model.test(set14_dl, checkpoint_filename="esrgan.ckpt")
