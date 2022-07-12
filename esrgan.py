import PIL
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
    batch = set14_ds[4]
    lr_img = batch["lr_img"]
    hr_img = batch["hr_img"]
    
    model = sr_gans.ESRGAN(
        DEVICE, 3, GEN_CHANNELS, IMG_SIZE, RES_SCALE,
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.999)},
        {"psnr": -1000., "ssim": -1000.}).to(device=DEVICE)
    model.load_checkpoint("esrgan.ckpt")
    gen_hr_img = model.predict(lr_img)
    
    lr_img = torchvision.transforms.functional.to_pil_image(lr_img)
    lr_img.save("lr_img.jpg")
    hr_img = torchvision.transforms.functional.to_pil_image(hr_img)
    hr_img.save("hr_img.jpg")
    gen_hr_img = torchvision.transforms.functional.to_pil_image(gen_hr_img)
    gen_hr_img.save("gen_hr_img.jpg")
