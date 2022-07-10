import torch
import torch.jit
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms
import torchvision.transforms.functional

import sr_gans

if __name__ == "__main__":
    BATCH_SIZE = 16
    IMG_SIZE = 128
    DEVICE = torch.device("cpu")

    psum = torch.zeros((3,), dtype=torch.float)
    psqsum = torch.zeros((3,), dtype=torch.float)

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

    div2k_train_ds = sr_gans.DIV2KDataset(
        True, DEVICE, lr_transforms, hr_transforms)
    div2k_train_dl = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())

    num_batches = len(div2k_train_dl)
    for i, batch in enumerate(div2k_train_dl):
        hr_imgs = batch["hr_imgs"]
        psum = psum + torch.mean(hr_imgs, dim=[0, 2, 3])
        psqsum = psqsum + torch.mean(hr_imgs ** 2, dim=[0, 2, 3])
        print(f"[Batch {i + 1:3d}/{num_batches:3d}]")
    
    mean = psum / num_batches
    std = torch.sqrt((psqsum / num_batches) - (mean ** 2))
    print(f"Mean {mean.tolist()}")
    print(f"Std  {std.tolist()}")
