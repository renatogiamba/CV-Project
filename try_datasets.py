import matplotlib
import matplotlib.pyplot
import torch
import torch.nn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms

import sr_gans


if __name__ == "__main__":
    lr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(224)
    )
    hr_transforms = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(896)
    )
    div2k_train_ds = sr_gans.DIV2KDataset(True, lr_transforms, hr_transforms)
    div2k_val_ds = sr_gans.DIV2KDataset(False, lr_transforms, hr_transforms)
    set14_ds = sr_gans.Set14Dataset(lr_transforms, hr_transforms)
    div2k_train_dataloader = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=128, shuffle=True,
        collate_fn=sr_gans.DataCollator()
    )
    div2k_val_dataloader = torch.utils.data.DataLoader(
        div2k_val_ds, batch_size=128, shuffle=False,
        collate_fn=sr_gans.DataCollator()
    )
    set14_dataloader = torch.utils.data.DataLoader(
        set14_ds, batch_size=8, shuffle=False,
        collate_fn=sr_gans.DataCollator())

    for batch in div2k_train_dataloader:
        print(batch["lr_imgs"].shape)
        print(batch["hr_imgs"].shape)
        print(10 * "=")
    print(20 * "*")
    for batch in div2k_val_dataloader:
        print(batch["lr_imgs"].shape)
        print(batch["hr_imgs"].shape)
        print(10 * "=")
    print(20 * "*")
    for batch in set14_dataloader:
        print(batch["lr_imgs"].shape)
        print(batch["hr_imgs"].shape)
        print(10 * "=")
