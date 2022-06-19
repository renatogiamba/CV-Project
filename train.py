import torch
import torch.nn
import torch.optim
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
    div2k_train_dataloader = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=128, shuffle=True,
        collate_fn=sr_gans.DataCollator()
    )
    div2k_val_dataloader = torch.utils.data.DataLoader(
        div2k_val_ds, batch_size=128, shuffle=False,
        collate_fn=sr_gans.DataCollator()
    )

    model = sr_gans.SRGAN()

    gen_optimizer = torch.optim.Adam(model.gen.parameters(), lr=0.001)
    disc_optimizer = torch.optim.Adam(model.disc.parameters(), lr=0.001)

    model.fit(div2k_train_dataloader, 0, gen_optimizer, disc_optimizer)
