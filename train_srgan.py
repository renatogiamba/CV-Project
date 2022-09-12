import argparse
import torch
import torch.cuda
import torch.jit
import torch.utils
import torch.utils.data

import sr_gans


if __name__ == "__main__":
    BATCH_SIZE = 16
    IMG_SIZE = 96
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--num_epochs", action="store", type=int, default=50,
        help="the number of epochs to train")
    cli.add_argument(
        "--start_epoch", action="store", type=int, default=0,
        help="the epoch to restart the training in")
    cli.add_argument(
        "--psnr", action="store", type=float, default=0.,
        help="the current psnr when the model stopped training")
    cli.add_argument(
        "--ssim", action="store", type=float, default=0.,
        help="the current ssim when the model stopped training")
    cli.add_argument(
        "--lr", action="store", type=float, default=1.e-4,
        help="an eventually new learning rate")
    
    args = cli.parse_args()
    
    print(f"[Device in use: {DEVICE}.]\n")

    print("[Loading the training dataset...]\n")
    div2k_train_ds = sr_gans.DIV2KDataset(True, DEVICE)
    div2k_train_dl = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sr_gans.DataCollator())
    print("[Training dataset loaded.]\n")

    print("[Loading the validation dataset...]\n")
    div2k_val_ds = sr_gans.DIV2KDataset(False, DEVICE)
    div2k_val_dl = torch.utils.data.DataLoader(
        div2k_val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())
    print("[Validation dataset loaded.]\n")

    print("[Building the model...]\n")
    model = sr_gans.SRGAN(DEVICE, 3, 3, 16).to(device=DEVICE)
    print("[Model built.]\n")
    
    model.fit(
        div2k_train_dl, div2k_val_dl,
        args.num_epochs, args.start_epoch,
        args.psnr, args.ssim, args.lr)
