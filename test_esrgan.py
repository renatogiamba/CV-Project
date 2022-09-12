import argparse
import torch
import torch.cuda
import torch.jit
import torch.utils
import torch.utils.data

import sr_gans


if __name__ == "__main__":
    BATCH_SIZE = 16
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--ckpt_filename", action="store",
        default="./models/ESRGAN_GAN-epoch_350-psnr_19.35-ssim_0.58.ckpt",
        help="the name of the file with a checkpoint for the pretraining "
            "or the training phase")
    cli.add_argument(
        "--is_PSNR_ckpt", action="store_true", default=False,
        help="whether the checkpoint is for pretraining or not")

    args = cli.parse_args()

    sr_gans.set_seed(0xDEADBEEF)

    print(f"[Device in use: {DEVICE}.]\n")

    print("[Loading the test dataset...]\n")
    set14_ds = sr_gans.Set14Dataset(DEVICE)
    set14_dl = torch.utils.data.DataLoader(
        set14_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sr_gans.DataCollator())
    print("[Test dataset loaded.]\n")

    print("[Building the model...]\n")
    model = sr_gans.ESRGAN(DEVICE, 3, 3, 23).to(device=DEVICE)
    print("[Model built.]\n")

    model.test(
        set14_dl,
        not args.is_PSNR_ckpt, args.checkpoint_filename)
