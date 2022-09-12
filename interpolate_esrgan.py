import argparse
import torch
import torch.cuda
import torch.jit

import sr_gans


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--PSNR_ckpt_filename", action="store",
        default="./models/ESRGAN_PSNR-epoch_16-psnr_21.97-ssim_0.65.ckpt",
        help="the name of the file with a checkpoint for the pretraining phase")
    cli.add_argument(
        "--GAN_ckpt_filename", action="store",
        default="./models/ESRGAN_GAN-epoch_350-psnr_19.35-ssim_0.58.ckpt",
        help="the name of the file with a checkpoint for the training phase")
    cli.add_argument(
        "--alpha", action="store", type=float, default=0.75,
        help="the coefficient for the interpolation")
    
    args = cli.parse_args()

    print("[Interpolating the model weights...]\n")
    sr_gans.esrgan_interpolation(
        args.PSNR_ckpt_filename, args.GAN_ckpt_filename, args.alpha, DEVICE)
    print("[Model weights interpolated...]\n")

    print("[Building the model...]\n")
    model = sr_gans.ESRGAN(DEVICE, 3, 3, 23).to(device=DEVICE)
    model.load_GAN_checkpoint_for_test(
        f"./models/ESRGAN_interp_{args.alpha:.2f}.ckpt")
    model.gen_model.requires_grad_(requires_grad=False)
    model.gen_model.eval()
    model.disc_model.requires_grad_(requires_grad=False)
    model.disc_model.eval()
    print("[Model built.]\n")

    print("[Optimizing and saving the model...]\n")
    jit_gen_model = torch.jit.script(model.gen_model)
    jit_gen_model = torch.jit.freeze(jit_gen_model)
    torch.jit.save(
        jit_gen_model, f"./models/esrgan_interp_{args.alpha:.2f}_generator_jit.pt")
    jit_disc_model = torch.jit.script(model.disc_model)
    jit_disc_model = torch.jit.freeze(jit_disc_model)
    torch.jit.save(
        jit_disc_model, f"./models/esrgan_interp_{args.alpha:.2f}_discriminator_jit.pt")
    print("[Model optimized and saved.]\n")
