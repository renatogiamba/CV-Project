import argparse
import PIL
import PIL.Image
import torch
import torch.nn
import torch.nn.functional
import torch.cuda
import torch.jit
import torchvision
import torchvision.utils
import torchvision.transforms
import torchvision.transforms.functional

import sr_gans


@torch.no_grad()
def upscale_img(
        img: torch.Tensor,
        gen_model: torch.jit.ScriptModule,
        model_img_size: int) -> torch.Tensor:
    _, _, height, width = img.shape

    pad_height = ((model_img_size // 4) - height %
                  (model_img_size // 4)) % (model_img_size // 4)
    pad_width = ((model_img_size // 4) - width %
                 (model_img_size // 4)) % (model_img_size // 4)
    padded_img = torch.nn.functional.pad(
        img, (0, pad_width, 0, pad_height), mode="reflect")

    num_channels = padded_img.shape[1]
    patches = padded_img.unfold(
        2, (model_img_size // 4), (model_img_size // 4))\
        .unfold(3, (model_img_size // 4), (model_img_size // 4))
    _, _, num_patches_height, num_patches_width, _, _ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(
        -1, num_channels, model_img_size // 4, model_img_size // 4)

    idx = 0
    upscaled_patches = torch.zeros(
        (patches.shape[0], num_channels, model_img_size, model_img_size),
        dtype=torch.float, device=DEVICE)
    while idx < patches.shape[0]:
        start = idx
        end = start + 32 if start + 32 < patches.shape[0] else patches.shape[0]
        print(
            f"[Upscaling {start + 1}-{end}/{patches.shape[0]} patches...]\n")
        upscaled_patches[start:end] = gen_model(patches[start:end])
        print(
            f"[Patches {start + 1}-{end}/{patches.shape[0]} upscaled.]\n")
        idx = end

    padded_upscaled_img = torchvision.utils.make_grid(
        upscaled_patches, nrow=num_patches_width, padding=0)
    upscaled_img = torch.nn.functional.pad(
        padded_upscaled_img, (0, - 4 * pad_width, 0, -4 * pad_height),
        mode="reflect")
    
    return upscaled_img


if __name__ == "__main__":
    BATCH_SIZE = 16
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--gen_model_filename", action="store",
        default="./models/ESRGAN_interp_0.75_generator_jit.pt",
        help="the name of the file with a PyTorch optimized generator model")
    cli.add_argument(
        "--input_image_filename", action="store",
        default="input.jpg",
        help="the name of the input image file"
    )
    cli.add_argument(
        "--output_image_filename", action="store",
        default="output.jpg",
        help="the name of the input image file"
    )

    args = cli.parse_args()

    sr_gans.set_seed(0xDEADBEEF)

    print(f"[Device in use: {DEVICE}.]\n")

    print("[Reading the input image...]\n")
    img = PIL.Image.open(args.input_image_filename).convert("RGB")
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze_(0)
    print("[Input image read.]\n")

    print("[Building the model...]\n")
    model = torch.jit.load(args.gen_model_filename, map_location=DEVICE)
    print("[Model built.]\n")

    print("[Optimizing the model for this particular platform...]\n")
    model = torch.jit.optimize_for_inference(model)
    print("[Model optimized for this particular platform.]\n")

    print("[Upscaling the image...]\n")
    up_img = upscale_img(img, model, IMG_SIZE)
    print("[Image upscaled.]\n")

    print("[Writing the output image...]\n")
    up_img = torchvision.transforms.functional.to_pil_image(up_img.squeeze_())
    up_img.save(args.output_image_filename)
    print("[Output image written.]\n")
