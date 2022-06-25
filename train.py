import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms
import config
from models import GeneratorRN , Discriminator
from torch import optim
from utils import VGGLoss, load_checkpoint, save_checkpoint, plot_examples
from tqdm import tqdm
import numpy as np
import sys

import sr_gans

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, criterion_GAN, criterion_content, vgg_loss,epoch):
    loop = tqdm(loader, leave=True)

    for idx, imgs in enumerate(loop):
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        imgs_lr = torch.autograd.Variable(imgs["lr_img"].type(Tensor)).to(config.DEVICE)
        imgs_hr = torch.autograd.Variable(imgs["hr_img"].type(Tensor)).to(config.DEVICE)
        if cuda:
                  gen = gen.cuda()
                  disc = disc.cuda()
                  vgg_loss = vgg_loss.cuda()
                  criterion_GAN = criterion_GAN.cuda()
                  criterion_content = criterion_content.cuda()

        valid = torch.autograd.Variable(Tensor(np.ones((imgs_lr.size(0), *disc.output_shape))),requires_grad=False)
        fake = torch.autograd.Variable(Tensor(np.zeros((imgs_lr.size(0), *disc.output_shape))),requires_grad=False)
        
        opt_gen.zero_grad()

        gen_hr = gen(imgs_lr)

        loss_GAN = criterion_GAN(disc(gen_hr),valid)

        gen_features = vgg_loss(gen_hr)
        real_features = vgg_loss(imgs_hr)
        loss_content = criterion_content(gen_features,real_features.detach())

        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        opt_gen.step()

        opt_disc.zero_grad()

        loss_real = criterion_GAN(disc(imgs_hr),valid)
        loss_fake = criterion_GAN(disc(gen_hr.detach()),fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        opt_disc.step()

        sys.stdout.write(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                         % (epoch, config.NUM_EPOCHS, idx, len(loader), loss_D.item(), loss_G.item())  
                    )

        if idx % 200 == 0:
            plot_examples("/content/data/Set14/", gen)

def main():
    
    div2k_train_ds = sr_gans.DIV2KDataset(True, config.lr_transforms, config.hr_transforms)
    div2k_val_ds = sr_gans.DIV2KDataset(False, config.lr_transforms, config.hr_transforms)
    loader = torch.utils.data.DataLoader(
        div2k_train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=config.NUM_WORKERS
    )
      
    hr_shape=(config.HIGH_RES,config.HIGH_RES)
    gen = GeneratorRN(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc = Discriminator(input_shape=(config.IMG_CHANNELS,*hr_shape)).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, criterion_GAN, criterion_content, vgg_loss,epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)



if __name__ == "__main__":
    main()