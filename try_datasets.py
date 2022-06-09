import matplotlib
import matplotlib.pyplot

import sr_gans


if __name__ == "__main__":
    set14_ds = sr_gans.Set14Dataset("./data/Set14")

    imgs = set14_ds[0]
    fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    axes[0].axis("off")
    axes[0].imshow(imgs["lr_img"])
    axes[1].axis("off")
    axes[1].imshow(imgs["hr_img"])
    matplotlib.pyplot.show()
