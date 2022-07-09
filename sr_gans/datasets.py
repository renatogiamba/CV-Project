import os
import PIL
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from typing import *


class DIV2KDataset(torch.utils.data.Dataset):
    """
    Pytorch subclass for handling the DIV2K dataset.
    """

    def __init__(
            self, train: bool, device: torch.device,
            lr_transforms: torch.nn.Sequential,
            hr_transforms: torch.nn.Sequential) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        train (bool): Flag to choose wheter to open the training set (if True) or the 
            validation set (if False).

        device (torch.device): The Pytorch device to put data tensors on.

        lr_transforms (torch.nn.Sequential): A Pytorch module containing a sequence of 
            image transformation that will be applied to the LR images.

        hr_transforms (torch.nn.Sequential): A Pytorch module containing a sequence of 
            image transformation that will be applied to the HR images.
        """

        super(DIV2KDataset, self).__init__()

        # store the device and the image transformations
        self.device = device
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

        # prepare filename lists both for low resolution (LR) and high resolution (HR)
        # images
        self.lr_filenames = list()
        self.hr_filenames = list()

        # iterate over the root directory and split LR image paths and HR image paths
        with os.scandir(f"./data/DIV2K_{'train' if train else 'valid'}_LR") as file_it:
            for entry in file_it:
                self.lr_filenames.append(entry.path)
        with os.scandir(f"./data/DIV2K_{'train' if train else 'valid'}_HR") as file_it:
            for entry in file_it:
                self.hr_filenames.append(entry.path)

        # sort the filenames
        self.lr_filenames.sort()
        self.hr_filenames.sort()

    def __len__(self) -> int:
        """
        Get the number of image couples (LR, HR) in the dataset.

        Returns:
        ========
        (int): The number of image couples in the dataset.
        """
        return len(self.lr_filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve an image couple (LR, HR) from the dataset by its index.

        Parameters:
        ===========
        idx (int): The index of the image couple in the dataset.
            Must be between 0 and len(dataset) - 1.

        Returns:
        ========
        (Dict[str, torch.Tensor]): The desired image couple stored in a Python
            dictionary. The dictionary has two keys:
            - 'lr_img' that has the LR image as value;
            - 'hr_img' that has the HR image as value.
        """

        # open the images with Pillow, convert them into a Pytorch tensor
        # and apply the image transformations
        lr_img = PIL.Image.open(self.lr_filenames[idx]).convert("RGB")
        lr_img = torchvision.transforms.functional.to_tensor(lr_img)
        lr_img = lr_img.to(device=self.device)
        lr_img = self.lr_transforms(lr_img)
        hr_img = PIL.Image.open(self.hr_filenames[idx]).convert("RGB")
        hr_img = torchvision.transforms.functional.to_tensor(hr_img)
        hr_img = hr_img.to(device=self.device)
        hr_img = self.hr_transforms(hr_img)

        return {
            "lr_img": lr_img,
            "hr_img": hr_img
        }


class Set14Dataset(torch.utils.data.Dataset):
    """
    Pytorch subclass for handling the Set14 dataset.
    """

    def __init__(
            self, device: torch.device,
            lr_transforms: torch.nn.Sequential,
            hr_transforms: torch.nn.Sequential) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        device (torch.device): The Pytorch device to put data tensors on.

        lr_transforms (torch.nn.Sequential): A Pytorch module containing a sequence of 
            image transformation that will be applied to the LR images.

        hr_transforms (torch.nn.Sequential): A Pytorch module containing a sequence of 
            image transformation that will be applied to the HR images.
        """

        super(Set14Dataset, self).__init__()

        # store the device and the image transformations
        self.device = device
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

        # prepare filename lists both for low resolution (LR) and high resolution (HR)
        # images
        self.lr_filenames = list()
        self.hr_filenames = list()

        # iterate over the root directory and split LR image paths and HR image paths
        with os.scandir("./data/Set14") as file_it:
            for entry in file_it:
                if entry.name.endswith("LR.png"):
                    self.lr_filenames.append(entry.path)
                elif entry.name.endswith("HR.png"):
                    self.hr_filenames.append(entry.path)
                else:
                    pass

        # sort the filenames
        self.lr_filenames.sort()
        self.hr_filenames.sort()

    def __len__(self) -> int:
        """
        Get the number of image couples (LR, HR) in the dataset.

        Returns:
        ========
        (int): The number of image couples in the dataset.
        """
        return len(self.lr_filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve an image couple (LR, HR) from the dataset by its index.

        Parameters:
        ===========
        idx (int): The index of the image couple in the dataset.
            Must be between 0 and len(dataset) - 1.

        Returns:
        ========
        (Dict[str, torch.Tensor]): The desired image couple stored in a Python
            dictionary. The dictionary has two keys:
            - 'lr_img' that has the LR image as value;
            - 'hr_img' that has the HR image as value.
        """

        # open the images with Pillow, convert them into a Pytorch tensor
        # and apply the image transformations
        lr_img = PIL.Image.open(self.lr_filenames[idx]).convert("RGB")
        lr_img = torchvision.transforms.functional.to_tensor(lr_img)
        lr_img = lr_img.to(device=self.device)
        lr_img = self.lr_transforms(lr_img)
        hr_img = PIL.Image.open(self.hr_filenames[idx]).convert("RGB")
        hr_img = torchvision.transforms.functional.to_tensor(hr_img)
        hr_img = hr_img.to(device=self.device)
        hr_img = self.hr_transforms(hr_img)

        return {
            "lr_img": lr_img,
            "hr_img": hr_img
        }


class DataCollator():
    """
    Pytorch utility that turns a sequence of data points into a batch of data points.
    """

    def __init__(self) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        """

    def __call__(
            self,
            img_couples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform the convertion into batches.

        Parameters:
        ===========
        img_couples (List[Dict[str, torch.Tensor]]): The sequence of image couples to 
            convert into a batch.

        Returns:
        ========
        (Dict[str, torch.Tensor]): The batch with the provided image couples stored in 
            a Python dictionary. The dictionary has two keys:
            - 'lr_imgs' that has the batch of LR images as value;
            - 'hr_imgs' that has the batch of HR images as value.
        """

        # prepare the needed lists
        lr_images = list()
        hr_images = list()

        # iterate over the sequence of image couples and apply the image
        # transformations
        for img_couple in img_couples:
            lr_images.append(img_couple["lr_img"])
            hr_images.append(img_couple["hr_img"])

        # convert into a batch
        lr_imgs_batch = torch.stack(lr_images, dim=0)
        hr_imgs_batch = torch.stack(hr_images, dim=0)

        return {
            "lr_imgs": lr_imgs_batch,
            "hr_imgs": hr_imgs_batch
        }
