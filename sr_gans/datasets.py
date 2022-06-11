import numpy
import os
import PIL
import torch
import torch.utils
import torch.utils.data
from typing import *


class Set14Dataset(torch.utils.data.Dataset):
    """
    Pytorch subclass for handling the Set14 dataset.
    """

    def __init__(self, dir_path: str) -> None:
        """
        Constructor for the class.

        Parameters:
        ===========
        dir_path (str): The complete local path to the root directory of the dataset.
        """

        super(Set14Dataset, self).__init__()

        # prepare filename lists both for low resolution (LR) and high resolution (HR)
        # images
        self.lr_filenames = list()
        self.hr_filenames = list()

        # iterate over the root directory and split LR image paths and HR image paths
        with os.scandir(dir_path) as file_it:
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

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """
        Retrieve an image couple (LR, HR) from the dataset by its index.

        Parameters:
        ===========
        idx (int): The index of the image couple in the dataset.
            Must be between 0 and len(dataset) - 1.

        Returns:
        ========
        (Dict[str, numpy.ndarray]): The desired image couple stored in a Python
            dictionary. The dictionary has two keys:
            - 'lr_img' that has the LR image as value;
            - 'hr_img' that has the HR image as value.
        """

        # open the images with Pillow and convert them into a Numpy array
        lr_img = numpy.array(PIL.Image.open(self.lr_filenames[idx]))
        hr_img = numpy.array(PIL.Image.open(self.hr_filenames[idx]))

        return {
            "lr_img": lr_img,
            "hr_img": hr_img
        }


# next thing to implement. it's a utility for the Pytorch dataloader
class Set14DataCollator():
    def __init__(self) -> None:
        pass

    def __call__(self) -> None:
        return None
