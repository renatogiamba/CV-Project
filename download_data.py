import argparse
import os
import os.path
import shutil
import torchvision
import torchvision.datasets
import torchvision.datasets.utils


def download_and_decompress(
        dataset_name: str, dataset_branch_name: str, dataset_url: str,
        archive_path: str, dataset_path: str, store_root: str) -> None:
    """
    Download and decompress the images from the requested branch of the requested
    dataset. This function tries to be lazy: if it detects that the data is already
    downloaded and/or already decompressed in the provided folder, it won't perform 
    the detected operation/operations.

    Parameters
    ==========
    dataset_name (str) The name of the dataset to retrieve. 
        Used only for printing the infos during the execution of the function.

    dataset_branch_name (str) The name of the dataset branch to retrieve. 
        Used only for printing the infos during the execution of the function.

    dataset_url (str) The complete URL of a zip archive containing the dataset 
        to retrive.

    archive_path (str) -- The complete local path of the downloaded zip archive.

    dataset_path (str) -- The complete local path of the decompressed dataset.

    store_root (str) -- The complete local path of the directory where to store the
        downloaded zip archive and/or the decompressed dataset.
    """

    print(
        f"Start downloading the `{dataset_branch_name}` set "
        f"of the {dataset_name} dataset...")

    # check if the dataset has been downloaded and decompressed.
    # if both checks fail, download and decompress the dataset
    if (not os.path.exists(archive_path)) and (not os.path.exists(dataset_path)):
        torchvision.datasets.utils.download_and_extract_archive(
            dataset_url, store_root, store_root)
    else:
        print(
            f"The `{dataset_branch_name}` set "
            f"of the {dataset_name} dataset has already been downloaded.")
    print("Download finished.")

    # check if the downloaded archive has already been decompressed.
    # if not, decompress it.
    print("Start decompressing the downloaded archive...")
    if not os.path.exists(dataset_path):
        torchvision.datasets.utils.extract_archive(archive_path, store_root)
    else:
        print("The downloaded archive has already been decompressed.")
    print("Decompression finished.")


if __name__ == "__main__":
    # set the general CLI commands and options
    cli = argparse.ArgumentParser()
    sub_clis = cli.add_subparsers()

    # set the CLI commands and options for the DIV2K dataset
    div2k_cli = sub_clis.add_parser(
        "div2k", help="wheter to download the DIV2K dataset or not")
    div2k_cli.set_defaults(dataset="div2k")
    div2k_cli.add_argument(
        "--download_root", action="store", default="./data/",
        help="the folder where the data are downloaded in")
    div2k_cli.add_argument(
        "--train_LR", action="store_true", default=False,
        help="wheter to download the training set of LR images or not")
    div2k_cli.add_argument(
        "--train_HR", action="store_true", default=False,
        help="wheter to download the training set of HR images or not")
    div2k_cli.add_argument(
        "--valid_LR", action="store_true", default=False,
        help="wheter to download the validation set of LR images or not")
    div2k_cli.add_argument(
        "--valid_HR", action="store_true", default=False,
        help="wheter to download the validation set of HR images or not")

    # set the CLI commands and options for the Set14 dataset
    set14_cli = sub_clis.add_parser(
        "set14", help="wheter to download the Set14 dataset or not")
    set14_cli.set_defaults(dataset="set14")
    set14_cli.add_argument(
        "--download_root", action="store", default="./data/",
        help="the folder where the data are downloaded in")

    # parse the CLI and store the commands/options in a sort of dictionary
    args = cli.parse_args()

    # download options
    if args.dataset == "div2k":
        if args.train_LR:
            download_and_decompress(
                "DIV2K", "train_LR",
                "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip",
                "./data/DIV2K_train_LR_unknown_X4.zip",
                "./data/DIV2K_train_LR_unknown",
                args.download_root)

            # more adjustments for the DIV2K dataset
            if not os.path.exists("./data/DIV2K_train_LR"):
                os.makedirs("./data/DIV2K_train_LR", exist_ok=True)
                shutil.copytree(
                    "./data/DIV2K_train_LR_unknown/X4",
                    "./data/DIV2K_train_LR",
                    dirs_exist_ok=True)
                shutil.rmtree("./data/DIV2K_train_LR_unknown")
        if args.train_HR:
            download_and_decompress(
                "DIV2K", "train_HR",
                "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
                "./data/DIV2K_train_HR.zip",
                "./data/DIV2K_train_HR",
                args.download_root)
        if args.valid_LR:
            download_and_decompress(
                "DIV2K", "valid_LR",
                "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip",
                "./data/DIV2K_valid_LR_unknown_X4.zip",
                "./data/DIV2K_valid_LR_unknown",
                args.download_root)

            # more adjustments for the DIV2K dataset
            if not os.path.exists("./data/DIV2K_valid_LR"):
                shutil.copytree(
                    "./data/DIV2K_valid_LR_unknown/X4",
                    "./data/DIV2K_valid_LR",
                    dirs_exist_ok=True)
                shutil.rmtree("./data/DIV2K_valid_LR_unknown")
        if args.valid_HR:
            download_and_decompress(
                "DIV2K", "valid_HR",
                "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                "./data/DIV2K_valid_HR.zip",
                "./data/DIV2K_valid_HR",
                args.download_root)
    elif args.dataset == "set14":
        download_and_decompress(
            "Set14", "complete",
            "https://github.com/jbhuang0604/SelfExSR/archive/refs/heads/master.zip",
            "./data/master.zip",
            "./data/SelfExSR-master",
            args.download_root
        )

        # more adjustments for the Set14 dataset
        if not os.path.exists("./data/Set14"):
            os.makedirs("./data/Set14", exist_ok=True)
            shutil.copytree(
                "./data/SelfExSR-master/data/Set14/image_SRF_4",
                "./data/Set14",
                dirs_exist_ok=True)
            shutil.rmtree("./data/SelfExSR-master")
