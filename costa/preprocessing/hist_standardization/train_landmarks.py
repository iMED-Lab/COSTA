# This script aims to perform histgram standardization of Multi-site MRA (MsMRA) images
# Please note that we use a large scale unlabeled multi-site (or various manufactors) MRA scans,
# in order to learn the global histogram representation of different MRA scans.
# We use torchio to perform histogram standardization

import argparse
import glob
import torch
from torchio.transforms import HistogramStandardization
from batchgenerators.utilities.file_and_folder_operations import *
import costa


def histogram_standardization_train(in_folder, mask_folder):
    """
    train a standardization operator
    Histogram standardization of multi-site MRA scans, saved to local device
    Supports for nifit images
    :param in_folder:
    :param out_folder:
    :return: histogram standardization landmarks.pth
    """
    print("Step 1: histogram standardization operator training...")
    # change the file format to support other type of scans
    img_paths = sorted(glob.glob(join(in_folder, "*.nii.gz")))
    mask_paths = sorted(glob.glob(join(mask_folder, "*.nii.gz")))
    landmarks = HistogramStandardization.train(img_paths, cutoff=(0.0, 1.0), mask_path=mask_paths)
    landmarks_dict = {
        'image': landmarks
    }
    # we save it to reuse in the future
    save_landmarks_path = join(costa.__path__[0], "preprocessing", "hist_standardization", "landmarks.pth")
    torch.save(landmarks_dict, save_landmarks_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="The input folder of your own files")
    parser.add_argument("-m", "--mask_folder", help="Corresponding brain mask folder")

    args = parser.parse_args()
    input_folder = args.input_folder
    mask_folder = args.mask_folder

    print("Train histogram standardization configurations from existing images")
    histogram_standardization_train(in_folder=input_folder, mask_folder=mask_folder)
    print("Training done. The landmarks.pth is saved to {}".format(
        join(costa.__path__[0], "preprocessing", "hist_standardization", "landmarks.pth")))


if __name__ == '__main__':
    main()
