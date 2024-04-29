# This script aims to perform histgram standardization of Multi-site MRA (MsMRA) images
# Please note that we use a large scale unlabeled multi-site (or various manufactors) MRA scans,
# in order to learn the global histogram representation of different MRA scans.
# We use torchio to perform histogram standardization

import argparse
import glob
import os
import shutil

import torchio
import tqdm
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import costa


def perform_histogram_standardization(in_folder, landmarks, out_folder):
    for file in tqdm.tqdm(glob.glob(join(in_folder, "*"))):
        transform = torchio.HistogramStandardization(landmarks, masking_method=lambda x: x > 0)
        subject = torchio.Subject(
            image=torchio.ScalarImage(file)
        )
        transformed = transform(subject)
        transformed_image = transformed['image'].as_sitk()
        sitk.WriteImage(transformed_image, os.path.join(out_folder, os.path.basename(file)))


def plan_the_costa_input_dir(raw_dir, normed_dir):
    raw_dir = os.path.abspath(raw_dir)
    normed_dir = os.path.abspath(normed_dir)
    if not os.path.exists(normed_dir):
        raise ValueError("Histogram standardized folder does not exist")

    output_dir = os.path.join(os.path.dirname(raw_dir), "costa_inputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_num = 0
    for file in tqdm.tqdm(glob.glob(os.path.join(raw_dir, "*"))):
        file_basename = os.path.basename(file)[:-7]
        raw_file = file
        normed_file = os.path.join(normed_dir, os.path.basename(file))
        dst_raw_file = os.path.join(output_dir, file_basename + "_0000.nii.gz")
        dst_normed_file = os.path.join(output_dir, file_basename + "_0001.nii.gz")
        shutil.copyfile(src=raw_file, dst=dst_raw_file)
        shutil.copyfile(src=normed_file, dst=dst_normed_file)
        file_num += 1

    print("Done!")
    print("Total {} files are copied to {}".format(file_num, output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, help="The input folder of your own files")
    parser.add_argument("-o", "--output_folder", required=False, default=None,
                        help="The output folder of your own files")

    args = parser.parse_args()
    input_folder = args.input_folder
    if len(os.listdir(input_folder)) <= 0:
        raise Exception("The input folder is empty")
    input_folder = os.path.abspath(input_folder)

    images_dir = os.path.basename(input_folder)
    # if images_dir not in ["imagesTr", "imagesTs"]:
    #     raise Exception("The input folder should be named as `imagesTr` or `imagesTs`")

    if args.output_folder is None:
        output_folder = os.path.join(os.path.dirname(input_folder), images_dir + "_normed")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    else:
        output_folder = os.path.abspath(args.output_folder)
        output_dir = os.path.basename(args.output_folder)
        if output_dir not in ["imagesTr_normed", "imagesTs_normed"]:
            raise Exception("The output folder should be named as `imagesTr_normed` or `imagesTs_normed`")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    landmarks_path = join(costa.__path__[0], "preprocessing", "hist_standardization", "landmarks.pth")
    print("Perform intensity histogram standardization using the pre-trained configurations")
    perform_histogram_standardization(in_folder=input_folder, landmarks=landmarks_path, out_folder=output_folder)


if __name__ == '__main__':
    main()
