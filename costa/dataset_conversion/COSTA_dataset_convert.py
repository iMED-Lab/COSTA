#   This dataset format conversation script is builted to convert COSTA dataset to CESAR network input format
#   Author: iMED group
#   Time: 2023/5/8
#   Thanks: nnUNet framework

import glob
import os
import re
from multiprocessing.pool import Pool
import numpy as np
from collections import OrderedDict
import argparse

from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import shutil
from costa.paths import nnUNet_RAW_DATA, nnUNet_PREPROCESSED, nnUNet_TRAINED_MODELS


def find_task_name_from_task_id(task_id):
    nnunet_raw_data = join(nnUNet_RAW_DATA, "nnUNet_raw_data")
    all_task_dirs = listdir(nnunet_raw_data)
    target_task = None
    for tpe in all_task_dirs:
        if tpe.startswith("Task" + str(task_id) + "_"):
            target_task = tpe
    if target_task is None:
        raise ValueError("Cannot find target task based on task id %d" % task_id)

    return target_task


def generate_task_name_from_task_id(task_id, task):
    task_name = "Task%03.0d" % task_id + "_" + task.split("_")[-1]
    return task_name


def convert_dataset_to_cesar_format(task_id: int):
    # please input your dataset root path
    nnUNet_raw_data = join(nnUNet_RAW_DATA, "nnUNet_raw_data")
    task = find_task_name_from_task_id(task_id)
    downloaded_data_dir = join(nnUNet_raw_data, task)
    new_task_name = generate_task_name_from_task_id(task_id, task)

    target_base = join(nnUNet_raw_data, new_task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    for tpe in ["imagesTr"]:
        cur = join(downloaded_data_dir, tpe)
        for p in sorted(os.listdir(cur)):
            patient_name = p[:-7]
            patient_names.append(patient_name)
            ori = join(cur, patient_name + ".nii.gz")
            # imagesTr_normed folder contains the intensisty histogram-standardized TOF-MRA images.
            normed = join(downloaded_data_dir, "imagesTr_normed", patient_name + ".nii.gz")
            seg = join(downloaded_data_dir, "labelsTr", patient_name + ".nii.gz")

            assert all([
                isfile(ori),
                isfile(normed),
                isfile(seg)
            ]), "%s" % patient_name

            shutil.copy(ori, join(target_imagesTr, patient_name + "_0000.nii.gz"))  # the raw image channel=0000
            shutil.copy(normed,
                        join(target_imagesTr, patient_name + "_0001.nii.gz"))  # the standardized image channel=0001
            if not os.path.exists(join(target_labelsTr, patient_name + ".nii.gz")):
                shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))

            if task == new_task_name:  # we should remove the original raw image
                os.remove(ori)
                os.remove(normed)

    patient_namesTs = []
    for tpe in ["imagesTs"]:
        cur = join(downloaded_data_dir, tpe)
        for p in sorted(os.listdir(cur)):
            patient_name = p[:-7]
            patient_namesTs.append(patient_name)
            ori = join(cur, patient_name + ".nii.gz")
            normed = join(downloaded_data_dir, "imagesTs_normed", patient_name + ".nii.gz")

            assert all([
                isfile(ori),
                isfile(normed),
            ]), "%s" % patient_name

            shutil.copy(ori, join(target_imagesTs, patient_name + "_0000.nii.gz"))  # the raw image channel=0000
            shutil.copy(normed,
                        join(target_imagesTs, patient_name + "_0001.nii.gz"))  # the standardized image channel=0001

    json_dict = OrderedDict()
    json_dict['name'] = "COSTA"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = "Research Only"
    json_dict['release'] = "1.0"
    json_dict['modality'] = {
        "0": "MRA",
        "1": "MRA",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "artery",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = len(patient_namesTs)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in patient_namesTs]

    save_json(json_dict, join(target_base, "dataset.json"))

    del tpe, cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", type=int, required=True, default=1, help="Task id of the dataset")
    args = parser.parse_args()
    convert_dataset_to_cesar_format(args.task_id)


if __name__ == "__main__":
    main()
