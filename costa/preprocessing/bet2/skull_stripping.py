# Created by: COSTA
import os
import glob
import tqdm
import argparse
import shutil

'''
You should install BET and BET2 from FSL installation guides:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default=None, help='input dir')
    parser.add_argument('-f', type=float, default=0.04, help='f')
    parser.add_argument('-g', type=float, default=0.00, help='g')

    args = parser.parse_args()
    input_dir = args.input_dir
    f = str(args.f)
    g = str(args.g)

    input_dir = os.path.abspath(input_dir)
    if input_dir is None:
        raise ValueError('input_dir should not be None')

    output_dir = os.path.join(os.path.dirname(input_dir), "SkullStripped")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("You should install BET and BET2 from FSL installation guides: \n"
          "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation\n")
    print("Brain Extraction Start (BET2) ...")
    for file in tqdm.tqdm(glob.glob(os.path.join(input_dir, '*.nii.gz'))):
        save_file = os.path.join(output_dir, os.path.basename(file))
        cmdStr = 'bet2' + ' ' + file + ' ' + save_file + ' -f ' + f + ' -g ' + g + ' -m'
        os.system(cmdStr)

    # The brain mask and the skull stripped images are stored in one folder
    # So, I need to move the brain mask to another folder, named "BrainMask"
    output_brainmask_dir = os.path.join(os.path.dirname(input_dir), "BrainMask")
    if not os.path.exists(output_brainmask_dir):
        os.makedirs(output_brainmask_dir)
    for file in glob.glob(os.path.join(output_dir, '*mask.nii.gz')):
        src_file = file
        dst_file = os.path.join(output_brainmask_dir, os.path.basename(file)[:-12])
        shutil.move(src_file, dst_file)
