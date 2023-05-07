# **COSTA**: A Multi-center Multi-vendor TOF-MRA Dataset and A Novel Cerebrovascular Segmentation Network

## COSTA Dataset
![COSTA](./assets/costa.png)

The COSTA dataset contains TOF-MRA images acquired at two different magnetic field strengths (1.5T and 3.0T), including 423 TOF-MRA images from 8 disparate data centers, which utilize MRI scanners from 4 distinct vendors. ```Six``` subsets of COSTA with manual annotations are online available. Please go to [HERE](https://imed.nimte.ac.cn/costa.html) to access the COSTA dataset and for more information.

---

## CESAR: CErebrovaSculAR segmentation network
Our code and trained models will be made publicly available following acceptance of the paper.

### Getting started
#### Requirements
- Linux (Ubuntu 20.04)
- NVIDIA RTX 3090 (24 GB), CUDA (12.0 is recommended)
- [PyTorch](https://pytorch.org/) (version >=1.7.0)
- [SimpleITK](https://simpleitk.org/), [MONAI](https://monai.io/)

#### Installation
```
git clone https://github.com/iMED-Lab/COSTA.git
cd xxx
pip install -e .
```

#### Dataset format
We follow the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) dataset format. Or you cluld use the script (```xxx```) to convert your dataset format.

#### Preprocessing pipeline
The input of the CESAR model needs to be preprocessed as the following steps:
1. Skull Stripping: You can use [Brain Extraction Tool (BET)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) to remove the skull in a TOF-MRA image.
2. Intensity histogram standardization: You can perform intensity histogram standardization based on the script located in ```xxx```.
3. Dynamic voxel spacing resampling (DyR): The DyR is implemented in ```xxx```