# **COSTA**: A Multi-center Multi-vendor TOF-MRA Dataset and A Novel Cerebrovascular Segmentation Network

## COSTA Dataset
![COSTA](./assets/costa.png)

The COSTA dataset contains TOF-MRA images acquired at two different magnetic field strengths (1.5T and 3.0T), including 423 TOF-MRA images from 8 disparate data centers, which utilize MRI scanners from 4 distinct vendors. ```Six``` subsets of COSTA with manual annotations are online available. Please go to [HERE](https://imed.nimte.ac.cn/costa.html) to access the COSTA dataset and for more information.

## CESAR: CErebrovaSculAR segmentation network

### 1. **Requirements**
To successfully run the COSTA framework, please ensure the following requirements are met:
- Operating System: Linux (Ubuntu 20.04)
- Graphics Processing Unit (GPU): NVIDIA RTX 3090 with 24 GB VRAM
- CUDA: It is recommended to have CUDA version 12.0 installed for optimal performance.

### 2. **Installation**
To install the necessary components for COSTA, please follow the steps below:

- Create a new Python 3.9 environment named COSTA using Conda:
```shell
conda create -n COSTA python=3.9
```

- Activate the COSTA environment:
```shell
conda activate COSTA
```

- Clone the COSTA repository from GitHub:
```shell
git clone https://github.com/iMED-Lab/COSTA.git
```

- Navigate to the COSTA directory:
```shell
cd ./COSTA
```

- Install the required dependencies:
```shell
pip install -e .
```

After running these commands, the CESAR network and nnUNet will be installed automatically. The terminal commands available include:
- `COSTA_plan_and_preprocess`: Executes the data preprocessing pipeline used in this work.
- `COSTA_train`: Trains the CESAR network.
- `COSTA_predict`: Performs cerebrovascular segmentation.
- Additionally, all nnUNet commands are available.

Finally, please follow the instructions provided in the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) repository to set up the necessary data environment variables according to their guidelines.

### 3. **Data Preparation**
- Click [HERE](https://imed.nimte.ac.cn/costa.html) to request the download of the COSTA dataset.

- Skull Stripping (For your own dataset only) \
 The first step involves performing skull stripping using the [Brain Extraction Tool (BET)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) to remove the non-brain regions from the TOF-MRA images. [NOTE: THIS STEP IS EXTREMELY IMPORTANT!]

- Intensity Histogram Standardization \
Use the `COSTA/costa/utils/intensity_histogram_standardization.py`  script to obtain histogram matching configurations based on the `imagesTr` set. This will generate a configuration file named `landmarks.pth`, which can be used to perform intensity standardization. Alternatively, you can perform intensity standardization using the pre-trained `landmarks.pth` provided by us.

- Dynamic Voxel Spacing Resampling (DyR) \
The implementation of DyR can be found in `COSTA/costa/preprocessing/preprocessing.py` (lines 240 - 250). DyR is utilized to automatically resample the training, validation, and test sets during data preprocessing, model training, and inference.

- Dataset Format Conversion \
To convert the dataset format according to the principles of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), use the `COSTA/costa/dataset_conversion/COSTA_Dataset_Convert_2023.py` script. Make sure to modify the dataset paths in the following code snippet with your own paths:
```
if __name__ == "__main__":
    ...
    nnUNet_raw_data = "XXXX/nnUNet_raw_data"
    task_name = "Task099_COSTA"
    download_data_dir = "XXXX/nnUNet_raw_data/Task99_COSTA"
    ...
```

### 4. **Experiment Planning**
To preprocess the COSTA dataset in a Linux terminal, please utilize the following command:
```python
COSTA_plan_and_preprocess -t XX # XX is the Task ID, e.g., 99
```

### 5. **Run Training**
```
CUDA_VISIBLE_DEVICES=0 COSTA_train CESAR COSTA 99 0 --use_ssl_pretrained=True

# CESAR: model name
# 99: Task ID No.
# 0: fold number (0, 1, 2, 3, 4)
# --use_ssl_pretrained: whether to use self-supervised learning (SSL) pretrained weights or not. True or False (defult is False)
```
The SSL pretrained weights can be download form [MONAI Project](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain#pre-trained-models). Or you can download it from [Google Drive](https://drive.google.com/drive/folders/1tN9mYEmXcIrYX2ir1QjUZqxr-IZQ65Jo?usp=share_link)

### 6. **Inference**
There are two options available for performing cerebrovascular segmentation: 

Option 1:
- Train the CESAR model from scratch using the provided steps.

Option 2:

- Download the pre-trained CESAR model from the [Google Drive](https://drive.google.com/drive/folders/1HDL2CrqWldkNiFlVnPFTw79bPcHEZw82?usp=share_link).
- Place the downloaded model files in the ```nnUNet_trained_models/nnUNet/CESAR/``` folder. 

Finally, execute the following commands in a Linux terminal to run the inference process:
```
CUDA_VISIBLE_DEVICES=0 COSTA_predict -i /Path/to/your/own/TOF-MRA/files/ -o /Path/to/save/the/predictions/ -t 99 -m CESAR -f 0 -chk model_best
# The value of -f can be 0, 1, 2, 3, 4
```

### 7. **Performance Evaluation**
All evaluation metrics can be found in [DeepMind/surface-distance](https://github.com/deepmind/surface-distance)

### 8. **Potential Usages**
We have constructed the COSTA database, which encompasses heterogeneous Time-of-Flight Magnetic Resonance Angiography (TOF-MRA) images of cerebrovascular structures from multiple centers and vendors. This comprehensive database not only serves as a valuable resource for developing models for cerebrovascular structure segmentation but also provides a precious asset for evaluating the generalization performance of segmentation models on cross-center/cross-device data. 

The COSTA database offers a diverse collection of TOF-MRA images, incorporating variations arising from different imaging centers and vendors. This diversity reflects real-world scenarios and enhances the applicability of segmentation models developed using this database. Researchers can utilize this resource to train, validate, and refine their segmentation algorithms, ensuring their effectiveness across various imaging setups and improving clinical outcomes. 

Moreover, the COSTA database presents an opportunity for interested researchers to explore domain adaptation and domain generalization methods. By utilizing this database, researchers can investigate techniques that facilitate the transfer of knowledge from one imaging domain to another or enhance the performance of segmentation models on unseen data from different centers or devices. This enables the development of robust and versatile segmentation algorithms that can be applied in a wide range of clinical settings.

### 9. **Citation**
If you find this work useful to you, feel free to cite the following reference:
```
Citation information
```

### Usef links
[![](https://img.shields.io/badge/Dataset-TubeTK-blue)](https://public.kitware.com/Wiki/TubeTK/Data)
[![](https://img.shields.io/badge/Dataset-IXI%20Dataset-blue)](http://brain-development.org/ixi-dataset/)
[![](https://img.shields.io/badge/Dataset-ADAM%20Challenge-blue)](https://adam.isi.uu.nl/)
[![](https://img.shields.io/badge/Dataset-ICBM-blue)](https://www.nitrc.org/projects/icbmmra)
[![](https://img.shields.io/badge/Software-3D%20Slicer-orange)](https://www.slicer.org/)
[![](https://img.shields.io/badge/Software-Brain%20Extraction%20Tool%20(BET)-orange)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET)
[![](https://img.shields.io/badge/Software-Histogram%20standardization-orange)](https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.HistogramStandardization)
[![](https://img.shields.io/badge/Software-batchgenerators-orange)](https://pypi.org/project/batchgenerators/)
[![](https://img.shields.io/badge/Software-VTK%3A%20Surface%20generation-orange)](https://examples.vtk.org/site/Python/Medical/GenerateModelsFromLabels/)

---
## **Acknowledgements**
The model was trained, validated, and tested using the nnUNet framework. The SSL pre-trained weights were obtained from Project-MONAI, and we would like to express our sincere gratitude to [DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [Project-MONAI/research-contributions](https://github.com/Project-MONAI/research-contributions) for their contributions and support in providing these valuable resources..
