# **COSTA**: A Multi-center Multi-vendor TOF-MRA Dataset and A Novel Cerebrovascular Segmentation Network

## CESAR: CErebrovaSculAR segmentation network

### 1. **Requirements**

To successfully run the COSTA framework, please ensure the following requirements are met:

- Operating System: Linux (Ubuntu 20.04)
- Graphics Processing Unit (GPU): NVIDIA RTX 3090 with 24 GB VRAM
- CUDA: It is recommended to have CUDA version 12.0 installed for optimal performance.

### 2. **Installation**

To install the necessary components for COSTA, please follow the steps below:

- Create a new Python 3.9 environment named COSTA using Conda:
  
  ```bash
  conda create -n COSTA python=3.9
  ```
  
  > Python 3.8 or Python 3.10 is also acceptable.

- Activate the COSTA environment:
  
  ```bash
  conda activate COSTA
  ```

- Clone the COSTA repository from GitHub:
  
  ```bash
  git clone https://github.com/iMED-Lab/COSTA.git
  ```

- Navigate to the COSTA directory:
  
  ```bash
  cd ./COSTA
  ```

- Install the required dependencies:
  
  ```shell
  pip install -e .
  ```

After running these commands, the CESAR network and nnUNet will be installed automatically. The terminal commands available include:

- `COSTA_brain_extraction`: Skull stripping performed with the [BET2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET) toolbox (please ensure that the [BET2(FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) toolbox is correctly installed).
- `COSTA_train_landmarks`: Acquire the landmark configuration necessary for the intensity histogram standardization process.
- `COSTA_standardization`: Intensity histogram standardization based on generated landmarks.
- `COSTA_convert_dataset`: Transforms data into the nnUNet-like format.
- `COSTA_plan_and_preprocess`: Executes the data preprocessing pipeline used in this work.
- `COSTA_train`: Trains the CESAR network.
- `COSTA_predict`: Performs cerebrovascular segmentation.
- `COSTA_plan_inference_input`: Prepare the input format incorporating intensity histogram standardization for utilization when employing the trained model(s) to predict external data. Please ensure prior skull stripping of the TOF-MRA images.
- Additionally, all `nnUNet` commands are available.

Finally, please follow the instructions provided in the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) repository to set up the necessary data environment variables according to their guidelines.

### 3. **Data Preparation**

### Skull Stripping (For your own dataset only)
  The first step involves performing skull stripping using the [Brain Extraction Tool (BET2)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) (or [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and [iCVMapp3r](https://icvmapp3r.readthedocs.io/en/latest/)) to remove the non-brain regions from the TOF-MRA images. The [BET2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) is recommended.
  ```bash
  COSTA_brain_extraction -i INPUT_DIR [-f 0.04 -g 0.05]
  ```
  The -f and -g options are discretionary. For further details on the usage of -f and -g, kindly refer to the [BET User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide). Following the skull stripping process, two files, namely `SkullStripped` and `BrainMask`, will be generated in the same directory as the input folder. These files contain the TOF-MRA image and the corresponding brain mask after skull stripping, respectively.

### Train Landmarks Configuration
  In the second phase, utilize the TOF-MRA images post-skull stripping to train the required landmarks for intensity histogram standardization.
  ```bash
  COSTA_train_landmarks -i INPUT_DIR_OF_TOFMRA_IMAGES -m MASK_DIR_OF_BRAIN_MASK
  ```
  Here, `-i` represents the path to the TOF-MRA image post-skull stripping, and `-m` corresponds to the path of the associated brain mask. Following this pipeline, a file named `landmarks.pth` is generated within the `costa/preprocessing/hist_standardization/` directory.

### Intensity Histogram Standardization 
  When conducting skull stripping, it is necessary to create a folder in the `nnUNet_raw_data_base` directory following the `TaskXX_XXX` format. This folder should encompass four subfolders: `imagesTr`, `imagesTs`, `labelsTr`, and `labelsTs`. These folders respectively store the training and test images obtained after skull stripping, as well as the ground truth for training and testing.

  Then, use the
   ```bash
   COSTA_histogram_standardization -i INPUT_DIR [-o OUTPUT_DIR]
   ``` 
   command to obtain histogram standardized images. You can perform intensity standardization using the pre-trained `landmarks.pth` provided by us.
   After completing this step, folders are automatically generated with names starting from the input folder and ending with "_normed" in the `TaskXX_XXX` directory, such as "imagesTr_normed".

   The ultimate directory structure within the TaskXX_XXX folder is as follows:
   ```
   TaskXX_XXX
   ├── imagesTr
   ├── imagesTr_normed
   ├── imagesTs
   ├── imagesTs_normed
   ├── labelsTr
   └── labelsTs
   ```

### Dataset Format Conversion
  To convert the dataset format according to the principles of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), use the 
  ```bash
  COSTA_convert_dataset -t Task_ID # e.g. COSTA_convert_dataset -t 99
  ```
This process will automatically explore the directories mentioned above, based on the task_id, and execute a format conversion to create a new repository for raw images used in model training. This repository will be generated at the same directory level as TaskXX_XXX and will include the following:

- `imagesTr`: Original TOF-MRA images, with filenames ending in `_0000.nii.gz`, and normalized TOF-MRA images, with filenames ending in `_0001.nii.gz`.
- `imagesTs`: Same as above.
- `labelsTs`: Ground truth data for training models.
- `dataset.json`

### 4. **Experiment Planning**

To preprocess the COSTA dataset in a Linux terminal, please utilize the following command:

```python
COSTA_plan_and_preprocess -t XX # XX is the Task ID, e.g., 99
```

### 5. **Run Training**

```
CUDA_VISIBLE_DEVICES=0 COSTA_train CESAR COSTA 99 0 --use_ssl_pretrained=True
```
> CESAR: model name \
> 99: Task ID No. \
> 0: fold number (0, 1, 2, 3, 4) \
> --use_ssl_pretrained: whether to use self-supervised learning (SSL) pretrained weights or not. True or False (defult is False)

The SSL pretrained weights can be download form [MONAI Project](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain#pre-trained-models).

### 6. **Inference**

There are two options available for performing cerebrovascular segmentation: 

#### Option 1:

- Train the CESAR model from scratch using the provided steps.
  - `COSTA_plan_inference_input -i INPUT_TOF-MRA_DIR`: Prepare inputs for the trained model. This process will create a folder containing histogram-standardized images (with names ending in "_normed") and a folder named "consta_inputs" with converted COSTA inputs in the sibling directory `INPUT_TOF-MRA_DIR`.
  - `CUDA_VISIBLE_DEVICES=0 COSTA_predict -i /Path/to/your/own/TOF-MRA/files/ -o /Path/to/save/the/predictions/ -t 99 -m CESAR -f 0 -chk model_best`: Perform cerebrovascular predictions

#### Option 2:

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

### Useful links

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
