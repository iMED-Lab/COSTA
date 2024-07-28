<img src="./assets/logo.png#pic_center" style="zoom:25%;" />

## COSTA: A Multi-center Multi-vendor TOF-MRA Dataset and A Novel Cerebrovascular Segmentation Network

![](assets/costa.png)

COSTA dataset download from [here](https://imed.nimte.ac.cn/costa.html) or [Zenodo link](https://doi.org/10.5281/zenodo.11025761)

### 1. **Requirements**

To successfully run the COSTA framework, please ensure the following requirements are met:

<center>Ubuntu 20.04 LTS + NVIDIA RTX 3090 + CUDA version 12.0</center>

### 2. **Installation & Quick Start**

To install the necessary components for COSTA, please follow the steps below:

- Create a new Python 3.9 environment named COSTA using Conda:
  
  ```bash
  conda create -n COSTA python=3.9 # Python 3.8 or Python 3.10 is also acceptable.
  ```
  
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

1. `COSTA_brain_extraction`: Skull stripping performed with the [BET2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET) toolbox (please ensure that the [BET2(FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) toolbox is correctly installed). 

   > COSTA_brain_extraction -i input_folder -f 0.04 -g 0.00

   After this command runs, "SkullStripped" and "BrainMask" folders are generated in the same level directory as input_dir, containing the TOF-MRA images after skull stripping and brain masks, respectively. masks, respectively.

2. `COSTA_train_landmarks`: Acquire the landmark configuration necessary for the intensity histogram standardization process.

   > COSTA_train_landmarks -i input_folder -m brain_mask_folder

   After this command, the intensity histogram standardization landmarks configuration will be saved to `costa/preprocessing/hist_standardization/landmarks.pth`. 

3. `COSTA_standardization`: Intensity histogram standardization based on generated landmarks based on step 2.

   > COSTA_standardization -i input_folder -o output_dir[optional]

   A folder named "XXX_normed" will be generated along the `input_dir`, "XXX" is the folder name of `input_dir` if `-o` not specificed. The "XXX_normed" folder contains the intensity histogram standardized TOF-MRA images. 

4. `COSTA_convert_dataset`: Transforms data into the nnUNet-like format.

   > COSTA_convert_dataset -t TaskID

5. `COSTA_plan_and_preprocess`: Executes the data preprocessing pipeline used in this work.

   > COSTA_plan_and_process -t TaskID

6. `COSTA_train`: Trains the CESAR network.

   > COSTA_train -net NetworkName -tr TrainerName -t TaskID -f Fold --use_ssl_pretrained=Ture/False

7. `COSTA_predict`: Performs cerebrovascular segmentation.

   > COSTA_predict -i input_folder -o output_folder -t TaskID -tr TrainerName -m NetworkName -f Fold -chk model_best_is_default

8. `COSTA_plan_inference_input`: Prepare the input format incorporating intensity histogram standardization for utilization when employing the trained model(s) to predict external data. Please ensure prior skull stripping of the TOF-MRA images.

   > COSTA_plan_inference_input -i input_folder

9. Additionally, all `nnUNet` commands are available.

Finally, please follow the instructions provided in the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) repository to set up the necessary data environment variables according to their guidelines.

### 3. **Data Preparation**

#### 3.1 Skull Stripping (For your own dataset only)

  The first step involves performing skull stripping using the [Brain Extraction Tool (BET2)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) (or [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and [iCVMapp3r](https://icvmapp3r.readthedocs.io/en/latest/)) to remove the non-brain regions from the TOF-MRA images. The [BET2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) is recommended.

```bash
COSTA_brain_extraction -i INPUT_DIR [-f 0.04 -g 0.05]
```

  The -f and -g options are discretionary. For further details on the usage of -f and -g, kindly refer to the [BET User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide). Following the skull stripping process, two files, namely `SkullStripped` and `BrainMask`, will be generated in the same directory as the input folder. These files contain the TOF-MRA image and the corresponding brain mask after skull stripping, respectively.

#### 3.2 Train Landmarks Configuration

  In the second phase, utilize the TOF-MRA images post-skull stripping to train the required landmarks for intensity histogram standardization.

```bash
COSTA_train_landmarks -i INPUT_DIR_OF_TOFMRA_IMAGES -m MASK_DIR_OF_BRAIN_MASK
```

  Here, `-i` represents the path to the TOF-MRA image post-skull stripping, and `-m` corresponds to the path of the associated brain mask. Following this pipeline, a file named `landmarks.pth` is generated within the `costa/preprocessing/hist_standardization/` directory.

#### 3.3 Intensity Histogram Standardization

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

#### 3.4 Dataset Format Conversion

  To convert the dataset format according to the principles of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), use the 

```bash
COSTA_convert_dataset -t Task_ID # e.g. COSTA_convert_dataset -t 99
```

This process will automatically explore the directories mentioned above, based on the task_id, and execute a format conversion to create a new repository for raw images used in model training. This repository will be generated at the same directory level as TaskXX_XXX and will include the following:

- `imagesTr`: Original TOF-MRA images, with filenames ending in `_0000.nii.gz`, and normalized TOF-MRA images, with filenames ending in `_0001.nii.gz`.
- `imagesTs`: Same as above.
- `labelsTs`: Ground truth data for training models.
- `dataset.json`

### 4. Data Preparation (for COSTA dataset)

- Download COSTA dataset from [https://imed.nimte.ac.cn/costa.html](https://imed.nimte.ac.cn/costa.html)

- The structure of the download dataset is as follows:

  > TaskXX_XXX
  > ├── imagesTr
  > ├── imagesTs
  > ├── labelsTr
  > └── labelsTs

  In light of this, we should employ the following terminal command to carry out histogram standardization:

  > `COSTA_standardization -i ./imagesTr` 
  >
  > and
  >
  > `COSTA_standardization -i ./imagesTs`

  This command will generate `imagesTr_normed` and `imagesTs_normed`, each containing the standardized training and testing images.

### 5. **Experiment Planning**

To preprocess the COSTA dataset in a Linux terminal, please utilize the following command:

```python
COSTA_plan_and_preprocess -t XX # XX is the Task ID, e.g., 99
```

### 6. **Run Training**

```bash
CUDA_VISIBLE_DEVICES=0 COSTA_train -net CESAR -tr COSTA -t 99 -f 0 --use_ssl_pretrained=True
```

You can download the SSL pretrained weights from the [MONAI Project](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain#pre-trained-models) and place them in the `costa/ssl_pretrained_weights/` folder.

### 7.  **Run Inference**

There are two options available for performing cerebrovascular segmentation: 

#### Option 1:

- Model train: train the CESAR model from scratch using the provided steps.
- Prepare inputs for the trained model: `COSTA_plan_inference_input -i INPUT_TOF-MRA_DIR` . This process will create a folder containing histogram-standardized images (with names ending in "_normed") and a folder named "consta_inputs" with converted COSTA inputs in the sibling directory `INPUT_TOF-MRA_DIR`.
- Perform segmentation: `CUDA_VISIBLE_DEVICES=0 COSTA_predict -i /Path/to/your/own/TOF-MRA/files/ -o /Path/to/save/the/predictions/ -t 99 -m CESAR -f 0 -chk model_best`

#### Option 2:

- Download the pre-trained CESAR model from the [Google Drive](https://drive.google.com/drive/folders/1HDL2CrqWldkNiFlVnPFTw79bPcHEZw82?usp=share_link) or [Zenodo](https://zenodo.org/records/10957925).

- Place the downloaded model files in the ```nnUNet_trained_models/COSTA/CESAR/``` folder. 
  
  Finally, execute the following commands in a Linux terminal to run the inference process:
  
  ```
  CUDA_VISIBLE_DEVICES=0 COSTA_predict -i /Path/to/your/own/TOF-MRA/files/ -o /Path/to/save/the/predictions/ -t 99 -m CESAR -f 0 -chk model_best
  # The value of -f can be 0, 1, 2, 3, 4
  ```

### 8. **Performance Evaluation**

All evaluation metrics can be found in [DeepMind/surface-distance](https://github.com/deepmind/surface-distance). We provided a terminal command to perform evaluation in terms of ASD, HD95, DICE, and clDice.

> COSTA_eval -p PredictionsFolder -g GroundTruthFolder -id WhateverYouLike

This command will produce an overall performance result in a `.txt` file and detailed evaluation results for each individual image collected in an Excel (`.xlsx`) file.

### 9. **Citation**

If you find this work useful to you, feel free to cite the following reference:

```
@ARTICLE{10599360,
  author={Mou, Lei and Yan, Qifeng and Lin, Jinghui and Zhao, Yifan and Liu, Yonghuai and Ma, Shaodong and Zhang, Jiong and Lv, Wenhao and Zhou, Tao and Frangi, Alejandro F. and Zhao, Yitian},
  journal={IEEE Transactions on Medical Imaging}, 
  title={COSTA: A Multi-center TOF-MRA Dataset and A Style Self-Consistency Network for Cerebrovascular Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Image segmentation;Annotations;Magnetic resonance imaging;Image resolution;Feature extraction;Magnetic fields;Hospitals;Multi-center and multi-vector;TOF-MRA;heterogeneity;style self-consistency;cerebrovascular segmentation},
  doi={10.1109/TMI.2024.3424976}}

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

Pretrained CESAR weghts: [Google Drive](https://drive.google.com/drive/folders/1HDL2CrqWldkNiFlVnPFTw79bPcHEZw82?usp=share_link).

### **Acknowledgements**

The model was trained, validated, and tested using the nnUNet framework. The SSL pre-trained weights were obtained from Project-MONAI, and we would like to express our sincere gratitude to [DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [Project-MONAI/research-contributions](https://github.com/Project-MONAI/research-contributions) for their contributions and support in providing these valuable resources.

### References:

1. *F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, 2021.*
2. *La. G. Nyul, J. K. Udupa, and X. Zhang, “New variants of a method of mri scale standardization,” IEEE Transactions on Medical Imaging, vol. 19, no. 2, pp. 143–150, 2000.*
3. *S.M. Smith. Fast robust automated brain extraction. Human Brain Mapping, 17(3):143-155, November 2002.*
4. *M. Jenkinson, M. Pechaud, S. Smith, et al., “BET2: Mr-based estimation of brain, skull and scalp surfaces,” in Eleventh Annual Meeting of the Organization for Human Brain Mapping. Toronto., 2005, vol. 17, p. 167.*
