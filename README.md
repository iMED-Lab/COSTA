# COSTA package for MRAtoBG_brain_vessel_segmentation

---

## ⚠️ System Requirements

Please ensure your system meets these requirements for full compatibility:

- **OS**: Ubuntu 20.04
- **Python**: 3.8
- **GPU**: NVIDIA V100
- **CUDA**: Version 11.7

---


## 2. **Requirements Installation**

Before proceeding, ensure you have completed step `1. Clone the Repository` [here](https://github.com/jshe690/MRAtoBG_brain_vessel_segmentation).

Once the `MRAtoBG_brain_vessel_segmentation` repo is ready, set up its system variable:

```bash
export MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH=/user/repos/MRAtoBG_brain_vessel_segmentation
```

To install the necessary COSTA packages for `MRAtoBG_brain_vessel_segmentation`, please follow these steps:

- Create a new Python 3.8 environment named MRAtoBG-brain-vessel-segmentation at:

  ```bash
  python -m venv $MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv
  ```

- Activate the environment:

  ```bash
  source MRAtoBG_BRAIN_VESSEL_SEGMENTATION_PATH/venv/bin/activate
  ```

- Clone this COSTA repository for MRAtoBG_brain_vessel_segmentation:

  ```bash
  git clone https://github.com/jshe690/COSTA.git
  ```

- Navigate to the COSTA directory:

  ```bash
  cd ./COSTA
  ```

- Install the required dependencies:

  ```shell
  pip install -e .
  ```
