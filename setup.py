from setuptools import setup, find_namespace_packages

setup(name='costa',
      packages=find_namespace_packages(include=["costa", "costa.*"]),
      version='1.0.1',
      description="COSTA: Novel multi-center TOF-MRA cerebrovascular segmentation dataset and netwokr",
      url="https://github.com/iMED-Lab/COSTA",
      license='Apache License Version 2.0, January 2004',
      install_requires=[
          "einops",
          "monai",
          "nnunet",
          "torch==2.0.0",
          "torchio",
          "thop",
          "prettytable",
          "rich",
          "openpyxl"
          "numpy==1.23.2"
      ],
      entry_points={
          'console_scripts': [
              'COSTA_plan_and_preprocess = costa.experiment_planning.COSTA_plan_and_preprocess:main',  # 5
              'COSTA_train = costa.run.run_training:main',  # 6
              'COSTA_predict = costa.inference.predict_sample:main',  # 8
              'COSTA_train_landmarks = costa.preprocessing.hist_standardization.train_landmarks:main',  # 2
              "COSTA_standardization = costa.preprocessing.hist_standardization.intensity_hist_stand:main",  # 3
              "COSTA_convert_dataset = costa.dataset_conversion.COSTA_dataset_convert:main",  # 4
              "COSTA_plan_inference_input = costa.inference.plan_inference_input:main",  # 7
              "COSTA_brain_extraction = costa.preprocessing.bet2.skull_stripping:main",  # 1
              "COSTA_eval = costa.run.run_evaluation:main"
          ],
      },
      )
