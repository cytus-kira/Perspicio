# Perspicio

## Installation

```
pip install -r requirements.txt
```

## Get feature

Before running this code, you need to go to [this link](https://drive.google.com/drive/folders/1Cqp25c-DpbZV9Ozm3MNbOVDD0mC3cyBo?usp=drive_link) to download the `additional_file/model_file.zip` file and extract it to the `get_feature` folder locally.After that, download the dataset from  [this link](https://drive.google.com/drive/folders/1Cqp25c-DpbZV9Ozm3MNbOVDD0mC3cyBo?usp=drive_link) to download the `dataset/dataset.zip`and extract it to the `dataset` folder. Next, you can run `get_gaze_head.py` and modify the features to be extracted for head or gaze. You can extract the other three features by running `get_other_feature.py`.

## Model

To test the model in this file, you can run `python test_model.py` for model testing. If you wish to retrain this model, you can download `train_data.zip` from [this link](https://drive.google.com/drive/folders/1Cqp25c-DpbZV9Ozm3MNbOVDD0mC3cyBo?usp=drive_link) and unzip it into the `/new_feature/final_data` directory. After that, modify the code in the `main` function and execute the `train` function.