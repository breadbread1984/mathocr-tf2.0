# mathocr-tf2.0

OCR from manual written math equations to latex base on [math-formula-recognition](https://github.com/jungomi/math-formula-recognition)

## prepare the dataset

The dataset is from [CROHME: Competition on Recognition of Online Handwritten Mathematical
Expressions](https://www.isical.ac.in/~crohme/). We use the formatted version of the dataset made by jungomi which can be downloaded from [here](https://www.floydhub.com/jungomi/datasets/crohme-png).

Generate the dataset in tfrecord format by executing command

```python
python3 create_datasets.py
```

Executing the script successfully, you can get two tfrecord files which are for training and evaluating respectively. If you want to evaluate on older version of the dataset, please change the ground truth path on line 137.

## training

Train the model by executing the following command

```python
python3 train_mathocr.py
```
## convert checkpoint to hdf5 model

Convert the model with the command

```python
python3 save_model.py
```

## convert hdf5 model to saved model

Convert the model with the command

```python
python3 convert_model.py
```

