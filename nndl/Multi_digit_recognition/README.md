# Multi- Digit Recognition

A TensorFlow implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf) 


## Requirements

* Python 3.5
* Tensorflow
* h5py

    ```
    In Ubuntu:
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```

## Setup

1. Download the source code

2. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

3. Extract to data folder, which should look like:
    ```
    Multi_digit_recognition
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ```


## Usage


1. Create tfrecords for the data

    ```
    $ python create_tfrecords.py --data_dir ./data
    ```

2. Train the model by using the training data

    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train
    ```

3. Can retrain the model by using the checkpoint created from the last trained model
    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train2 --restore_checkpoint ./logs/train/latest.ckpt
    ```

4. Evaluate the model by using test data

    ```
    $ python evaluator.py --data_dir ./data --checkpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

5. Visualize the tensorboard for the model

    ```
    $ tensorboard --logdir ./logs
    ```

6. Can make an inference for the test data or the data from the external data 

    ```
    Can check the inference of test data by opening `inference.ipynb` in Jupyter
    Can check the inference of external data by opening `inference_of_external data.ipynb` in Jupyter
    ```

