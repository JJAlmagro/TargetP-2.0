# TargetP-2.0: Detecting Sequence Signals in Targeting Peptides Using Deep Learning

## Synopsis
Repository with the code used to train and test the tool TargetP-2.0.

## Authors
J. J. Almagro Armenteros, M. Salvatore, O. Emanuelsson, O. Winther, G. von Heijne, A. Elofsson, H. Nielsen.

## Software requirements
Python 3 and Tensorflow 1.7 where used to train and test the model. 

## Data

The protein sequences are encoded in BLOSUM62 and trimmed up to 200 amino acids. The sequences are stored in one `npz` file named `targetp_data.npz` in the data folder.

## Training

The training is performed running the script `train.py`. This is a minimal example:

`python train.py -d targetp_data.npz`

By default, the training will run on the CPU. To select the GPU where to run the training, for example GPU 0, use:

`python train.py -d targetp_data.npz -g 0`

## Testing
Once the training is finished run the `test.py` script to get the final performance of the model. It is necessary to define the folder where the trained models have been saved.
`python test.py -d targetp_data.npz -m saved_models`


