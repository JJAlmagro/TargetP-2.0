# TargetP-2.0: Detecting Sequence Signals in Targeting Peptides Using Deep Learning

## Synopsis
Repository with the code used to train and test the tool TargetP-2.0.

## Authors
J. J. Almagro Armenteros, M. Salvatore, O. Emanuelsson, O. Winther, G. von Heijne, A. Elofsson, H. Nielsen.

## Software requirements
Python 3 and Tensorflow 1.7 where used to train and test the model. Additional packages are numpy (processing of the data) and sklearn (metrics).

## Data

The protein sequences are encoded in BLOSUM62 and trimmed up to 200 amino acids. The sequences are stored in one `npz` file named `targetp_data.npz` in the data folder. The file contains the following arrays:
+ Protein sequences (`x`) encoded in BLOSUM62 and trimmed up to 200 amino acids. Sequences smaller than 200 amino acids are padded.
+ Peptide types (`y_type`) that the input protein can be classified as. They can be noTP `0`, SP `1`, mTP `2`, cTP `3` or luTP `4`.
+ Cleavage site (`y_cs`) is the position where the sorting signal is cleaved. This is enconded as a zero vector of length 200 with `1` in the cleavage site position. 
+ Protein organism (`org`) defines whether the protein is from plant `1` or non-plant `0`. 
+ Sequence length (`len_seq`) of the proteins, being the maximum length 200.
+ Partition assignment (`fold`) of each example. The protein data was divided in 5 partitions based on their sequence similarity. This vector contains the partition that each protein belongs to, which is used in the cross-validation procedure. 
+ Accession number (`ids`) of the proteins. 

## Training

The training is performed running the script `train.py`. This is a minimal example:

`python train.py -d targetp_data.npz`

By default, the training will run on the CPU. To select the GPU to run the training, for example GPU 0, use:

`python train.py -d targetp_data.npz -g 0`

## Testing
Once the training is finished run the `test.py` script to get the final performance of the model. It is necessary to define the folder where the trained models have been saved.
`python test.py -d targetp_data.npz -m saved_models`


