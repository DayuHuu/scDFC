# scDFC
A Deep fusion clustering method for single-cell RNA-seq data
# Requirements

Python --- 3.6 

Tensorflow --- 1.12.0 

Keras --- 2.1.0

Numpy --- 1.19.5

Scipy --- 1.5.4

Pandas --- 1.1.5

Sklearn --- 0.24.2

# Usage
## inpput 
All the original tested datasets (Biase, Björklund, Brown, Chung, Sun.1, Sun.2, Sun.3 and Habib) can be downloaded.

For example, the original expression matrix data.tsv of dataset Biase is downloaded and put into /data/Biase. Before clustering, low-quality cells and genes can be filtered by running the following command:
