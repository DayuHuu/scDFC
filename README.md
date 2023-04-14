# scDFC
scDFC is a deep fusion clustering method for single-cell RNA-seq data. Existing methods either consider the attribute information of each cell or the structure information between different cells. In other words, they cannot sufficiently make use of all of this information simultaneously. To this end, we propose a novel single-cell deep fusion clustering model, which contains two modules, i.e., an **attributed feature** clustering module and a **structure-attention** feature clustering module. More concretely, two elegantly designed autoencoders are built to handle both features regardless of their data types.

# Requirements

Python --- 3.6 

Tensorflow --- 1.12.0 

Keras --- 2.1.0

Numpy --- 1.19.5

Scipy --- 1.5.4

Pandas --- 1.1.5

Sklearn --- 0.24.2

# Implement
## input 
The link of datasets 
Biase:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249
Darmanis
Enge
Bjorklund:
Sun.1 
Fink
Sun.2
Sun.3
Brown
Habib

## demo
The example expression matrix data.tsv of dataset Biase is put into data/Biase. To change datasets, you should type the iuput of code:
```python
parser.add_argument('--dataset_str', default='Biase', type=str, help='name of dataset')

parser.add_argument('--n_clusters', default=3, type=int, help='expected number of clusters')

parser.add_argument('--label_path', default='data/Biase/label.ann', type=str, help='true labels')
```

