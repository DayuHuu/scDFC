# scDFC
scDFC is a deep fusion clustering method for single-cell RNA-seq data. Existing methods either consider the attribute information of each cell or the structure information between different cells. In other words, they cannot sufficiently make use of all of this information simultaneously. To this end, we propose a novel single-cell deep fusion clustering model, which contains two modules, i.e., an **attributed feature** clustering module and a **structure-attention** feature clustering module. More concretely, two elegantly designed autoencoders are built to handle both features regardless of their data types.

# Requirements

Python --- 3.6.2

Pandas --- 1.1.5

Tensorflow --- 1.12.0 

Keras --- 2.1.0

Numpy --- 1.19.5

Scipy --- 1.5.4

Pandas --- 1.1.5

Scikit-learn --- 0.19.0

# Implement
## The link of datasets 

Biase:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249

Darmanis:https://pubmed.ncbi.nlm.nih.gov/26060301/

Enge:https://pubmed.ncbi.nlm.nih.gov/28965763/

Bjorklund:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580

Sun.1:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066

Fink:https://www.sciencedirect.com/science/article/abs/pii/S1534580722004932

Sun.2:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066

Sun.3:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066

Brown:https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710


## Examples
The example expression matrix data.tsv of dataset Biase is put into data/Biase. To change datasets, you should type the iuput of code:
```python
parser = argparse.ArgumentParser(description='scEMC')
parser.add_argument('--n_clusters', default=data_para['K'], type=int)
parser.add_argument('--lr', default=1, type=float)
```
## Run 
```python
python scDFC.py
```
## Citation 
Please cite our paper:

```python
@article{hu2023scdfc,
  title={scDFC: A deep fusion clustering method for single-cell RNA-seq data},
  author={Hu, Dayu and Liang, Ke and Zhou, Sihang and Tu, Wenxuan and Liu, Meng and Liu, Xinwang},
  journal={Briefings in Bioinformatics},
  pages={bbad216},
  year={2023},
  publisher={Oxford University Press}
}
```

