# scDFC: Deep Fusion Clustering for Single-Cell RNA-seq Data

[![Python](https://img.shields.io/badge/Python-3.6.2-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-1.12.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**scDFC** is a novel deep fusion clustering framework designed specifically for Single-Cell RNA-seq (scRNA-seq) data analysis.

Existing methods either consider the attribute information of each cell or the structure information between different cells independently. In other words, they cannot sufficiently make use of all heterogeneous information simultaneously. To this end, **scDFC** proposes a novel model containing two modules:

1.  **Attributed Feature Clustering Module**: Extracts deep representations of gene expression profiles.
2.  **Structure-Attention Feature Clustering Module**: Captures high-order topological relationships between cells.

More concretely, two elegantly designed autoencoders are built to handle both features regardless of their data types, achieving superior clustering performance.

> **Note**: For detailed methodology, please refer to our paper published in *Briefings in Bioinformatics*.



## 🛠 Requirements

Please ensure your environment meets the following requirements:

* **Python** == 3.6.2
* **Tensorflow** == 1.12.0
* **Keras** == 2.1.0
* **Numpy** == 1.19.5
* **Pandas** == 1.1.5
* **Scipy** == 1.5.4
* **Scikit-learn** == 0.19.0

### Installation
You can install the dependencies using the following command:

```bash
pip install tensorflow==1.12.0 keras==2.1.0 numpy==1.19.5 pandas==1.1.5 scipy==1.5.4 scikit-learn==0.19.0
````

-----

## 📂 Data Availability

We evaluated scDFC on several benchmark scRNA-seq datasets. The original data sources can be accessed via the links below:

| Dataset | Accession / Source | Link |
| :--- | :--- | :--- |
| **Biase** | GSE57249 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249) |
| **Darmanis**| PubMed 26060301 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26060301/) |
| **Enge** | PubMed 28965763 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/28965763/) |
| **Bjorklund** | GSE70580 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580) |
| **Sun (1-3)** | GSE128066 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066) |
| **Fink** | Science Direct | [Article](https://www.sciencedirect.com/science/article/abs/pii/S1534580722004932) |
| **Brown** | GSE137710 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710) |

-----

## 🚀 Usage

### 1\. Data Preparation

The example expression matrix `data.tsv` for the **Biase** dataset is put into `data/Biase`. The directory structure should look like this:

```text
data/
├── Biase/
│   ├── data.tsv
│   └── label.ann
└── [Your_Dataset_Name]/
    ├── data.tsv
    └── label.ann
```

### 2\. Run the Model

To run the model with the default dataset (**Biase**), simply execute:

```bash
python scDFC.py
```

### 3\. Arguments

To change datasets or modify parameters, use the following arguments:

```bash
python scDFC.py --dataset_str Brown --n_clusters 4
```

**Key Parameters in `scDFC.py`:**

```python
parser.add_argument('--dataset_str', default='Biase', type=str, help='name of dataset')
parser.add_argument('--n_clusters', default=3, type=int, help='expected number of clusters')
parser.add_argument('--label_path', default='data/Biase/label.ann', type=str, help='true labels')
```

-----

## 📧 Contact

If you have any questions about the code or the paper, please feel free to contact:

**Dayu Hu**
Email: [hudy@bmie.neu.edu.cn](mailto:hudy@bmie.neu.edu.cn)

-----

## 📝 Citation

If you find this work useful, please consider citing:

**Text:**

> Hu, D., Liang, K., Zhou, S., Tu, W., Liu, M., & Liu, X. (2023). scDFC: A deep fusion clustering method for single-cell RNA-seq data. *Briefings in Bioinformatics*, bbad216. Oxford University Press.

**BibTeX:**

```bibtex
@article{hu2023scdfc,
  title={scDFC: A deep fusion clustering method for single-cell RNA-seq data},
  author={Hu, Dong and Liang, Ke and Zhou, Sihang and Tu, Wenxuan and Liu, Meng and Liu, Xinwang},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={4},
  pages={bbad216},
  year={2023},
  publisher={Oxford University Press}
}
```

```
```
