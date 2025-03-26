The experimental scRNA-seq datasets were downloaded from the **Gene Expression Omnibus (GEO)** database, with the following accession numbers:

| Dataset  | Accession Number | Cell Type                      |
| -------- | ---------------- | ------------------------------ |
| hHEP     | GSE81252         | Human Hepatocytes              |
| hESC     | GSE75748         | Human Embryonic Stem Cells     |
| mESC     | GSE98664         | Mouse Embryonic Stem Cells     |
| Mouse DC | GSE48968         | Mouse Dendritic Cells          |
| mHSC     | GSE81682         | Mouse Hematopoietic Stem Cells |

Once you have the authority for the dataset, download the dataset and extract the csv files to `data/`  in this project. For specific data processing, please refer to [BEELINE]([Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data | Nature Methods](https://www.nature.com/articles/s41592-019-0690-6)/).We only retained the genes present in the data expression matrix in the Ground Truth networks.

