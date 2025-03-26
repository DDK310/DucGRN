# DuCGRN

## Overview

DuCGRN is a novel deep learning framework for inferring Gene Regulatory Networks (GRNs) from single-cell sequencing data. The model leveraging Graph Neural Networks (GNNs) to capture transcription factor (TF)-gene interactions.

## Requirements Packages

```
python 3.9
numpy 1.23.5
pandas 2.1.4
pytorch 2.1.1
```



## **Installation**

### **1. Clone the Repository**

```
git clone https://github.com/DDK310/DuCGRN.git
cd DuCGRN
```

### 2. **Set Up the Environment**

We recommend using Python **3.9** with the required dependencies.

```
pip install -r requirements.txt
```

## **Usage**

### **1. Data Preparation**

Prepare your single-cell RNA sequencing dataset in the required format. See data/README.md for details.

### **2. Train the Model**

```
python train.py 
```

### 3. Configuration

Please see `train.py` for detailed configurations.



