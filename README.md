# GMBDT
Gaussian Mixture Binary Decision Tree
<div align="center">
<p>
   <img width="600" src="https://user-images.githubusercontent.com/99949549/155826201-2f3814ce-d6ca-487a-a320-1d1ba4fadb3b.PNG"></a>
</p>

<div align="left">
  
## Installation
All python dependencies are in [`requirements.txt`](requirements.txt).
```
pip install -r requirements.txt
```  
## Introduction

Gaussian Mixture Binary Decision Tree (GMBDT) is anomaly detection model for numerical datasets.

GMBDT is very inspired by [Isolation Forest](https://ieeexplore.ieee.org/document/4781136) (But It's not an ensemble forest model). 

GMBDT is just tree model now, but I'm working to make this tree to the forest.

## Core Concepts

Anomalous data is easier to isolate than normal data by Isolation Forest.

The number of components in Gaussian Mixture Model is fixed to 2.
The model sepreates dataset to two clusters recursively. The larger size cluster which is bigger than the other one is target of recursive seperating.
(Assuming the larger size cluster has more possibility to be a normal dataset)
  
The earlier data classified into smaller one has more possibility to predict as anomaly.
  
The anomaly score is decided by Log Likelyhood Ratio Test p-value and probability to be in a larger size cluster.
GMBDT predicts anomaly from calculation of anomaly score matrix and weight array.
The anomaly score matrix is concatenated matrix of anomaly scores from each depth of tree.
The weight array makes more weight to seperated data from shallow depth of tree.
  
## Note

1. It's not an ensemble model.
2. Bayesian Gaussian Mixture Model based.
3. Supports 1 dimensional datasets.
