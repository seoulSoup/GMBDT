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

Gaussian Mixture Binary Decision Tree is anomaly detection model for numerical datasets.

It's very inspired by [Isolation Forest](https://ieeexplore.ieee.org/document/4781136). (But It's not an ensemble forest model). 

It' just tree now. But I'm working to make this tree to the forest...

## Core Cencepts

Anomalous data is easier to isolate than normal data(by Isolation Forest).

Number of components in Gaussian Mixture Model is fixed to 2.
The model sepreates dataset to 2 clusters recursively. In 2 clusters, larger size cluster is target of recursive seperating.
(larger size cluster is treated as having more possibility to be a normal dataset)
  
The earlier data classified into smaller one has more possibility to predict as anomaly.
  
The anomaly score is decided by Log Likelyhood Ratio Test p-value and probability to be in a larger cluster.
The Model predicts anomaly by calculation of anomaly score matrix and weight array.
The anomaly score matrix is concatenated matrix of anomaly scores from each depth of tree.
The weight array makes more weight to seperated data from shallow depth of tree.
  
## Note

1. It's not an ensemble model.
2. Bayesian Gaussian Mixture Model based.
3. Supports 1 dimensional datasets.
