# session-based-recommendation

This repository contains code and data to train and test a few recent session-based recommendation models, mainly using the pytorch and Deep Graph Library([DGL](https://github.com/dmlc/dgl)).<br/>
The following session-base recommendation models were implemented on both external(yoochoose & diginetica) and internal data(Amex-log): <br/>

(1)	NARM : [Neural Attentive Session-based Recommendation](https://arxiv.org/pdf/1711.04725.pdf) <br/>
        <br/>
(2)	 SRGNN : [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/pdf/1811.00855.pdf) <br/>
        <br/>
(3)	 NISER : [Normalized Item and Session Representations to Handle Popularity Bias](https://arxiv.org/pdf/1909.04276.pdf) <br/>
        <br/>
(4)	 TAGNN: [Target Attentive Graph Neural Networks for Session-based Recommendation](https://arxiv.org/pdf/2005.02844.pdf) <br/>
        <br/>
(5)	 LESSR : [Handling Information Loss of Graph Neural Networks for Session-based Recommendation](https://www.cse.ust.hk/~raywong/paper/kdd20-informationLoss-GNN.pdf) <br/>
        <br/>
(6)	 MSGIFSR : [Learning Multi-granularity User Intent Unit for Session-based Recommendation](https://arxiv.org/pdf/2112.13197.pdf) <br/>
        <br/>
## Dataset
Download and extract the following datasets and put the files in the corresponding folder <br/>
Experiments of session-based recommender system used both yoochoose/diginetica(external) data and Amex-log (Internal) data. <br/>
* Yoochoose(https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
* Diginetica(https://competitions.codalab.org/competitions/11161#learn_the_details-data2) 
* MOBI Amex-explorpoi-poi_catogory

(1) run the notebook to prepreocess yoochoose data in folder **YOOCHOOSE_data**<br/>
```python 
Data_preprocess.ipynb
```
(2) run the notebook to prepreocess diginetica data in folder **diginetica_data**<br/>
```python 
Data_preprocess.ipynb
```
(3) run the notebook to prepreocess Amex-explorpoi-poi_catogory data in folder **dataset**<br/>
```python 
run_preprocess.ipynb
```
Some standard data preprocessing strategies were utilized to clean session sequential data. <br/>
* Filter out items appearing less than 5 times and sessions consisting of less than 2 items
* Split out training and test set based on sessions of the last day.
* Ignore the clicked items that do not appear in the training set (Cold start issue)

## How to use
first create the enveriment.
```
conda env create -f environment.yaml
```
then in each model folder 
```
bash run.sh
```
