# Few-shot Graph Learning for Molecular Property Prediction

## Introduction
This is the source code and dataset for the following paper: 

**Few-shot Graph Learning for Molecular Property Prediction. In WWW 2021.**

Contact Zhichun Guo (zguo5@nd.edu), if you have any questions.

## Datasets
The datasets uploaded can be downloaded to train our model directly.

The original datasets are downloaded from [Data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). We utilize Original_datasets/splitdata.py to split the datasets according to the molecular properties and save them in different files in the Original_datasets/[DatasetName]/new. Then run main.py, the datasets will be automatically preprocessed by loader.py and the preprocessed results will be saved in the Original_datasets/[DatasetName]/new/[PropertyNumber]/propcessed.

## Usage

### Installation
We used the following Python packages for the development by python 3.6.
```
- torch = 1.4.0
- torch-geometric = 1.6.1
- torch-scatter = 2.0.4
- torch-sparse = 0.6.1
- scikit-learn = 0.23.2
- tqdm = 4.50.0
- rdkit
```

### Run code

Datasets and k (for k-shot) can be changed in the last line of main.py.
```
python main.py
```

## Performance
The performance of meta-learning is not stable for some properties. We report two times results and the number of the iteration where we obtain the best results here for your reference.

| Dataset    | k    | Iteration | Property   | Results   || k    | Iteration | Property  | Results   |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  | ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  |
| Sider | 1 | 307/599 | Si-T1| 75.08/75.74 | | 5 | 561/585 | Si-T1 | 76.16/76.47 | 
|  |  | | Si-T2| 69.44/69.34 | |  | | Si-T2 | 68.90/69.77 | 
|  |  | | Si-T3| 69.90/71.39 | |  | | Si-T3 | 72.23/72.35 | 
|  |  | | Si-T4| 71.78/73.60 | |  | | Si-T4 | 74.40/74.51 | 
|  |  | | Si-T5| 79.40/80.50 | |  | | Si-T5 | 81.71/81.87 | 
|  |  | | Si-T6| 71.59/72.35 | |  | | Si-T6 | 74.90/73.34 | 
|  |  | | Ave.| 72.87/73.82 | |  | | Ave. | 74.74/74.70 | 
| Tox21 | 1 | 1271/1415 | SR-HS | 73.72/73.90 | | 5 | 1061/882 | SR-HS | 74.85/74.74 | 
|  |  | | SR-MMP | 78.56/79.62 | |  | | SR-MMP | 80.25/80.27 | 
|  |  | | SR-p53| 77.50/77.91 | |  | | SR-p53 | 78.86/79.14 | 
|  |  | | Ave.| 76.59/77.14 | |  | | Ave. | 77.99/78.05 | 

## Acknowledgements

The code is implemented based on [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns).

## Reference

```
@article{guo2021few,
  title={Few-Shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2102.07916},
  year={2021}
}
```
