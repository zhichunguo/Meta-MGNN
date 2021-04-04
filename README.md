# Few-shot Graph Learning for Molecular Property Prediction

## Introduction
This is the source code and dataset for the following paper: 

**Few-shot Graph Learning for Molecular Property Prediction. In WWW 2021.**

Contact Zhichun Guo (zguo5@nd.edu), if you have any questions.

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

```
python main.py
```

### Result
The performance of meta-learning is not stable. We run the experiments for two times. And we report the average results and the number of the iteration where we obtain the best result here for your reference.

## Reference

```
@inproceedings{guo2020graseq,
  title={Few-shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  booktitle={WWW},
  year={2021}
}
```
