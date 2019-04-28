# 2019 Seminar in DS - Itay Levinas
This project implements the method of Graph Convolutinal Matrix Completion, based on the following paper:

Rianne van den Berg, Thomas N. Kipf, Max Welling, [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017)

In addition to the datasets used in the paper, this project is intended to running the method on a new dataset:
[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
## Installation

```python setup.py install```

## Requirements

  * Python 2.7
  * TensorFlow (1.4)
  * pandas


## Previous Usage

To reproduce the experiments mentioned in the paper you can run the following commands:


**Douban**
```bash
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing 
```

**Flixster**
```bash
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```

**Yahoo Music**
```bash
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```

**Movielens 100K on official split with features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing
```

**Movielens 100K on official split without features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing
```

**Movielens 1M**
```bash
python train.py -d ml_1m --data_seed 1234 --accum sum -do 0.7 -nsym -nb 2 -e 3500 --testing
```

**Movielens 10M** 
```bash
python train_mini_batch.py -d ml_10m --data_seed 1234 --accum stack -do 0.3 -nsym -nb 4 -e 20 --testing
```
Note: 10M dataset training does not fit on GPU memory (12 Gb), therefore this script uses a naive version of mini-batching.
Script can take up to 24h to finish.

## Usage On The New Dataset
The parameter choice was not based on optimization. It is a guess of parameters similar to other datasets.
The command --write_summary outputs a TensorFlow summary of the run for a view of the results.

Without side-information:
```bash
python train.py -d book_crossing --accum stack -do 0.7 -nleft -nb 2 -e 200 --testing --write_summary
```

With side-information:
```bash
python train.py -d book_crossing --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 10 --testing --write_summary
```

## View Results
The results are represented in a TensorFlow summary. In order to see them, run the following command in the virtual environment (alternatively, in command line using ```python -m tensorboard.main --logdir=path/to/dir```):
```bash
tensorboard --logdir=path/to/log/dir
# Path example: logs/MovieLens_100K_hidden_feat
```
