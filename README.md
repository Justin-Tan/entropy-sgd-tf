# entropy-sgd-tf
TensorFlow implementation of [Entropy SGD: Biasing gradient descent into wide valleys](https://arxiv.org/pdf/1611.01838.pdf). 

## Instructions
The model should automatically download and process the CIFAR-10 dataset when first run.
```
$ git clone https://github.com/Justin-Tan/entropy-sgd-tf.git
$ cd entropy-sgd-tf
# Check command line arguments
$ python3 train.py -h
# Run
$ python3 train.py <args> 
```
## Results
```
# Plots showing convergence of entropy-sgd v. sgd, Adam here.
```

## Dependencies
* Python 3.6
* [TensorFlow 1.4](https://www.tensorflow.org/)

## Related work
* [Original lua implementation](https://github.com/ucla-vision/entropy-sgd).
* [Simplfying neural nets by discovering flat minima](https://papers.nips.cc/paper/899-simplifying-neural-nets-by-discovering-flat-minima.pdf) Related work by Schmidhuber and Hochreiter.
