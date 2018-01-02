# entropy-sgd-tf
TensorFlow implementation of [Entropy SGD: Biasing gradient descent into wide valleys](https://arxiv.org/pdf/1611.01838.pdf). Entropy-SGD uses geometric information about the energy landscape to bias the optimization algorithm toward flat regions, which may aid generalization.

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
The CIFAR-10 model is trained with the same hyperparameters specified in the original paper. Hyperparameters for CIFAR-100 are optimized using Hyperband. Convergence when training on both datasets is compared with vanilla SGD, Adam and momentum-based optimizers.

### CIFAR-10
```
# Plots showing convergence of entropy-sgd v. sgd, Adam here.
```
### CIFAR-100
```
# Plots showing convergence of entropy-sgd v. sgd, Adam here.
```

## Dependencies
* Python 3.6
* [TensorFlow 1.4](https://www.tensorflow.org/)

## Related work
* [Original lua implementation](https://github.com/ucla-vision/entropy-sgd).
* [Simplfying neural nets by discovering flat minima](https://papers.nips.cc/paper/899-simplifying-neural-nets-by-discovering-flat-minima.pdf) Related work by Schmidhuber and Hochreiter.
* [Stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf#cite.RobCas2004a). Used to approximate the expectation in the update step
* [PDEs for optimizing deep neural networks](https://arxiv.org/pdf/1704.04932.pdf). Followup work by Chaudhari et. al
