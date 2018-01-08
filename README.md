# entropy-sgd-tf

TensorFlow implementation of [Entropy SGD: Biasing gradient descent into wide valleys](https://arxiv.org/pdf/1611.01838.pdf). The entropy-SGD optimization algorithm uses geometric information about the energy landscape to bias the optimization algorithm toward flat regions of the loss function, which may aid generalization.

-----------------------------
## Instructions
The CIFAR-10 dataset will be automatically downloaded and converted to tfrecord format when first run. The default is
to run on CIFAR-10 with SGD and Nesterov momentum on a wide residual network (28x10). For a complete list of options run `python3 train.py -h`, e.g. to run
on CIFAR-10 using the entropy-sgd optimizer with 20 Langevin iterations:
```
# Check command line arguments
$ python3 train.py -h
# Run
$ python3 train.py -opt entropy-sgd -L 20
```
The default hyperparameters (used to report all results) can be accessed and set in the `config.py` file under `config_train`. Most should
be self-explanatory. For parameters labelled 'entropy-sgd specific', you may need to refer to the original paper.
Checkpoints and Tensorboard scalars are saved beneath their respective directories. 

### Multi-GPU

Coming soon...

-----------------------------

## Results
Both CIFAR-10/CIFAR-100 models are trained with the same hyperparameters and learning rate schedule specified in the original paper. The dataset is subjected to meanstd preprocessing and random rotations+reflections. Convergence when training on both datasets is compared with vanilla SGD and SGD with Nesterov momentum. The accuracy reported is the average of 5 runs with random weight initialization.

Models trained without entropy-SGD are run for 200 epochs, models trained with entropy-SGD are run with L=20 for 10
epochs, with the hyperparameters specified as in the CIFAR-10 run in the original paper. 

### CIFAR-10
```
# Plots showing convergence of entropy-sgd v. sgd here.
```
### CIFAR-100
```
# Plots showing convergence of entropy-sgd v. sgd here.
```
-----------------------------

## Dependencies
* Python 3.6
* [TensorFlow 1.4](https://www.tensorflow.org/)

## Related work
* [Original lua implementation](https://github.com/ucla-vision/entropy-sgd).
* [Simplfying neural nets by discovering flat minima](https://papers.nips.cc/paper/899-simplifying-neural-nets-by-discovering-flat-minima.pdf) Related work by Schmidhuber and Hochreiter.
* [Stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf#cite.RobCas2004a). Used to approximate the expectation in the update step
* [PDEs for optimizing deep neural networks](https://arxiv.org/pdf/1704.04932.pdf). Followup work by Chaudhari et. al
