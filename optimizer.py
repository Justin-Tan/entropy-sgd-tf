"""
Entropy-SGD TensorFlow implementation
Original paper: arXiv 1611.01838
Justin Tan 2017
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops

from sgld import local_entropy_sgld

class EntropySGD(optimizer.Optimizer):

    def __init__(self, params, config={}, iterator, training_phase, global_step,
                 use_locking=False, name='EntropySGD'):
        # Construct entropy-sgd optimizer - ref. arXiv 1611.01838
        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True, L=0,
                 eps=1e-4, g0=1e-2, g1=0, alpha=0.75,
                 lr_prime=0.1)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(use_locking, name)
        self.config = config
        self.iterator = iterator
        self.training_phase = training_phase
        self._global_step = global_step

        # Parameter tensors
        self._lr_tensor = None
        self._lr_prime_tensor = None
        self._epsilon_tensor = None
        self._g0_tensor = None
        self._g1_tensor = None
        self._alpha_tensor = None
        self._wd_tensor = None
        self._momentum_tensor = None

        self.sgld_opt = local_entropy_sgld(eta_prime=lr_prime, epsilon=epsilon,
                                  gamma=gamma, momentum=momentum, alpha=alpha)

    def _prepare(self):
        self._lr_tensor = ops.convert_to_tensor(self.config['lr'],
                                                name="learning_rate")
        self._lr_prime_tensor = ops.convert_to_tensor(self.config['lr_prime'],
                                                      name="learning_rate_prime")
        self._epsilon_tensor = ops.convert_to_tensor(self.config['eps'],
                                                     name="epsilon")
        self._g0_tensor = ops.convert_to_tensor(self.config['g0'], name="gamma_0")
        self._g1_tensor = ops.convert_to_tensor(self.config['g1'], name="gamma_1")
        self._alpha_tensor = ops.convert_to_tensor(self.config['alpha'],
                                                   name="gamma_1")
        self._wd_tensor = ops.convert_to_tensor(self.config['weight_decay'],
                                                   name="decay")
        self._momentum_tensor = ops.convert_to_tensor(self.config['momentum'],
                                                      name="momentum")

    def _create_slots(self, var_list):
        # Manage variables that accumulate updates
        # Creates slots for x', the expectation μ = <x'> and current weights
        for v in var_list:
            mu = tf.train.Optimizer.get_slot(v, 'mu')
            gamma = tf.train.Optimizer.get_slot(v, 'gamma')


    def _langevin_ops(self):
        self.example, self.labels = self.iterator.get_next()
        self.logits = Network.cnn(self.example, config, self.training_phase)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            labels=self.labels))

        self.inner = self.sgld_opt.minimize(self.cost)


    def _apply_dense(self, config, grad, var):
        # define your favourite variable update
        # for example:
        '''
        # Here we apply gradient descents by substracting the variables
        # with the gradient times the learning_rate (defined in __init__)
        var_update = state_ops.assign_sub(var, self.learning_rate * grad)
        '''
        #The trick is now to pass the Ops in the control_flow_ops and
        # eventually groups any particular computation of the slots your
        # wish to keep track of:
        # for example:
        '''
        m_t = ...m... #do something with m and grad
        v_t = ...v... # do something with v and grad
        '''
        lr_t = math_ops.cast(self._lr_tensor, var.dtype.base_dtype)
        lr_prime_t = math_ops.cast(self._lr_prime_tensor, var.dtype.base_dtype)
        eps_t = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)
        g0_t = math_ops.cast(self._g0_tensor, var.dtype.base_dtype)
        g1_t = math_ops.cast(self._g1_tensor, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_tensor, var.dtype.base_dtype)

        mu = self.get_slot(var, 'mu')
        gamma = self.get_slot(var, 'gamma')

        gamma_t = gamma.assign(g0_t*tf.pow((1+g1), self._global_step))

        for l in range(L):
            _langevin_ops()

        mu_t = mu.assign(self.sgld_opt.get_slot(var, 'mu'))
        var_update = state_ops.assign_sub(var, lr_t*gamma_t*(var-mu_t))

        return control_flow_ops.group(*[var_update, mu_t, gamma_t])

    def _apply_sparse(self, config, grad, var_list):
        raise NotImplementedError("Optimizer does not yet support sparse gradient updates.")
