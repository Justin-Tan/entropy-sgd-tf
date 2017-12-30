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


class local_entropy_sgld(optimizer.Optimizer):

    def __init__(self, start, eta_prime, epsilon, gamma, momentum, alpha,
                 use_locking=False, name='le_sgld'):
        # Inner loop Langevin dynamics
        super(local_entropy_sgld, self).__init__(use_locking, name)

        self._start = start
        self._lr_prime = eta_prime
        self._epsilon = epsilon
        self._gamma = gamma
        self._momentum = momentum
        self._alpha = alpha

        # Parameter tensors
        self._start_t = None
        self._lr_prime_t = None
        self._epsilon_t = None
        self._gamma_t = None
        self._momentum_t = None
        self._alpha_t = None

    def _prepare(self):

        self._start_t = ops.convert_to_tensor(self._start,
                                              name="start")
        self._lr_prime_t = ops.convert_to_tensor(self._lr_prime,
                                                 name="learning_rate_prime")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon,
                                                name="epsilon")
        self._gamma_t = ops.convert_to_tensor(self._gamma,
                                              name="gamma")
        self._momentum_t = ops.convert_to_tensor(self._momentum,
                                                 name="momentum")
        self._alpha_t = ops.convert_to_tensor(self._alpha,
                                              name="alpha")

    def _create_slots(self, var_list):
        # Manage variables that accumulate updates
        # Creates slots for x', the expectation μ = <x'> and current weights
        for v in var_list:
            wc = self._zeros_slot(v, "wc", self._name)
            xp = self._zeros_slot(v, "xp", self._name)
            mu = self._zeros_slot(v, "mu", self._name)

    def _apply_dense(self, grad, var):
        # define your favourite variable update
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
        start_t = math_ops.cast(self._start_t, var.dtype.base_dtype)
        lr_prime_t = math_ops.cast(self._lr_prime_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        wc = self.get_slot(var, 'wc')
        xp = self.get_slot(var, 'xp')
        mu = self.get_slot(var, 'mu')

        wc_t = tf.cond(start_t,
            lambda: wc.assign(var),
            lambda: wc)

        eta = tf.random_normal(shape=var.get_shape())
        dx = grad - gamma_t*(wc-var)
        update = -lr_prime_t*dx + tf.sqrt(lr_prime)*epsilon_t*eta
        xp_t = xp.assign(var - update)
        mu_t = mu.assign((1.0-alpha_t)*mu + alpha_t*(var-update))

        var_update = state_ops.assign_sub(var, update)

        return control_flow_ops.group(*[var_update, wc_t, xp_t, mu_t])

    def _apply_sparse(self, grad, var_list):
        raise NotImplementedError("Optimizer does not yet support sparse gradient updates.")
