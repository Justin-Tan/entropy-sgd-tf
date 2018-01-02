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

    def __init__(self, eta_prime, epsilon, gamma, alpha, momentum, L,
                 sgld_global_step, use_locking=False, name='le_sgld'):
        # Run inner loop Langevin dynamics
        super(local_entropy_sgld, self).__init__(use_locking, name)

        self._lr_prime = eta_prime
        self._epsilon = epsilon
        self._gamma = gamma
        self._momentum = momentum
        self._alpha = alpha
        self._L = L
        self.sgld_global_step = sgld_global_step

        # Parameter tensors
        self._lr_prime_t = None
        self._epsilon_t = None
        self._gamma_t = None
        self._momentum_t = None
        self._alpha_t = None
        self._L_t = None
        self._sgld_gs_t = None

    def _prepare(self):
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
        self._L_t = ops.convert_to_tensor(self._L,
                                          name="L")
        self._sgld_gs_t = ops.convert_to_tensor(self.sgld_global_step,
                                                name="sgld_global_step")

    def _create_slots(self, var_list):
        # Manage variables that accumulate updates
        # Creates slots for x', the expectation μ = <x'> and current weights
        for v in var_list:
            wc = self._zeros_slot(v, "wc", self._name)
            xp = self._zeros_slot(v, "xp", self._name)
            mu = self._zeros_slot(v, "mu", self._name)

    def _apply_dense(self, grad, var):
        # Updates dummy weights during SGLD
        # Reassign to original weights upon completion of inner loop
        lr_prime_t = math_ops.cast(self._lr_prime_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        wc = self.get_slot(var, 'wc')
        xp = self.get_slot(var, 'xp')
        mu = self.get_slot(var, 'mu')

        wc_t = tf.cond(tf.logical_not(tf.cast(tf.mod(self.sgld_global_step, self._L_t), tf.bool)),
            lambda: wc.assign(var),
            lambda: wc)

        eta = tf.random_normal(shape=var.get_shape())
        eta_t = math_ops.cast(eta, var.dtype.base_dtype)

        # update = -lr_prime_t*(grad-gamma_t*(wc-var)) + tf.sqrt(lr_prime)*epsilon_t*eta_t
        xp_t = xp.assign(var-lr_prime_t*(grad-gamma_t*(wc-var))+tf.sqrt(lr_prime_t)*epsilon_t*eta_t)
        mu_t = mu.assign((1.0-alpha_t)*mu + alpha_t*xp)

        var_update = state_ops.assign_sub(var,
            lr_prime_t*(grad-gamma_t*(wc-var))-tf.sqrt(lr_prime_t)*epsilon_t*eta_t)

        return control_flow_ops.group(*[var_update, wc_t, xp_t, mu_t])

    def _apply_sparse(self, grad, var_list):
        raise NotImplementedError("Optimizer does not yet support sparse gradient updates.")
