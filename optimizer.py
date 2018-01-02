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

from network import Network
from sgld import local_entropy_sgld
from config import config_train

class EntropySGD(optimizer.Optimizer):

    def __init__(self, iterator, training_phase, sgld_global_step, config={},
                 use_locking=False, name='EntropySGD'):

        # Construct entropy-sgd optimizer - ref. arXiv 1611.01838
        defaults = dict(lr=1e-3, gamma=1e-3, momentum=0, damp=0, weight_decay=0,
                        nesterov=True, L=2, epsilon=1e-4, g0=3e-2, g1=1e-3,
                        alpha=0.75, lr_prime=0.1)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(use_locking, name)
        self.config = config
        self.iterator = iterator
        self.training_phase = training_phase
        self.sgld_global_step = sgld_global_step

        self._learning_rate = config['lr']
        self._gamma = config['gamma']

        # Scalar parameter tensors
        self._lr_tensor = None
        self._gamma_tensor = None
        self._wd_tensor = None
        self._momentum_tensor = None

        self.sgld_opt = local_entropy_sgld(eta_prime=config['lr_prime'],
            epsilon=config['epsilon'], gamma=self._gamma, alpha=config['alpha'],
            momentum=config['momentum'], L=config['L'],
            sgld_global_step=self.sgld_global_step)

    def _prepare(self):
        self._lr_tensor = ops.convert_to_tensor(self._learning_rate,
                                                name="learning_rate")
        self._gamma_tensor = ops.convert_to_tensor(self._gamma,
                                                   name="gamma")
        self._wd_tensor = ops.convert_to_tensor(self.config['weight_decay'],
                                                   name="decay")
        self._momentum_tensor = ops.convert_to_tensor(self.config['momentum'],
                                                      name="momentum")

    def _create_slots(self, var_list):
        # Manage variables that accumulate updates
        # Creates slots for x', the expectation μ = <x'> and current weights
        for v in var_list:
            wc = self._zeros_slot(v, "wc", self._name)
            mu = self._zeros_slot(v, "mu", self._name)


    def _apply_dense(self, grad, var):
        # Apply weight updates
        lr_t = math_ops.cast(self._lr_tensor, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_tensor, var.dtype.base_dtype)

        wc = self.get_slot(var, "wc")
        mu = self.get_slot(var, "mu")

        wc_t = wc.assign(self.sgld_opt.get_slot(var, "wc"))
        mu_t = mu.assign(self.sgld_opt.get_slot(var, "mu"))

        # Reset weights to pre-SGLD state, then execute update
        var_reset = state_ops.assign(var, wc_t)

        with tf.control_dependencies([var_reset]):
            var_update = state_ops.assign_sub(var, lr_t*gamma_t*(var-mu_t))

        return control_flow_ops.group(*[var_update, mu_t, wc_t])

    def _apply_sparse(self, grad, var_list):
        raise NotImplementedError("Optimizer does not yet support sparse gradient updates.")
