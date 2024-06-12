"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict
from email.policy import default 


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if not param.requires_grad:
                continue
            grad = param.grad.data + param.data * self.weight_decay
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            param.data -= self.lr * self.u[param]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad_norm = np.linalg.norm(param.grad.data)
            if grad_norm > max_norm:
                param.grad.data = param.grad.data / grad_norm * max_norm
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if not param.requires_grad:
                continue
            grad = param.grad.data + param.data * self.weight_decay
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
