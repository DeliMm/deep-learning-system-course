import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    b = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in * fan_out, low=-b, high=b, **kwargs).reshape((fan_in, fan_out))
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in * fan_out, mean=0, std=std, **kwargs).reshape((fan_in, fan_out))
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    b = math.sqrt(6.0 / fan_in)
    if shape is not None:
        b = math.sqrt(6.0 / math.prod(shape[:-1]))
        return rand(*shape, low=-b, high=b, **kwargs)
    return rand(fan_in * fan_out, low=-b, high=b, **kwargs).reshape((fan_in, fan_out))
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2.0 / fan_in)
    return randn(fan_in * fan_out, mean=0, std=std, **kwargs).reshape((fan_in, fan_out))
    ### END YOUR SOLUTION