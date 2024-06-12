from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes == (-1, ):
            self.axes= (len(Z.shape) - 1, )
        max_z = Z.max(axis=self.axes)
        z_dim = Z.max(axis=self.axes, keepdims=True)
        return array_api.log(array_api.exp(Z - z_dim.broadcast_to(Z.shape)).sum(axis=self.axes)) + max_z 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        rshape = list(lhs.shape)
        if isinstance(self.axes, tuple):
            for axis in self.axes:
                rshape[axis] = 1
        elif isinstance(self.axes, int):
            rshape[self.axes] = 1
        else:
            rshape= [1] * len(lhs.shape)
        a = out_grad.reshape(rshape).broadcast_to(lhs.shape)
        b = exp(lhs - logsumexp(lhs, axes=self.axes).reshape(rshape).broadcast_to(lhs.shape))
        return a * b
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

