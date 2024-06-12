"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy
import math 

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad * self.scalar * (lhs ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -lhs * out_grad / (rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(a.ndim))
        if self.axes is None:
            self.axes = axes[-2:]
        axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
        return array_api.transpose(a, axes)   
    
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        cor = []
        x, y = list(lhs.shape), list(self.shape)
        if (len(x) != len(y)):
            x = [1] * (len(y) - len(x)) + x 
            
        for idx in range(len(x)):
            if x[idx] != y[idx]:
                cor.append(idx) 
        for c in cor:
            out_grad = summation(out_grad, (c,), keepdims=True)
        return reshape(out_grad, lhs.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes, self.keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        cor = list(lhs.shape)
      
        if type(self.axes) is int:
            cor[self.axes] = 1
        elif isinstance(self.axes, tuple):
            for x in self.axes:
                cor[x] = 1
        else:
            cor = [1] * len(cor)
        return broadcast_to(reshape(out_grad, tuple(cor)), lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad, rhs_grad = matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        if len(lhs.shape) != len(lhs_grad.shape):
            lhs_grad = summation(lhs_grad, tuple(range(len(lhs_grad.shape) - len(lhs.shape))))
        if len(rhs.shape) != len(rhs_grad.shape):
            rhs_grad = summation(rhs_grad, tuple(range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data() > 0, device=node.device, dtype="float32")
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input_data = node.inputs[0]
        return out_grad * (1 - tanh(input_data) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = len(args)
        tshape = args[0].shape
        rshape = tshape[:self.axis] + (x,) + tshape[self.axis:]
        out = array_api.full(rshape, device=args[0].device, dtype=args[0].dtype, fill_value=0)
        idx = [slice(None)] * len(rshape)
        for i in range(x):  
            idx[self.axis] = slice(i, (i + 1))
            out[tuple(idx)] = args[i]
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out = []
        idx = [slice(None)] * len(A.shape)
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            idx[self.axis] = slice(i, i + 1)
            out.append(A[tuple(idx)].compact().reshape(new_shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        idx = [slice(None)] * len(a.shape)
        for i in range(len(self.axes)):
            shape[self.axes[i]] *= self.dilation + 1
            idx[self.axes[i]] = slice(None, None, self.dilation + 1)
        out = array_api.full(tuple(shape), fill_value=0, dtype=a.dtype, device=a.device)
        out[tuple(idx)] = a
        return out 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        idx = [slice(None)] * len(a.shape)
        for i in range(len(self.axes)):
            shape[self.axes[i]] = shape[self.axes[i]] // (self.dilation + 1)
            idx[self.axes[i]] = slice(None, None, self.dilation + 1)

        out = array_api.empty(tuple(shape), dtype=a.dtype, device=a.device)
        out = a[tuple(idx)] 
        return out 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        n, h, w, c = A.shape
        k, _, _, c_out = B.shape
        ns, hs, ws, cs = A.strides
        inner_dim = k * k * c
        im_A = A.as_strided((n, (h - k) // self.stride + 1, (w - k) // self.stride + 1, k, k, c), (ns, hs * self.stride, ws * self.stride, hs, ws, cs)).compact().reshape((-1, inner_dim))
        im_B = B.compact().reshape((-1, c_out))
        out = (im_A @ im_B).compact().reshape((n, (h - k) // self.stride + 1, (w - k) // self.stride + 1, c_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_data = lhs.realize_cached_data()
        lhs_data = lhs_data.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) 
        n, h, w, _ = lhs_data.shape  
        k, _, _, c_o = rhs.shape
        # lhs_grad
        out_grad_lhs = dilate(out_grad, (1, 2), self.stride - 1)  
        lhs_grad = conv(out_grad_lhs, transpose(flip(rhs, (0, 1)), (2, 3)), 1, k - 1)
        lhs_grad = lhs_grad.realize_cached_data()[:, self.padding:h - self.padding, self.padding:w - self.padding, :]
        # rhs_grad 
        out_grad_rhs = array_api.full((n, h - k + 1, w - k + 1, c_o), fill_value=0, dtype=rhs.dtype, device=rhs.device)
        out_grad_rhs[:, 0::self.stride, 0::self.stride, :] = out_grad.realize_cached_data().compact()
        out_grad_rhs = out_grad_rhs.permute((1, 2, 0, 3))
        lhs_r = lhs_data.permute((3, 1, 2, 0))
        rhs_grad = conv(Tensor(lhs_r, device=out_grad.device), Tensor(out_grad_rhs, device=out_grad.device), 1, 0).realize_cached_data().permute((1, 2, 0, 3))
        return Tensor(lhs_grad, device=out_grad.device).detach(), Tensor(rhs_grad, device=out_grad.device).detach()
        ### END YOUR SOLUTION

def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)
