"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows*cols)
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        Y = np.frombuffer(f.read(), dtype=np.uint8)

    return (X, Y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    tmp = ndl.exp(Z) / ndl.exp(Z).sum(axes=(1)).reshape((-1, 1)).broadcast_to(Z.shape)
    return -(y_one_hot * ndl.log(tmp)).sum() / y_one_hot.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, X.shape[0], batch):
        x, label = ndl.Tensor(X[i: i + batch], dtype="float32"), y[i: i + batch]
        z = ndl.relu((x @ W1)) @ W2
        I2 = np.zeros((batch, W2.shape[1]))
        I2[range(batch), label] = 1 
        y_ont_hot = ndl.Tensor(I2, dtype="int8")
        loss = softmax_loss(z, y_ont_hot)
        loss.backward()
        W2.data -= lr * W2.grad.data
        W1.data -= lr * W1.grad.data
    return (W1, W2)
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    acc, loss, cnt = [], [], 0
    if opt is not None:
        model.train()
        for (X, y) in dataloader:
            bs = X.shape[0]
            cnt += bs
            X, y = ndl.Tensor(X, device=ndl.cpu()), ndl.Tensor(y, device=ndl.cpu())
            pred = model(X)
            loss_v = loss_fn(pred, y)
            acc.append(np.sum(np.argmax(pred.numpy(), axis=1) == y.numpy()))
            opt.reset_grad()
            loss.append(loss_v.numpy() * bs)
            loss_v.backward()
            opt.step()
    else :
        model.eval()
        for (X, y) in dataloader:
            bs = X.shape[0]
            cnt += bs
            X, y = ndl.Tensor(X), ndl.Tensor(y)
            pred = model(X)
            loss_v = loss_fn(pred, y)
            acc.append(np.sum(np.argmax(pred.numpy(), axis=1) == y.numpy()))
            loss.append(loss_v.numpy() * bs)

    return sum(acc) / cnt, sum(loss) / cnt
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = 0, 0
    opt = optimizer(model.parameters(), lr = lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=opt)
    return avg_acc, avg_loss 
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader, model, loss_fn=loss_fn())
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
 
    acc, loss, cnt = [], [], 0
    if opt is not None:
        model.train()
        for i in range(len(data) - seq_len - 1):
            X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
            bs = X.shape[1]
            cnt += bs * seq_len
            pred, _ = model(X)
            loss_v = loss_fn(pred, y)
            acc.append(np.sum(np.argmax(pred.numpy(), axis=1) == y.numpy()))
            opt.reset_grad()
            loss.append(loss_v.numpy() * bs * seq_len)
            loss_v.backward()
            opt.step()
    else :
        model.eval()
        for i in range(len(data) - seq_len - 1):
            X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
            bs = X.shape[0]
            cnt += bs * seq_len
            pred, _ = model(X)
            loss_v = loss_fn(pred, y)
            acc.append(np.sum(np.argmax(pred.numpy(), axis=1) == y.numpy()))
            loss.append(loss_v.numpy() * bs * seq_len)

    return sum(acc) / cnt, sum(loss) / cnt
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr = lr, weight_decay=weight_decay)
    loss_fn = loss_fn()
    avg_acc, avg_loss = 0, 0
    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn, opt, clip, device, dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
