"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import needle.nn as nn 
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        a = Tensor(init.ones_like(x), device=x.device, requires_grad=False)
        return a / (1 + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size * hidden_size, low=-k, high=k, device=device).reshape((input_size, hidden_size)), device=device, dtype=dtype, requires_grad=True)
        self.W_hh = Parameter(init.rand(hidden_size * hidden_size, low=-k, high=k, device=device).reshape(((hidden_size, hidden_size))), device=device, dtype=dtype, requires_grad=True)
        self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device), device=device, dtype=dtype, requires_grad=True) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device), device=device, dtype=dtype, requires_grad=True) if bias else None
        self.act = None 
        if nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'relu':
            self.act = nn.ReLU()
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = Tensor(init.zeros(X.shape[0] * self.hidden_size, device=X.device).reshape((X.shape[0], self.hidden_size)), device=X.device)
        
        out_i, out_h = X @ self.W_ih, h @ self.W_hh
        if self.bias_hh is not None:
            out_h += self.bias_hh.broadcast_to(out_h.shape)
        if self.bias_ih is not None:
            out_i += self.bias_ih.broadcast_to(out_i.shape)
        if self.act is not None:
            return self.act(out_h + out_i)
        return out_h + out_i 
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        self.rnn_cells.extend([RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(1, num_layers)] )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        L, B, _ = X.shape
        seq = X
        if h0 is None:
            h0 = Tensor(init.zeros(self.num_layers * B * self.hidden_size).reshape((self.num_layers, B, self.hidden_size)), device=X.device, requires_grad=True)
        out, attn = [], []
        for idx, m in enumerate(self.rnn_cells):
            h = ops.split(h0, axis=0)[idx]
            for s in range(L):
                h = m(ops.split(seq, axis=0)[s], h)
                out.append(h)
            attn.append(h)
            seq = ops.stack(out, axis=0)
            out = []
        return seq, ops.stack(attn, axis=0)                         
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size * 4 * hidden_size, low=-k, high=k, device=device).reshape((input_size, 4 * hidden_size)), device=device, dtype=dtype, requires_grad=True)
        self.W_hh = Parameter(init.rand(hidden_size * 4 * hidden_size, low=-k, high=k, device=device).reshape(((hidden_size, 4 * hidden_size))), device=device, dtype=dtype, requires_grad=True)
        self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device), device=device, dtype=dtype, requires_grad=True) if bias else None
        self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device), device=device, dtype=dtype, requires_grad=True) if bias else None
        self.tanh = nn.Tanh()
        self.sigmoid = Sigmoid()
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = Tensor(init.zeros(X.shape[0] * self.hidden_size, device=X.device).reshape((X.shape[0], self.hidden_size)), device=X.device)
            c0 = Tensor(init.zeros(X.shape[0] * self.hidden_size, device=X.device).reshape((X.shape[0], self.hidden_size)), device=X.device)
        else:
            h0, c0 = h 

        out_i, out_h = X @ self.W_ih, h0 @ self.W_hh
        if self.bias_hh is not None:
            out_h += self.bias_hh.broadcast_to(out_h.shape)
        if self.bias_ih is not None:
            out_i += self.bias_ih.broadcast_to(out_i.shape)
        
        out = list(ops.split(out_h + out_i, axis=1))
        i, f = ops.stack(out[:self.hidden_size], axis=1), ops.stack(out[self.hidden_size: 2* self.hidden_size], axis=1)
        g, o = ops.stack(out[2*self.hidden_size:3*self.hidden_size], axis=1), ops.stack(out[3*self.hidden_size:], axis=1)
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c = f * c0 + i * g
        h = o * self.tanh(c)
        return h, c 
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        self.lstm_cells.extend([LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(1, num_layers)])
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        L, B, C = X.shape
        if h is not None:
            h0, c0 = h
        else:
            h0 = Tensor(init.zeros(self.num_layers * X.shape[1] * self.hidden_size, device=X.device).reshape((self.num_layers, X.shape[1], self.hidden_size)), device=X.device)
            c0 = Tensor(init.zeros(self.num_layers * X.shape[1] * self.hidden_size, device=X.device).reshape((self.num_layers, X.shape[1], self.hidden_size)), device=X.device)
        seq = X
        out, hn, cn = [], [], []
        for idx, m in enumerate(self.lstm_cells):
            h, c = ops.split(h0, axis=0)[idx], ops.split(c0, axis=0)[idx]
            for s in range(L):
                h, c = m(ops.split(seq, axis=0)[s], (h, c))
                out.append(h)
            seq = ops.stack(out, axis=0)
            out = []
            hn.append(h), cn.append(c)
        return seq, (ops.stack(hn, axis=0), ops.stack(cn, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings * embedding_dim, device=device, dtype=dtype).reshape((num_embeddings, embedding_dim)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype, requires_grad=False)
        return (x.reshape((-1, self.num_embeddings)) @ self.weight).reshape((x.shape[0], x.shape[1], self.embedding_dim))
        ### END YOUR SOLUTION