# Sequence Modeling and Recurrent Networks

## Sequence modeling 

In practice, many cases where the input/output pairs are given in specific sequence, and we need to use the information about this sequence to help us make predictions. Some examples:

- Part of speech tagging 
- Speech to text 
- autogressive prediction 

## Recurrent neural networks

Recurrent neural networks mantian a hidden state over time, which is a function of the current input and previous hidden state.

$$
h_t = f(W_{hh}h_{t-1}W_{hx}x_t + b_h) \\
y_t = g(W_{ht}h_t + b_y)
$$

We can train an RNN using backpropagation through time, which just involves "unrolling" the RNN over the length of the sequence, then relying mostly on auto-diff.

```python
opt = optimizer(params=(w_hh, w_hx, w_ht, b_h, b_y))
h[0] = 0
l = 0
for t = 1,...,T:
    h[t] = f(w_hh*h[t-1] + w_hx*x[t] + b_h)
    y[t] = g(w_yh*h[t] + b_y)
    l += Loss(y[t], y_star[t])
l.backward()
opt.step()
```

Just like normal neural networks, RNNS can be stacked together, treating the hidden unit of one layer as the input to the next layer, to form "deep" RNNS.

The challenge for training RNNS is simliar to that of training deep MLP networks. If the weights/activation of the RNN are scaled poorly, the hidden activations(and therefore also the gradients) will **grwo unboundedly with sequence length.**

Similaryly, if weights are too small then information from the inputs will quicklu decay with time(and it is precisely the "long range" dependencies that we would often like to model with sequence models)

One obvious problem with the ReLU is that it can grow unboundly; does using bounded activations "fix" this problem? 

No, creating large encough weights to not cause activations/gradients to vanish requires being in the "saturating" regions of the activations, where gradients are very small, still have vanishing gradients. 

## LSTMS

Long short term memory cells are a particular form of hidden unit update that avoids the problems of vanilla RNNs.

- Divide the hidden unit into two components, called (confusingly) the hidden state and the cell state.

- Use a very formula to update the hidden state and cell state(throwing in some other names, like "forget gate", "input gate", "output gate" for good measure).

$$
\begin{bmatrix} i_t \\ f_t \\ g_t \\ o_t \end {bmatrix} = \begin{bmatrix} \text{sigmoid} \\ \text{sigmoid} \\ \text{tanh} \\ \text{sigmoid}\end{bmatrix} (W_{hh}h_{t - 1} + W_{hx}x_t + b_h) \\
c_t = c_{t-1} \circ f_t + i_t \circ g_t \\
h_t = \tanh(c_t) \circ o_t  
$$



## Beyond "simple" sequential models


- Sequence-to-Sequence models
- Bidirectional RNNs 
    - Bi-directionnal RNNS: stack a forward-running RNN with a backward-running RNN: information form the entire sequence to propagate to hidden state.