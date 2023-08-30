# "Manual" Neural Networks

## 1. Nonlinear hypothesis class 

We need a hypothesis function to map inputs in $\mathbb{R}^n$ to outpus $\mathbb{R}^k$, so we initialy used the linear hypothesis class:

$$
h_\theta(x) = \theta^T x, \theta \in \mathbb{R} ^{n \times k}
$$

If the data can't be seperated by a set of linear regions? 

**One idea**: apply a linear classifier to some (potentially higher-dimensional) features of the data:

$$
h_\theta(x) = \theta^T \phi(x), \theta \in \mathbb{R} ^{d \times k}, \phi: \mathbb{R}^n \to \mathbb{R}^d 
$$

The way for creating the feature function $\phi$:
1. Through manual engineering of features relevant to the problem (the "old" way of doing machine learning)
2. In a way that itself is learned from data (the "new" way)

The from of the second way is $\phi(x) = \sigma(W^T x)$ 

If we want to train $W$ to minimize loss or we want tot compose multiple features together? And we can use **Neural Networks** to handle these problems. 

## 2. Neural Networks

### Universal function approximation 

**Theorem (1D case)**: Given any smooth function $f:\mathbb{R} \to \mathbb{R}$, closed region $\mathcal{D} \subset \mathbb{R}$, and $\epsilon \gt 0$, we can construct a one-hidden-layer neural network $\hat f$ such that 

$$
\max_{x \in \mathcal{D}} |f(x) - \hat f(x)| \le \epsilon
$$

We can use Lagrange polynomial to construct the funcion.

A more generic form of a $L-$ layer neural network a.k.a MLP, feedforward network, fully-connected network  

$$
Z_{i + 1} = \sigma_i (Z_iW_i), i = 1, \cdots, L \\
Z_1 = X  \\
h_\theta(X) = Z_{L + 1} \\
[Z_i \in \mathbb{R}^{m\times n_i}, W_i \in \mathbb{R}^{n_i \times n_{i+1}}]  \\
\theta = \{W_1, \cdots, W_L\}
$$

Why deep networks? 

It seems like the work better for a fixed parameter count!!!

## Backpropagation 

The gradient of a two-layer network, it's form is:
$$
\nabla_{W_1, W_2} \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)
$$

The gradient w.r.t. $W_2$ looks identical to the softmax regression case:

$$
\frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial W_2} = \frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial (\sigma(XW_1)W_2, y )} \cdot  \frac{\partial (\sigma(XW_1)W_2, y )}{\partial W_2} \\ = (S - I_y) \cdot \sigma(XW_1)
$$

So the gradient is :
$$
\frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial W_2} = \sigma(XW_1)^T(S-I_y)
$$
The gradient w.r.t $W_1$:

$$
\frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial W_1} = \frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial \sigma(XW_1)W_2} \cdot \frac{\partial \sigma(XW_1)W_2}{\partial \sigma(XW_1)} \cdot \frac{\partial \sigma(XW_1)}{\partial XW_1} \cdot \frac{\partial XW_1}{\partial W_1} \\ 
= (S - I_y) \cdot W_2 \cdot (\sigma'(XW_1)) \cdot X
$$

also the gradient is:

$$
\frac{\partial \mathcal{L}_{ce}(\sigma(XW_1)W_2, y)}{\partial W_1} = X^T ((S - I_y)W^T_2 \circ \sigma'(XW_1))
$$

#### Backpropagation "in general"

consider a fully-connected network:

$$
Z_{i+1} = \sigma_i(Z_i W_i), i = 1, \cdots, L
$$

Then

$$
\begin{align}
\frac{\partial \mathcal{L}(Z_{L+1}, y)}{\partial W_i}  &= \underbrace{\frac{\partial \mathcal L}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_{L}} \cdots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}}}_{G_{i+1} = \frac{\partial \mathcal{L}(Z_{L+1}, y)}{\partial Z_{i+1}}} \cdot \frac{\partial Z_{i+1}}{\partial W_i}
\end{align}
$$

Then we have a simple "backward" iteration to compute the $G_i$'s 

$$
G_i = G_{i+1} \cdot \frac{\partial Z_{i +1}}{\partial Z_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_i W_i)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i }{\partial Z_i} = G_{i + 1} \cdot \sigma'(Z_i W_i) \cdot W_i
$$

similar formula for actual parameter gradients $\nabla_{W_i} \mathcal{L}(Z_{L+1}, y)$ is 

$$
\nabla_{W_i} \mathcal{L}(Z_{L+1}, y) = Z^T_i(G_{i+1} \circ \sigma' (Z_i W_i)) 
$$

#### Putting it all together 
We can effeciently compute all the gradients we need for a neural network by following the procedure bellow:

1. Initiliaze: $Z_1 = X$
2. iterate : $Z_{i+1} = \sigma_i (Z_i, W_i), i = 1, \cdots, L$ 
3. Initialize: $G_{L+1} = \nabla_{Z_{L+1}} \mathcal{L} (Z_{L+1}, y) = S - I_y$
4. Iterate: $G_i = (G_{i+1} \circ \sigma'_i(Z_iW_i))W^T_i , i = L, \cdots, 1$

And we can compute all the needed gradients along the way:

$$
\nabla_{W_i} \mathcal L (Z_{k+1}, y) = Z^T_i (G_{i+1} \circ \sigma'(Z_i W_i))
$$

Backpropagation is just chain rule + intelligent caching of intermediate results.
Each layer needs to be able to multiply the incoming backward gradient $G_{i+1}$ by it's derivatives $\frac{\partial Z_{i+1}}{\partial W_i}$ an operation called **"vector Jacobian product"**. 