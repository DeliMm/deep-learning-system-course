# Fully connected networks, optimization, initialization

## 1. Fully connected networks

A $L-$ layer neural network a.k.a MLP, feedforward network, fully-connected network, now with an explicit bias term, is defined by the iteration:  

$$
Z_{i + 1} = \sigma_i (Z_iW_i + b_i), i = 1, \cdots, L \\
Z_1 = x  \\
h_\theta(X) = Z_{L + 1} \\
[Z_i \in \mathbb{R}^{m\times n_i}, W_i \in \mathbb{R}^{n_i \times n_{i+1}}]  \\
\theta = \{W_{1:L}, b_{1:L}\}
$$

where $\sigma_i(x)$ is the nonlinear activattion, usually with $\sigma_L (x) = x$

**Broadcasting** does not copy any data(desribed more in later lecture)


 
## 2. Optimization 

### Gradient descent 

for a general funciotn $f$, and writing iterate number $t$ explicitly

$$
\theta_{t+1} = \theta_t - \alpha\nabla_\theta f(\theta_t)
$$

where $\alpha \gt 0$ is step size, $\nabla_\theta f(\theta_t)$ is gradient evaluated at the parameters $\theta_t$  

#### Newton's Method 

Newton's method scales gradient according to inverse of the Hessian(matrix of second derivatives):

$$
\theta_{t + 1} = \theta_t - \alpha \left(\nabla_\theta^2 f(\theta_t)\right)^{-1}\nabla_\theta f(\theta_t)
$$

where $\nabla_\theta^2 f(\theta_t)$ is the Hessian, $n \times n$ matrix of all second derivatives. This method is equivalent to approximating the function as quadratic using second-order Taylor expansion, then solving for optimal solution. 

Full step given by $\alpha = 1$, otherwise called damped Newton method. 

### Momentum 

Momentum takes into account a moving average of multiple previous gradients

$$
\mu_{t + 1} = \beta \mu_t + (1 - \beta)\nabla_\theta f(\theta_t) \\
\theta_{t+1} = \theta_t - \alpha\mu_{t + 1}
$$

where $\alpha$ is step size as before, and $\beta$ is momentum averaging parameter

### Nesterov Momentum 

This method computes momentum update at "next" point.

$$
\mu_{t + 1} = \beta \mu_t + (1 - \beta)\nabla_\theta f(\theta_t - \alpha \mu_{t + 1}) \\
\theta_{t+1} = \theta_t - \alpha\mu_{t + 1}
$$


### Adam 

Adam algorithm combines momentum and adaptive scale estimation:

$$
u_{t + 1} = \beta_1 u_t + (1 - \beta_1)\nabla_\theta f(\theta_t) \\
v_{t + 1} = \beta_2 v_t + (1 - \beta_1)(\nabla_\theta f(\theta_t)) ^ 2\\ 
\theta_{t + 1} = \theta_t - \alpha \frac{u_{t + 1}}{\sqrt{v_{t + 1}} + \epsilon}
$$

## Initialization 

Key idea

- choice of initialization matters
- Weights don't move "that much" 

if we choose the initial values of $W_i$ and $b_i$ to be zero, the $\nabla_{W_i}\mathcal{l}(h_\theta(X), y) = 0$. This is a very bad optimum of the objective.

Let's just initialize weights "randomly", e.g., $W_i \sim \mathcal{N}(0, \sigma^2I)$, the choice of variance $\sigma^2$ will affect two quantities. 

Consider independent random variables $x \sim \mathcal{N}(0, 1), w \sim \mathcal{N}(0, \frac1n)$, then 

$$
\mathbf{E}(x_iw_i) = \mathbf{E}(x_i) \mathbf{E}(w_i) = 0 \\
\mathbf{Var}(x_i w_i) = \mathbf{Var}(x_i)\mathbf{Var}(w_i) = \frac1 n \\
\mathbf{E}(w^Tx)  = 0 \\
\mathbf{Var}(w^Tx)  = 1 
$$

Thus, informally speaking if we used a linear activation, the $z_i \sim \mathcal{N}(0, I)$.

If we use a ReLU nonlinearity, then "half" the components of $z_i$ will be set to zero, so we need twice the variance on $W_i$ to achieve the same final variance, hence $W_i \sim \mathcal{N}(0, \frac2n I)$ (Kaiming normal initialization)


