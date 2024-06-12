# Normalization and Regularization

## Normalization 

Won't the scale of the initial weights be "fixed" after a few iterations of optimization ?

No! A deep network with poorly-chosen weights will never train(at least with vanilla SGD)

Even when trained successfully, the effects/scales present at initialization persist throughout training. 

**Initialization matters a lot for training, and can vary over the course of training to no longer be "consistent" across layers/networks.**  

### Layer Normalization 

Let's normalize(mean zero and variance one) activations at each layer; this is known as layer normalization:

$$
\hat{z_{i + 1}} = \sigma_i(W_i^T z_i + b_i) \\
z_{i + 1} = \frac{\hat{z_{i + 1} - \mathbb{E}[\hat{z_{i + 1}}]}}{(\mathbb{Var}[\hat{z_{i + 1}}] + \epsilon) ^ {\frac12}}
$$

Also commont to add an additional scalar weight and bias to each term(only change representation e.g. if we put normalization prior to nonlinearity instead)

In practice, for standard FCN, harder to train resulting networks to low loss (relative norms of examples are a useful discriminative feature)

### Batch Normalization 

Layer normalization is equivalent to normalizing the rows of this matrix. What if, we normalize it's columns? This is called batch normalization. 

One oddity to BatchNorm is that it makes the predictions for each example dependent on the entire batch. 

Common solution is to compute a running average of mean/variance for all features at each layer $\hat{\mu_{i + 1}}$, $\hat{\sigma_{i + 1}^2}$, and at test time normalize by these quantities. 

$$
(z_{i + 1})_j = \frac{(z_{i + 1})_j - (\hat{\mu_{i + 1}})_j}{((\hat{\sigma_{i + 1}^2})_j + \epsilon) ^{1/2}}
$$

## Regularization 

Regularization is the process of "limiting the complexity of the function class" in order to ensure that networks will generalize better to new data.

- Implict regularization refers to the manner in which our existing algorithms or architectures already limit functions considered 

- Explicit regularization refers to modifications made to the network and training procedure explicitly intended to regularize the network.

### L2 Regularization

Classically, the magnitude of a model's parameters are often a reasonable proxy for complexity, so we can minimize loss while also keeping parameters small.

$$
\min_{W_{1:L}} \frac1m \sum_{i=1}^m l (h_{W_{1:L}}(x^{(i)}), y^{(i)}) + \frac{\lambda}{2}\sum_{i=1}^L \| W_i\|_2^2
$$

Results in the gradient descent updates:

$$
W_i = (1 - \alpha \lambda)W_i - \alpha\nabla_{W_i}(l(h(x), y))
$$

at each iteration we shrink the weights by a factor $(1 - \alpha\lambda)$ before taking the gradient step.

### Dropout 

Randomly set some fraction of the activations at each layer to zero.

$$
\hat{z_{i + 1}} = \sigma_i(W_i^T z_i + b_i) \\
(z_{i+1})_j = \begin{cases}
\hat{z_{i + 1}} / (1 - p) & \text{with 1 - p} \\
0 & \text{with p}   
\end{cases}
$$

Instructive to consider Dropout as bringing a similar stochastic approximation as SGD to the setting of individual activations. 

## Interaction of optimization, initialization, normalization, regularization 

It seems to be possible to get similarly good results with widly different architectural and methodological chocies. 


