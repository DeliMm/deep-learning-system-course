# Softmax Regression 

## Basisc of machine learning 
The **(supervisied) ML approach**: collect a training set of images with konwn labels and feed these into a machine learning algorighm, which will automatically produce a "program" that solve this task.  

Every machine learning algorithm consists of three different elements:

1. The hypothesis class
2. The loss function 
3. An optimization method

## Softmax Regression  

Let's consider a $k-class$ classification setting, where we have:

- Training data: $x^{(i)} \in \mathbb{R}^n$, $y^{(i)} \in \{1, \cdots, k \}$  for $i = 1, \cdots m$

- $n$ = dimensionality of the input data 
- $k$ = number of different classes / labels
- $m$ = number of points in the traing set 
 
 Our hypothesis function maps inputs $x\in \mathbb{R}^n$ to $k-$dimensional vectors.
 $$
 h:\mathbb{R}^n \rightarrow \mathbb{R}^k 
 $$

 A linear hypothesis function uses a linear operator for this transformation

 $$
 h_\theta (x) = \Theta^T x
 $$

 for parameters $\theta \in \mathbb{R}^{n \times k}$

### Loss function 
#### classification error 
The simplest loss function to use in classification is just the classification error, i.e., whether the classifier makes a mistake or not
$$
\mathcal{l}_{err}(h(x), y) = \begin{cases}
0 & \text{if} \argmax_i h_i(x) = y \\
1 & \text{otherwise}
\end{cases}
$$

#### softmax/ cross-entropy loss 
convert the hypothesis function to a "probability" by exponentiating and normalizing its entries.

$$
z_i = p(\text{label = i}) = \frac{\exp(h_i(x))}{\sum_{j=1}^k \exp(h_j(x))} \Leftrightarrow z = \text{normalize}(\exp(h(x)))
$$

The format of softmax or cross-entropy loss is as following:

$$
\mathcal{l}_{ce}(h(x), y) = -\log p(\text(label=y)) = -h_y(x) + \log \sum_{j=1}^k \exp(h_j(x)) 
$$

And the problem of minimizing the average loss on the training set for softmax regerssion is

$$
\min_{\theta} \frac1m \sum_{i=1}^m \mathcal{l}_{ce}(\Theta^Tx^{(i)}, y^{(i)})
$$

The gradient descent algorithm proceeds by iteratively taking steps in the direction of the negative gradient

$$
\theta := \theta - \alpha\nabla_\theta f(\theta)
$$

How do we compute the gradient $\nabla_\theta l_{ce}(\theta^Tx, y)$?

- Use matrix differential calculus, jacobians, kronecker products, and vectorization

- Pretend everything is a scalar, use the typical chain rule, and then rearrange/transpose matrices/vectors to make the sizes work

If we use "matrix batch", the form of the loss is as following:
$$
\nabla_\theta l_{ce}(X\theta, y) \in \mathbb{R}^{n \times k} = X^T(Z - I_y), Z = \text{normalize}(\exp(X\theta))
$$


The final algorithm is :

- Repeat until parameters/loss converges
 1. Iterate over minibatches $X\in \mathbb{R}^{B\times n}, y \in \{1, \cdots, k\}^B$ of training set
 2. Update the parameters $\theta := \theta - \frac\alpha B X^T(Z - I_y)$

This gets less than **8%** error in classifying MNIST digits 