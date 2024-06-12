# Transformers and Attention 

## The two approaches to time series modeling 

A time series prediction task is the task of predicting 

$$
y_{1:T} = f_\theta(x_{1:T})
$$

where $y_t$ can depend only on $x_{1:t}$. There are multiple methods for doing so, which may or may not involve the latent state representation of RNNS. 

#### The RNN "latent state" approach

We have already seen the RNN approach to time series: maintain "latent state" $h_t$ that summarizes all information up until that point.

- Pros: Potentially "infinite" history, compact representation 

- Cons: Long "compute path" between history and current time $\Rightarrow$ vanishing / exploding gradients, hard to learn. 


#### The "direct prediction" approach

In contraset, can also directly predict output $y_t$, just need a function that can make predictions of differently-sized inputs.

- Pros: Often can map from past to current state with shorter compute path.

- Cons: No compact state representation, finite history in practice.

#### CNNs for direct prediction 

One of the most straightforward ways to specify the function $f_{\theta}$: (fully) convolutional netowrks, a.k.a. temporal convolutional networks(TCNs). 

The main constraint is that the convolutions be casual: $z_t^{(i + 1)}$ can only depend on $z^{(i)}_{t-k : t}$. 

CNNs have a notable disadvantage for time series prediction: the receptive field of each convolution is usually relateively small $\Rightarrow$ need deep networks to actually incorporate past information. Potential solutions:

- Increase kernel size: also increase the parameters of the network. 
- Pooling layers: not as well suited to dense prediction, where we want to predict all of $y_{1:T}$
- Dilated convolutions: "Skips over" some past state / inputs.

## Self-attention and Transformers 

"Attention" in deep networks generally refers to any mechanism where individual states are weighted and then combined.

$$
z_t = \theta ^T h_t^{(k)} \\ 
w = \text{softmax}(z) \\ 
\bar h = \sum_{t = 1}^T w_t h^{(k)}_t 
$$

Used originally in RNNs when one wanted to combine latent states over all times in a more general manner than "just" looking at the last state.  

**Self-attention** refers to a particular form of attention mechanism. we define the self attention operation as:

$$
\text{Self-Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{d^{1/2}}\right)V
$$

Properties of self-attention:

- invariant(generallu, equivariant) to permutations of the $Q, K, V$ matrices 
- Allows influence between $q_t$, $k_t$, $v_t$ over all times 
- Compute cost is $O(T^2 + Td)$ (cannot be easily reduced due to nonlinearity applied to full $T \times T$ matrix)

The Transformer architecture processes all time stpes parallely, avoids the need for sequential processing as in RNNs.

We can apply the Transformer block to the "direct" prediction method for time serise, instead of using a convolutional block. 

- Pros:
    - Full receptive field within a single layer(i.e., can immediately use past data)
    - Mixing over time doesn't increase parameter count (unlike convolutions)

- Cons:
    - All outputs depend on all inputs(no good e.g. for autogressive tasks)
    - No ordering of data(rember that transformers are equivariant to permutations of the sequence)

To solve the problem of **"acausal" dependencies**, we can mask the softmax operator to assign zero weight to any "future" time steps.

To solve the problem of "order invariance", we can add a **positional encoding** to the input, which associates each input with it's position in the sequence.

## Transformers beyond time serise(very brifely)

Recent work has observed that transformer blocks are extremely powerful beyond just time series:

- Vision Transformers: Apply transformer to iamge(represented by a collection of patch embeddings), works better than CNNs for large data sets.
- Graph Transformers: Capture graph structure in the attention matrix.

In all cases, some challenges are:
 - How to represent data such that $O(T^2)$ operations are feasible 
- How to form positional embeddings 
- How to form the mask matrix

