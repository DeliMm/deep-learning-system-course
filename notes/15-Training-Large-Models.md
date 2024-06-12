# Training Larage Models

## Techniques for memory saving 

Sources of memory consumption

1. Model weights 
2. Optimizer states 
3. Intermediate activation values

We only need $O(1)$ memory for computing the final output of a N layer deep network by cycling through two buffers.

Because the need to keep intermediate value around(checkpoint) for the gradient steps. Training a N-layer neural network would require $O(N)$ memory.

we can use checkpointing techniques to only checkpoint necessary nodes in forward computation and recompute the missing intermediate nodes in small segments.

For a $N$ layer neural network, if we checkpoint every $K$ layers:

$$
\text{Memory cost} = O(\frac{N}{K}) + O(K)
$$

## Parallel and distributed training

### Data parallel training 

Let each worker access $\frac{B}{K}$ fraction of the minibatch, and run gradient compuatation then sum up all gradients together. Every worker runs the same replica of the model.

- All-reduce 
- Parameter Server

Many opportunities to continue computation while sending data over the network.

### Model parallel training 

Maps parts of the computation graph to workers.

Tensor Parallelism: Partitions tensor data across devices. 


