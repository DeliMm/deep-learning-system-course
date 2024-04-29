# Auto Differentiation 

## 1. General introduction to different differentiation methods

### Numerical differentiation 

Directly compute the partial gradient by definnition:

$$
\frac{\partial f(\theta)}{\partial \theta_i} = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon e_i) - f(\theta)}{\epsilon}
$$

A more numerically accurate way to approximate the gradient: 
$$
\frac{\partial f(\theta)}{\partial \theta_i} = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon} + o(\epsilon ^2)
$$

Suffer from numerical error, less efficient to compute. 

#### Numerical gradient checking 
This method is a powerful tool to check an implement of an automatic differentiation algorithm in unit test cases.

$$
\delta ^T \nabla_\theta f(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon\delta)}{2\epsilon} +  o(\epsilon ^2)
$$

Pick $\theta$ form unti ball, check the above variance. 

### Symbolic differentiation 

Write down the formulas, derive the gradient by sum, product and chain rules. 

The computational graph is usually a DAG. Each node represent an (intermediate) value in the computation. 

### Forward mode automatic differentiation(AD)

Define $\dot v_i = \frac{\partial v_i}{\partial x_1}$，we can then compute the $\dot v_i$ iteratively in the forward topological order of the computation graph. 

For $f: \mathbb{R}^n \to \mathbb{R}^k$, we need $n$ forward passes to get the gradient with respect to each input. 

We mostly care about the cases where $k=1$ ang large $n$. 

In order to resolve the problem efficiently, we need to use another kind of AD. 

## 2. Reverse mode automatic differentation 

Define adjoint $\overline {v_i} = \frac{\partial y}{\partial v_i}$, we can then compute the $\bar v_i$ iteratively in the reverse topological order of the computational graph.

Define $\overline{v_{i \to j}} = \overline{v_j} \frac{\partial v_j}{\partial v_i}$ for each input output node pair $i$ and $j$ :

$$
\overline {v_i} = \sum_{j \in next(i)} \overline{v_{i \to j}} 
$$

We can compute partial adjonts seperately then sum them together.

Reverse AD algorithm:

```txt
def gradient(out):
    node_to_grad = {out: [1]}
    for in reverse_topo_order(out):
        v_bar_i = sum(node_to_grad[i])

        for k in inputs(i):
            v_bar_k_2_i = v_bar_i * partial(vi /vk)
            append v_bar_k_2_i to node_to_grad[k]
    return adjoint of input  
```

Reverse mode AD vs Backprop 

- Backprop:
    -  Run backward operations the same forward graph 
    - Used in first generation deep learning frameworks(caffe, cuda-convnet)
- Reverse mode AD:
    - Construct seperate graph nodes for adjoints 
    - Used by modern deep learning frameworks 
    - Handling gradient of gradient 


     