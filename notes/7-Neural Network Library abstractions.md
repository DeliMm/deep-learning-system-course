# Neural Network Library Abstractions

## Programming abstractions 

### Forward and backward layer interface 

```python 
class Layer:

    def forward(bottom, top):
        pass 
    
    def backward(top, propagate_down, bottom):
        pass

```
used in cuda-convnet(the AlexNet framewrok)

### decalartive programming 

```python 
import tensorflow as tf 

v1 = tf.Variable()
v2 = tf.exp(1)
v3 = v2 + 1
v4 = v2 * v3

sess = tf.session()
value4 = sess.run(v4, feed_dict={v1 : numpy.array([1])}) 
```
Frist declare the computational graph, then execute the graph by feeding input value.

### Imperative automatc differentiation 

```python
import needle as ndl

v1 = ndl.Tensor([1])
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3
```

Executes computation as we construct the computational graph. Allow easy mixing of python control flow and construction.

## High level modular library components 

`nn.Module` Compose things together, for Layer(Tensor in, Tensor out):

1. For given inputs, how to compute outputs
2. Get the list of parameters
3. Ways to initialize the parameters


Loss function as a special kind of module, follwing the rule(Tensor in, scalar out).

Optimizer takes a list of weights from the model perform steps of optimization nad keeps tracks of auxiliary states(momentum).

There are two ways to incorporate regularization:

- Implement as part of loss function 
- Directly incorporate as part of optimizer update

Initialization can be folded into the construction phase of a `nn.Module`. 

Data loading and augmentation is also compositional in nature. 

 