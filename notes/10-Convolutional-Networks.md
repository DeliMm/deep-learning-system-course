# Convolutional Networks

## Convolutional operators in deep networks 

The problem with fully connected networks:

1. For image input, it needs too much parameter 

2. Does not capture any of the intuitive invariances that we expect to have in images 

Convolutions combine two ideas that are well-suited to processing images

- Require that activations between layers occur only in a "local" manner, and treat hidden layers themselves as spatial images 

- Share weights across all spatial locations 

Convolutions are a common operation in many computer vision applications: convolution network just move to learned filters.

Convolutions in deep networks are virtually always multi-channel convolutions: map multi-channel inputs to multi-channel hidden units

- $x \in \mathbb{R}^{h \times w \times c_{in}}$ denotes $c_{in}$ channel, size $h \times w$ image input 

- $z \in \mathbb{R}^{h \times w \times c_{out}}$ denotes $c_{out}$ channel, size $h \times w$ image input 

- $W \in  \mathbb{R}^{c_{in} \times c_{out} \times k \times k}$ denotes convolutional filter

## Elements of practical convolutions

### Padding 

"Naive" convolutions produce a smaller output than input image. 

For (odd) kernel size $k$, pad input with (k - 1) / 2 zeros on all slides, results in an output that is the same size as the input. 

### Stride Convolutions / Pooling 

Convolutions keep the same resolution of the input at each layer, don't naively allow for representations at different "resolutions"?

- incorporate max or average pooling layers to aggregate information 

- slide convolutional filter over image in increments

### Grouped Convolutions 

for large numbers of input/output channels, filters can still have a large number of weights, can lead to overfitting + slow computation

Group together channels, so that groups of channels in output only depend on corresponding groups of channels in input(equivalently, enforce filter weight matrices to be block-diagonal)

### Dilations 

Convolutions each have a relatively small receptive field size

Dilate convolution filter, so that it convers more of the image; note that getting an image of the same size again requires adding more padding

## Differentiating convolutions

if we define our operation:

$$
z = \text{conv}(x, w)
$$

how do we multiply by the adjoints:

$$
\bar v \frac{\partial \text{conv}(x, W)}{\partial W}, \bar v \frac{\partial \text{conv}(x, W)}{\partial x}
$$

what is the "transpose" of a convolution? 

**the operation $\hat {W} v$ it itself just a convolution with the "flipped" filter.** 


