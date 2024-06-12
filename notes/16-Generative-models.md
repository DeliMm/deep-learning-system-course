# Generative Models

## Generative adversarial training(GAN)

To build effective training mechanism, we need to define a "distance" between generated and real datasets and use that to drive the training.

What we really wanted, in text: make sure that the generated samples "looks real".

Assume that we have an oracle discriminator that can tell the difference between real and fake data. Then we need train the generator to "fool" the oracle discriminator. We need to maximiza the discriminator loss:

$$
\max_G - \mathbb{E}_{z \sim Noise} \log(1 - D(G(Z)))
$$

We can learn discriminator using the real and generated fake data.

$$
\min_D \{-\mathbb{E}_{x \sim data}\log D(x) - \mathbb{E}_{z \sim Noise} \log(1 - D(G(Z)))\}
$$


putting it together:

$$
\min_D \max_G\{-\mathbb{E}_{x \sim data}\log D(x) - \mathbb{E}_{z \sim Noise} \log(1 - D(G(Z)))\}
$$

In practice, we usually optimize $G$ to maximize the probability that discriminator predicts generated image is real. 

Iterative process:

- Discriminator update
    
    - Sample minibatch of $D(G(Z))$, get a minibatch of $D(x)$ 
    - Update $D$ to minimize $min_D \{-\mathbb{E}_{x \sim data}\log D(x) - \mathbb{E}_{z \sim Noise} \log(1 - D(G(Z)))\}$

- Generator update:

    - Sample minibatch of $D(G(Z))$
    - update $G$ to minimize $\max_G - \mathbb{E}_{z \sim Noise} \log(1 - D(G(Z)))$, this can be done by feeding label=1 to the model 



## Modular Design 

Deep learning is modular in nature. 

GAN is not exactly like a loss function, as it involves an iterative update recipe. But we can compose it with other neural network modules in a similar way like loss function.

DCGAN: Deep convolutional generative adversial networks:

Generator: Convolutional units with Conv2dTranpose. 

CycleGAN: the goal of CycleGAN is to learn bi-directional translator between two unpaired collections of data. 