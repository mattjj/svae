Code for [Composing graphical models with neural networks for structured representations and fast inference](http://arxiv.org/abs/1603.06277), a.k.a. structured variational autoencoders.

**NOTE:** This code isn't yet compatible with a recent rewrite of autograd. To
use an older, compatible version of autograd, clone
[autograd](https://github.com/hips/autograd) and check out commit 0f026ab.


###Abstract

We propose a general modeling and inference framework that composes probabilistic graphical models with deep learning methods and combines their respective strengths.
Our model family augments graphical structure in latent variables with neural network observation models.
For inference we extend variational autoencoders to use graphical model approximating distributions, paired with recognition networks that output conjugate potentials.
All components of these models are learned simultaneously with a single
objective, giving a scalable algorithm that leverages stochastic
variational inference, natural gradients, graphical model message passing, and
the reparameterization trick.
We illustrate this framework with several example models and an application to
mouse behavioral phenotyping.


By

* [Matthew James Johnson](http://www.mit.edu/~mattjj/)
* [David Duvenaud](http://people.seas.harvard.edu/~dduvenaud/)
* [Alexander B. Wiltschko](https://github.com/alexbw)
* [Sandeep R. Datta](http://datta.hms.harvard.edu/)
* [Ryan P. Adams](https://www.seas.harvard.edu/directory/rpa)
