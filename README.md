# BilinearModels

<!--
[![Build Status](https://travis-ci.org/jwmi/BilinearModels.jl.svg?branch=master)](https://travis-ci.org/jwmi/BilinearModels.jl)
-->

## About

BilinearModels is a Julia package for estimation and inference in generalized bilinear models. 

Please cite the following publication if you use this package in your research:
> J. W. Miller and S. L. Carter, **Inference in generalized bilinear models**. [arXiv preprint arXiv:2010.04896](https://arxiv.org/abs/2010.04896), 2020.


## Installation

- Install [Julia](http://julialang.org/downloads/).

- Start Julia and run the following command at the `julia>` prompt:
```julia
using Pkg; Pkg.add(url="https://github.com/jwmi/BilinearModels.jl")
```


## Quick start

At the `julia>` prompt:

```julia
using BilinearModels

Y = [1 2 3; 4 5 6; 7 8 0]  # toy data matrix
I,J = size(Y)              # dimensions of Y
X = ones(I,1)              # feature covariate matrix
Z = ones(J,1)              # sample covariate matrix
M = 0                      # number of latent factors

# fit the model
A,B,C,D,U,V,S,T,omega,logp = BilinearModels.fit(Y,X,Z,M)  

# compute standard errors
se_A,se_B,se_C,se_U,se_V,se_S,se_T = BilinearModels.infer(Y,X,Z,A,B,C,D,U,V,S,T,omega)

```


## Tutorial

A tutorial is provided in the [Jupyter notebook here](https://nbviewer.jupyter.org/github/jwmi/BilinearModelsExamples/blob/main/tutorial/Tutorial%20for%20BilinearModels%20package.ipynb).
Or, for a simple HTML rendering of the tutorial, [see here](http://jwmi.github.io/software/BilinearModels-tutorial.html).


## Questions and issues

Please feel free to post any [issues here](https://github.com/jwmi/BilinearModels.jl/issues), or submit a [pull request](https://github.com/jwmi/BilinearModels.jl/pulls) if you make improvements to the code.
For general questions, feel free to contact me ([Jeff Miller](http://jwmi.github.io/)).


## Licensing

This package is released under the MIT "Expat" license. See [LICENSE.md](LICENSE.md). 



<!--## References
> J. W. Miller and S. L. Carter, **Inference in generalized bilinear models**. [arXiv preprint arXiv:2010.04896](https://arxiv.org/abs/2010.04896), 2020.
-->

