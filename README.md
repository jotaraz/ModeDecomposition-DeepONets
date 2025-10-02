# Mode-Decomposition in DeepONets: Generalization and Coupling Analysis

This repository contains some of the code of my [master thesis](https://repository.tudelft.nl/record/uuid:e8a0439c-ecfa-4adc-8ea7-2679847995eb). 
The thesis was written at TU Delft under the supervision of [Dr. Alexander Heinlein](https://searhein.github.io/).

## What is the thesis about?

The [DeepONet](https://arxiv.org/abs/1910.03193) is the standard neural network architecture in operator learning: learning an operator (mapping function to function) from data.
To investigate why/where approximation errors arise, we decompose the approximation error twice:
1. The total approximation error is decomposed into trunk and branch errors, heavily building on the framework derived by [Lanthaler et al.](https://arxiv.org/abs/2102.09618)
2. The branch error is decomposed into mode errors: how well are we approximating the coefficient of a given spatial basis function (or mode)?

## What are the key findings?
- For large inner dimension (and 'smooth' problems) the branch error dominates.
- The error corresponding to modes with intermediate (neither very large nor very small) singular values dominates the branch error.
- Generalization performance of gradient descent can be improved by re-weighting the mode losses.
- Mode-decomposition establishes connections between operator learning and multi-task-learning.
- Mode-decomposition enables us to study the coupling between the modes.

## What does the repository contain?
- Code to train DeepONets
- Code to analyze DeepONet errors (specifically mode-wise errors)
- Some training data (Burger's equation)
- Parameters of some trained DeepONets
