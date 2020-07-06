
# Scaffold constrained generation

Implementation of the methods described in:

We build on existing Recurrent Neural Network models for SMILES to perform scaffold constrained generation.
Scaffold constrained generation and optimization is not a well studied problem, yet when dealing with drug discovery projects (especially in late stage optimization of compounds) it is the problem we are trying to solve.

# Reproducing the results of the paper

There are four different notebooks:

* Distribution_learning_benchmarks.ipynb to reproduce results for distribution learning related benchmarks, on SureChEMBL and on DRD2
* Focused_learning_experiments.ipynb to reproduce the figure showing the results of focused learning
* MMP12_experiments.ipynb that shows the results on the MMP 12 experiments
* Minimal_working_example.ipynb is a starting point for using the model with a given scaffold

To reproduce the results of a given notebook, simply launch the notebook and run all cells.

# Why build on REINVENT's codebase?

This repository is a fork from the Reinvent's codebase (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x).
One of the major advantages of the algorithms proposed in our work on scaffold constrained generation is that we build on existing models. To go through with this idea of building on existing solutions rather than designing a totally new system, we provide code for our method that builds on an existing, popular codebase: https://github.com/MarcusOlivecrona/REINVENT. This way people who are already familiar with this codebase can get started very easily instead of having to learn how a totally new repository works.
We thank the authors of the original paper for providing a clear codebase for reproducing their work and building on it.


## Requirements

This package requires:
* Python 3.6
* PyTorch >= 0.4.1 
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Scikit-Learn (for QSAR scoring function)
* tqdm (for training Prior)



