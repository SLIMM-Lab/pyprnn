# Physically Recurrent Neural Networks

<p align="center">
<img src="https://raw.githubusercontent.com/MarinaMaia2021/supportMaterial/main/multiscaleWithPRNN.gif" width="75%" height="75%"/>
</p>

**Intact** constitutive models embedded in an encoder-decoder MLP architecture.

If you have accurate material models at the microscale and would like to perform computational homogenization, those same models can be directly embedded into a hybrid architecture to make macroscale predictions.

Because the models in the architecture are the exact same as in the micromodel, a number of features can be directly inherited and therefore not learned from data:

- Path dependency (loading/unloading/reloading) without training for it
- Strain rate dependency while training with only a single rate
- Consistent step size dependence (independent for inviscid models; correct dependence for viscous models)
- Between $10\times$ and $100\times$ less training data than RNNs for comparable performance

<p align="center">
<img src="https://raw.githubusercontent.com/MarinaMaia2021/supportMaterial/main/matPoint_2d_paper1.gif" width="75%" height="75%"/>
</p>

## **NEW**: Related code

- You can quickly try out our [demo notebook](https://colab.research.google.com/drive/1LmBUjF0ZrEm3ur1Bt8_XUgoLtQHdU4lM?usp=sharing) directly in Google Colab.

- For >10x faster training and inference times, try this [JAX-based PRNN implementation](https://github.com/SLIMM-Lab/jax-prnn). 

## Journal papers and preprints

- MA Maia, IBCM Rocha, P Kerfriden, FP van der Meer (2023), [PRNNs for 2D composites, elastoplastic](https://www.sciencedirect.com/science/article/pii/S0045782523000579)

- MA Maia, IBCM Rocha, FP van der Meer (2024), [PRNNs for 3D composites, finite-strain thermoviscoplasticity, creep and fatigue](https://www.sciencedirect.com/science/article/pii/S0167663624002370)

- N Kovacs, MA Maia, IBCM Rocha, C Furtado, PP Camanho, FP van der Meer (2024), [PRNNs for micromodels including distributed cohesive damage](https://arxiv.org/abs/2410.13774)

- MA Maia, IBCM Rocha, D Kovacevic, FP van der Meer (2025), [Reproducing creep and fatigue experiments in thermoplastics using PRNNs](https://arxiv.org/abs/2501.10193)

## In this repository

The code in this repository contains a standalone demonstration of PRNNs for a 2D micromodel with $J_2$ elastoplasticity (matrix) and linear elasticity (fibers):

- `prnn-demo.ipynb`: Jupyter notebook with a few ready-to-run examples. **START HERE!**
- `J2Tensor_vect.py`: a simple $J_2$ plasticity model in plane stress. This code comes directly from an FE package, demonstrating how PRNNs can embed existing material models with little to no changes in code;
- `prnn.py`: A PyTorch network class that implements the PRNN, with single-layer encoder and decoder; 
- `rnn.py`: Implements GRU and LSTM networks with variational Gaussian dropout. For comparing predictions and learning performance with PRNNs;
- `utils.py`: Implements a custom dataset class for handling stress and strain paths, and a class for training and evaluating networks, with the ability of saving and loading checkpoints;
- `pyprnn.yml`: Conda environment file that should take care of all dependencies for running the code.

The demonstration notebook also provides three different types of strain path for training, validation and testing:

- A set of 18 **canonical** paths, comprising uniaxial and biaxial combinations of tension/compression and shear. This dataset is made to mimic traditional fitting of constitutive models. PRNNs already perform remarkably well even when trained only on these simple paths;
- A set of 100 **proportional** paths in random directions in stress space containing a single unloading-reloading cycle;
- A set of 100 non-proportional **GP** paths, sampled from suitable Gaussian Process priors and designed to be as general as possible.

<p align="center">
<img src="https://raw.githubusercontent.com/ibcmrocha/public/main/levels.png" width="75%" height="75%"/>
</p>

