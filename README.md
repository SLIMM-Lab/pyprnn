# Physically Recurrent Neural Networks

<img src="https://github.com/SLIMM-Lab/pyprnn/blob/main/animations/PRNNanalogy.gif" width="75%" height="75%"/>

**Intact** constitutive models embedded in an encoder-decoder MLP architecture.

If you have accurate material models at the microscale and would like to perform computational homogenization, those same models can be directly embedded into a hybrid architecture to make macroscale predictions.

Because the models in the architecture are the exact same as in the micromodel, a number of features can be directly inherited and therefore not learned from data:

- Path dependency (loading/unloading/reloading) without training for it
- Strain rate dependency while training with only a single rate
- Consistent step size dependency (independent for inviscid models; correct dependence for viscous models)
- Between $10\times$ and $100\times$ less training data than RNNs for comparable performance

## Journal papers and preprints

- MA Maia, IBCM Rocha, P Kerfriden, FP van der Meer (2023), [PRNNs for 2D composites, elastoplastic](https://www.sciencedirect.com/science/article/pii/S0045782523000579)

- MA Maia, IBCM Rocha, FP van der Meer (2024), [PRNNs for 3D composites, finite-strain thermoviscoelasticity, creep and fatigue](https://arxiv.org/abs/2404.17583)

- MA Maia, IBCM Rocha, D Kovacevic, FP van der Meer (2024), Reproducing creep and fatigue experiments in thermoplastics using PRNNs -- **COMING SOON**

- N Kovacs, MA Maia, IBCM Rocha, C Furtado, PP Camanho, FP van der Meer (2024), PRNNs for micromodels including distributed cohesive damage -- **COMING SOON**

## In this repository

The code in this repository contains a standalone demonstration of PRNNs for a 2D micromodel with $J_2$ elastoplasticity (matrix) and linear elasticity (fibers):

- `J2Tensor.py` and `J2Tensor_vect.py`: a simple $J_2$ plasticity model in plane stress. This code comes directly from an FE package, demonstrating how PRNNs can embed existing material models with little to no changes in code;
- `prnn.py`: a wrapper class for a PyTorch network that implements the PRNN, with single-layer encoder and decoder. Training is also performed here;
- `prnn_GPcurve_0_6_18.pth`: a PyTorch checkpoint with a pretrained PRNN with two fictitious $J_2$ material points and trained with 18 strain paths coming from suitable Gaussian Processes (non-monotonic and non-proportional paths);
- `prnn-demo.ipynb`: simple Jupyter notebook with a ready-to-run example to get you started with the code.

## Illustrated concepts

- In the PRNN, the internal variables of the path and/or rate-dependent material models are used as the physics-based memory of the network. Here, we illustrate the concept considering a fictitious material point evaluated by a $J_2$ elastoplastic model (matrix), as well as the case in which the material point is evaluated by a linear elastic model (fibers). Note that for the latter, no internal variables are needed. 
<img src="https://github.com/SLIMM-Lab/pyprnn/blob/main/animations/materialPointWithAndWithoutIntVars.gif" width="75%" height="75%"/>

- This time, we show how the fictitious material point with strain rate-dependency differs from the previous fictitious material point with only path-dependency. At the end, we also refer to another interesting feature to be explored in future works: the ability of the network to extrapolate to a different set of material properties without the need to retrain.
<img src="https://github.com/SLIMM-Lab/pyprnn/blob/main/animations/matPointWRateDepAndDifProps.gif" width="50%" height="50%"/>

