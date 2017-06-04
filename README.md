# Auto-Encoder-toolbox
This is the Auto-Encoder toolbox for the paper "Fengling Mao, Wei Xiong, Bo Du and Lefei Zhang. Stochastic Decorrelation Constraint Regularized Auto-Encoder for Visual Recognition. MultiMedia Modeling(MMM). Springer International Publishing, 2017" --Fengling Mao.


--Introduction----------------------------

1. A matlab toolbox for Auto-Encoder with whitening, convolution, pooling, classification.

2. This Auto-Encoder toolbox is programmed based on the Deep Learn Toolbox (https://github.com/rasmusbergpalm/DeepLearnToolbox).

3. The simple example of using this toolbox is a small experiment of our paper-"Stochastic Decorrelation Constraint Regularized Auto-Encoder for Visual Recognition" on the MNIST dataset.

4. If you use this toolbox in your research please cite https://github.com/FenglingM/Auto-Encoder-toolbox .

--How to use the example------------------

1. addpath the toolbox.
2. open the initL2.m to initialize the network parameters.
3. run the DAE_MNIST.m.
4. we add the Stochastic Decorrelation Constraint Regularizer in AEcost.m
