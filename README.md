# NanoporeDataAugmenter

Contains method for augmenting Nanopore read data, although it can be applied to other 1D data. 

The customTFOps has a routine to compute Brownian Motion Data Augmentation on tensorflow. The CuPy implementation can augment using noise addition, linear stretching, magnitude multiplication and Brownian Motion data augmentation together, in an optimized function. If you use this code for a research article, please cite "Brownian motion data augmentation: a method to push neural network performance on nanopore sensors".
