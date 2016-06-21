# Painless Gaussian Mixture Models 

## Introduction to Painless GMM

A Gaussian Mixture Model (GMM) is a probability distribution defined as a linear combination of weighted Gaussian distributions. It is commonly used in computer vision and image processing tasks, such as estimating a color distribution for foreground/background segmentation. This project is intended as an *educational* tool on how to properly implement a Gaussian Mixture Model.

GMMs are annoying to implement. The math behind GMMs is very easy to understand, but it is not possible to take the formulas and implement them directly. A straight implementation of the GMM formulas leads to underflow errors, singular matrices, divisions-by-zero, and NaNs. The likelihoods involved in GMM are very frequently too small to be directly represented as floating-point numbers (and, even more so, their multiplication). In the following paragraphs and code, I show the changes needed
to take GMM from theory to a robust real-world implementation. Therefore, this is an implementation of GMM without the pain: a Painless GMM.

## GMM: The theory

A GMM is a probability distribution defined as a linear combination of ![equation](https://latex.codecogs.com/gif.latex?k) weighted Gaussian distributions,

![equation](https://latex.codecogs.com/gif.latex?P_%7BGMM%7D%28z_i%20%7C%20%5Cvi%7B%5Cpi%7D%2C%20%5Cvi%7B%5Cmu%7D%2C%20%5Cvi%7B%5CSigma%7D%29%20%3D%20%5Csum_k%20%5Cpi_k%20%5Cmathcal%7BN%7D%28z_i%20%7C%20%5Cmu_k%2C%20%5CSigma_k%29%2C)

with weights ![equation](https://latex.codecogs.com/gif.latex?%24%5Cpi_k%24), means ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmu_k%24) and covariance matrices ![equation](https://latex.codecogs.com/gif.latex?%24%5CSigma_k%24). We simplify this notation in the following section as ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%20%7C%20k%29%20%3D%20%5Cmathcal%7BN%7D%28z_i%20%7C%20%5Cmu_k%2C%20%5CSigma_k%29%24), and ![equation](https://latex.codecogs.com/gif.latex?%24P%28k%29%20%3D%20%5Cpi_k%24). The GMM likelihood then becomes ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%29%20%3D%20%5Csum_k%20p%28z_i%7Ck%29%20P%28k%29%24). 

For more information about GMMs, visit Reynold's [gmm tutorial](http://www.ee.iisc.ernet.in/new/people/faculty/prasantg/downloads/GMM_Tutorial_Reynolds.pdf) or the [Wikipedia page](https://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model).

## Training a GMM with Expectation-Maximization (EM)
We start with a data set ![equation](https://latex.codecogs.com/gif.latex?%24%5Cvi%7Bz%7D) of ![equation](https://latex.codecogs.com/gif.latex?N) ![equation](https://latex.codecogs.com/gif.latex?d)-dimensional feature vectors ![equation](https://latex.codecogs.com/gif.latex?z_i) (e.g., ![equation](https://latex.codecogs.com/gif.latex?d%3D3) for RGB color pixels), an initial set of ![equation](https://latex.codecogs.com/gif.latex?K) Gaussian distributions (initialized as described below), and ![equation](https://latex.codecogs.com/gif.latex?K) weights ![equation](https://latex.codecogs.com/gif.latex?%24P%28k%29%24).  We use the Expectation-Maximization (EM) algorithm to optimize the Gaussian distributions and weights that maximize the global GMM likelihood ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%20%3D%20%5Cprod_i%20p%28z_i%29%24), that is, the mixture of Gaussian distributions and weights that best fit the data set ![equation](https://latex.codecogs.com/gif.latex?%24%5Cvi%7Bz%7D).

The EM algorithm is an optimization algorithm which maximizes ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%24) by coordinate ascent, alternating between expectation steps (E-steps) and maximization steps (M-steps). The algorithm starts with an initial E-step. 


