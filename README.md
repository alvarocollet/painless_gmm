# Tutorial on Painless Gaussian Mixture Models 

## Introduction to Painless GMM

A Gaussian Mixture Model (GMM) is a probability distribution defined as a linear combination of weighted Gaussian distributions. It is commonly used in computer vision and image processing tasks, such as estimating a color distribution for foreground/background segmentation. This project is intended as an *educational* tool on how to properly implement a Gaussian Mixture Model.

GMMs are annoying to implement. The math behind GMMs is very easy to understand, but it is not possible to take the formulas and implement them directly. A straight implementation of the GMM formulas leads to underflow errors, singular matrices, divisions-by-zero, and NaNs. The likelihoods involved in GMM are very frequently too small to be directly represented as floating-point numbers (and, even more so, their multiplication). In the following paragraphs and code, I show the changes needed
to take GMM from theory to a robust real-world implementation. Therefore, this is an implementation of GMM without the pain: a Painless GMM.

## GMM: The theory

A GMM is a probability distribution defined as a linear combination of ![equation](https://latex.codecogs.com/gif.latex?k) weighted Gaussian distributions,

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![equation](https://latex.codecogs.com/gif.latex?P_%7BGMM%7D%28z_i%20%7C%20%5Cvi%7B%5Cpi%7D%2C%20%5Cvi%7B%5Cmu%7D%2C%20%5Cvi%7B%5CSigma%7D%29%20%3D%20%5Csum_k%20%5Cpi_k%20%5Cmathcal%7BN%7D%28z_i%20%7C%20%5Cmu_k%2C%20%5CSigma_k%29%2C)

with weights ![equation](https://latex.codecogs.com/gif.latex?%24%5Cpi_k%24), means ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmu_k%24) and covariance matrices ![equation](https://latex.codecogs.com/gif.latex?%24%5CSigma_k%24). We simplify this notation in the following section as ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%20%7C%20k%29%20%3D%20%5Cmathcal%7BN%7D%28z_i%20%7C%20%5Cmu_k%2C%20%5CSigma_k%29%24), and ![equation](https://latex.codecogs.com/gif.latex?%24P%28k%29%20%3D%20%5Cpi_k%24). The GMM likelihood then becomes ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%29%20%3D%20%5Csum_k%20p%28z_i%7Ck%29%20P%28k%29%24). 

For more information about GMMs, visit Reynold's [gmm tutorial](http://www.ee.iisc.ernet.in/new/people/faculty/prasantg/downloads/GMM_Tutorial_Reynolds.pdf) or the [Wikipedia page](https://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model).

## Training a GMM with Expectation-Maximization (EM)
We start with a data set ![equation](https://latex.codecogs.com/gif.latex?%24%5Cvi%7Bz%7D) of ![equation](https://latex.codecogs.com/gif.latex?N) ![equation](https://latex.codecogs.com/gif.latex?d)-dimensional feature vectors ![equation](https://latex.codecogs.com/gif.latex?z_i) (e.g., ![equation](https://latex.codecogs.com/gif.latex?d%3D3) for RGB color pixels), an initial set of ![equation](https://latex.codecogs.com/gif.latex?K) Gaussian distributions (initialized as described below), and ![equation](https://latex.codecogs.com/gif.latex?K) weights ![equation](https://latex.codecogs.com/gif.latex?%24P%28k%29%24).  We use the Expectation-Maximization (EM) algorithm to optimize the Gaussian distributions and weights that maximize the global GMM likelihood ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%20%3D%20%5Cprod_i%20p%28z_i%29%24), that is, the mixture of Gaussian distributions and weights that best fit the data set ![equation](https://latex.codecogs.com/gif.latex?%24%5Cvi%7Bz%7D).

The EM algorithm is an optimization algorithm which maximizes ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%24) by coordinate ascent, alternating between expectation steps (E-steps) and maximization steps (M-steps). The algorithm starts with an initial E-step. 

In the **E-step**, we determine the responsibility ![equation](https://latex.codecogs.com/gif.latex?p%28k%7Cz_i%29) of each Gaussian distribution for each training data point ![equation](https://latex.codecogs.com/gif.latex?p%28z_i%29), as

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![equation](https://latex.codecogs.com/gif.latex?%24%24p_%7Bki%7D%20%5Ctriangleq%20p%28k%7Cz_i%29%20%3D%20%5Cfrac%7Bp%28z_i%7Ck%29P%28k%29%7D%7Bp%28z_i%29%7D%2C%24%24)

that is, we estimate how likely each Gaussian distribution is to generate the data point ![equation](https://latex.codecogs.com/gif.latex?z_i).

In the **M-step**, we re-estimate the gaussian distributions and weights given the responsibilities ![equation](https://latex.codecogs.com/gif.latex?p_%7Bki%7D).  In particular, we update ![equation](https://latex.codecogs.com/gif.latex?P%28k%29), ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmu_k%24) and ![equation](https://latex.codecogs.com/gif.latex?%24%5CSigma_k%24) as

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?%24%24%20P%28k%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_i%20p_%7Bki%7D%20%2C%24%24)  &nbsp; &nbsp; &nbsp; ![equation](https://latex.codecogs.com/gif.latex?%24%24%20%5Cmu_k%20%3D%20%5Cfrac%7B%5Csum_i%20p_%7Bki%7Dz_i%7D%7B%5Csum_i%20p_%7Bki%7D%7D%20%2C%24%24) &nbsp; &nbsp; &nbsp; ![equation](https://latex.codecogs.com/gif.latex?%5CSigma_k%20%3D%20%5Cfrac%7B%5Csum_i%20p_%7Bki%7D%28z_i%20-%20%5Cmu_k%29%28z_i%20-%20%5Cmu_k%29%5ET%7D%7B%5Csum_i%20p_%7Bki%7D%7D.)

Note that ![equation](https://latex.codecogs.com/gif.latex?p%28z_i%29) and ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmu_k%24) are considered column vectors, so that the outer product ![equation](https://latex.codecogs.com/gif.latex?%24%28z_i%20-%20%5Cmu_k%29%28z_i%20-%20%5Cmu_k%29%5ET%24) results in a ![equation](https://latex.codecogs.com/gif.latex?%243%5Ctimes3%24) matrix.

We then alternate between the E-Step and M-step until ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%24) does not increase significantly anymore. For example, a common stopping criterion is when ![equation](https://latex.codecogs.com/gif.latex?%24p%28%5Cvi%7Bz%7D%29%24) increases less than 0.01%, or after 100 iterations.

## Avoiding underflow errors
The procedure described above for GMM training is correct, but it is not possible to implement directly. A straight implementation of the previous formulas leads to underflow errors and singular matrices, which we must avoid in a robust implementation.

The likelihoods and responsibilities involved in GMM are very frequently too small to be directly represented as floating-point numbers (and, even more so, their multiplication). An effective solution of the underflow problem is to use log likelihoods and the `logsumexp` trick.

First, we must use the Gaussian log likelihood ![equation](https://latex.codecogs.com/gif.latex?%24l%28z_i%7Ck%29%20%3D%20%5Clog%20p%28z_i%20%7C%20k%29%24) instead of
the linear likelihood, as

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?%24%24%20l%28z_i%7Ck%29%20%3D%20%5Clog%20p%28z_i%7Ck%29%20%3D%20-%5Cfrac%7B1%7D%7B2%7D%20%5Clog%20%28%20%282%5Cpi%29%5Ed%20%5C%7C%5CSigma_k%5C%7C%29%20-%20%5Clog%20%5Cleft%28%20%5Cfrac%7B1%7D%7B2%7D%20%28z_i%20-%20%5Cmu_k%29%5ET%20%5CSigma_k%5E%7B-1%7D%20%28z_i%20-%20%5Cmu_k%29%20%5Cright%29.%20%24%24)

We must also use the log likelihood of the whole GMM instead of the linear likelihood, but this calculation is slightly more convoluted. Given that the formula ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%29%20%3D%20%5Csum_k%20p%28z_i%7Ck%29%20P%28k%29%24) performs likelihood additions, the use of ![equation](https://latex.codecogs.com/gif.latex?%24%5Clog%20p%28z_i%29%24) does not pose any immediate advantage (because we cannot directly add log likelihoods).  We use instead the `logexpsum` trick `LSE()`, which
states that

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?LSE%28z_i%29%20%5Ctriangleq%20%5Clog%20%5Cleft%28%20%5Csum_i%20z_i%20%5Cright%29%20%3D%20%5Clog%20z_%5Ctext%7Bmax%7D%20&plus;%20%5Clog%20%5Cleft%28%20%5Csum_i%20%5Cexp%20%5Cleft%28%20%5Clog%20z_i%20-%20%5Clog%20z_%5Ctext%7Bmax%7D%20%5Cright%29%20%5Cright%29.)

In `LSE()`, we scale the ![equation](https://latex.codecogs.com/gif.latex?N) terms in the summation by the largest term ![equation](https://latex.codecogs.com/gif.latex?z_%5Ctext%7Bmax%7D) and convert the scaled terms to linear domain instead.

Let us give an example of the `logexpsum` trick at work.  Let ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_1%29%20%3D%20%5Cexp%28-1000%29%24) and ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_2%29%20%3D%20%5Cexp%28-1001%29%24). We wish to compute ![equation](https://latex.codecogs.com/gif.latex?%24X%20%3D%20%5Clog%28p%28z_1%29&plus;p%28z_2%29%29%24).  The direct evaluation of ![equation](https://latex.codecogs.com/gif.latex?%24X%20%3D%20%5Clog%28p%28z_1%29&plus;p%28z_2%29%29%24)
requires calculating ![equation](https://latex.codecogs.com/gif.latex?%24%5Cexp%28-1000%29%24) and ![equation](https://latex.codecogs.com/gif.latex?%24%5Cexp%28-1001%29%24), which causes underflow (regardless of the representation, `float` or `double`), and therefore ![equation](https://latex.codecogs.com/gif.latex?%24X%20%3D%20%5Clog%20%280%20-%200%29%20%3D%20%5Clog%200%20%3D%20-%5Cinf%24). Using `logexpsum`, this is ![equation](https://latex.codecogs.com/gif.latex?%24X%20%3D%20-1000%20&plus;%20log%20%5Cleft%28%5Cexp%280%29%20&plus;%20%5Cexp%28-1%29%5Cright%29%20%5Capprox%20-999.7%24).

The GMM log likelihood can be expressed as ![equation](https://latex.codecogs.com/gif.latex?%24l%28z_i%29%20%3D%20%5Clog%20p%28z_i%29%20%3D%20LSE%28p%28z_i%7Ck%29%20P%28k%29%29%24). Given that we already calculate ![equation](https://latex.codecogs.com/gif.latex?%24l%28z_i%7Ck%29%24) instead of ![equation](https://latex.codecogs.com/gif.latex?%24p%28z_i%7Ck%29%24), the GMM log likelihood becomes

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?l_%5Ctext%7Bmax%7D%28z_i%29%20%26%3D%26%20%5Cmax_k%20%28l%28z_i%7Ck%29%20&plus;%20%5Clog%20P%28k%29%29)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?l%28z_i%29%20%26%3D%26%20%5Clog%20%5Csum_k%20p%28z_i%7Ck%29%20P%28k%29%20%3D%20l_%5Ctext%7Bmax%7D%28z_i%29%20&plus;%20%5Csum_k%20%5Cexp%28%20l%28z_i%7Ck%29%20&plus;%20%5Clog%20P%28k%29%20-%20l_%5Ctext%7Bmax%7D%28z_i%29%29.)


Analogously, the E-step becomes

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?%24%24l_%7Bki%7D%20%5Ctriangleq%20l%28k%7Cz_i%29%20%3D%20%5Clog%20p%28k%7Cz_i%29%20%3D%20l%28z_i%7Ck%29%20&plus;%20%5Clog%20P%28k%29%20-%20l%28z_i%29%24%24)

and the responsibilities ![equation](https://latex.codecogs.com/gif.latex?p_%7Bki%7D) are computed from ![equation](https://latex.codecogs.com/gif.latex?l_%7Bki%7D), i.e., ![equation](https://latex.codecogs.com/gif.latex?%24p_%7Bki%7D%20%3D%20%5Cexp%28l_%7Bki%7D%29%24).

The M-step does not require any changes to prevent underflows. Finally, 
the global GMM log likelihood becomes

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![equation](https://latex.codecogs.com/gif.latex?l%28%5Cvi%7Bz%7D%29%20%3D%20%5Clog%20p%28%5Cvi%7Bz%7D%29%20%3D%20%5Csum_i%20l%28z_i%29)




