# expanding-ejecta
[comment]:[![arXiv](https://img.shields.io/badge/arXiv-2301.00822%20-green.svg)](https://arxiv.org/abs/24XX.XXXXX)

This GitHub page is the home of expanding-ejecta; the code for generating P Cygni line profile in a ellipsoidal supernova, calculating the intensity correlator of the image as seen by an intensity interferometer described in [25XX.XXXXX](https://arxiv.org/abs/24XX.XXXXX) capable of measuring supernova morphology and Hubble constant. This code is JAX compatible with just-in-time speed up and autograd for potential faster Markov-Chain Monte-Carlo (MCMC) methods.

See the Jupyter Notebook `demo.ipynb` for a quick demonstration on how to create an image of a supernova and its intensity correlator as well as the JAX compatibility of the code

![RingFlux](/plots/ellipsoid.png)
![alternative text](/plots/EEM_animation.gif)

If this pipeline is used in published work, please cite [24XX.XXXXX](https://arxiv.org/abs/24XX.XXXXX).

## Authors

- I-Kai Chen
- David Dunsky
- Junwu Huang
- Ken Van Tilburg

## model

The `model` folder contains the code for generating the image of P Cygni profile, including spatially integrated spectra, 2D image in polar coordinates, and the intensity correlator (which is the Fourier transform of the image):
- `P_cygni_ellipsoid_jax.py` contains the class `P_cygni_ellipsoid` which create a supernova with the parameters specified in [25XX.XXXXX](https://arxiv.org/abs/24XX.XXXXX). It contains methods to calculate the velocity field, optical depth, and geometric dilution factor of the ejecta surrounding the supernova. It then uses these to calculate the spectral radiance from absorption and emission from the line-forming ejecta.
- `hankel_ellipsoid_jax.py` contains the class `Correlator_ellipsoid` which is an inherit class of `P_cygni_ellipsoid`. This class contains methods to calculate the intensity correlator of the image generated with the parent class using JAX compatible fast Hankel transformation up to the 7th multipole expansion. It also contains method to calculate the uncertainty of the intensity correlator given an observation of a pair of intensity interferometer telescope. A method to calculate the spatially integrated spectrum is also included in this class.
- `black_body_jax.py` contains the helper function for calculating the spectral radiance of an ideal black body spectrum. This is called by the class `Correlator_ellipsoid` to calculate the spatially integrated spectrum.
- `table` contains a table precomputed table of the geometric dilution factor `w_table.csv` which is the solid angle subtended by the photosphere as seen from a location away from the supernova. This table serves as an interpolation table when the geometric dilution factor is needed for the emission line profile when calculated in the class `P_cygni_ellipsoid`. This table is precomputed with high resolution using Monte-Carlo method.

## plots

The `PaperPlots` folder contains the Jupyter Notebook scripts used to generate the plots shown in [25XX.XXXXX](https://arxiv.org/abs/25XX.XXXXX).
