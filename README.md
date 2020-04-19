# NamastePymc3: An MCMC Analysis of Single Transiting Exoplanets with Pymc3

NamastePymc3 is an update to my [previous Namaste code](http://github.com/hposborn/Namaste) using the ["exoplanet" models of Dan Foreman-Mackey et al](http://github.com/dfm/exoplanet). 
This uses a PyMC3/Theano back-end, a differentiable Keplerian orbital model, and the Gaussian process modeling of [Celerite](http://github.com/dfm/celerite) to fit exoplanetary transits.

Here we simply provide a pre-written  ["exoplanet"](http://github.com/dfm/exoplanet) model specific to single transits, and plot/analyse the results.

See [Example.ipynb](https://github.com/hposborn/NamastePymc3/blob/master/Example.ipynb) for an example of how to use it.
If that doesn't render, try [this nbviewer link](https://nbviewer.jupyter.org/github/hposborn/NamastePymc3/blob/master/Example.ipynb).


![alt text](https://github.com/hposborn/NamastePymc3/blob/master/EPIC00248847494/EPIC00248847494_2019-09-06_0_TransitFit.png)

NamastePymc3 natively includes eccentricity and fits for period (rather than scaled transit velocity as before).
The period prior can be applied according to your prefered index - P^-8/3 (as suggested by [Kipping et al](https://iopscience.iop.org/article/10.3847/2515-5172/aaf50c) ), or the less-steep P^-5/3 (as used by [Osborn et al 2016](https://academic.oup.com/mnras/article/457/3/2273/2588921) and [Sandford et al 2019](https://arxiv.org/abs/1908.08548))

If stellar parameters are not given, Dan Huber's ["Isoclassify"](https://github.com/danxhuber/isoclassify) is required to estimate density given info scraped in the `stellar` module from Gaia, input catalogues, wide-field surveys, spectra, etc. A modified version is included. However, the "mesa.h5" file must be downloaded using `wget https://www.dropbox.com/s/vrr8hc7qav1fzyb/mesa.h5?dl=0` in `stellar/isochrones`.
