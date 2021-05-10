# MonoTools: A python package for planets of uncertain period

MonoTools is a package specific to the detection, vetting and modelling of transiting exoplanets, with a specific emphasis on Monotransiting planets, and those with unknown periods.

MonoSearch includes scripts specifically for searching and assessing a lightcurve for the presence of monotransits. This includes:
 - Searching for monotransits
 - Searching for periodic planets
 - Performing a best-fit transit model
 - Vetting whether detected monotransits are variability, asteroids (using background flux), background EB (with centroids), etc.
 - Assessing whether those transits are linked to any detected multi-transiting planet candidate, or with each other - e.g. a 2-transit duo.
 - Setting up the monotransit fitting code.
 - Fitting planets in a Bayesian way to account for uncertain periods (i.e. mono or duo-transits), lightcurve gaps, stellar variability, etc.
 - Compiling all the steps and various plots into a report for each planet candidate

Full documentation is are taking shape at [readthedocs](https://monotools.readthedocs.io/en/latest/).

MonoTools.fit is an update to my [previous Namaste code](http://github.com/hposborn/Namaste) using the ["exoplanet" models of Dan Foreman-Mackey et al](http://github.com/dfm/exoplanet).

This uses a PyMC3/Theano back-end, a differentiable Keplerian orbital model, and the Gaussian process modeling of [Celerite](http://github.com/dfm/celerite) to fit exoplanetary transits.

Here we wrap the ["exoplanet"](http://github.com/dfm/exoplanet) core model to specify it for single transits, and plot/analyse the results.

#### Installing
To install, I recommend using a virtual environment, as some of the packages required are not at their most recent versions.
MonoTools should be pip installable, therefore run `pip install MonoTools`.

Alternatively, to run the most up-to-date development version, you can run `git clone http://github.com/hposborn/MonoTools`, `cd` into the MonoTools folder, then run `pip install .` (plus make sure the folder where MonoTools is installed is included in your `$PYTHONPATH`, e.g. by adding `export PYTHONPATH=/path/to/dir:$PYTHONPATH` to your `.bashrc` file).

MonoTools will look at the `$MONOTOOLSPATH` bash variable as the location to store files, and defaults to `MonoTools/data`. Include this in your `.bashrc` file to modify this location.

Mac OSX users may need to make sure GCC is correctly installed (.e.g with `brew install gcc`) and C libraries are present.

#### Using MonoTools

For examples how to run the tools (e.g. lightcurve, stellar parameters, etc) and the search (monotransit search and vetting) functions, see [Search_Example.ipynb](https://github.com/hposborn/MonoTools/blob/master/examples/Search_Example.ipynb).

For examples on how to run the fitting code, see [Fit_Example.ipynb](https://github.com/hposborn/MonoTools/blob/master/examples/Fit_Example.ipynb).

For info on how to include RVs in the fitting, see [Fit_with_RVs_Example.ipynb](https://github.com/hposborn/MonoTools/blob/master/examples/Fit_with_RVs_Example.ipynb).

<!--See [Search_Example.ipynb](https://github.com/hposborn/MonoTools/blob/master/Example.ipynb) for an example of how to use it.-->
<!--If that doesn't render, try [this nbviewer link](https://nbviewer.jupyter.org/github/hposborn/MonoTools/blob/master/Example.ipynb)-->

<!--To run the entire process from the command line, you can use `python -m MonoTools.main 000ID00 MISSION`-->

![alt text](https://github.com/hposborn/MonoTools/blob/master/data/TIC00270341214/00270341214_Monotransit_Search.png)

<!--MonoTools natively includes eccentricity and fits for period (rather than scaled transit velocity as before).-->
<!--The period prior can be applied according to your prefered index - P^-8/3 (as suggested by [Kipping et al](https://iopscience.iop.org/article/10.3847/2515-5172/aaf50c) ), or the less-steep P^-5/3 (as used by [Osborn et al 2016](https://academic.oup.com/mnras/article/457/3/2273/2588921) and [Sandford et al 2019](https://arxiv.org/abs/1908.08548))-->

#### Extra steps to install:

If stellar parameters are not given and not accessible from the input catalogues provided, Dan Huber's ["Isoclassify"](https://github.com/danxhuber/isoclassify) is required to estimate density given info scraped in the `stellar` module from Gaia, input catalogues, wide-field surveys, spectra, etc. A modified version is included. However, the "mesa.h5" file must be downloaded using `wget https://www.dropbox.com/s/vrr8hc7qav1fzyb/mesa.h5?dl=0` in `stellar/isochrones`. mwdust modules may also require specific installation.
