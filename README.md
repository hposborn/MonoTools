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

Full documentation is are taking shape at [readthedocs](https://monotools.readthedocs.io/en/main/).

MonoTools.fit is an update to my [previous Namaste code](http://github.com/hposborn/Namaste) using the ["exoplanet" models of Dan Foreman-Mackey et al](http://github.com/dfm/exoplanet).

This uses a PyMC3/Theano back-end, a differentiable Keplerian orbital model, and the Gaussian process modeling of [Celerite](http://github.com/dfm/celerite) to fit exoplanetary transits.

Here we wrap the ["exoplanet"](http://github.com/dfm/exoplanet) core model to specify it for single and "duo"-transits, and plot/analyse the results.

The first use of `MonoTools` in a published paper is out now in [Osborn+ (2022)](http://arxiv.org) where MonoTools helped recover the orbits of the two outer planets in the TOI-2076 system.

### Full installation and usage advice on ["ReadTheDocs"](https://monotools.readthedocs.io/en/main)

#### Installing
To install, I recommend using a virtual environment, as some of the packages required are not at their most recent versions. This avoids dependency management and adoids any clashes with your system packages. To create a new virtual environment (which we call MONO, but you can call it whatever you like):

```
python -m venv MONO
source MONO/bin/activate
```
To 'exit' the virtual environment, simply type ```deactivate``` in your terminal. If you are running MonoTools in a jupyter notebook using a virtual environment you have to make sure you install a kernel which matches your virtual environment and then select it for the notebook. This is done by executing the following in a terminal:

```ipython kernel install --name "MONO" --user```

MonoTools should be pip installable, therefore run `pip install MonoTools`.

Alternatively, to run the most up-to-date development version, you can run `git clone http://github.com/hposborn/MonoTools`, `cd` into the MonoTools folder, then run `pip install .` (plus make sure the folder where MonoTools is installed is included in your `$PYTHONPATH`, e.g. by adding `export PYTHONPATH=/path/to/dir:$PYTHONPATH` to your `.bashrc` file).

MonoTools will look at the `$MONOTOOLSPATH` bash variable as the location to store files, and defaults to `MonoTools/data`. Include this in your `.bashrc` file to modify this location.

Mac OSX users may need to make sure GCC is correctly installed (.e.g with `brew install gcc`) and C libraries are present.

#### Using MonoTools

For a tutorial on how to deal with the in-built lightcurve class, see [using_lightcurve.ipynb](https://github.com/hposborn/MonoTools/blob/main/docs/using_lightcurve.html).

For examples how to run the tools (e.g. lightcurve, stellar parameters, etc) and the search (monotransit search and vetting) functions, see [using_fit.ipynb](https://github.com/hposborn/MonoTools/blob/main/docs/using_search.html) [TBD].
https://github.com/hposborn/MonoTools/blob/master/MonoTools/examples/Search_Example.ipynb).

For examples on how to run the fitting code (including with RVs), see [using_fit.ipynb](https://github.com/hposborn/MonoTools/blob/main/docs/using_fit.ipynb).

#### Extra steps to install:
If stellar parameters are not given and not accessible from the input catalogues provided, Dan Huber's ["Isoclassify"](https://github.com/danxhuber/isoclassify) is required to estimate density given info scraped in the `stellar` module from Gaia, input catalogues, wide-field surveys, spectra, etc. A modified version is included. However, the "mesa.h5" file must be downloaded using `wget https://www.dropbox.com/s/vrr8hc7qav1fzyb/mesa.h5?dl=0` in `stellar/isochrones`. mwdust modules may also require specific installation.-->
