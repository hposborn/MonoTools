# MonoTools: A python package for planets of uncertain period

MonoTools is a package specific to modelling transiting exoplanets with unknown periods (Monos, Duos, Trios, etc).

Full documentation is are taking shape at [readthedocs](https://monotools.readthedocs.io/en/new/).

MonoTools.fit is an update to my [previous Namaste code](http://github.com/hposborn/Namaste) using the ["exoplanet" models of Dan Foreman-Mackey et al](http://github.com/dfm/exoplanet).

This uses a `PyMC`(v5)/`pytensor` back-end, a differentiable Keplerian orbital model, and the Gaussian process modeling of [Celerite](http://github.com/dfm/celerite) to fit exoplanetary transits. For the old and buggy PyMC3 backend-based, manually install the `old_pymc3` branch via github.

Here we wrap the ["exoplanet"](http://github.com/dfm/exoplanet) core model to specify it for single and "duo"-transits, and plot/analyse the results.

The first use of `MonoTools` in a published paper can be found in [Osborn+ (2022)](https://arxiv.org/abs/2203.03194) where MonoTools helped recover the orbits of the two outer planets in the TOI-2076 system. Additionally, it can be directly cited thanks it's position the [Astrophysics Source Code Library](https://www.ascl.net/2204.020).

### Full installation and usage advice on ["ReadTheDocs"](https://monotools.readthedocs.io/en/main)

#### Installing
To install, I recommend using a virtual environment, as some of the packages required are not at their most recent versions.
MonoTools should be pip installable, therefore run `pip install MonoTools`.

Alternatively, to run the most up-to-date development version, you can run `git clone http://github.com/hposborn/MonoTools`, `cd` into the MonoTools folder, then run `pip install .` (plus make sure the folder where MonoTools is installed is included in your `$PYTHONPATH`, e.g. by adding `export PYTHONPATH=/path/to/dir:$PYTHONPATH` to your `.bashrc` file).

MonoTools will look at the `$MONOTOOLSPATH` bash variable as the location to store files, and defaults to `MonoTools/data`. Include this in your `.bashrc` file to modify this location.

#### Using MonoTools

For a tutorial on how to deal with the in-built lightcurve class, see [using_lightcurve.ipynb](https://github.com/hposborn/MonoTools/blob/main/docs/using_lightcurve.html).

Various examples for fitting applications can be found for:
- [Modelling Duotransits](https://github.com/hposborn/MonoTools/blob/pymc/docs/Example_1_Duotransit_Example.ipynb)
- [Modelling a Monotransit](https://github.com/hposborn/MonoTools/blob/pymc/docs/Example_2_Monotransit_and_duotransit.ipynb)
- [Modelling a lightcurve with five transits but an ambiguous period](https://github.com/hposborn/MonoTools/blob/pymc/docs/Example_3_Model_Ambiguous_Transits_simple.ipynb)
- [and modelling the same pentatransit including TTVs and a GP](https://github.com/hposborn/MonoTools/blob/pymc/docs/Example_4_Modelling_Ambiguous_Transits_with_TTVs_and_GPs.ipynb)
