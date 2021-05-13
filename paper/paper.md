---
title: 'MonoTools -- A python package for planets of uncertain period'

tags:
  - Python
  - astronomy
  - exoplanets
  - transit

authors:
  - name: Hugh P. Osborn
    orcid: 0000-0002-4047-4724
    affiliation: 1, 2


affiliations:
 - name: NCCR/Planet S, Centre for Space and Habitability, University of Bern, Switzerland
   index: 1
 - name: Kavli Institute for Space Sciences, Massacussets Institute of Technology, Cambridge, MA, USA
   index: 2
date: 1 June 2021
bibliography: paper.bib

---

# Summary

The transit method has proved the most productive technique for detecting extrasolar planets, especially since the era of space-based photometric survey missions began with *CoRoT* [@auvergne2009corot] and *Kepler* [@borucki2010kepler] in the late 2000s.
This continued with *K2* [@howell2014k2] and *TESS* [@ricker2014transiting], and will extend into the 2030s with *PLATO* [@rauer2014plato].
Typically, the planets detected by these surveys show multiple consecutive transits.
This means planet candidates are most often detected through algorithms which search the frequency domain [e.g. @kovacs2002box, @hippke2019optimized], vetted using metrics that require multiple detected transits [e.g. @thompson2018planetary, @shallue2018identifying], and modelled (and sometimes statistically validated) using the assumption that the orbital period is well-constrained and approximated by a Gaussian distribution [e.g. @eastman2013exofast, @morton2012efficient].
However, planet candidates continue to be found that do not show multiple consecutive transits - the single (or "Mono-") transits [e.g. @wang2015planet, @osborn2016single, @gill2020ngts].
For these transit candidates - where orbital period is not a priori known from the detection - a special approach to exoplanet detection, vetting and modelling must be taken.

In this work, we detail ``MonoTools``, a python package capable of performing detection, vetting and modelling of Mono (and Duo) transit candidates.
First we will describe briefly what Mono (and Duo-) transits are, and the challenges associated with them.
Then in the following three sections we will outline the basis of the three parts of the code.
Following that, we will validate the code using limited examples of planets with known orbital periods.

# Use Cases - Mono & Duo transits

Mono-transits, which have also variously been called "single transits" or "orphan transits", are the transits of long-period planet candidates, ideally planets but also astrophysical false positives, which occur only once during photometric observations.
In these cases, the orbital period is not directly evident as we do not have subsequent transits.

Another special case is worth noting - that of two non-consecutive transits where intermediate transit events were not observed, therefore the orbital period cannot be directly observed.
Here I class these cases as "Duotransits" in contrast to "Monotransits" and "Multitransits" which we will use as short-hand for planet candidates which show multiple (and consecutive) transit events.

Transit shape is universally governed by the same simple geometry [e.g. @mandel2002analytic, @seager2003unique].
As they must be strongly detected in a single transit, the typical per-transit signal-to-noise of monotransits is often higher than for multitransits, allowing their shape to be well-constrained.
It is this shape, eventually, which allows vetting and modelling of the planet candidate.
Transit duration is weakly dependent on planetary period ($t_D \alpha P^{1/3}$), therefore long-period planets typically have longer-duration transits.
Indeed the longest-duration transit yet found belonged to a monotransit detected in K2 [@giles2018transiting] at 54 hours


# Search

# Vetting

# Fitting

# Validation

# Installation

# Acknowledgements

# References
