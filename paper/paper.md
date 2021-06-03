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
This means planet candidates are most often detected through algorithms which search the frequency domain [e.g.; @kovacs2002box; @hippke2019optimized], vetted using metrics that require multiple detected transits [e.g.; @thompson2018planetary; @shallue2018identifying], and modelled (and sometimes statistically validated) using the assumption that the orbital period is well-constrained and approximated by a Gaussian distribution [e.g.; @eastman2013exofast; @morton2012efficient].
However, planet candidates continue to be found that do not show multiple consecutive transits - the single (or "Mono-") transits [e.g.; @wang2015planet; @osborn2016single; @gill2020ngts].
For these transit candidates - where orbital period is not a priori known from the detection - a special approach to exoplanet detection, vetting and modelling must be taken.

In this work, we detail ``MonoTools``, a python package capable of performing detection, vetting and modelling of Mono (and Duo) transit candidates.
First we will describe briefly what Mono (and Duo-) transits are, and the challenges associated with them.
Then in the following three sections we will outline the basis of the three parts of the code.
Following that, we will validate the code using limited examples of planets with known orbital periods.

# Mono- & Duo-transits

Mono-transits, which have also variously been called "single transits" or "orphan transits", are the transits of long-period planet candidates which occur only once during photometric observations.
In these cases, the orbital period is not directly evident as we do not have subsequent transits.
However, the planet's orbit can be constrained using the transit event, as we will explore later in this section.

Another special case is worth noting - that of two non-consecutive transits where intermediate transit events were not observed, therefore the orbital period is not directly constrained by the transit events.
Here I class these cases as "Duotransits" in contrast to "Monotransits" and "Multitransits" which we will use as short-hand for planet candidates which show multiple (and consecutive) transit events.
In these cases, the resulting planet candidate may have both a highly uncertain period ($20{\rm d}< P <750{\rm d}$ in the case of two transits separated by a 2-year gap) and yet a well-constrained array of possible periods to search ($P \in (t_{{\rm tr},2}-t_{{\rm tr},1})/\{1,2,3, \cdots, N_{\rm max}\}$).

``MonoTools`` is explicitly dealt to deal with both the monotransit and duotransit cases.

Transit shape is universally governed by the same simple geometry [e.g. @mandel2002analytic, @seager2003unique].
As they must be strongly detected in a single transit, the typical per-transit signal-to-noise of monotransits is often higher than for multitransits, allowing their shape to be well-constrained.
This shape is important for detection, vetting and modelling of such planet candidates.
Transit duration is weakly dependent on planetary period ($t_D \propto P^{1/3}$), therefore long-period planets typically have longer-duration transits.
Indeed the longest-duration transit yet found belonged to a monotransit detected in K2 [@giles2018transiting] at 54 hours.

# Input Information

- **Detrended Lightcurve.**

- **Supplementary Photometric data.**

- **Stellar parameters.**


# Search

## ``MonoTools.search.MonoTransitSearch``

This function iteratively fits both a transit model and a polynomial to the lightcurve to detect monotransits in space telescope photometry, which we detail here.

We first create a series of reference transit models (default 5) to iterate across the lightcurve using ``exoplanet`` [@foreman2021exoplanet].
The derived stellar parameters are used, along with a default planet-to-star radius ratio of 0.1.
As input periods, logspaced values between 0.4 and 2.5 times the duration of continuous observations (in the case of lightcurves with gaps longer than 5 days, the longest individual region was used).
The impact parameters were chosen such that the maximum duration transit (with $P=2.5P_{\rm mono}$) is given $b=0.0$ while successively shorter durations linearly spaced up to $b=0.85$ producing ever-shorter duration transits.
500 in-transit steps are generated for each model with exposure times fixed to that of the lightcurve, and then interpolated.
This interpolated transit function forms the model which is minimized at each step in the lightcurve.

Each of the models (with differing transit durations) are then iterated over the lightcurve, where transit centres are shifted some small fraction of transit duration each iteration (default 5\%).
At each position, a 7-transit-duration-long window around the transit time is fitted to three different models which are minimised using ``scipy.optimize``. These models are:
- The interpolated transit model with varying depth (reparameterised to $\log{\rm depth}$ to avoid negative depths) plus a 1D gradient in the out-of-transit flux.
- A 3-degree polynomial.
- A "wavelet" model with the following equation, designed to fit dips due to stellar variability where $t_D$ is the transit duration (set, in our case, from the interpolated transit models), and $a$ is the depth. As with the transit, a gradient was also included to account for any non-linear out-of-eclipse flux trend.
$$
t' = 2\pi x / (2 t_D);  
F = {a}(\exp{((-t'^2) / (2\pi^2))}\sin{(t'-\pi/2)})
$$
<!---The shift of 0.1 is included to depress the median flux below 0.0, which is the case for dips which mimic a transit.
For each model, a likelihood is calculated and the function minimised. Bayesian Information Criterion and likelihood ratios are then calculated between each false positive model and the transit model. A transit SNR is also calculated assuming white noise.-->
For each of these three models, the minimised log likelihood is used to compute a Bayesian Information Criterion.
Significant detections are therefore found by choosing all transit model fits which have a log likelihood ratio with respect to non-transit models greater than some threshold (default: 4.5) as well as an SNR (calculated from the depth, transit duration, and out-of-transit RMS) greater than some SNR threshold (default: 6.0).

Multiple iterations (either in transit time or duration) may find the same significant dip.
In this case the minimum DeltaBIC between transit & polynomial model is used to choose the representative detecion, and all nearby detections within $0.66 t_D$ of this candidate are removed to avoid double counting.
<!---TThis it iterated until no detections are classed as significant, or 8 high-SNR transit have been found.-->

## ``MonoTools.search.PeriodicPlanetSearch``
Many multitransiting planets produce high-SNR individual transits that would be detected using ``MonoTransitSearch``, therefore we also require a method of detecting periodic planets, as well as selecting between the monotransit and multitransit cases.

To search for periodic transits, we first flatten long-timescale variation from the lightcurve.
This is performed by fitting polynomials to sections of the lightcurve while also iteratively removing anomalies, as was adapted from [@armstrong2014abundance].
For each small step along the lightcurve, a wide window around (but not including) each step is used to fit a polynomial.
Points in this window that had already been identified as either outliers (i.e. from detrending) or within detected monotransits (from the Monotransit search), can be excluded from the polynomial fitting.
A log likelihood is computed on each of ten iterated polynomial fits, and each time a new pseudo-random mask is generated by excluding points whose scaled residual to the model is greater than a randomly-generated absolute normal distribution with unit standard deviation (thereby, on average, excluding points with offset residuals).
This best-fit polynomial, is then subtracted from the small central masked region.
For Periodic Planet searches, a window with duration 11 times the likely maximum duration and a stepsize of 0.1 days are typically used to ensure transits do not influence the polynomial fit.

``transit least squares`` [TLS; @hippke2019optimized] is used to perform the periodic planet search.
We iteratively run this TLS search and masked the detected transits until no more candidates are found above the SNR threshold (default:6)

During the TLS search, we necessitated a minimum of three transits.
This is preferred over a limit of two for a handful reasons:
- The implementation of period-epoch values in ``transit least squares`` means that allowing two transits also lets monotransits be detected, thereby duplicating our effort with the above search technique.
- Multi-transit search is not strict about assigning only similar dips together and may connect either two monotransits, or the wrong two transits from a multi-transiting planet. Requiring three dips ensures the correct periodicity
- Individual transits of the majority of good duo-transiting planet are likely to be individually detectable on their own right, as the individual transits have SNR's only $1/\sqrt{2}$ (30\%) lower than the combination of both events.
To make sure that at least 3 transits were detected, we excluding any candidates where one or two individual transits dominated the combined SNR (defined by computing an expected SNR from the sum of each individual transit SNRs and assuring solid detections have ${\rm SNR}_i > 0.5 {\rm SNR}_{\rm expected}$).
If the highest periodogram peak in the TLS corresponds to a multi-transiting planet with a SNR higher than our threshold (default: 6), and

In either case, if a signal with SNR higher that the threshold is found, we mask the detected transits by replacing all points associated with the transit with flux values randomly taken from the rest of the lightcurve.
The lightcurve is then re-scanned with \texttt{TLS} until no high-SNR candidates remain.

# Vetting

# Fitting

## Typical Monotransit fitting approaches

We have the following information available from a monotransit:

- Epoch of transit, $t_0$

- Transit duration, $t_D$

- Ingress & Egress duration $\tau$

- Transit depth, $\delta$

- In-transit shape

- Stellar parameters (e.g. stellar radius and density)

- Orbital period information from the lack of additional transits in the lightcurve. At the least we have a minimum possible period below which obvious transits would be observed, and at the most we may have a complex sequence of period islands.

- Additional planets

- Complimentary observations (e.g. radial velocities)

From these observables, there are then second order parameters. These can either be derived from the observables or, more commonly, can be used directly in fitting as reparameterisations of the observed parameters:

- **Limb-darkening parameters** - These parameters due to the change in optical depth as a function of position on the stellar surface correspond to the in-transit shape and are also constrainable from the stellar parameters (as theoretical limb-darkening parameters can be calculated for a given star).

- **Radius ratio, $R_p/R_s$** - This is most directly linked to transit depth $\delta$ ($R_p/R_s \sim \sqrt{\delta}$), although limb-darkening and dilution can play effects here (as well as impact parameter in the case of a grazing transit/eclipse).

- **Impact parameter, $b$** - Impact parameter refers to the location of the transit chord between the centre and edge of the stellar disc. In the case of multitransiting planets impact parameter constraints come from both the transit shape and the known orbital distance compared with the transit duration. With monotransits we do not have this luxury and instead only the transit shape constrains b (i.e. the radius ratio, ingress duration, transit duration).

These parameters can then in turn be linked to orbital parameters.
Typical transit modelling includes parameters for both transit shape (e.g. impact parameter, radius ratio, \& limb-darkening parameters), semi-major axis (typically parameterised as $a/R_s$), and orbital period.
Splitting orbital parameters into both $a/R_s$ & $P$ is superfluous for planets with uncertain periods.

Instead, the typical approach is to use only the transit shape parameters to constrain as few orbitla parameters as possible.
For example, if the impact parameter can be constrained from the shape alone, then in combination with the transit duration we can estimate the velocity of a planet across the star.
In the case of a purely circular orbit, this velocity then directly produces a period.
Including samples from some eccentricity and omega (argument of periasteron) distributions, these will then modify the resulting period.

There have been numerous past efforts and theoretical works exploring fitting such transits:

- @yee2008characterizing provided a theoretical perspective on modelling such transits even before Kepler began finding them.

- @wang2015planet adapted a transit model which included both circular period and semi-major axis ($a/R_s$) without specific priors on these quantities.

- @foreman2016population included eccentricity and reparameterised the orbital semi-major axis \& inclination into two parameters ($\sqrt{a}\sin{i}$ & $\sqrt{a}\cos{i}$), with an effective prior on the period ($P^{-2/3}$).

- @osborn2016single fitted impact parameter and a scaled velocity parameter (which encapsulates a prior equating to $P^{-5/3}$) to predict planetary periods, with the same approach being used in @giles2018transiting.

- @kipping2018orbital provided a purely theoretical view of the correct prior to place on such analyses, combining the geometric transit probability, a window effect prior, and the intrinsic perior prior to produce a value of $P^{-8/3}$.

- @sandford2019estimation created the ``single`` python package which used gaia parallaxes as a source of stellar density and allowed eccentricity to vary (with a period prior of $P^{-5/3}$)

- @becker2018discrete modelled the duotransit system HIP41378 using discrete period aliases and a $P^{-1}$ prior.

As can be seen from this array, the approach and prior varies widely between study.
Some directly model orbital period while others reparameterise in terms of parameters closer to the observed transit information.
Some use eccentricity but most assume circular orbits.
Some use information from interior multitransiting planets (e.g. @becker2018discrete) but most treat only the outer planet individually.

## ``MonoTools.fit`` approach

The ``monoModel`` class of ``MonoTools.fit`` uses the ``exoplanet`` package [@foreman2021exoplanet] and ``PyMC3`` [@exoplanet:pymc3] to build flexible a transit model which can be easily and efficiently sampled using ``PyMC3``'s Hamiltonian Monte Carlo approach.

The key development of ``MonoTools`` over past monotransit and duotransit tools is that it natively supports bayesian marginalisation over discontinous period space.
In the case of duotransits, this means the multiple period aliases, while in the case of monotransits, this means the multiple period gaps that can occur due to non-continuous photometric coverage.

### Calculating Period Aliases & Gaps

For Duotransits, period is not a modelled quantity in `MonoTools.fit`, but is instead derived from modelling two transit centres $t_0$ and $t_1$, with the period being part of the set $P \in (t_{{\rm tr},2}-t_{{\rm tr},1})/\{1,2, \cdots, N\}$).
Potential aliases therefore lie between $P_{\rm max}=t_{{\rm tr},2}-t_{{\rm tr},1}$ and $P_{\rm min}$, a minimum period, and are calculated by `compute_duo_period_aliases`.
To calculate $P_{\rm min}$, this function iterates over all potential aliases between $P_{\rm max}$ and 10d.
For each period, the data is phase-folded and the known transits masked.
Only period aliasess for which there there are no significant in-transit observations found elsewhere (defined as 15% of the central 90% of the transit duration) are kept in the model.

For monotransiters, a similar process is applied to find regions of the period parameter space that are rejected by photometry using `compute_period_gaps`.
First, an RMS timeseries of the light curve is computed.
This iterates through the flattened light curve in steps that are typically $1/7 t_D$ wide, performing a weighted average \& standard deviation for photometry in a $1 t_D$ wide.
The resulting timeseries can be converted into a theoretical transit SNR given the depth of the known transit.
This timeseries can be converted to a function of period space (i.e. by phase-folded around the know transit), with regions without photometric data being given SNR values of 0.0.
Period gaps can then be defined as regions in period space where the computed SNR is below some threshold value (default: $4\sigma$).

### Marginalisation

Here we have some number of discrete regions in one parameter space that we want to sample.
Typically, samplers such as MCMC fail with multiple local minima, especially in the case where the gaps between regions are far wider than the regions themselves.
One way to avoid this problem is to treat each region of this discontinuous parameter space as seperate and therefore sample each one individually.
We can then perform marginalisation over N submodels with parameters that correspond to each N period gaps.
By computing the log likelihood with respect to the data and the log prior of the parameters used, their sum gives us the probability of each submodel for a given step.

$p(\theta \mid y) = \sum_{k=1}^K p(\theta \mid y, M_{P=i}) \; p(M_{P=i} \mid y)$

The normalised probability for each period gap or alias are then the marginalised probabilities, and the marginalised parameters are simply the average of the submodel parameters weighted by this probability.
However, if all of the parameters in the model are marginalised, this can effectively require a huge number of parameters - $N_{params} \times N_{models}$.
Therefore, to improve efficiency, we must choose which parameters to marginalise and which to fit only once.

In the case of a transit where we want to marginalise over multiple sub-models at different orbital periods, we only need marginalise over parameters that substantially vary as a function of orbital period.
Other parameters, such as transit time, limb darkening and radius ratio, can be fitted as global parameters.

In the simplest case, ``MonoTools`` allows some degree of flexibility in what parameters to marginalise using the `fit_params` and `marginal_params` lists as inputs to the `monoModel` class.
Period is always marginalised, but so can $t_D$ or $b$.

However, this implementation of marginalisation can still be slow, and suffers from drawbacks.
For $t_D$ and $b$ one must always be globally fitted and the other marginalised.
But, their connection to the orbital period means that across this marginalisation there is always going to be many aliases which do not well represent the data.
For example, if a 15d planet with $b=0.2$ fits the transit well, a 150d planet is sure to produce far too long a transit duration, and therefore a very low likelihood.
And, despite the fact a 150d plant might be able to well explain the data at b=0.8, this part of the parameter space is not explored and our 150d alias is given an artificially low marginal probability.

### Marginalising with derived in-transit velocity

The solution to this problem is to not marginalise duration or impact parameter which are both intimitely connected to the observed transit shape.
By keeping all the parameters required to fit transit shape global, we can remove the need to perform likelihood calculations for each of the different period parameters, greatly improving speed and sampling efficiency.
Instead, we use the duration and impact parameter to derive an instantaneous velocity across the star, as was performed in @osborn2016single.
For each of the period aliases and the sampled stellar parameters, we can calculate a circular velocity.
The derived transit velocity as a ratio of circular velocity ($v/v_{\rm circ}$) for each period alias/gap then becomes the important quantity to marginalise.
Of course this is incompatible with the assumption of a circular orbit - we require an eccentricity distribution for this method to work.

As we are assuming the likelihood for each alias is identical (or at least negligible), all that is important is deriving a log prior for each.
The key part of this log prior comes from the assumed eccentricity distribution.
Observations of exoplanets show that low eccentricities are typically preferred over high ones.
Two distributions are typically used to quantify this - the beta distribtion of @kipping2013parametrizing for typically single-planet RV systems, and the Rayleigh distribution of @van2015eccentricity for multi-planet transiting systems.

Another observational constraint on eccentricity comes from the distribution of perihelion distances - exoplanet orbits do not typically pass within $2R_s$.
In terms of semi-major axis, we require that $e < 1 - 2R_s/a$.
We can also include another upper limit on eccentricity here - stable exoplanetary systems require that a planet's orbit does not cross the orbit of interior candidates.
So in the case of transiting multi-planet systems we can use $e < 1 - R_s/a_{\rm inner}$.

Each given $v/v_{\rm circ}$ we must calculate the possible eccentricity and argument of periastron.
From @barnes2007effects (Eq 12) we know that a planet's azimuthal velocity can be defined as:

$\frac{v_f}{v_{circ}} = \frac{1+e\cos{f}}{\sqrt{1-e^2}}$ where $f_{\rm tr}=(\omega-\pi/2)$.

Rearranging to give eccentricity gives two roots, although the second root is only applicable for cases where $v/v_{\rm circ}$<1.0:

$e_1 = (-v^2 \sqrt{\frac{v^2 (\sin{\omega}^2+v^2-1)}{(\sin{\omega}^2+v^2)^2}} - \sin{\omega}^2 \sqrt{\frac{v^2 (\sin^2{\omega}+v^2-1)}{(\sin^2{\omega}+v^2)^2}} - \sin{\omega})/(\sin{\omega}^2 + v^2)$

$e_2 = (v^2 \sqrt{\frac{v^2 (\sin{\omega}^2+v^2-1)}{(\sin{\omega}^2+v^2)^2}} + \sin{\omega}^2 \sqrt{\frac{v^2 (\sin^2{\omega}+v^2-1)}{(\sin{\omega}^2+v^2)^2}}-\sin{\omega})/(\sin{\omega}^2 + v^2)$

These two roots make it impractical to solve for the probability of $v$ analytically, so we instead compute this numerically.
Ultimately, we must derive the probability of each velocity ($v/v_{\rm circ}$, or $v$ hereafter) given that a transit occurs by marginalising over all compatible eccentricities and arguments of perasteron:

$$ p(v \mid {\rm Tr}, e_{\rm max}) = \frac{\int_0^{2\pi} \int_{0}^{e_{\rm max}} p(e, \omega \mid v) p({\rm Tr} \mid e, \omega, v) de d\omega}{\int_{v_{\rm min}}^{v_{\rm max}} \int_0^{2\pi} \int_{0}^{e_{\rm max}} p(e, \omega \mid v) p({\rm Tr} \mid e, \omega, v) de d\omega dv} $$

Using the equations for $e$, we can feasibly generate eccentricity for each $v/v_{\rm circ}$ & $\omega$ sample.
As the geometric probability of transit is a function of the distance in-transit, and eccentricity \& argument of periasteron directly affect this quantity, we also calculate a geometric correction (i.e. the distance at transit compared to semi major axis):

$$ \frac{d_{\rm Tr}}{a} = \frac{1 + e \sin{\omega}}{1 - e^2}$$

Therefore the probability of each grid position is then determined by the probability derived from selected prior distribution (i.e. `'kipping'`,`'vaneylen'` or `'uniform'`) multiplied by the geometric correction.
In the case that the derived eccentricity is above $e_{\rm max}$, the probability is set to zero.

As all velocities here are normalised to circular velocities and the joint argument of periastron -- eccentricity distributions remain constant with period, these calculations should remain constant for any period across all model samples.
However, the maximum permitted eccentricity ($e_{\rm max}$) can vary for each sample due to e.g. the sampled stellar radius and parameters for the orbits of interior planets.
Therefore, we need a way to compute on-the-fly a prior probability for a particular velocity and $e_{\rm max}$, as well as a marginal eccentricity and argument of periastron.
We choose to generate a 2D interpolation function for each eccentricity prior distribution.

Effectively the equation required to produce the marginalised probability distribution for $v$ (given some maximum eccentricity and the fact that a transit occurs) is:

$$ p(v \mid {\rm Tr}, e_{\rm max}) = \int_0^{2\pi} \int_{0}^{e_{\rm max}} p(e, \omega \mid v, e_{\rm max}) p({\rm Tr} \mid e, \omega, v) de d\omega$$

Where, for example in the case of the @kipping2013parametrizing $\beta$ distribution where $\alpha=0.867$ and $\beta=3.03$, the probability on $p(e \mid v, e_{\rm max})$ (and, therefore, $p(e,\omega \mid v, e_{\rm max})$ as $\omega$ is uniform) is:
$$p(e \mid v, e_{\rm max}) = \begin{cases}
    0 & \text{if } e > e_{\rm max} \\ % & is your "\tab"-like command (it's a tab alignment character)
    \frac{\exp{(\alpha - 1)}(1-e)^{\beta-1}}{{\rm B}(\alpha,\beta)} & \text{otherwise.}
\end{cases}$$

By generating a grid of $v/v_{\rm circ}$ (16000-steps flat in $\log_{10}{}$ between 0.07 and 14), omega (8000 steps flat between 0 & $2\pm$) and $e_{\rm max}$ (96 steps sampled using $e_{\rm max} \in 1 - 10^{\{-3,...,-0.05\}}$), we can derive eccentricities for each point in the $\omega$ - $v$ plane and therefore compute marginalised probabilities for each point on the $v$ - $e_{\rm max}$ plane.
For each of the $e_{\rm max}$ steps, the sum of probabilities for $v$ must sum to 1.0, therefore we must renormalise the above equation using the integral over all possible velocities using the following normalisation factor:

$$ \int_{v_{\rm min}}^{v_{\rm max}} \int_0^{2\pi} \int_{1\times10^-4}^{e_{\rm max}} p(e,\omega \mid v) p({\rm Tr} \mid e, \omega, \log{v}) de d\omega dv $$

The resulting $v$ - $e_{\rm max}$ distributions can be seen in Figure XX.


### Choice of transit shape parameters

Transit duration and impact parameter can both

### Treatment of limb darkening parameters



### Treatment of period gaps & aliases


### Eccentricity distribution marginalisation




# Validation

# Installation

# Acknowledgements

# References
