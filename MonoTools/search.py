# 
# Let's use this python script to:
# - perform transit "Quick fit"
# - Search for dips in a lightcurve which match that detected
# - Search for secondaries
# - Perform quick/dirty EB modelling
#

import numpy as np
import pandas as pd
import pickle
import os
import glob

from copy import deepcopy
from datetime import datetime

from scipy import optimize
import exoplanet as xo
import scipy.interpolate as interp
import scipy.optimize as optim
import matplotlib.pyplot as plt
import matplotlib

from astropy.coordinates import SkyCoord
from astropy import units as u

import seaborn as sns
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

MonoData_tablepath = os.path.join('/'.join(os.path.dirname( __file__ ).split('/')[:-1]),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join('/'.join(os.path.dirname( __file__ ).split('/')[:-1]),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')

if not os.path.isdir(MonoData_savepath):
    os.mkdir(MonoData_savepath)

os.environ["CXXFLAGS"]="-fbracket-depth=512" if not "CXXFLAGS" in os.environ else "-fbracket-depth=512,"+os.environ["CXXFLAGS"]
os.environ["CFLAGS"] = "-fbracket-depth=512" if not "CFLAGS" in os.environ else "-fbracket-depth=512,"+os.environ["CFLAGS"]

#creating new hidden directory for theano compilations:
theano_dir=MonoData_savepath+'/.theano_dir_'+str(np.random.randint(8))

theano_pars={'device':'cpu','floatX':'float64',
             'base_compiledir':theano_dir,"gcc.cxxflags":"-fbracket-depth=1024"}
'''if MonoData_savepath=="/Users/hosborn/python/MonoToolsData" or MonoData_savepath=="/Volumes/LUVOIR/MonoToolsData":
    theano_pars['cxx']='/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9'

if MonoData_savepath=="/Users/hosborn/python/MonoToolsData" or MonoData_savepath=="/Volumes/LUVOIR/MonoToolsData":
    theano_pars['cxx']='cxx=/Library/Developer/CommandLineTools/usr/bin/g++'
'''
if os.environ.get('THEANO_FLAGS') is None:
    os.environ["THEANO_FLAGS"]=''
for key in theano_pars:
    if key not in os.environ["THEANO_FLAGS"]:
        os.environ["THEANO_FLAGS"] = os.environ["THEANO_FLAGS"]+','+key+"="+theano_pars[key]

import theano.tensor as tt
import pymc3 as pm
import theano
theano.config.print_test_value = True
theano.config.exception_verbosity='high'

from . import tools
from . import fit

def transit(pars,x):
    log_per,b,t0,log_r_pl,u1,u2=pars
    # Compute a limb-darkened light curve using starry
    light_curve = (
        xo.LimbDarkLightCurve([np.clip(u1,0,1),np.clip(u2,0,1)]).get_light_curve(orbit=xo.orbits.KeplerianOrbit(period=np.power(10,log_per), b=b, t0=t0),
                                                 r=np.power(10,log_r_pl), t=x).eval()
    )
    return light_curve.ravel()

def least_sq(pars,x,y,yerr):
    model=transit(pars,x)
    chisq=-1*np.sum((y-model)**2*yerr**-2)
    return chisq

def QuickMonoFit(lc,it0,dur,Rs=None,Ms=None,Teff=None,useL2=False,fit_poly=True,force_tdur=False,
                 polyorder=2, ndurs=4.2, how='mono', init_period=None,fluxindex='flux_flat',mask=None, **kwargs):
    # Performs simple planet fit to monotransit dip given the detection data.
    #Initial depth estimate:
    dur=0.3 if dur/dur!=1.0 else dur #Fixing duration if it's broken/nan.
    winsize=np.clip(dur*ndurs,0.75,3.5)
    
    if mask is None and ((fluxindex=='flux_flat')|(fluxindex=='flux')):
        mask=lc['mask']
    elif mask is None:
        mask=np.isfinite(lc[fluxindex])
    
    timeindex='bin_time' if 'bin_' in fluxindex else 'time'
    fluxerrindex='bin_flux_err' if 'bin_' in fluxindex else 'flux_err'
    
    if how=='periodic':
        assert init_period is not None
        xinit=(lc[timeindex]-it0-init_period*0.5)%init_period-init_period*0.5
        nearby=(abs(xinit)<winsize)
    else:
        xinit=lc[timeindex]-it0
        nearby=abs(xinit)<winsize
        assert np.sum(nearby)>0
    cad = np.nanmedian(np.diff(lc[timeindex])) if 'bin_' in fluxindex else float(int(max(set(list(lc['cadence'][nearby])), key=list(lc['cadence'][nearby]).count)[1:]))/1440

    
    if fluxindex=='flux_flat' and 'flux_flat' not in lc:
        #Reflattening and masking transit:
        lc=tools.lcFlatten(lc, winsize=9*dur, stepsize=0.1*dur, transit_mask=abs(xinit)>dur*0.5)
    if how=='periodic':
        assert np.sum(nearby&mask)>0
        #print(lc[timeindex][nearby])
        
        x = xinit[nearby&mask]+it0 #re-aligning this fake "mono" with the t0 provided
        y = lc[fluxindex][nearby&mask][np.argsort(x)]
        y-=np.nanmedian(y)
        yerr=lc[fluxerrindex][nearby&mask][np.argsort(x)]
        x=np.sort(x).astype(np.float64)

        oot_flux=np.nanmedian(y[(abs(x-it0)>0.65*dur)])
        int_flux=np.nanmedian(y[(abs(x-it0)<0.35*dur)])
        #print(it0,dur,oot_flux,int_flux,abs(oot_flux-int_flux)/lc['flux_unit'],x,y,yerr)
        fit_poly=False
        init_poly=None
    else:
        #Mono case:
        x=lc[timeindex][nearby&mask].astype(np.float64)
        yerr=lc[fluxerrindex][nearby&mask]
        if not fit_poly or np.sum(abs(x-it0)<0.6)==0.0:
            y=lc[fluxindex][nearby&mask]
            y-=np.nanmedian(y)
            oot_flux=np.nanmedian(y[(abs(x-it0)>0.65*dur)])
            int_flux=np.nanmedian(y[(abs(x-it0)<0.35*dur)])
            fit_poly=False
            init_poly=None
        else:
            y=lc[fluxindex][nearby&mask]
            y-=np.nanmedian(y)
            #print(np.sum(abs(x-it0)<0.6),it0,np.min(x),np.max(x))
            init_poly=np.polyfit(x[abs(x-it0)>0.7*dur]-it0,y[abs(x-it0)>0.7*dur],polyorder)
            oot_flux=np.nanmedian((y-np.polyval(init_poly,x-it0))[abs(x-it0)>0.65*dur])
            int_flux=np.nanmedian((y-np.polyval(init_poly,x-it0))[abs(x-it0)<0.35*dur])
    dep=abs(oot_flux-int_flux)*lc['flux_unit']
    
    #print(dep,dur,it0,x,y,init_poly)
    
    with pm.Model() as model:
        # Parameters for the stellar properties
        if fit_poly:
            trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(polyorder+1)[::-1], shape=polyorder+1,testval=init_poly)
            flux_trend = pm.Deterministic("flux_trend", tt.dot(np.vander(x - it0, polyorder+1), trend))
            #trend = pm.Uniform("trend", upper=np.tile(1,polyorder+1), shape=polyorder+1,
            #                  lower=np.tile(-1,polyorder+1), testval=init_poly)
            #trend = pm.Normal("trend", mu=np.zeros(polyorder+1), sd=5*(10.0 ** -np.arange(polyorder+1)[::-1]), 
            #                  shape=polyorder+1, testval=np.zeros(polyorder+1))
            #trend = pm.Uniform("trend", upper=np.tile(10,polyorder+1),lower=np.tile(-10,polyorder+1),
            #                   shape=polyorder+1, testval=np.zeros(polyorder+1))
        else:
            mean = pm.Normal("mean", mu=0.0, sd=3*np.nanstd(y))
            flux_trend = mean
        
        r_star = Rs if Rs is not None and not np.isnan(Rs) else 1.0
        m_star = Ms if Ms is not None and not np.isnan(Ms) else 1.0
        Ts = Teff if Teff is not None and not np.isnan(Teff) else 5500.0

        u_star = tools.getLDs(Ts)[0]
        #xo.distributions.QuadLimbDark("u_star")
        if how=='periodic' and init_period is not None:
            log_per = pm.Normal("log_per", mu=np.log(init_period),sd=0.4)
        else:
            rhostar=m_star/r_star**3
            init_per = abs(18226*rhostar*((2*np.sqrt((1+dep**0.5)**2-0.41**2))/dur)**-3)
            # Orbital parameters for the planets
            log_per = pm.Uniform("log_per", lower=np.log(dur*5),upper=np.log(3000),
                                 testval=np.clip(np.log(init_per),np.log(dur*6),np.log(3000))
                                 )

        tcen = pm.Bound(pm.Normal, lower=it0-0.7*dur, upper=it0+0.7*dur)("tcen", 
                                            mu=it0,sd=0.25*dur,testval=it0)
        
        b = pm.Uniform("b",upper=1.0,lower=0.0,testval=0.2)
        log_ror = pm.Uniform("log_ror",lower=-6,upper=-0.5,testval=np.clip(0.5*np.log(dep),-6,-0.5))
        ror = pm.Deterministic("ror", tt.exp(log_ror))
        #ror, b = xo.distributions.get_joint_radius_impact(min_radius=0.0075, max_radius=0.25,
        #                                                  testval_r=np.sqrt(dep), testval_b=0.41)
        #logror = pm.Deterministic("logror",tt.log(ror))
        
        
        #pm.Potential("ror_prior", -logror) #Prior towards larger logror

        r_pl = pm.Deterministic("r_pl", ror*r_star*109.1)

        period = pm.Deterministic("period", tt.exp(log_per))

        # Orbit model
        orbit = xo.orbits.KeplerianOrbit(r_star=r_star,m_star=m_star,
                                         period=period,t0=tcen,b=b)
        
        vx, vy, vz = orbit.get_relative_velocity(tcen)
        vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/r_star)
        
        #tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)
        #correcting for grazing transits by multiplying b by 1-rp/rs
        tdur=pm.Deterministic("tdur",(2*tt.sqrt((1+ror)**2-b**2))/vrel)
        
        if force_tdur:
            #Adding a potential to force our transit towards the observed transit duration:
            pm.Potential("tdur_prior", -0.05*len(x)*abs(tt.log(tdur/dur)))        
        
        # The 2nd light (not third light as companion light is not modelled) 
        # This quantity is in delta-mag
        if useL2:
            deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
            third_light = pm.Deterministic("third_light", tt.power(2.511,-1*deltamag_contam)) #Factor to multiply normalised lightcurve by
        else:
            third_light = 0.0

        # Compute the model light curve using starry
        light_curves = (
            xo.LimbDarkLightCurve(u_star).get_light_curve(
                orbit=orbit, r=r_pl/109.1, t=x, texp=cad))*(1+third_light)/lc['flux_unit']
        transit_light_curve = pm.math.sum(light_curves, axis=-1)
        
        light_curve = pm.Deterministic("light_curve", transit_light_curve + flux_trend)

        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)
        #print(model.check_test_point())
        if fit_poly:
            map_soln = xo.optimize(start=model.test_point,vars=[trend],verbose=False)
            map_soln = xo.optimize(start=map_soln,vars=[trend,log_ror,log_per,tcen],verbose=False)

            # Fit for the maximum a posteriori parameters
        else:
            map_soln = xo.optimize(start=model.test_point,vars=[mean],verbose=False)
            map_soln = xo.optimize(start=map_soln,vars=[mean,log_ror,log_per,tcen],verbose=True)
        
        '''
        map_soln = xo.optimize(start=map_soln,vars=[b, log_ror, log_per, tcen],verbose=False)
        
        if useL2:
            map_soln = xo.optimize(start=map_soln,vars=[log_ror, b, deltamag_contam],verbose=False)
        map_soln = xo.optimize(start=map_soln,verbose=False)
        if fit_poly:
            map_soln = xo.optimize(start=map_soln,vars=[trend],verbose=False)
        map_soln = xo.optimize(start=map_soln,verbose=False)
        map_soln = xo.optimize(start=map_soln,verbose=False)
        '''
        
        map_soln, func = xo.optimize(start=map_soln,verbose=False,return_info=True)
        '''
        interpy=xo.LimbDarkLightCurve(map_soln['u_star']).get_light_curve(
                    r=float(map_soln['r_pl']),
                    orbit=xo.orbits.KeplerianOrbit(
                        r_star=float(map_soln['r_star']),
                        m_star=float(map_soln['m_star']),
                        period=float(map_soln['period']),
                        t0=float(map_soln['t0']),
                        b=float(map_soln['b'])),
                    t=interpt
                )/(1+map_soln['third_light'])
        '''
    #print(func)
    interpt=np.linspace(map_soln['tcen']-winsize,map_soln['tcen']+winsize,600)
    if 'third_light' not in map_soln:
        map_soln['third_light']=np.array(0.0)
    
    transit_zoom = (xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=xo.orbits.KeplerianOrbit(r_star=r_star,m_star=m_star,
                                                       period=map_soln['period'],t0=map_soln['tcen'],b=map_soln['b']),
                                                       r=map_soln['r_pl']/109.1, t=interpt, texp=cad
                                                      )*(1+map_soln['third_light'])
                   ).eval().ravel()/lc['flux_unit']

    #Reconstructing best-fit model into a dict:
    best_fit={'log_lik_mono':-1*func['fun'],'model_success':str(func['success'])}
    for col in map_soln:
        if 'interval__' not in col:
            if map_soln[col].size==1:
                best_fit[col]=float(map_soln[col])
            else:
                best_fit[col]=map_soln[col].astype(float)
    #print({bf:type(best_fit[bf]) for bf in best_fit})
    #print(best_fit["vrel"],best_fit["b"],map_soln["tdur"],best_fit["tcen"])
    if np.isnan(map_soln["tdur"]):
        best_fit['tdur']=dur
    #Adding depth:
    best_fit['transit_zoom']=transit_zoom
    best_fit['monofit_x']=x
    best_fit['monofit_ymodel']=map_soln['light_curve']
    best_fit['monofit_y']=y
    best_fit['monofit_yerr']=yerr
    best_fit['depth']=np.max(transit_zoom)-np.min(transit_zoom)
    #err = std / sqrt(n_pts in transit)
    best_fit['depth_err']=np.nanstd(y[abs(x-best_fit['tcen'])<0.475*best_fit["tdur"]])/\
                          np.sqrt(np.sum(abs(x-best_fit['tcen'])<0.475*best_fit["tdur"]))
    best_fit['snr']=best_fit['depth']/(best_fit['depth_err'])

    '''
    print("time stuff:", best_fit['tcen'],interpt[0],interpt[-1],
          "\nfit stuff:",best_fit['r_pl'],best_fit['b'], best_fit['logror'],best_fit['tdur'],(1+best_fit['third_light'])/lc['flux_unit'],
          "\ndepth stuff:",best_fit['depth'],best_fit['depth_err'],lc['flux_unit'],np.min(transit_zoom),np.min(map_soln['light_curve']))'''
    #Calculating std in typical bin with width of the transit duration, to compute SNR_red
    
    if how=='mono' or init_period is None:
        oot_mask=mask*(abs(lc[timeindex]-best_fit['tcen'])>0.5)*(abs(lc[timeindex]-best_fit['tcen'])<25)
        binlc=tools.bin_lc_segment(np.column_stack((lc[timeindex][oot_mask],lc[fluxindex.replace('_flat','')][oot_mask],
                                                    lc[fluxerrindex][oot_mask])),best_fit['tdur'])
        best_fit['cdpp'] = np.nanmedian(abs(np.diff(binlc[:,1])))#np.nanstd(binlc[:,1])
        best_fit['Ntrans']=1
    else:
        phase=(abs(lc['time']-best_fit['tcen']+0.5*best_fit['tdur'])%init_period)
        if len(mask)==len(phase):
            oot_mask=mask*(phase>best_fit['tdur'])
        else:
            oot_mask=lc['mask']*(phase>best_fit['tdur'])
        binlc=tools.bin_lc_segment(np.column_stack((lc['time'][oot_mask],lc['flux'][oot_mask],
                                                    lc['flux_err'][oot_mask])),best_fit['tdur'])
        #Counting in-transit cadences:
        durobs=0
        for cad in np.unique(lc['cadence']):
            if len(mask)==len(phase):
                durobs+=np.sum(phase[mask&(lc['cadence']==cad)]<best_fit['tdur'])*float(int(cad[1:]))/1440
            else:
                durobs+=np.sum(phase[lc['mask']&(lc['cadence']==cad)]<best_fit['tdur'])*float(int(cad[1:]))/1440

        best_fit['Ntrans']=durobs/best_fit['tdur']
        best_fit['cdpp'] = np.nanmedian(abs(np.diff(binlc[:,1])))#np.nanstd(binlc[:,1])

    best_fit['snr_r']=best_fit['depth']/(best_fit['cdpp']/np.sqrt(best_fit['Ntrans']))
    best_fit['interpmodel']=interp.interp1d(np.hstack((-10000,interpt-best_fit['tcen'],10000)),
                                     np.hstack((0.0,transit_zoom,0.0)))
    if how=='periodic':
        for col in ['period','vrel']:
            par = best_fit.pop(col) #removing things which may spoil the periodic detection info (e.g. derived period)
            best_fit[col+'_mono']=par
        assert 'period' not in best_fit
        best_fit['period']=init_period
    return best_fit

def MonoTransitSearch(lc,ID,mission, Rs=None,Ms=None,Teff=None,
                      mono_SNR_thresh=6.5,mono_BIC_thresh=-6,n_durs=5,poly_order=3,
                      n_oversamp=20,binsize=15/1440.0,custom_mask=None,
                      transit_zoom=3.5,use_flat=False,use_binned=True,use_poly=True,
                      plot=False, plot_loc=None ,n_max_monos=8, use_stellar_dens=True, **kwargs):
    #Searches LC for monotransits - in this case without minimizing for duration, but only for Tdur
    '''
    lc
    ID
    mono_SNR_thresh=6.5 - threshold in sigma
    mono_BIC_thresh=-10 - threshold in BIC to be used for a significant monotransit
    n_durs=5
    poly_order=3 - polynomial order to use for a "no transit" fit
    n_oversamp=10 - oversampling factor wrt to durations from which to match to lightcurve
    binsize=1/96. - size of bins in days
    transit_zoom=3.5 - size (in transit durations) to cut around transit when minimizing transit model
    use_flat=True - flattens lightcurve before monotransit search
    use_binned=True
    use_poly=True - fits transits (and sin) with a polynomial (order = poly_order-1) to account for variability
    Rs=None
    Ms=None
    Teff=None
    plot=False
    plot_loc=None
    use_stellar_dens=True - use stellar density to produce transit templates to search.
    '''
    
    #Computing a fine x-range to search:
    search_xrange=[]
    
    Rs=1.0 if (Rs is None or not use_stellar_dens or np.isnan(Rs)) else float(Rs)
    Ms=1.0 if (Ms is None or not use_stellar_dens or np.isnan(Ms)) else float(Ms)
    Teff=5800.0 if Teff is None else float(Teff)

    mincad=np.min([float(cad[1:])/1440 for cad in np.unique(lc['cadence'])])
    interpmodels, tdurs = get_interpmodels(Rs,Ms,Teff,lc['time'],lc['flux_unit'],
                                           n_durs=n_durs,texp=mincad, mission=mission)
    
    #print("Checking input model matches. flux:",np.nanmedian(uselc[:,0]),"std",np.nanstd(uselc[:,1]),"transit model:",
    #       interpmodels[0](10.0),"depth:",interpmodels[0](0.0))
        
    search_xranges=[]
    
    mask = lc['mask'] if custom_mask is None else custom_mask
    #Removing gaps bigger than 2d (with no data)
    for n in range(n_durs):
        search_xranges_n=[]
        if np.max(np.diff(lc['time'][mask]))<tdurs[n]:
            lc_regions=[lc['time'][mask]]
        else:
            lc_regions = np.array_split(lc['time'][mask],1+np.where(np.diff(lc['time'][mask])>tdurs[n])[0])
        for arr in lc_regions:
            search_xranges_n+=[np.arange(arr[0]+0.33*tdurs[n],arr[-1]-0.33*tdurs[n],tdurs[n]/n_oversamp)]
        search_xranges+=[np.hstack(search_xranges_n)]
    
    print(str(ID)+" - Searching "+str(np.sum([len(xr) for xr in search_xranges]))+" positions with "+str(n_durs)+" durations:",','.join(list(np.round(tdurs,3).astype(str))))

    #Looping through search and computing chi-sq at each position:
    outparams=pd.DataFrame()

    def trans_model_neglnlik(params,x,y,sigma2,init_log_dep,interpmodel):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = depth, duration, poly1, poly2
        model=np.exp(params[0]-init_log_dep)*interpmodel(x)
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
    
    def sin_model_neglnlik(params,x,y,sigma2,tcen,dur):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = depth, duration, poly1, poly2
        newt=x/(2.6*dur)*2*np.pi
        amp=np.exp(-1*np.power(newt, 2.) / (2 * np.power(np.pi, 2.)))
        model=params[0]*(amp*np.sin(newt-np.pi*0.5)-0.1)
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
    
    def trans_model_poly_neglnlik(params,x,y,sigma2,init_log_dep,interpmodel):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = depth, gradient
        model=x*params[1]+np.exp(params[0]-init_log_dep)*interpmodel(x)
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
    
    def sin_model_poly_neglnlik(params,x,y,sigma2,tcen,dur):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = log_depth, poly1, poly2
        newt=x/(2*dur)*2*np.pi
        amp=np.exp(-1*np.power(newt, 2.) / (2 * np.power(np.pi, 2.)))
        model=x*params[1]+np.exp(params[0])*(amp*np.sin(newt-np.pi*0.5)-0.1)
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
    
    #from progress.bar import IncrementalBar
    #bar = IncrementalBar('Searching for monotransit', max = np.sum([len(xr) for xr in search_xranges]))

    #For each duration we scan across the lightcurve and compare a transit model to others:
    for n,search_xrange in enumerate(search_xranges):
        tdur=tdurs[n%n_durs]
        
        #flattening and binning as a function of the duration we are searching in order to avoid
        if use_binned:
            #Having may points in a lightcurve makes flattening difficult (and fitting needlessly slow)
            # So let's bin to a fraction of the tdur - say 9 in-transit points.
            lc=tools.lcBin(lc,binsize=tdur/9, use_flat=False, extramask=custom_mask)
            if use_flat and not use_poly:
                lc=tools.lcFlatten(lc,winsize=tdur*13,stepsize=tdur*0.333,use_bin=True)
                uselc=np.column_stack((lc['bin_time'],lc['bin_flux_flat'],lc['bin_flux_err']))
            else:
                uselc=np.column_stack((lc['bin_time'],lc['bin_flux'],lc['bin_flux_err']))
        else:
            if use_flat and not use_poly:
                lc=tools.lcFlatten(lc,winsize=tdur*13,stepsize=tdur*0.333)
                uselc=np.column_stack((lc['time'][mask],lc['flux_flat'][mask],lc['flux_err'][mask]))
            else:
                uselc=np.column_stack((lc['time'][mask],lc['flux'][mask],lc['flux_err'][mask]))
                
        #Making depth vary from 0.1 to 1.0
        init_dep_shifts=np.exp(np.random.normal(0.0,n_oversamp*0.01,len(search_xrange)))
        randns=np.random.randint(2,size=(len(search_xrange),2))
        cad=np.nanmedian(np.diff(uselc[:,0]))
        p_transit = np.clip(3/(len(uselc[:,0])*cad),0.0,0.05)
        
        #What is the probability of transit given duration (used in the prior calculation) - duration
        methods=['SLSQP','Nelder-Mead','Powell']
        logmodeldep=np.log(abs(interpmodels[n](0.0)))

        for n_mod,x2s in enumerate(search_xrange):
            
            #bar.next()
            #minimise single params - depth
            round_tr=abs(uselc[:,0]-x2s)<(transit_zoom*tdur)
            #Centering x array on epoch to search
            x=uselc[round_tr,0]-x2s
            in_tr=abs(x)<(0.45*tdur)
            if len(x[in_tr])>0 and not np.isnan(x[in_tr]).all() and len(x[~in_tr])>0 and not np.isnan(x[~in_tr]).all():
                
                y=uselc[round_tr,1]
                oot_median=np.nanmedian(y[~in_tr])
                y-=oot_median
                yerr=uselc[round_tr,2]
                sigma2=yerr**2
                init_log_dep=np.log(np.clip(-1*(np.nanmedian(y[in_tr]))*init_dep_shifts[n_mod],0.00000001,1000))
                init_noise=np.std(y)
                
                poly_fit=np.polyfit(x,y,poly_order)
                poly_neg_llik=0.5 * np.sum((y - np.polyval(poly_fit,x))**2 / sigma2 + np.log(sigma2))

                if use_poly:
                    init_grad=np.polyfit(x[~in_tr],y[~in_tr],1)[0]
                    res_trans=optim.minimize(trans_model_poly_neglnlik,np.hstack((init_log_dep,init_grad)),
                                             args=(x,y,sigma2,
                                                   logmodeldep,interpmodels[n%n_durs]),
                                             method = methods[randns[n_mod,0]])
                    res_sin=optim.minimize(sin_model_poly_neglnlik, 
                                           np.hstack((init_log_dep,init_grad)),
                                           args=(x,y,sigma2,x2s,tdur),
                                           method = methods[randns[n_mod,1]])
                else:
                    res_trans=optim.minimize(trans_model_neglnlik,(init_log_dep),
                                             args=(x,y,sigma2,
                                                   logmodeldep,interpmodels[n%n_durs]),
                                             method = methods[randns[n_mod,0]])
                    res_sin=optim.minimize(sin_model_neglnlik, (init_log_dep),
                                           args=(x,y,sigma2,x2s,tdur),
                                           method = methods[randns[n_mod,1]])                
                log_len=np.log(np.sum(round_tr))

                #BIC = log(n_points)*n_params - 2*(log_likelihood + log_prior)
                #BIC = log(n_points)*n_params + 2*(neg_log_likelihood-log_prior). 
                # Setting log prior of transit as sum of:
                #   - 0.5*tdur/len(x)  - the rough probability that, given some point in time, it's in the centre of a transit
                #   - normal prior on log(duration) to be within 50% (0.5 in logspace)
                #        'BIC_trans':log_len*len(res_trans.x)+2*(res_trans.fun-np.log(p_transit)),
                #        'BIC_sin':log_len*len(res_sin.x)+2*(res_sin.fun-np.log(1-p_transit)),
                #        'BIC_poly':log_len*len(poly_fit)+2*(poly_neg_llik-np.log(1-p_transit)),
                n_params=np.array([len(res_trans.x),len(res_sin.x),len(poly_fit)])
                
                outdic={'tcen':x2s,
                        'llk_trans':-1*res_trans.fun,
                        'llk_sin':-1*res_sin.fun,
                        'llk_poly':-1*poly_neg_llik,
                        'BIC_trans':log_len*n_params[0]+2*(res_trans.fun-np.log(p_transit)),
                        'BIC_sin':log_len*n_params[1]+2*(res_sin.fun-np.log(5*p_transit)),
                        'BIC_poly':log_len*n_params[2]+2*(poly_neg_llik-np.log(1-(6*p_transit))),
                        'init_dep':np.exp(init_log_dep),
                        'init_dur':tdur,
                        'trans_dep':np.exp(res_trans.x[0]),
                        'sin_dep':np.exp(res_sin.x[0]),
                        'n_mod':n,
                        'tran_success':int(res_trans.success),'sin_success':int(res_sin.success),
                        'all_success':int(res_trans.success&res_sin.success)}
                outdic['trans_snr']=outdic['trans_dep']/(init_noise/np.sqrt(outdic['init_dur']/cad))
                outdic.update({'poly_'+str(n):poly_fit[n] for n in range(poly_order+1)})
                if use_poly:
                    outdic['trans_grad']=res_trans.x[1]
                    #outdic.update({'trans_poly_'+str(n):res_trans.x[1+n] for n in range(poly_order)})
                    outdic['sin_grad']=res_sin.x[1]
                    #outdic.update({'sin_poly_'+str(n):res_sin.x[1+n] for n in range(poly_order)})
                outparams=outparams.append(pd.Series(outdic,name=len(outparams)))
        #print(n,len(outparams))
    #bar.finish()
    outparams=outparams.sort_values('tcen')
    #Transit model has to be better than the sin model AND the DeltaBIC w.r.t to the polynomial must be <-10.
    # transit depth must be <0.0,
    # the sum of DeltaBICs has to be < threshold (e.g. -10)
    outparams['sin_llk_ratio']=outparams['llk_trans'].values-outparams['llk_sin'].values
    outparams['poly_llk_ratio']=outparams['llk_trans'].values-outparams['llk_poly'].values
    outparams['sin_DeltaBIC']=outparams['BIC_trans'].values-outparams['BIC_sin'].values
    outparams['poly_DeltaBIC']=outparams['BIC_trans'].values-outparams['BIC_poly'].values
    outparams['sum_DeltaBICs']=outparams['sin_DeltaBIC']+outparams['poly_DeltaBIC']
    outparams['mean_DeltaBICs']=0.5*outparams['sum_DeltaBICs']
    #if use_poly:
        #In the case of co-fitting for polynomials, BICs fail, but log likelihoods still work.
        #We will use 1.5 (4.5x) over sin and 3 (20x) over polynomial as our threshold:
    #    signfct=np.where((outparams['sin_llk_ratio']>1.5)&(outparams['poly_llk_ratio']>3)&(outparams['trans_dep']>0.0))[0]
    #else:
    #    signfct=np.where((outparams['sin_llk_ratio']>1.0)&(outparams['poly_DeltaBIC']<mono_BIC_thresh)&(outparams['trans_dep']>0.0))[0]
    signfct=(outparams['sin_llk_ratio']>1.5)&(outparams['poly_llk_ratio']>1.5)&(outparams['poly_DeltaBIC']<mono_BIC_thresh)&(outparams['trans_snr']>(mono_SNR_thresh-0.5))
    n_sigs=np.sum(signfct)
    if n_sigs>0:
        best_ix=[]
        nix=0
        detns={}
        while n_sigs>0 and nix<=n_max_monos:
            #Getting the best detection:
            signfct_df=outparams.loc[signfct]
            
            #Placing the best detection info into our dict:
            detn_row=signfct_df.iloc[np.argmin(signfct_df['poly_DeltaBIC'])]
            detns[str(nix).zfill(2)]={}
            detns[str(nix).zfill(2)]['llk_trans']   = detn_row['llk_trans']
            detns[str(nix).zfill(2)]['llk_sin']     = detn_row['llk_sin']
            detns[str(nix).zfill(2)]['llk_poly']    = detn_row['llk_poly']
            detns[str(nix).zfill(2)]['BIC_trans']   = detn_row['BIC_trans']
            detns[str(nix).zfill(2)]['BIC_sin']     = detn_row['BIC_sin']
            detns[str(nix).zfill(2)]['BIC_poly']    = detn_row['BIC_poly']
            detns[str(nix).zfill(2)]['sin_DeltaBIC']= detn_row['sin_DeltaBIC']
            detns[str(nix).zfill(2)]['poly_DeltaBIC']=detn_row['poly_DeltaBIC']
            detns[str(nix).zfill(2)]['tcen']        = detn_row['tcen']
            detns[str(nix).zfill(2)]['period']      = np.nan
            detns[str(nix).zfill(2)]['period_err']  = np.nan
            detns[str(nix).zfill(2)]['DeltaBIC']    = detn_row['poly_DeltaBIC']
            detns[str(nix).zfill(2)]['tdur']        = detn_row['init_dur']
            detns[str(nix).zfill(2)]['depth']       = detn_row['trans_dep']
            detns[str(nix).zfill(2)]['orbit_flag']  = 'mono'
            detns[str(nix).zfill(2)]['snr']         = detn_row['trans_snr']
            #Calculating minimum period:
            detns[str(nix).zfill(2)]['P_min']       = calc_min_P(uselc[:,0],detn_row['tcen'],detn_row['init_dur'])
            
            #Removing the regions around this detection from our array
            #print(np.sum(abs(outparams['tcen']-detn_row['tcen'])<np.where(outparams['init_dur']<detn_row['init_dur'],
            #                                                    0.66*detn_row['init_dur'], 0.66*outparams['init_dur'])))
            #print(np.sum(signfct[abs(outparams['tcen']-detn_row['tcen'])<np.where(outparams['init_dur']<detn_row['init_dur'],
            #                                                    0.66*detn_row['init_dur'], 0.66*outparams['init_dur'])]))
            away_from_this_detn=abs(outparams['tcen']-detn_row['tcen'])>np.where(outparams['init_dur']<detn_row['init_dur'],
                                                                0.66*detn_row['init_dur'], 0.66*outparams['init_dur'])
            signfct=signfct&away_from_this_detn
            n_sigs=np.sum(signfct)
            #print(n_sigs,detns[str(nix).zfill(2)]['poly_DeltaBIC'],np.sum(signfct[abs(outparams['tcen']-detn_row['tcen'])<np.where(outparams['init_dur']<detn_row['init_dur'],0.66*detn_row['init_dur'], 0.66*outparams['init_dur'])]))
            nix+=1
        #print(np.percentile(outparams['poly_DeltaBIC'],[5,16,50,84,95]))
    else:
        print("n_sigs == 0")
        detns={}
    if plot:
        fig_loc= PlotMonoSearch(lc,ID,outparams,detns,interpmodels,tdurs,custom_mask=custom_mask,
                             use_flat=use_flat,use_binned=use_binned,use_poly=use_poly,plot_loc=plot_loc)
        return detns, outparams, fig_loc
    else:
        return detns, outparams, None

def calc_min_P(time,tcen,tdur):
    abs_times=abs(time-tcen)
    abs_times=np.sort(abs_times)
    whr=np.where(np.diff(abs_times)>tdur*0.75)[0]
    if len(whr)>0:
        return abs_times[whr[0]]
    else:
        return np.max(abs_times)

def PlotMonoSearch(lc, ID, monosearchparams, mono_dic, interpmodels, tdurs, 
                   use_flat=True,use_binned=True,use_poly=False,transit_zoom=2.5,plot_loc=None,custom_mask=None,**kwargs):
    if plot_loc is None:
        plot_loc = str(ID).zfill(11)+"_Monotransit_Search.pdf"
    elif plot_loc[-1]=='/':
        plot_loc = plot_loc+str(ID).zfill(11)+"_Monotransit_Search.pdf"
    
    if use_flat and not use_poly:
        lc=tools.lcFlatten(lc,winsize=np.median(tdurs)*7.5,stepsize=0.2*np.median(tdurs))
        lc=tools.lcBin(lc, 30/1440, use_masked=True, use_flat=True, extramask = custom_mask)
        flux_key='flux_flat'
    else:
        flux_key='flux'
        lc=tools.lcBin(lc,30/1440,use_masked=True,use_flat=False)
    mask = lc['mask'] if custom_mask is None else custom_mask
    fig = plt.figure(figsize=(11.69,8.27))
    import seaborn as sns
    sns.set_palette(sns.set_palette("RdBu",14))
    axes=[]
    axes +=[fig.add_subplot(411)]
    axes[0].set_title(str(ID).zfill(7)+" - Monotransit Search")
    
    #rast=True if np.sum(lc['mask']>12000) else False
    axes[0].plot(lc['time'][mask],lc[flux_key][mask],'.k',alpha=0.28,markersize=0.75, rasterized=True)
    if use_flat:
        axes[0].plot(lc['bin_time'],lc['bin_flux_flat'],'.k',alpha=0.7,markersize=1.75, rasterized=True)
    else:
        axes[0].plot(lc['bin_time'],lc['bin_flux'],'.k',alpha=0.7,markersize=1.75, rasterized=True)

    axes[0].set_ylim(np.percentile(lc[flux_key][mask],(0.25,99.75)))
    axes[0].set_ylabel("flux")
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    axes +=[fig.add_subplot(412)]
    axes[1].plot([monosearchparams['tcen'].values[0],monosearchparams['tcen'].values[-1]],[-10,-10],'--k',alpha=0.25)
    #plt.plot(monosearchparams_2['tcen'],monosearchparams_2['worstBIC'],'.',c='C5',alpha=0.4)
    for n,dur in enumerate(np.unique(monosearchparams['init_dur'])):
        if n==0:
            axes[1].plot(monosearchparams.loc[monosearchparams['init_dur']==dur,'tcen'],
                     monosearchparams.loc[monosearchparams['init_dur']==dur,'poly_DeltaBIC'],
                     c=sns.color_palette()[n],alpha=0.6,label='Transit - polynomial', rasterized=True)
            axes[1].plot(monosearchparams.loc[monosearchparams['init_dur']==dur,'tcen'],
                     monosearchparams.loc[monosearchparams['init_dur']==dur,'sin_DeltaBIC'],
                     c=sns.color_palette()[-1*(n+1)],alpha=0.6,label='Transit - wavelet', rasterized=True)
        else:
            axes[1].plot(monosearchparams.loc[monosearchparams['init_dur']==dur,'tcen'],
                     monosearchparams.loc[monosearchparams['init_dur']==dur,'poly_DeltaBIC'],
                         c=sns.color_palette()[n],alpha=0.6, rasterized=True)
            axes[1].plot(monosearchparams.loc[monosearchparams['init_dur']==dur,'tcen'],
                     monosearchparams.loc[monosearchparams['init_dur']==dur,'sin_DeltaBIC'],
                         c=sns.color_palette()[-1*(n+1)],alpha=0.6, rasterized=True)
    axes[1].legend(prop={'size': 5})
    ix=(np.isfinite(monosearchparams['sum_DeltaBICs']))&(monosearchparams['sin_DeltaBIC']<1e8)&(monosearchparams['poly_DeltaBIC']<1e8)
    min_bic=np.nanmin(monosearchparams.loc[ix,'poly_DeltaBIC'])
    maxylim=np.nanmax([np.percentile(monosearchparams.loc[ix,'poly_DeltaBIC'],98),
                    np.percentile(monosearchparams.loc[ix,'sin_DeltaBIC'],98)])
    axes[1].set_ylim(maxylim,min_bic)
    axes[1].set_ylabel("Delta BIC")
    axes[1].set_xlabel("Time [BJD-"+str(lc['jd_base'])+"]")
    
    if len(mono_dic)>1:
        
        trans_model_mins=[]
        
        n_poly=int(np.sum([1 for n in range(10) if 'poly_'+str(n) in mono_dic[list(mono_dic.keys())[0]]]))
        for nm,monopl in enumerate(mono_dic):
            tdur=mono_dic[monopl]['tdur']
            tcen=mono_dic[monopl]['tcen']

            transit_zoom=2.25

            axes[1].text(tcen,np.clip(np.min([mono_dic[monopl]['sin_DeltaBIC'],mono_dic[monopl]['poly_DeltaBIC']]),
                                  min_bic,1e6),monopl)

            axes +=[fig.add_subplot(4,len(mono_dic),nm+2*len(mono_dic)+1)]
            axes[-1].set_xticks([])
            axes[-1].set_xticklabels([])
            axes[-1].text(tcen+0.1,np.clip(np.min([mono_dic[monopl]['sin_DeltaBIC'],mono_dic[monopl]['poly_DeltaBIC']]),
                                  min_bic,1e6),
                     monopl)
            for n,dur in enumerate(np.unique(monosearchparams['init_dur'])):
                index=(monosearchparams['init_dur']==dur)&(abs(monosearchparams['tcen']-tcen)<transit_zoom*tdur)
                axes[-1].plot(monosearchparams.loc[index,'tcen'], 
                              monosearchparams.loc[index,'BIC_trans']-monosearchparams.loc[index,'BIC_poly'],
                              c=sns.color_palette()[n], alpha=0.6, rasterized=True)
                axes[-1].plot(monosearchparams.loc[index,'tcen'], 
                              monosearchparams.loc[index,'BIC_trans']-monosearchparams.loc[index,'BIC_sin'],
                              c=sns.color_palette()[-1*n], alpha=0.6, rasterized=True)
            #plt.plot(monosearchparams['tcen'],monosearchparams['worstBIC'],',k')
            axes[-1].plot([tcen,tcen],[-150,130],'--k',linewidth=1.5,alpha=0.25)
            axes[-1].set_xlim(tcen-tdur*transit_zoom,tcen+tdur*transit_zoom)
            axes[-1].set_ylim(maxylim,min_bic)
            if nm==0:
                axes[-1].set_ylabel("Delta BIC")
            else:
                axes[-1].set_yticks([])
                axes[-1].set_yticklabels([])
            
            mono_dets=monosearchparams.loc[monosearchparams['tcen']==mono_dic[monopl]['tcen']].iloc[0]
            axes +=[fig.add_subplot(4,len(mono_dic),nm+3*len(mono_dic)+1)]
            
            nmod=np.arange(len(tdurs))[np.argmin(abs(tdurs-tdur))]
            round_tr=mask&(abs(lc['time']-tcen)<(transit_zoom*tdur))
            x=(lc['time'][round_tr]-tcen)
            y=lc[flux_key][round_tr]
            
            y_offset=np.nanmedian(lc[flux_key][round_tr&(abs(lc['time']-tcen)>(0.7*tdur))]) if use_poly else 0
            y-=y_offset
            
            #Plotting polynomial:
            axes[-1].plot(lc['time'][round_tr],np.polyval([mono_dets['poly_'+str(n)].values[0] for n in range(n_poly)],x),'--',
                     c=sns.color_palette()[-4],linewidth=2.0,alpha=0.6, rasterized=True)
            
            #Plotting transit:
            modeldep=abs(interpmodels[nmod](0.0))
            
            if use_flat and not use_poly:
                trans_model=(mono_dets['trans_dep']/modeldep)*interpmodels[nmod](x)
            else:
                trans_model=mono_dets['trans_grad']*x+(mono_dets['trans_dep']/modeldep)*interpmodels[nmod](x)

            #Plotting sin wavelet:
            newt=x/(2*tdur)*2*np.pi
            amp=np.exp(-1*np.power(newt*1.25, 2.) / (2 * np.power(np.pi, 2.)))
            if use_flat and not use_poly:
                sin_model=mono_dets['sin_dep']*(amp*np.sin(newt-np.pi*0.5)-0.1)
            else:
                sin_model=mono_dets['sin_grad']*x+mono_dets['sin_dep']*(amp*np.sin(newt-np.pi*0.5)-0.1)

            if nm==0:
                axes[-1].set_ylabel("Flux")
            else:
                axes[-1].set_yticks([])
                axes[-1].set_yticklabels([])
            axes[-1].plot(lc['time'][round_tr],y,'.k',markersize=1.5,alpha=0.3, rasterized=True)
            round_tr_bin=abs(lc['bin_time']-tcen)<(transit_zoom*tdur)
            
            axes[-1].plot(lc['bin_time'][round_tr_bin],lc['bin_flux'][round_tr_bin]-y_offset,
                          '.k',alpha=0.7,markersize=2.5, rasterized=True)
            axes[-1].plot(lc['time'][round_tr],trans_model,'-',
                          c=sns.color_palette()[0],linewidth=2.0,alpha=0.85, rasterized=True)
            axes[-1].plot(lc['time'][round_tr],sin_model,'-.',
                          c=sns.color_palette()[-1],linewidth=2.0,alpha=0.6, rasterized=True)
            axes[-1].set_xlim(tcen-tdur*transit_zoom,tcen+tdur*transit_zoom)
            trans_model_mins+=[np.min(trans_model)]
        #print(trans_model_mins)
        trans_model_min=np.min(np.array(trans_model_mins))
        for nm,monopl in enumerate(mono_dic):
            axes[2+2*nm+1].set_ylim(trans_model_min*1.2,np.percentile(lc['bin_flux'],97.5))

    fig.subplots_adjust(wspace=0, hspace=0.15)
    #plt.tight_layout()
    fig.savefig(plot_loc, dpi=400)
    #plt.xlim(1414,1416)
    return plot_loc
'''
    #smearing this out by 0.3*tdur to avoid "micro peaks" in the array making multiple detections
    def gaussian(x , s):
        #Simple gaussian given position-adjusted x and sigma in order to convolve chisq spectrum
        return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )
    chisqs_conv=np.convolve(chisqs, np.fromiter( (gaussian(x, n_oversamp*0.5) for x in range(-1*n_oversamp, n_oversamp, 1 ) ), np.float ), mode='same' )
    
    #Finding all points below some threshold:
    rms_chisq=np.std(chisqs_conv[chisqs_conv>np.percentile(chisqs_conv,20)])
    bool_in_trans=chisqs_conv<(-1*sigma_threshold*rms_chisq)
    
    print("N_trans_points",np.sum(bool_in_trans))
    
    ix_in_trans=[]
    #Looping through each "region" where chisq is lower than threshold and finding minimum value:
    starts = search_xrange[:-1][np.diff(bool_in_trans.astype(int))>0]
    stops  = search_xrange[1:][np.diff(bool_in_trans.astype(int))<0]
    for ns in range(len(starts)):
        ix_bool=(search_xrange>starts[ns])&(search_xrange<stops[ns])
        ix_in_trans+=[search_xrange[ix_bool][np.argmin(chisqs[ix_bool])]]
    
    if len(ix_in_trans)==0:
        #No extra eclipse found
        return None
    elif len(ix_in_trans)==1:
        #Found single eclipse-like feature.
        return {'t0':search_xrange[ix_in_trans[0]],'chisq':chisqs[ix_in_trans[0]],
                'dep':outparams[ix_in_trans[0],0],'dur':outparams[ix_in_trans[0],1]}
    elif len(ix_in_trans)>1:
        #Found multiple eclipse-like features?
        peak=ix_in_trans[np.argmax(chisqs[ix_in_trans])]
        return {'t0':search_xrange[peak],'chisq':chisqs[peak],
                'dep':outparams[peak,0],'dur':outparams[peak,1]}
'''

def PeriodicPlanetSearch(lc, ID, planets, use_binned=False, use_flat=True, binsize=15/1440.0, n_search_loops=5,
                         rhostar=None, Ms=1.0, Rs=1.0, Teff=5800,
                         multi_FAP_thresh=0.00125, multi_SNR_thresh=7.0,
                         plot=False, plot_loc=None, mask_prev_planets=True, **kwargs):
    #Searches an LC (ideally masked for the monotransiting planet) for other *periodic* planets.
    from transitleastsquares import transitleastsquares
    print("Using TLS on ID="+str(ID)+" to search for multi-transiting planets")
    
    #Using bins if we have a tonne of points (eg >1 sector). If not, we can use unbinned data.
    use_binned=True if len(lc['flux'])>10000 else use_binned
    
    #Max period is half the total observed time NOT half the distance from t[0] to t[-1]
    p_max=0.66*np.sum(np.diff(lc['time'])[np.diff(lc['time'])<0.4])
    #np.clip(0.75*(np.nanmax(lc[prefix+'time'])-np.nanmin(lc[prefix+'time'])),10,80)
    suffix='_flat' if use_flat else ''
    if use_flat:
        #Setting the window to fit over as 5*maximum duration
        rhostar=1.0 if rhostar==0.0 or rhostar is None else rhostar
        durmax = (p_max/(3125*rhostar))**(1/3)
        
        plmask_0 = np.tile(False,len(lc['time']))
        if mask_prev_planets:
            #Masking each of those transit events we detected during the MonoTransit search
            for pl in planets:
                plmask_0+=abs(lc['time']-planets[pl]['tcen'])<0.6*planets[pl]['tdur']
            
        lc=tools.lcFlatten(lc,winsize=11*durmax,transit_mask=~plmask_0)
        print("after flattening:",np.sum(lc['mask']))

    if use_binned:
        lc=tools.lcBin(lc,binsize=binsize,use_flat=use_flat)
        #print("binned",lc.keys,np.sum(lc['mask']))
        suffix='_flat'
    #else:
    #    print(use_binned, 'bin_flux' in lc)
    prefix='bin_' if use_binned else ''
    
    if plot:
        sns.set_palette("viridis",10)
        fig = plt.figure(figsize=(11.69,8.27))
        
        rast=True if np.sum(lc['mask']>12000) else False

        #plt.plot(lc['time'][lc['mask']],lc['flux_flat'][lc['mask']]*lc['flux_unit']+(1.0-np.nanmedian(lc['flux_flat'][lc['mask']])*lc['flux_unit']),',k')
        #if prefix=='bin_':
        #    plt.plot(lc[prefix+'time'],lc[prefix+'flux'+suffix]*lc['flux_unit']+(1.0-np.nanmedian(lc[prefix+'flux'+suffix])*lc['flux_unit']),'.k')
        plt.subplot(312)
        plt.plot(lc['time'][lc['mask']],
                 lc['flux_flat'][lc['mask']]*lc['flux_unit']+(1.0-np.nanmedian(lc['flux_flat'][lc['mask']])*lc['flux_unit']),
                 '.k', markersize=0.5,rasterized=rast)
        lc=tools.lcBin(lc, binsize=1/48, use_flat=True)
        plt.plot(lc['bin_time'],
                 lc['bin_flux_flat']*lc['flux_unit']+(1.0-np.nanmedian(lc['bin_flux_flat'])*lc['flux_unit']),
                 '.k', markersize=4.0)

    #Looping through, masking planets, until none are found.
    #{'01':{'period':500,'period_err':100,'FAP':np.nan,'snr':np.nan,'tcen':tcen,'tdur':tdur,'rp_rs':np.nan}}
    if prefix+'mask' in lc:
        anommask=lc[prefix+'mask'][:]
    else:
        anommask=~np.isnan(lc[prefix+'flux'+suffix][:])
    plmask=np.tile(False,len(anommask))
    t_zero=np.nanmin(lc['time'])
    SNR_last_planet=100;init_n_pl=len(planets);n_pl=len(planets);results=[]
    while SNR_last_planet>multi_SNR_thresh and n_pl<(n_search_loops+init_n_pl):
        if len(planets)>1:
            planet_name=str(int(1+np.max([float(key) for key in planets]))).zfill(2)
        else:
            planet_name='00'
        #Making model. Making sure lc is at 1.0 and in relatie flux, not ppt/ppm:
        '''
        if np.sum(plmask)>0:
            #Re-doing flattening with other transits now masked (these might be causing 
            lc=tools.lcFlatten(lc,winsize=11*durmax,use_binned=use_binned,transit_mask=~plmask)
        '''
        modx = lc[prefix+'time']-t_zero
        mody = lc[prefix+'flux'+suffix] * lc['flux_unit']+(1.0-np.nanmedian(lc[prefix+'flux'+suffix][anommask])*lc['flux_unit'])
        #print(n_pl,len(mody),len(anommask),np.sum(anommask),len(plmask),np.sum(plmask))
        if np.sum(plmask)>0:
            mody[plmask] = mody[anommask][np.random.choice(np.sum(anommask),np.sum(plmask))][:]
        #anommask *= tools.CutAnomDiff(mody)
        #print(n_pl,"norm_mask:",np.sum(lc['mask']),"anoms:",np.sum(anommask),"pl mask",np.sum(plmask),"total len",len(anommask))
        model = transitleastsquares(modx[anommask], mody[anommask])
        results+=[model.power(period_min=1.1,period_max=p_max,duration_grid_step=1.0625,Rstar=Rs,Mstar=Ms,
                              use_threads=1,show_progress_bar=False, n_transits_min=3)]
        #print(results[-1])
        
        if 'FAP' in results[-1] and 'snr' in results[-1] and not np.isnan(results[-1]['snr']) and 'transit_times' in results[-1]:
            #Defining transit times as those times where the SNR in transit is consistent with expectation (>-3sigma)
            snr_per_trans_est=np.sqrt(np.sum(results[-1].snr_per_transit>0))
            trans=np.array(results[-1]['transit_times'])[results[-1].snr_per_transit>snr_per_trans_est/2]
        else:
            trans=[]
            
        phase_nr_trans=(lc[prefix+'time'][~plmask&anommask]-results[-1]['T0']-0.5*results[-1]['period'])%results[-1]['period']-0.5*results[-1]['period']
        if 'FAP' in results[-1] and 'snr' in results[-1] and np.sum(abs(phase_nr_trans)<0.5*np.clip(results[-1]['duration'],0.2,2))>3:
            plparams = QuickMonoFit(deepcopy(lc),results[-1]['T0'],np.clip(results[-1]['duration'],0.15,3),
                                    init_period=results[-1]['period'],how='periodic',ndurs=4.5,Teff=Teff,Rs=Rs,Ms=Ms,
                                    fluxindex=prefix+'flux'+suffix,mask=~plmask&anommask, **kwargs)
            SNR=np.max([plparams['snr'],results[-1].snr])
            FAP=results[-1]['FAP']
        else:
            SNR=0;FAP=0
        if (FAP<multi_FAP_thresh) and SNR>multi_SNR_thresh and len(trans)>2:
            SNR_last_planet=SNR
            planets[planet_name]=plparams
            planets[planet_name]['tcen']+=t_zero
            planets[planet_name].update({'period':results[-1].period, 'period_err':results[-1].period_uncertainty,
                                         'P_min':results[-1].period,
                                         'snr_tls':results[-1].snr, 'FAP':results[-1].FAP, 
                                         'orbit_flag':'periodic',
                                         'xmodel':results[-1].model_lightcurve_time+t_zero,
                                         'ymodel':results[-1].model_lightcurve_model, 'N_trans':len(trans)})
            if plot:
                plt.subplot(311)
                plt.plot([results[-1].period,results[-1].period],[0,1.4*results[-1].snr],'-',
                         linewidth=4.5,alpha=0.4,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det_'+str(n_pl))
                plt.plot(results[-1].periods,results[-1].power,c=sns.color_palette()[n_pl-init_n_pl])
                plt.subplot(312)
                plt.plot(results[-1]['model_lightcurve_time']+t_zero,results[-1]['model_lightcurve_model'],'.',
                         alpha=0.75,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det='+str(n_pl),
                         rasterized=True)

            '''#Special cases for mono and duos:
            if len(trans)==2:
                planets[planet_name]['period_err']=np.nan
                planets[planet_name]['tcen_2']=trans[1]
                planets[planet_name]['orbit_flag']='duo'
            elif len(trans)==1:
                planets[planet_name]['period']=np.nan
                planets[planet_name]['period_err']=np.nan
                planets[planet_name]['orbit_flag']='mono'
            '''
            #Removing planet from future data to be searched
            this_pl_masked=(((lc[prefix+'time']-planets[planet_name]['tcen']+0.7*planets[planet_name]['tdur'])%planets[planet_name]['period'])<(1.4*planets[planet_name]['tdur']))
            plmask=plmask+this_pl_masked#Masking previously-detected transits
            #print(n_pl,results[-1].period,plparams['tdur'],np.sum(this_pl_masked),np.sum(plmask))
            #print(n_pl,"pl_mask",np.sum(this_pl_masked)," total:",np.sum(plmask))
        elif SNR>multi_SNR_thresh:
            # pseudo-fails - we have a high-SNR detection but it's a duo or a mono.
            #print(plparams['tcen'],plparams['tdur'],"fails with transits at ",trans,"with durations",plparams['tdur'],"transits. Reserching")
            this_pl_masked=np.min(abs((lc[prefix+'time'][np.newaxis,:]-t_zero)-np.array(trans)[:,np.newaxis]),axis=0)<(0.7*plparams['tdur'])
            #this_pl_masked=(((lc[prefix+'time']-plparams['tcen']+0.7*plparams['tdur'])%results[-1].period)<(1.4*plparams['tdur']))
            #print(n_pl,results[-1].period,plparams['tdur'],np.sum(this_pl_masked))
            plmask = plmask+this_pl_masked
            SNR_last_planet=SNR
        else:
            # Fails
            #print(n_pl,"detection at ",results[-1].period," with ",len(trans)," transits does not meet SNR ",SNR,"or FAP",results[-1].FAP)
            SNR_last_planet=0
        n_pl+=1

    if plot:
        multis=[pl for pl in planets if planets[pl]['orbit_flag']=='periodic']
        if len(multis)==0:
            plt.subplot(311)
            plt.plot(results[0].periods,results[0].power)
        else:
            for n_m,mult in enumerate(multis):
                #print(planets[mult]['depth'], planets[mult]['interpmodel'](0.0), np.nanstd(lc[prefix+'flux'+suffix]),
                #      np.min(lc[prefix+'flux'+suffix]),np.max(lc[prefix+'flux'+suffix]))
                phase=(lc['time']-planets[mult]['tcen']-0.5*planets[mult]['period'])%planets[mult]['period']-0.5*planets[mult]['period']
                lc=tools.lcFlatten(lc,winsize=11*durmax,transit_mask=abs(phase)>0.6*planets[mult]['tdur'])

                plt.subplot(3, len(multis), len(multis)*2+1+n_m)
                #print("subplot ",3, len(multis), len(multis)*2+1+n_m)
                bin_phase=tools.bin_lc_segment(np.column_stack((np.sort(phase[abs(phase)<1.2]),
                                                                lc['flux'][abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])],
                                                                lc['flux_err'][abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])])),binsize=planets[mult]['tdur']*0.15)
                time_shift=0.4*np.nanstd(bin_phase[:,1])*(lc['time'][abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])] - \
                                                                                                planets[mult]['tcen'])/planets[mult]['period']
                #plt.scatter(phase[abs(phase)<1.2],time_shift+lc['flux'][abs(phase)<1.2],
                plt.scatter(phase[abs(phase)<1.2],lc['flux'][abs(phase)<1.2],
                            s=3,c='k',alpha=0.4)
                plt.scatter(bin_phase[:,0],bin_phase[:,1],s=8,
                            c=sns.color_palette()[n_pl-init_n_pl])
                plt.plot(np.sort(phase[abs(phase)<1.2]),
                         planets[mult]['interpmodel'](phase[abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])]),
                         c=sns.color_palette()[n_pl-init_n_pl],alpha=0.4)
                #plt.ylim(np.nanmin(bin_phase[:,1])-2*np.nanstd(bin_phase[:,1]),
                #         np.nanmax(bin_phase[:,1])+2*np.nanstd(bin_phase[:,1]))
                plt.gca().set_title(planet_name+'/det='+str(n_pl))
        if plot_loc is None:
            plot_loc = str(ID).zfill(11)+"_multi_search.pdf"
        elif plot_loc[-1]=='/':
            plot_loc = plot_loc+str(ID).zfill(11)+"_multi_search.pdf"
        plt.subplot(311)
        plt.legend(prop={'size': 5})
        plt.subplot(312)        
        plt.plot(results[-1]['model_lightcurve_time']+t_zero,results[-1]['model_lightcurve_model'],
                 alpha=0.5,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det_'+str(n_pl),linewidth=4)
        plt.legend(prop={'size': 5})
        if 'jd_base' in lc:
            plt.xlabel("time [BJD-"+str(lc['jd_base'])+"]")
        else:
            plt.xlabel("time")
        plt.subplot(311)
        plt.suptitle(str(ID).zfill(11)+"-  Multi-transit search")
        plt.tight_layout()
        plt.savefig(plot_loc, dpi=400)
    if plot:
        return planets, plot_loc
    else:
        return planets, None


def GenModelLc(lc,all_pls,mission,Rstar=1.0,rhostar=1.0,Teff=5800,logg=4.43):
    #Generates model planet lightcurve from dictionary of all planets
    u = tools.getLDs(Teff,logg=logg,FeH=0.0,mission=mission).ravel()
    cad=np.nanmedian(np.diff(lc['time'])) 
    light_curves=[]
    for pl in all_pls:
        if all_pls[pl]['orbit_flag']=='periodic':
            #generating periodic planet
            # The light curve calculation requires an orbit
            orbit = xo.orbits.KeplerianOrbit(r_star=Rstar, rho_star=rhostar,
                                             period=all_pls[pl]['period'], t0=all_pls[pl]['tcen'], b=0.4)
            # Compute a limb-darkened light curve using starry
            light_curves+=[xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=np.sqrt(all_pls[pl]['depth']), 
                                                                   t=lc['time'], texp=cad*0.98).eval()]
        elif all_pls[pl]['orbit_flag'] in ['mono','duo']:
            #generating two monos for duo
            per_guess=18226*rhostar*(2*np.sqrt(1-0.4**2)/all_pls[pl]['tdur'])**-3#from circular orbit and b=0.4
            orbit = xo.orbits.KeplerianOrbit(r_star=Rstar, rho_star=rhostar,
                                             period=per_guess, t0=all_pls[pl]['tcen'], b=0.4)
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=np.sqrt(all_pls[pl]['depth']), 
                                                                   t=lc['time'], texp=cad*0.98
                                                                   ).eval()
            light_curve[abs(lc['time']-all_pls[pl]['tcen'])>per_guess*0.4]=0.0
            if all_pls[pl]['orbit_flag'] == 'duo' and 'tcen_2' in all_pls[pl]:
                orbit2 = xo.orbits.KeplerianOrbit(r_star=Rstar, rho_star=rhostar,
                                                 period=per_guess, t0=all_pls[pl]['tcen_2'], b=0.4)
                light_curve2 = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=np.sqrt(all_pls[pl]['depth']), 
                                                                       t=lc['time'], texp=cad*0.98
                                                                       ).eval()
                light_curve2[abs(lc['time']-all_pls[pl]['tcen_2'])>per_guess*0.4]=0.0
                light_curve=light_curve+light_curve2
            light_curves+=[light_curve]
    return np.column_stack(light_curves)


def DoEBfit(lc,tc,dur):
    # Performs EB fit to primary/secondary.
    return None

def dipmodel_step(params,x,npolys):
    return np.hstack((np.polyval( params[1:1+npolys[0]], x[x<params[0]]),
                      np.polyval( params[-npolys[1]:], x[x>=params[0]]) ))
    
def log_likelihood_step(params,x,y,yerr,npolys):
    model=dipmodel_step(params,x,npolys)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def Step_neg_lnprob(params, x, y, yerr, priors, polyorder,npolys):
    #Getting log prior - we'll leave the polynomials to wander and us:
    lnprior=0
    lnp=log_gaussian(params[0], priors[0], priors[1])
    if lnp<-0.5 and lnp>-20:
        lnprior+=3*lnp
    elif lnp<-20:
        lnprior+=1e6*lnp
    #Getting log likelihood:
    llk=log_likelihood_step(params,x,y,yerr,npolys)
    return -1*(lnprior + llk)


def Poly_neg_lnprob(params,x,y,yerr,priors,polyorder):
    return -1*Poly_lnprob(params, x, y, yerr, priors, polyorder=polyorder)

def Poly_lnprob(params, x, y, yerr, priors, polyorder=2):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    for p in np.arange(polyorder+1):
        #Simple log gaussian prior here:
        lnp=log_gaussian(params[p], 0.0, priors[p],weight=1)
        if lnp<-0.5 and lnp>-20:
            lnprior+=3*lnp
        elif lnp<-20:
            lnprior+=1e6*lnp
    llk = log_likelihood_poly(params, x, y, yerr)
    return lnprior + llk

def log_likelihood_poly(params, x, y, yerr):
    model=np.polyval(params,x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def Sinusoid_neg_lnprob(params, x, y, yerr, priors, polyorder):
    return -1*Sinusoid_lnprob(params, x, y, yerr, priors, polyorder=polyorder)

def Sinusoid_lnprob(params, x, y, yerr, priors, polyorder=2):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    for p in np.arange(3):
        #Simple log gaussian prior here:
        lnp=log_gaussian(params[p], priors[p,0], priors[p,1],weight=1)
        if lnp<-0.5 and lnp>-20:
            lnprior+=3*lnp
        elif lnp<-20:
            lnprior+=1e6*lnp
    llk = log_likelihood_sinusoid(params, x, y, yerr)
    return lnprior + llk

def log_likelihood_sinusoid(params, x, y, yerr):
    model=dipmodel_sinusoid(params,x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def dipmodel_sinusoid(params,x):
    #Sinusoidal model aligned with dip.
    # tcen, log(dur), log(dep), [n x polynomials]
    newt=(x-params[0])/(4.5*np.exp(params[1]))*2*np.pi-np.pi*0.5
    return np.polyval(params[3:],x) + np.exp(params[2])*(np.sin(newt))

def Gaussian_neg_lnprob(params, x, y, yerr, priors, order=3):
    return -1*Gaussian_lnprob(params, x, y, yerr, priors, order=order)

def Gaussian_lnprob(params, x, y, yerr, priors, order=3):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    for p in np.arange(3):
        #Simple log gaussian prior here:
        lnp=log_gaussian(params[p], priors[p,0], priors[p,1],weight=1)
        if lnp<-0.5 and lnp>-20:
            lnprior+=3*lnp
        elif lnp<-20:
            lnprior+=1e6*lnp
    llk = log_likelihood_gaussian_dip(params, x, y, yerr)
    return lnprior + llk

def log_gaussian(x, mu, sig, weight=0.1):
    return -1*weight*np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

def log_likelihood_gaussian_dip(params, x, y, yerr):
    model=dipmodel_gaussian(params,x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2)

def dipmodel_gaussian(params,x):
    dip=1.0+np.exp(params[0])*np.exp(-1*np.power(x - params[2], 2.) / (2 * np.power((0.075*np.exp(params[1])), 2.)))
    mod = np.polyval(params[3:],x)*dip
    return mod


def centroid_neg_lnprob(params, t, x, y, xerr, yerr, priors, interpmodel, order=3):
    return -1*centroid_lnprob(params, t, x, y, xerr, yerr, priors, interpmodel, order=order)

def centroid_lnprob(params, t, x, y, xerr, yerr, priors, interpmodel, order=3):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    
    for p in np.arange(2):
        #Simple log gaussian prior here:
        gauss=log_gaussian(params[p], priors[p,0], priors[p,1],weight=1)
        #gauss=0.0 if gauss>-0.5 else gauss
        lnprior+=gauss
        
    llk = log_likelihood_centroid(params, t, x, y, xerr,yerr, interpmodel, order)
    return lnprior + llk

def log_likelihood_centroid(params, t, x, y, xerr, yerr, interpmodel, order):
    xmodel,ymodel=dipmodel_centroid(params,t,interpmodel,order)
    xsigma2 = xerr ** 2
    xllk=-0.5 * np.sum((x - xmodel) ** 2 / xsigma2 + np.log(xsigma2))
    ysigma2 = yerr ** 2
    yllk=-0.5 * np.sum((y - ymodel) ** 2 / ysigma2 + np.log(ysigma2))
    return xllk+yllk

def dipmodel_centroid(params,t,interpmodel,order):
    #params=xdep, ydep, xpolyvals, ypolyvals
    xdip = params[0]*interpmodel(t)
    ydip = params[1]*interpmodel(t)
    xmod = np.polyval(params[2:2+(order+1)],t)+xdip
    ymod = np.polyval(params[2+(order+1):],t)+ydip
    return xmod,ymod


def AsteroidCheck(lc,monoparams,ID,order=3,dur_region=3.5,
                  plot=False,plot_loc=None, return_fit_lcs=False, remove_earthshine=True, **kwargs):
    # Checking lightcure for background flux boost during transit due to presence of bright asteroid
    # Performing two model fits 
    # - one with a gaussian "dip" roughly corresponding to the transit combined with a polynomial trend for out-of-transit
    # - one with only a polynomial trend
    # These are then compared, and the BIC returned to judge model fit
    
    #Need the region around the model to be at least 1.5d, otherwise the polynomial will absorb it:
    dur_region=np.clip(dur_region,2/monoparams['tdur'],5/monoparams['tdur'])
    
    nearish_region=np.max([4.5,monoparams['tdur']*dur_region]) #For the background fit, we'll take a region 9d long
    nearishTrans=(abs(lc['time']-monoparams['tcen'])<nearish_region)&lc['mask']
    cad=np.nanmedian(np.diff(lc['time'][nearishTrans]))
    
    if 'bg_flux' in lc and np.sum(np.isfinite(lc['bg_flux'][nearishTrans]))>0:
        # Fits two models - one with a 2D polynomial spline and the interpolated "dip" model from the fit, and one with only a spline
        #If there's a big gap, we'll remove the far side of that
        if np.max(np.diff(lc['time'][nearishTrans]))>0.4:
            jump_n=np.argmax(np.diff(lc['time'][nearishTrans]))
            jump_time=0.5*(lc['time'][nearishTrans][jump_n]+lc['time'][nearishTrans][jump_n+1])
            if jump_time < monoparams['tcen']:
                nearishTrans=(lc['time']>jump_time)&((lc['time']-monoparams['tcen'])<nearish_region)&lc['mask']
            elif jump_time > monoparams['tcen']:
                nearishTrans=((lc['time']-monoparams['tcen'])>(-1*nearish_region))&(lc['time']<jump_time)&lc['mask']
                
        nearishTrans[nearishTrans]=np.isfinite(lc['bg_flux'][nearishTrans])

        bg_lc=np.column_stack((lc['time'][nearishTrans],
                               lc['bg_flux'][nearishTrans],
                               np.tile(np.nanstd(lc['bg_flux'][nearishTrans]),np.sum(nearishTrans))
                              ))
        
        bg_lc[:,0]-=monoparams['tcen']
        nearTrans=(abs(bg_lc[:,0])<monoparams['tdur']*dur_region)
        sigma2 = bg_lc[:,2]**2
        outTransit=(abs(bg_lc[:,0])>monoparams['tdur']*0.75)
        inTransit=(abs(bg_lc[:,0])<monoparams['tdur']*0.35)
        
        
        bg_lc[:,1:]/=np.nanmedian(bg_lc[outTransit,1])

        if remove_earthshine and lc['cadence'][np.argmin(abs(lc['time']-monoparams['tcen']))].lower()[0]=='t':
            #Removing Earthshine with a filter on frequencies of 1/0.5/0.33 days (for TESS lightcurves only):
            newt=np.arange(bg_lc[0,0],bg_lc[-1,0],cad)
            
            #Preparing the signal timeseries:            
            init_poly=np.polyfit(bg_lc[outTransit,0],bg_lc[outTransit,1],order) #do polynomial fit to make time series flat 
            news = (bg_lc[:,1]-np.polyval(init_poly,bg_lc[:,0]))[np.argmin(abs(bg_lc[:,0][:,np.newaxis]-newt[np.newaxis,:]),axis=0)] #Making sure the timeseries is uniform

            # Doing an FFT fit for the known frequencies associated with Earth rotation:
            n=news.size
            fr=np.fft.fftfreq(n,cad)  # a nice helper function to get the frequencies  
            fou=np.fft.fft(news) 
            freqs=[1,2,3] #Frequencies to filter - all multiples of 1

            #make up a narrow bandpass with a Gaussian
            df=0.066
            gpl= np.sum([np.exp(- ((fr-f)/(2*df))**2) for f in freqs],axis=0) # pos. frequencies
            gmn= np.sum([np.exp(- ((fr+f)/(2*df))**2) for f in freqs],axis=0) # neg. frequencies
            g=gpl+gmn    

            #ifft
            s2=np.fft.ifft(fou*g) #filtered spectrum = spectrum * bandpass 
            fft_model = np.real(s2)[np.argmin(abs(newt[:,np.newaxis] - bg_lc[:,0][np.newaxis,:]),axis=0)]
            #fft_model += np.polyval(init_poly, bg_lc[nearTrans,0]
            #print(len(fft_model),len(bg_lc[:,0]))
            
            #Checking model is actually a good fit around transit by doing 1D poly + FFT model vs 2D poly:
            bg_fft_model     = np.polyval(np.polyfit(bg_lc[nearTrans&outTransit,0],
                                                     bg_lc[nearTrans&outTransit,1] - fft_model[nearTrans&outTransit],0),
                                          bg_lc[:,0]) + \
                               fft_model
            llk_bg_model = -0.5 * np.sum((bg_lc[nearTrans&outTransit,1] - bg_fft_model[nearTrans&outTransit]) ** 2 / \
                                         sigma2[nearTrans&outTransit])
            
            bg_model = np.polyval(np.polyfit(bg_lc[nearTrans&outTransit,0],bg_lc[nearTrans&outTransit,1],2),
                                  bg_lc[:,0])
            llk_polyfit  = -0.5 * np.sum((bg_lc[nearTrans&outTransit,1] - bg_model[nearTrans&outTransit]) ** 2 / \
                                         sigma2[nearTrans&outTransit])
            #print("polyfit llk:", llk_polyfit, "fft llk:", llk_bg_model)
            if llk_polyfit > llk_bg_model-1:
                print("Earthshine model not useful in this case - polynomial model is better by", llk_polyfit - llk_bg_model)
                fft_model=np.tile(0,len(bg_lc[:,0]))
        else:
            fft_model=np.tile(0,len(bg_lc[:,0]))

        outTransit*=nearTrans
        
        #print(bg_lc)
        log_height_guess=np.log(2*np.clip((np.nanmedian(bg_lc[inTransit,1])-np.nanmedian(bg_lc[nearTrans&outTransit,1])),
                                          0.00001,1000) )
        priors= np.column_stack(([log_height_guess,np.log(np.clip(1.4*monoparams['tdur'],6*cad,3.0)),0.0],
                                 [3.0,0.25,0.75*monoparams['tdur']]))
        best_nodip_res={'fun':1e30,'bic':1e9,'llk':-1e20}
        best_dip_res={'fun':1e30,'bic':1e9,'llk':-1e20}
        methods=['L-BFGS-B', 'Nelder-Mead', 'Powell']
        n=0
        while n<21:
            #Running the minimization 7 times with different initial params to make sure we get the best fit
            
            #non-dip is simple poly fit. Including 10% cut in points to add some randomness over n samples
            
            rand_choice=np.random.random(len(bg_lc))<0.9
            nodip_res={'x':np.polyfit(bg_lc[nearTrans&rand_choice,0],
                                      bg_lc[nearTrans&rand_choice,1] - fft_model[nearTrans&rand_choice],
                                      order)}
            nodip_model = fft_model[nearTrans] + np.polyval(nodip_res['x'],bg_lc[nearTrans,0])
            nodip_res['llk'] = log_likelihood_gaussian_dip(np.hstack((-30,-30,0,nodip_res['x'])),
                                                           bg_lc[nearTrans,0], bg_lc[nearTrans,1], bg_lc[nearTrans,2])
            #nodip_res['llk'] = -0.5 * np.sum((bg_lc[nearTrans,1] - nodip_model)**2 / sigma2[nearTrans])
            nodip_res['prior']=0
            #np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
            #BIC = 2*(neg_log_likelihood-log_prior) + log(n_points)*n_params 
            nodip_res['bic'] = np.log(np.sum(nearTrans))*len(nodip_res['x']) - 2 * nodip_res['llk']
            
            #print('no dip:',nodip_res['bic'])
            if nodip_res['bic']<best_nodip_res['bic']:
                best_nodip_res=nodip_res
                
                
            
            #log10(height), log10(dur), tcen
            dip_args= np.hstack(([np.random.normal(log_height_guess,0.25)-(n%4)/2.0,
                                  np.log10(1.5*monoparams['tdur'])+abs(np.random.normal(0.0,0.5)),
                                  np.random.normal(0.0,0.5*monoparams['tdur'])],
                                  np.polyfit(bg_lc[outTransit&rand_choice,0],
                                             bg_lc[outTransit&rand_choice,1] - fft_model[outTransit&rand_choice],
                                             order)))

            dip_res=optim.minimize(Gaussian_neg_lnprob, dip_args,
                                     args=(bg_lc[nearTrans,0],
                                           bg_lc[nearTrans,1]-fft_model[nearTrans],
                                           bg_lc[nearTrans,2],
                                           priors,
                                           order),method=methods[n%3])
            
            dip_res['llk']=log_likelihood_gaussian_dip(dip_res['x'], 
                                                       bg_lc[nearTrans,0],
                                                       bg_lc[nearTrans,1]-fft_model[nearTrans],
                                                       bg_lc[nearTrans,2])
            dip_res['prior']=dip_res['fun']-dip_res['llk']
            dip_res['bic']= np.log(np.sum(nearTrans))*len(dip_args) - 2 * dip_res['llk']
            
            #print('dip:',dip_args,dip_res,dip_bic)
            if dip_res['bic']<best_dip_res['bic']:
                best_dip_res=dip_res
                
            #Increasing the polynomial order every odd number if we haven't yet found a good solution:
            if n>7 and n%2==1:
                order+=1
                #Need to expand on priors to include new polynomial orders in this case:
            #print(n,"dip:",dip_res['fun'],dip_res['bic'],dip_args,"nodip:",nodip_res['fun'],nodip_res['bic'],"order:",order)

            if n>=7 and (best_dip_res['llk']>-1e20) and (best_nodip_res['llk']>-1e20):
                break
            n+=1
            
        fit_dict={'asteroid_dip_'+col:best_dip_res[col] for col in best_dip_res}
        fit_dict.update({'asteroid_nodip_'+col:best_nodip_res[col] for col in best_nodip_res})
        
        # Also doing correlation coefficient between bg flux and real flux during transit:
        # In the case that the BG causes the lc to dip, we should find a negative correlation coefficient & low p-value:
        from scipy.stats import pearsonr
        fluxkey='rawflux' if 'rawflux' in lc else 'flux'
        just_oot=lc['mask'] * \
                 (abs(lc['time']-monoparams['tcen'])>monoparams['tdur']*0.75) * \
                 (abs(lc['time']-monoparams['tcen'])<monoparams['tdur']*2)
        inTransit=lc['mask'] * (abs(lc['time']-monoparams['tcen'])<monoparams['tdur']*0.625)
        #Removing a just-out-of-transit polynomial fit to both sets of in-transit data, so we should only have the dip
        bg_poly=np.polyval(np.polyfit(lc['time'][just_oot],
                                     lc['bg_flux'][just_oot], 1), lc['time'][inTransit])
        lc_poly=np.polyval(np.polyfit(lc['time'][just_oot],
                                     lc[fluxkey][just_oot], 1), lc['time'][inTransit])
        R=pearsonr(lc['bg_flux'][inTransit] - bg_poly, lc[fluxkey][inTransit]-lc_poly)
        fit_dict['R_bg_flux_corr']=R[0]
        fit_dict['P_bg_flux_corr']=R[1]
        
        if (best_dip_res['llk']>-1e20) and (best_nodip_res['llk']>-1e20):
            fit_dict['asteroid_DeltaBIC']=best_dip_res['bic']-best_nodip_res['bic'] # [prefer dip] < 0 < [prefer no dip]
            fit_dict['asteroid_log_llk_ratio']=(best_dip_res['llk'])-(best_nodip_res['llk'])
            fit_dict['asteroid_ampl']=np.exp(best_dip_res['x'][0])
            fit_dict['asteroid_dur']=np.exp(best_dip_res['x'][1])
            best_model_resids = bg_lc[nearTrans&outTransit,1] - \
                       (fft_model[nearTrans&outTransit] + dipmodel_gaussian(best_dip_res['x'],bg_lc[nearTrans&outTransit,0]))
            fit_dict['asteroid_bg_stdw']=np.nanmedian(abs(np.diff(best_model_resids)))
            fit_dict['asteroid_bg_stdr']=np.nanstd(best_model_resids)
            fit_dict['asteroid_snrw'] = fit_dict['asteroid_ampl'] / \
                                       ( fit_dict['asteroid_bg_stdw'] / np.sqrt(fit_dict['asteroid_dur']/cad) )
            fit_dict['asteroid_snrr'] = fit_dict['asteroid_ampl'] / \
                                       ( fit_dict['asteroid_bg_stdr'] / np.sqrt(fit_dict['asteroid_dur']/cad) )
            #print(ID,"| Ran Asteroid fitting "+str(n)+" times, and DeltaBIC="+str(DeltaBIC),
            #      "| params:",best_dip_res['x'],best_nodip_res['x'])
        #else:
        #    print(lc['bg_flux'][nearTrans]-fft_model[nearTrans])
        if plot:
            if plot_loc is not None and type(plot_loc)!=str:
                ax = plot_loc
            else:
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111)
                if plot_loc is None:
                    plot_loc=str(ID)+"_asteroid_check.pdf"
                elif plot_loc[-1]=='/':
                    plot_loc=plot_loc+str(ID)+"_asteroid_check.pdf"
                #print("Plotting asteroid",ID, np.sum(bg_lc[nearTrans,1]/bg_lc[nearTrans,1]==1.0),
                #      len(bg_lc[nearTrans,1]),plot_loc,best_nodip_res)
            if plot:
                #### PLOTTING ###
                ax.scatter(bg_lc[nearTrans,0],bg_lc[nearTrans,1],s=2,alpha=0.75,rasterized=True,zorder=1)
                
                ax.plot(bg_lc[nearTrans,0],np.nanmedian(bg_lc[nearTrans,1])+fft_model[nearTrans],':k',
                        alpha=0.2,linewidth=4,label='ft model of earthshine',zorder=2)

                ax.plot([-0.5*monoparams['tdur'],-0.5*monoparams['tdur']],[-2.0,2.0],':k',alpha=0.4,rasterized=True)
                ax.plot([0.0,0.0],[-2.0,2.0],'--k',linewidth=3,alpha=0.6,rasterized=True)
                ax.plot([0.5*monoparams['tdur'],0.5*monoparams['tdur']],[-2.0,2.0],':k',alpha=0.4,rasterized=True)
                ax.set_ylabel("Relative background flux")

                if (best_nodip_res['llk']>-1e20):
                    ax.plot(bg_lc[nearTrans,0],fft_model[nearTrans]+
                             np.polyval(best_nodip_res['x'],bg_lc[nearTrans,0]),c='C3',linewidth=2,
                             label='pure trend',alpha=0.6,rasterized=True,zorder=2)
                if (best_dip_res['llk']>-1e20):
                    ax.plot(bg_lc[nearTrans,0],
                             fft_model[nearTrans]+dipmodel_gaussian(best_dip_res.x,bg_lc[nearTrans,0]),c='C4',linewidth=2.5,
                             label='trend+asteroid',alpha=0.8,rasterized=True,zorder=3)
                try:
                    ax.set_ylim(np.nanmin(fft_model[nearTrans]+bg_lc[nearTrans,1]),
                                np.nanmax(fft_model[nearTrans]+bg_lc[nearTrans,1]))
                except:
                    b=0
                #plt.ylim(np.percentile(bg_lc[inTransit,1],[0.2,99.8]))
                ax.legend(prop={'size': 5})
                if (best_dip_res['llk']>-1e20) and (best_nodip_res['llk']>-1e20):
                    ax.set_title(str(ID)+" Asteroid. - "+["pass","fail"][int(fit_dict['asteroid_DeltaBIC']<-10)])
                else:
                    ax.set_title(str(ID)+" Asteroid. No fit ???")
                if type(plot_loc)==str:
                    fig.savefig(plot_loc, dpi=400)
                if return_fit_lcs:
                    return fit_dict, ax, np.column_stack((bg_lc[:,0],bg_lc[:,1],bg_lc[:,2],fft_model,
                                                         np.polyval(best_nodip_res['x'],bg_lc[:,0]),
                                                         dipmodel_gaussian(best_dip_res.x,bg_lc[:,0])))
                else:
                    return fit_dict, ax
        elif (best_dip_res['llk']>-1e20) and (best_nodip_res['llk']>-1e20):
            if return_fit_lcs:
                return fit_dict, None, None
            else:
                return fit_dict, None
        else:
            if return_fit_lcs:
                return None, None, None
            else:
                return None, None
    else:
        if return_fit_lcs:
            return None, None, None
        else:
            return None, None

def VariabilityCheck(lc, params, ID, modeltype='all',plot=False,plot_loc=None,ndurs=2.4, 
                     polyorder=1, return_fit_lcs=False, **kwargs):
    # Checking lightcure for variability flux boost during transit due to presence of bright asteroid
    # Performs two potential models:
    # - 1) 'sin': Variability sinusoid+polynomial model
    # - 2) 'step': Discontinuity Model (two polynomials and a gap between them)
    # the BIC returned to judge against the transit model fit
    
    #assuming QuickMonoFit has been run, we can replicate the exact x/y/yerr used there:
    x = params['monofit_x']-params['tcen']
    round_trans=(abs(x)<ndurs*np.clip(params['tdur'],0.1,5.0))
    x = x[round_trans]
    y = params['monofit_y'][round_trans]
    #mask = lc['mask'][np.isin(lc['time'],params['monofit_x'][abs(params['monofit_x']-params['tcen'])<ndurs*params['tdur']])]
    #assert len(mask)==len(x)
    
    #print(params['tdur'],np.sum(abs(x)>0.6*params['tdur']))
    
    yerr = params['monofit_yerr'][round_trans]
    y_trans = params['monofit_ymodel'][round_trans]
    sigma2 = yerr ** 2
    outTransit=abs(x)>0.6*params['tdur']
    yspan=np.diff(np.percentile(y,[5,95]))[0]
    priors={}
    best_mod_res={}
    mods=[]
    if modeltype=='sin' or modeltype=='both' or modeltype=='all':
        priors['sin']= np.column_stack(([0.0,np.log(params['tdur']),np.log(yspan)],
                                        [0.5*params['tdur'],3.0,4.0]))
        best_mod_res['sin']={'fun':1e30,'bic':1e9,'sin_llk':-1e9}
        mods+=['sin']
    if modeltype=='step' or modeltype=='both' or modeltype=='all':
        priors['step']= [0.0,0.5*params['tdur']]
        best_mod_res['step']={'fun':1e30,'bic':1e9,'sin_llk':-1e9,'npolys':[]}
        mods+=['step']
    if modeltype=='poly' or modeltype=='both' or modeltype=='all':
        best_mod_res['poly']={'fun':1e30,'bic':1e9,'sin_llk':-1e9,'npolys':[]}
        mods+=['poly']
    if modeltype!='none' and len(x)>polyorder+1 and len(y)>polyorder+1:
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        n=0
        while n<21:
            #print(np.sum(outTransit))
            #Running the minimization 7 times with different initial params to make sure we get the best fit
            if np.sum(outTransit)>20:
                rand_choice=np.random.random(len(x))<0.95
            else:
                rand_choice=np.tile(True,len(x))
            #non-dip is simple poly fit. Including 10% cut in points to add some randomness over n samples
            #log10(height), log10(dur), tcen
            if modeltype=='sin' or modeltype=='both' or modeltype=='all':
                mod_args= np.hstack(([np.random.normal(0.0,0.5*params['tdur']),
                                      np.log(params['tdur'])+np.random.normal(0.0,0.5)],
                                      np.log(params['depth'])+np.random.normal(0.0,0.5),
                                      np.polyfit(x[outTransit&rand_choice],
                                                 y[outTransit&rand_choice],polyorder)
                                    ))

                mod_res_sin=optim.minimize(Sinusoid_neg_lnprob, mod_args,
                                       args=(x[np.argsort(x)],y[np.argsort(x)],yerr[np.argsort(x)],priors['sin'],polyorder),
                                       method=methods[n%3])
                mod_res_sin['llk']=log_likelihood_sinusoid(mod_res_sin['x'], x[np.argsort(x)],
                                                           y[np.argsort(x)],yerr[np.argsort(x)])
                mod_res_sin['bic']=np.log(len(x))*len(mod_res_sin['x']) - 2 * mod_res_sin['llk']

                #print('dip:',dip_args,dip_res,dip_bic)
                if mod_res_sin['bic']<best_mod_res['sin']['bic']:
                    best_mod_res['sin']=mod_res_sin

            if modeltype=='step' or modeltype=='both' or modeltype=='all':
                points_either_side=False
                #Making sure we start off with a position that has points both before and afer the step:
                step_guess=np.random.normal(0.0,0.5*params['tdur'])
                if np.sum(x<step_guess)==0:
                    step_guess=x[4]
                elif np.sum(x>step_guess)==0:
                    step_guess=x[-5]
                
                #Making one side randomly have polyorder 1, and the other 3->6
                side=np.random.random()<0.5
                npolys=[np.clip(polyorder+1-int(side)*20,1,6),np.clip(polyorder+1-int(side)*20,1,6)]
                mod_args= np.hstack((step_guess,
                                     np.polyfit(x[(x<step_guess)&rand_choice],
                                                y[(x<step_guess)&rand_choice],npolys[0]),
                                     np.polyfit(x[(x>=step_guess)&rand_choice],
                                                y[(x>=step_guess)&rand_choice],npolys[1])
                                    ))
                #print(x[np.argsort(x)], y[np.argsort(x)], yerr[np.argsort(x)],
                #      mod_args, dipmodel_step(mod_args,x[np.argsort(x)],npolys))
                mod_res_step=optim.minimize(Step_neg_lnprob, mod_args,
                                       args=(x[np.argsort(x)],y[np.argsort(x)],yerr[np.argsort(x)],priors['step'],
                                             np.clip(polyorder+1,1,5),npolys),
                                       method=methods[n%3])
                mod_res_step['llk']=log_likelihood_step(mod_res_step['x'],x[np.argsort(x)],y[np.argsort(x)],
                                                        yerr[np.argsort(x)],npolys)
                mod_res_step['bic']=np.log(len(x))*len(mod_res_step['x']) - 2 * mod_res_step['llk']
                #(2*mod_res_step.fun + np.log(len(x))*len(mod_res_step.x))

                #print('dip:',dip_args,dip_res,dip_bic)
                if mod_res_step['bic']<best_mod_res['step']['bic']:
                    best_mod_res['step']=mod_res_step
                    best_mod_res['step']['npolys']=npolys
            if modeltype=='poly' or modeltype=='all':
                priors['poly'] = 10.0 ** -np.arange(polyorder+1)[::-1]
                mod_res_poly=optim.minimize(Poly_neg_lnprob, np.polyfit(x,y,polyorder),
                                            args=(x,y,yerr,priors['poly'],polyorder),
                                            method=methods[n%3])
                mod_res_poly['llk']=log_likelihood_poly(mod_res_poly['x'],x,y,yerr)
                mod_res_poly['bic']=np.log(len(x))*len(mod_res_poly['x']) - 2 * mod_res_poly['llk']
                #=(2*mod_res_poly.fun + np.log(len(x))*len(mod_res_poly.x))
                #print('dip:',dip_args,dip_res,dip_bic)
                if mod_res_poly['bic']<best_mod_res['poly']['bic']:
                    best_mod_res['poly']=mod_res_poly
                    best_mod_res['poly']['npolys']=npolys
                
            #Increasing the polynomial order every odd number if we haven't yet found a good solution:
            if n>7 and n%2==1:
                polyorder+=1
                #Need to expand on priors to include new polynomial orders in this case:
            #print(n,"dip:",dip_res['fun'],dip_res['bic'],dip_args,"nodip:",nodip_res['fun'],nodip_res['bic'],"order:",order)
            #
            if n>=7 and np.all([best_mod_res[mod]['fun']<1e9 for mod in mods]):
                break
            n+=1

        best_mod_res['trans']={}
        best_mod_res['trans']['llk']= -0.5 * np.sum((y - y_trans)**2 / sigma2 + np.log(sigma2))
        best_mod_res['trans']['bic']= np.log(len(x))*6 - 2 * best_mod_res['trans']['llk']
    
        if 'sin' in best_mod_res and best_mod_res['sin']['bic']<1e9:
            best_mod_res['sin']['llk_ratio']=log_likelihood_sinusoid(best_mod_res['sin']['x'], x, y, yerr) - best_mod_res['trans']['llk']
            best_mod_res['sin']['DeltaBIC']=(8-len(best_mod_res['sin']['x'])) - (best_mod_res['trans']['llk'] - best_mod_res['sin']['llk'])
        elif 'sin' in best_mod_res:
            best_mod_res['sin']['DeltaBIC']=np.nan;best_mod_res['sin']['llk_ratio']=np.nan
        if 'step' in best_mod_res and best_mod_res['step']['bic']<1e9:
            best_mod_res['step']['llk_ratio']=log_likelihood_step(best_mod_res['step']['x'], x, y, yerr,best_mod_res['step']['npolys']) - best_mod_res['trans']['llk']
            best_mod_res['step']['DeltaBIC']=(8-len(best_mod_res['step']['x'])) - (best_mod_res['trans']['llk'] - best_mod_res['step']['llk'])
        elif 'step' in best_mod_res:
            best_mod_res['step']['DeltaBIC']=np.nan;best_mod_res['step']['llk_ratio']=np.nan
        if 'poly' in best_mod_res and best_mod_res['poly']['bic']<1e9:
            best_mod_res['poly']['llk_ratio']=log_likelihood_poly(best_mod_res['poly']['x'], x, y, yerr) - best_mod_res['trans']['llk']
            best_mod_res['poly']['DeltaBIC']=(8-len(best_mod_res['poly']['x'])) - (best_mod_res['trans']['llk'] - best_mod_res['poly']['llk'])
        elif 'poly' in best_mod_res:
            best_mod_res['poly']['DeltaBIC']=np.nan;best_mod_res['poly']['llk_ratio']=np.nan

    else:
        best_mod_res={}
    #print("Variability models:",best_mod_res)
    #print("sin:", best_mod_res['sin_llk'], "trans:", best_mod_res['trans_llk'], "llk_ratio:", best_mod_res['llk_ratio'])
    #print("plot:",plot,kwargs)
    if plot:
        #### PLOTTING ###
        if plot_loc is not None and type(plot_loc)!=str:
            ax = plot_loc
        else:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            if plot_loc is None:
                plot_loc=str(ID)+"_variability_check.pdf"
            elif plot_loc[-1]=='/':
                plot_loc=plot_loc+str(ID)+"_variability_check.pdf"
        markers, caps, bars = ax.errorbar(x,y,yerr=yerr,
                                          fmt='.k',ecolor='#AAAAAA',markersize=3.5,alpha=0.6,rasterized=True)
        [bar.set_alpha(0.2) for bar in bars]
        [cap.set_alpha(0.2) for cap in caps]

        ax.plot([-0.5*params['tdur'],-0.5*params['tdur']],[-2.0,2.0],':k',alpha=0.6,zorder=2,rasterized=True)
        ax.plot([0.0,0.0],[-2.0,2.0],'--k',linewidth=3,alpha=0.8,zorder=2,rasterized=True)
        ax.plot([0.5*params['tdur'],0.5*params['tdur']],[-2.0,2.0],':k',alpha=0.6,zorder=2,rasterized=True)
        if lc['flux_unit']==0.001:
            ax.set_ylabel("Relative flux [ppm]")
        elif lc['flux_unit']==1:
            ax.set_ylabel("Relative flux [ppm]")
        if 'sin' in best_mod_res and best_mod_res['sin']['fun']<1e30:
            ax.plot(x[np.argsort(x)],dipmodel_sinusoid(best_mod_res['sin']['x'],x[np.argsort(x)]),c='C3',alpha=0.5,
                     label='sinusoid',linewidth=2.25,zorder=10,rasterized=True)
        if 'step' in best_mod_res and best_mod_res['step']['fun']<1e30:
            ax.plot(x[np.argsort(x)],dipmodel_step(best_mod_res['step']['x'],x[np.argsort(x)],best_mod_res['step']['npolys']),
                    c='C4',alpha=0.5,label='step model',linewidth=2.25,zorder=10,rasterized=True)
        if 'poly' in best_mod_res and best_mod_res['poly']['fun']<1e30:
            ax.plot(x, np.polyval(best_mod_res['poly']['x'],x),
                    c='C2',alpha=0.5,label='polynomial model',linewidth=2.25,zorder=2,rasterized=True)
        ax.plot(x, y_trans, '-', c='C1', alpha=0.75, label='transit model', linewidth=2.25, zorder=3, rasterized=True)

        try:
            ax.set_ylim(np.nanmin(y),np.nanmax(y))
        except:
            b=0
        #plt.ylim(np.percentile(bg_lc[inTransit,1],[0.2,99.8]))
        ax.legend(prop={'size': 5})
        if len(best_mod_res.keys())>0 and np.all([best_mod_res[mod]['fun']<1e30 for mod in mods]):
            ax.set_title(str(ID)+" ' - "+["pass","fail"][int(np.any([best_mod_res[mod]['llk_ratio']>0 for mod in mods]))])
        else:
            ax.set_title(str(ID)+" Variability. Bad fits ???")
        if plot_loc is not None and type(plot_loc)==str:
            fig.savefig(plot_loc, dpi=400)
            print("Saved varble plot to",plot_loc)
            if return_fit_lcs:
                return best_mod_res, plot_loc, np.column_stack((x[np.argsort(x)],y,y_trans,
                                                          dipmodel_sinusoid(best_mod_res['sin']['x'],x[np.argsort(x)]),
                                  dipmodel_step(best_mod_res['step']['x'],x[np.argsort(x)],best_mod_res['step']['npolys']),
                                                          np.polyval(best_mod_res['poly']['x'],x)))
            else:
                return best_mod_res, plot_loc
        else:
            if return_fit_lcs:
                return best_mod_res, ax, np.column_stack((x[np.argsort(x)],y,y_trans,
                                                          dipmodel_sinusoid(best_mod_res['sin']['x'],x[np.argsort(x)]),
                                  dipmodel_step(best_mod_res['step']['x'],x[np.argsort(x)],best_mod_res['step']['npolys']),
                                                          np.polyval(best_mod_res['poly']['x'],x)))
            else:
                return best_mod_res, ax
    elif np.all([best_mod_res[mod]['fun']<1e30 for mod in mods]):
        
        return best_mod_res, None, None
    else:
        return None, None, None

def CheckInstrumentalNoise(lc,monodic,jd_base=None, **kwargs):
    '''# Using the processed "number of TCEs per cadence" array, we try to use this as a proxy for Instrumental noise in TESS
    # Here we simply use the detected SNR over the instrumental noise SNR as a proxy
    INPUTS:
    - lc
    - monotransit dic
    - jd_base (assumed to be that of TESS)'''
    import io
    import gzip
    f=gzip.open(MonoData_tablepath+'/tces_per_cadence.txt.gz','rb')
    tces_per_cadence=np.loadtxt(io.BytesIO(f.read()))
    if 'jd_base' in lc and jd_base is None:
        jd_base=lc['jd_base']
    elif jd_base is None:
        jd_base=2457000
    tces_per_cadence[:,0]-=(jd_base-2457000)
    #print(jd_base,tces_per_cadence[0,0],tces_per_cadence[-1,0], np.nanmin(lc['time']),np.nanmax(lc['time']))
    tces_per_cadence=tces_per_cadence[(tces_per_cadence[:,0]>np.nanmin(lc['time']))*(tces_per_cadence[:,0]<np.nanmax(lc['time']))]
    inst_snr=1+np.average(tces_per_cadence[abs(tces_per_cadence[:,0]-monodic['tcen'])<monodic['tdur'],1])
    return monodic['snr']/np.clip(inst_snr,1.0,1000)

def GapCull(t0,t,dat,std_thresh=10,boolean=None,time_jump_thresh=0.4):
    #Removes data before/after gaps and jumps in t & y
    #If there's a big gap or a big jump, we'll remove the far side of that
    if boolean is None:
        boolean=np.tile(True,len(t))
    if np.max(np.diff(t[boolean]))>time_jump_thresh:
        jump_n=np.argmax(np.diff(t[boolean]))
        jump_time=0.5*(t[boolean][jump_n]+t[boolean][jump_n+1])
        #print("TIME JUMP IN CENTROID AT",jump_time)
        if jump_time < t0:
            boolean*=(t>jump_time)
        elif jump_time > t0:
            boolean*=(t<jump_time)
    #data must be iterable list
    for arr in dat:
        noise=np.nanmedian(abs(np.diff(arr[boolean])))
        #5-sigma x centroid jump - need to cut
        if np.sum(boolean)>0 and np.nanmax(abs(np.diff(arr[boolean])))>std_thresh*noise:
            jump_n=np.argmax(np.diff(arr[boolean]))
            jump_time=0.5*(t[boolean][jump_n]+t[boolean][jump_n+1])
            #print("X JUMP IN CENTROID AT",jump_time)
            if jump_time < t0:
                boolean*=(t>jump_time)
            elif jump_time > t0:
                boolean*=(t<jump_time)
    return boolean

def CentroidCheck(lc,monoparams,interpmodel,ID,order=2,dur_region=3.5, plot=True,plot_loc=None, return_fit_lcs=False, **kwargs):
    # Checking lightcure for centroid shift during transit.
    # Performing two model fits 
    # - one with a "dip" correlated to the transit combined with a polynomial trend
    # - one with only a polynomial trend
    # These are then compared, and the BIC returned to judge 
    if 'cent_1' in lc:
        
        if monoparams['orbit_flag']=='mono':
            roundTransit=(abs(lc['time']-monoparams['tcen'])<monoparams['tdur']*dur_region)&lc['mask']
            
            t=lc['time'][roundTransit]-monoparams['tcen']
            
            roundTransit=GapCull(monoparams['tcen'],lc['time'],[lc['cent_1'],lc['cent_2']],boolean=roundTransit)
            
            t = lc['time'][roundTransit*np.isfinite(lc['cent_1'])*np.isfinite(lc['cent_2'])]-monoparams['tcen']
            x = lc['cent_1'][roundTransit*np.isfinite(lc['cent_1'])*np.isfinite(lc['cent_2'])]
            y = lc['cent_2'][roundTransit*np.isfinite(lc['cent_1'])*np.isfinite(lc['cent_2'])]
            
            outTransit=(abs(t)>monoparams['tdur']*0.65)
            inTransit=(abs(t)<monoparams['tdur']*0.35)

        elif monoparams['orbit_flag']=='periodic':
            #Checking for centroid in periodic array. 
            #This needs more care as we have to sum each transit without adding noise/trends from each
            phase=(lc['time']-monoparams['tcen']-0.5*monoparams['period'])%monoparams['period']-0.5*monoparams['period']
            dur_region=np.min([0.25*monoparams['period'],dur_region*monoparams['tdur']])
            roundTransit=(abs(phase)<monoparams['tdur']*dur_region)&lc['mask']
            #roundtransit now becomes "islands" around each transit in time space:
            jumps=np.hstack((0,np.where(np.diff(lc['time'][roundTransit])>monoparams['period']*0.25)[0]+1,len(lc['time'][roundTransit]) )).astype(int)
            ts=[]
            cent1s=[]
            cent2s=[]
            for nj in range(len(jumps)-1):
                #Iteratively cutting jumps for each transit
                cent1_loc=lc['cent_1'][roundTransit][jumps[nj]:jumps[nj+1]]
                cent2_loc=lc['cent_2'][roundTransit][jumps[nj]:jumps[nj+1]]
                t_loc=phase[roundTransit][jumps[nj]:jumps[nj+1]]
                newbool=GapCull(0.0,t_loc,[cent1_loc,cent2_loc])&np.isfinite(cent1_loc)&np.isfinite(cent2_loc)
                #Using non-removed regions to fit 2D polynomial and subtract from cent curves
                if np.sum(newbool)>0:
                    ts+=[t_loc[newbool]]
                    cent1s+=[cent1_loc[newbool] - \
                             np.polyval(np.polyfit(t_loc[newbool],cent1_loc[newbool],order),t_loc[newbool])]
                    cent2s+=[cent2_loc[newbool] - \
                             np.polyval(np.polyfit(t_loc[newbool],cent2_loc[newbool],order),t_loc[newbool])]
            t=np.hstack(ts)
            x=np.hstack(cent1s)
            y=np.hstack(cent2s)

            y=y[np.argsort(t)]
            x=x[np.argsort(t)]
            t=np.sort(t)
            
            outTransit=(abs(t)>monoparams['tdur']*0.65)
            inTransit=(abs(t)<monoparams['tdur']*0.35)

            #At the point all arrays should be flat, so we can make order==1
            order=0
        if len(x)>order+1 and len(y)>order+1:
            xerr=np.std(x)
            x-=np.median(x)
            if len(x[inTransit])>0 and len(x[outTransit])>0:
                #Calculating candidate shift. Setting as ratio to depth
                xdep_guess=(np.median(x[inTransit])-np.median(x[outTransit]))/monoparams['depth']
                init_poly_x=np.polyfit(t[outTransit],x[outTransit],order)
            else:
                xdep_guess=0.0
                if len(x[inTransit])>0:
                    init_poly_x=np.polyfit(t,x,np.clip(order-1,0,10))
                else:
                    init_poly_x=np.zeros(np.clip(order,1,11))
                
            y-=np.median(y)
            yerr=np.std(y)
            if len(y[inTransit])>0 and len(y[outTransit])>0:
                #Calculating candidate shift. Setting as ratio to depth
                ydep_guess=(np.median(y[inTransit])-np.median(y[outTransit]))/monoparams['depth']
                init_poly_y=np.polyfit(t[outTransit],y[outTransit],order)
            else:
                ydep_guess=0.0
                if len(y[inTransit])>0:
                    init_poly_y=np.polyfit(t,x,np.clip(order-1,0,10))
                else:
                    init_poly_y=np.zeros(np.clip(order,1,11))
        else:
            return None, None, None
        
        #Prior on centroid shifts such that values within 4sigma of 0.0 are enhanced.
        n_sig = [4*xerr/np.sqrt(np.sum(inTransit)),4*yerr/np.sqrt(np.sum(inTransit))]
        #priors= np.column_stack(([0,0],[n_sig[0]/monoparams['depth'],n_sig[1]/monoparams['depth']]))
        priors = np.column_stack(([0,0],[xdep_guess,ydep_guess]))
        poly_priors = 10.0 ** -np.arange(order+1)[::-1]

        best_nodip_res={'fun':1e30,'bic':1e6}
        best_dip_res={'fun':1e30,'bic':1e6}
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        
        for n in range(7):
            #Doing simple polynomial fits for non-dips. Excluding 10% of data each time to add some randomness
            rand_choice=np.random.choice(len(x),int(len(x)//1.06),replace=False)
            xfit=optim.minimize(Poly_neg_lnprob, np.polyfit(t[rand_choice],x[rand_choice],order),
                                        args=(t,x,xerr,poly_priors,
                                        order),method=methods[n%3])
            yfit=optim.minimize(Poly_neg_lnprob, np.polyfit(t[rand_choice],y[rand_choice],order),
                                        args=(t,y,yerr,poly_priors,
                                        order),method=methods[n%3])
            nodip_res={'fun':yfit.fun+xfit.fun}
            nodip_res['x']=[xfit.x,yfit.x]
            #nodip_res['bic']=2*nodip_res['fun'] + np.log(2*np.sum(roundTransit))*(len(xfit)+len(yfit))
            nodip_res['llk']=log_likelihood_poly(xfit.x, t,y,yerr)+log_likelihood_poly(yfit.x, t,y,yerr)
            nodip_res['bic']=np.log(len(x)+len(y))*(len(xfit.x)+len(xfit.x)) - 2 * nodip_res['llk']
            if nodip_res['bic']<best_nodip_res['bic']:
                best_nodip_res=nodip_res

            dip_args= np.hstack((np.random.normal(xdep_guess,abs(0.25*xdep_guess)),
                                 np.random.normal(ydep_guess,abs(0.25*ydep_guess)),
                                 init_poly_x,init_poly_y ))

            dip_res=optim.minimize(centroid_neg_lnprob, dip_args,
                                     args=(t,x,y,xerr,yerr,
                                           priors,
                                           interpmodel,
                                           order),method=methods[n%3])
            dip_res['llk']=log_likelihood_centroid(dip_res['x'], t, x, y, xerr, yerr, interpmodel, order)
            #-1*dip_res['fun']
            dip_res['bic']=np.log(len(x)+len(y))*len(dip_res['x']) - 2 * dip_res['llk']
            #2*dip_res.fun + np.log(2*np.sum(roundTransit))*len(dip_res['x'])
            if dip_res['bic']<best_dip_res['bic']:
                best_dip_res=dip_res
            
        #Computing difference in Bayesian Information Criterium - DeltaBIC -  between "dip" and "no dip models"
        centinfo={}
        centinfo['centroid_DeltaBIC']   = best_dip_res['bic'] - best_nodip_res['bic'] # dip is better < 0 < no dip is better
        centinfo['centroid_llk_ratio']  = best_dip_res['fun'] - best_nodip_res['fun']
        #print(best_dip_res)
        if 'x' in best_dip_res:
            centinfo['x_centroid'] = best_dip_res['x'][0]*monoparams['depth']
            centinfo['y_centroid'] = best_dip_res['x'][1]*monoparams['depth']
            centinfo['x_centroid_SNR'] = np.sqrt(np.sum(inTransit))*abs(centinfo['x_centroid']) / \
                                         np.nanmedian(1.06*abs(np.diff(x[outTransit])))
            centinfo['y_centroid_SNR'] = np.sqrt(np.sum(inTransit))*abs(centinfo['y_centroid']) / \
                                         np.nanmedian(1.06*abs(np.diff(y[outTransit])))
        
        #print("init_guesses:",xdep_guess,ydep_guess,"best_fits:",best_dip_res['x'][0],best_dip_res['x'][1])
        #print("with centroid:",best_dip_res,"| without:",best_nodip_res)
        if 'x' in best_dip_res and return_fit_lcs:
            arrs=np.column_stack((t,x,np.polyval(best_nodip_res['x'][0],t),
                                 dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[0],
                                 y,np.polyval(best_nodip_res['x'][1],t),
                                 dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[1]))
        else:
            arrs=None
        if plot: 
            if plot_loc is not None and type(plot_loc)!=str:
                ax = plot_loc
            else:
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(133)
                if plot_loc is None:
                    plot_loc = str(ID)+"_centroid_shift.pdf"
                elif plot_loc[-1]=='/':
                    plot_loc=plot_loc+str(ID)+"_centroid_shift.pdf"
            
            '''
            #### PLOTTING ###
            if monoparams['orbit_flag']=='periodic':
                ax.plot(phase,lc['cent_1']-np.nanmedian(lc['cent_1'][roundTransit]),',k',rasterized=True)
                ax.plot(phase,lc['cent_2']-np.nanmedian(lc['cent_2'][roundTransit]),',k',rasterized=True)

            elif monoparams['orbit_flag']=='mono':
                ax.plot(lc['time']-monoparams['tcen'],lc['cent_1']-np.nanmedian(lc['cent_1'][roundTransit]),',k',rasterized=True)
                ax.plot(lc['time']-monoparams['tcen'],lc['cent_2']-np.nanmedian(lc['cent_2'][roundTransit]),',k',rasterized=True)
            '''
            ax.scatter(t,y,s=1.5,rasterized=True)
            ax.scatter(t,x,s=1.5,rasterized=True)

            ax.plot([-0.5*monoparams['tdur'],-0.5*monoparams['tdur']],[-2.0,2.0],':k',alpha=0.6,rasterized=True)
            ax.plot([0.0,0.0],[-2.0,2.0],'--k',linewidth=3,alpha=0.8,rasterized=True)
            ax.plot([0.5*monoparams['tdur'],0.5*monoparams['tdur']],[-2.0,2.0],':k',alpha=0.6,rasterized=True)
            ax.set_ylabel("Relative centroid [px]")
            try:
                if best_dip_res['fun']<1e29 and best_nodip_res['fun']<1e29 and len(best_nodip_res['x'])==2:
                    ax.plot(t,np.polyval(best_nodip_res['x'][0],t),'--',c='C3',linewidth=2.25,alpha=0.6,
                            label='pure trend - x',rasterized=True)
                    ax.plot(t,np.polyval(best_nodip_res['x'][1],t),'--',c='C4',linewidth=2.25,alpha=0.6,
                            label='pure trend - y',rasterized=True)
                    ax.plot(t,dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[0],c='C3',
                            linewidth=2.25,alpha=0.6,label='trend+centroid - x',rasterized=True)
                    ax.plot(t,dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[1],c='C4',
                            linewidth=2.25,alpha=0.6,label='trend+centroid - y',rasterized=True)
                    ax.legend(prop={'size': 5})

                    ax.set_title(str(ID)+" Centroid  - "+["pass","fail"][int(centinfo['centroid_llk_ratio']<-6)])
                else:
                    ax.set_title(str(ID)+" Centroid  - No fit ???")
            except:
                ax.set_title(str(ID)+" Centroid  - No fit ???")

            xlim=np.percentile(x,[0.2,99.8])
            ylim=np.percentile(y,[0.2,99.8])
            ax.set_ylim(np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]]))
            ax.set_xlim(np.min(t),np.max(t))

            if plot_loc is not None and type(plot_loc)==str:
                fig.savefig(plot_loc, dpi=400)
                return centinfo, plot_loc, arrs
            else:
                return centinfo, ax, arrs
        elif not plot:
            return centinfo, None,arrs
    else:
        return None, None

def CheckPeriodConfusedPlanets(lc,all_dets,mono_mono=True,multi_multi=True,mono_multi=True):
    #Merges dic of mono detections with a dic of periodic planet detections
    #Performs 3 steps: 
    # - Checks monos against themselves
    # - Checks periodic planets (and duos detected in the periodic search) against themselves
    # - Checks periodic planets (and duos detected in the periodic search) against monotransits
    # In each case, the signal with the highest SNR is kept (and assumed to be the correct one)
    # The other signal is removed from the list, but kept in the detn dictionary
    #
    #INPUTS:
    # - lc dict
    # - detection dict
    #RETURNS:
    # - detection dict
    # - list of monos
    # - list of multis/duos
    
    mono_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag']=='mono')&(all_dets[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability'])]
    #print([all_dets[pl]['flag'] for pl in mono_detns])
    if len(mono_detns)>1 and mono_mono:
        #removing monos which are effectively the same. Does this through tcen/tdur matching.
        for monopl in mono_detns:
            if all_dets[monopl]['orbit_flag'][:2]!='FP':
                other_dets = np.array([[other,all_dets[other]['tcen'],all_dets[other]['tdur']] for other in mono_detns if other !=monopl])
                trans_prox = np.min(abs(all_dets[monopl]['tcen']-other_dets[:,1].astype(float))) #Proximity to a transit
                prox_stats = abs(trans_prox/(0.5*(all_dets[monopl]['tdur']+other_dets[:,2].astype(float))))
                print("Mono-mono compare", monopl, all_dets[monopl]['tcen'], other_dets[:,1], prox_stats)
                if np.min(prox_stats)<0.5:
                    other=other_dets[np.argmin(prox_stats),0]
                    if all_dets[other]['snr']>all_dets[monopl]['snr']:
                        all_dets[monopl]['orbit_flag']='FP - confusion with '+other
                    else:
                        all_dets[other]['orbit_flag']='FP - confusion with '+monopl
    mono_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag']=='mono')&(all_dets[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability'])]
                    
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]
    if len(perdc_detns)>1 and multi_multi:
        #removing periodics which are effectively the same. Does this through cadence correlation.
        for perpl in perdc_detns:
            new_trans_arr=((lc['time'][lc['mask']]-all_dets[perpl]['tcen']+0.5*all_dets[perpl]['tdur'])%all_dets[perpl]['period'])<all_dets[perpl]['tdur']

            for perpl2 in perdc_detns:
                if perpl!=perpl2 and all_dets[perpl]['orbit_flag'][:2]!='FP' and all_dets[perpl2]['orbit_flag'][:2]!='FP':
                    new_trans_arr2=((lc['time'][lc['mask']]-all_dets[perpl2]['tcen']+0.5*all_dets[perpl2]['tdur'])%all_dets[perpl2]['period'])<all_dets[perpl2]['tdur']
                    sum_overlap=np.sum(new_trans_arr&new_trans_arr2)
                    prox_arr=np.hypot(sum_overlap/np.sum(new_trans_arr),sum_overlap/np.sum(new_trans_arr2))
                    #print("Multi-multi compare",perpl,all_dets[perpl]['period'],perpl2,all_dets[perpl2]['period'],prox_arr)
                    if prox_arr>0.6:
                        #These overlap - taking the highest SNR signal
                        if all_dets[perpl]['snr']>all_dets[perpl2]['snr']:
                            all_dets[perpl2]['orbit_flag']='FP - confusion with '+perpl2
                        else:
                            all_dets[perpl]['orbit_flag']='FP - confusion with '+perpl
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]
    trans_arr=[]
    #print(mono_detns,perdc_detns,len(mono_detns)>0 and len(perdc_detns)>0)
    if len(mono_detns)>0 and len(perdc_detns)>0 and mono_multi:
        confused_mono=[]
        #Looping over periodic signals and checking if the array of transit times (and durations) matches
        for perpl in perdc_detns:
            new_trans_arr=((lc['time'][lc['mask']]-all_dets[perpl]['tcen']+0.5*all_dets[perpl]['tdur'])%all_dets[perpl]['period'])<all_dets[perpl]['tdur']
            #trans_arr=np.hstack((np.arange(all_dets[perpl]['tcen'],np.nanmin(lc['time'])-all_dets[perpl]['tdur'],-1*all_dets[perpl]['period'])[::-1],np.arange(all_dets[perpl]['tcen']+all_dets[perpl]['period'],np.nanmax(lc['time'])+all_dets[perpl]['tdur'],all_dets[perpl]['period'])))
            #print(perpl,trans_arr)
            for monopl in mono_detns:
                if all_dets[monopl]['orbit_flag'][:2]!='FP' and all_dets[perpl]['orbit_flag'][:2]!='FP':
                    roundtr=abs(lc['time'][lc['mask']]-all_dets[monopl]['tcen'])<(2.5*all_dets[monopl]['tdur'])
                    new_trans_arr2=abs(lc['time'][lc['mask']][roundtr]-all_dets[monopl]['tcen'])<(0.5*all_dets[monopl]['tdur'])
                    sum_overlap=np.sum(new_trans_arr[roundtr]&new_trans_arr2)
                    prox_stat=sum_overlap/np.hypot(np.sum(new_trans_arr[roundtr]),np.sum(new_trans_arr2))
                    #adding depth comparison - if depths are a factor of >3different we start reducing prox_stat by the log ratio
                    prox_stat/=np.clip(abs(np.log(all_dets[perpl]['depth']/all_dets[monopl]['depth'])),1.0,20)
                    '''
                    nearest_trans=trans_arr[np.argmin(abs(all_dets[monopl]['tcen']-trans_arr))] #Proximity to a transit
                    trans_perdic=abs(lc['time'][lc['mask']]-nearest_trans)<(0.5*all_dets[perpl]['tdur'])
                    trans_mono=abs(lc['time'][lc['mask']]-all_dets[monopl]['tcen'])<(0.5*all_dets[monopl]['tdur'])
                    prox_stat=np.hypot(np.sum(trans_perdic&trans_mono)/np.sum(trans_perdic),
                                       np.sum(trans_perdic&trans_mono)/np.sum(trans_mono))
                    '''
                    #print("Multi-mono compare",perpl,all_dets[perpl]['tdur'],"|",monopl,all_dets[monopl]['tcen'],all_dets[monopl]['tdur'],prox_stat)
                    if prox_stat>0.33:
                        #These overlap - taking the highest SNR signal
                        #print("correlation - ",all_dets[perpl]['snr'],all_dets[monopl]['snr'])
                        if all_dets[perpl]['snr']>=all_dets[monopl]['snr']:
                            all_dets[monopl]['orbit_flag']= 'FP - confusion with '+perpl
                        elif all_dets[perpl]['snr']<all_dets[monopl]['snr']:
                            all_dets[perpl]['orbit_flag']= 'FP - confusion with '+monopl

    mono_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag']=='mono')&(all_dets[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability','step'])]
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]
    return all_dets, mono_detns, perdc_detns
    
def CheckMonoPairs(lc_time, all_pls,prox_thresh=3.5, **kwargs):
    #Loop through each pair of monos without a good period, and check:
    # - if they correspond in terms of depth/duration
    # - and whether they could be periodic given other data
    all_monos=[pl for pl in all_pls if (all_pls[pl]['orbit_flag']=='mono')&(all_pls[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability','FP - confusion'])]
    all_others=[pl for pl in all_pls if (all_pls[pl]['orbit_flag'] in ['periodic', 'duo'])&(all_pls[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability','FP - confusion'])]
    if len(all_monos)>1:
        prox_arr=np.tile(1e9,(len(all_monos),len(all_monos)))
        ordered_monos=np.array(all_monos)[np.argsort(np.array([all_pls[mon]['snr'] for mon in all_monos]))[::-1]]
        #print(ordered_monos,np.array([all_pls[mon]['snr'] for mon in ordered_monos]))
        found=[]
        for n1,m1 in enumerate(ordered_monos):
            proxs=[]
            for n2,m2 in enumerate(ordered_monos[n1+1:]):
                if m1 not in found and m2 not in found:
                    # 1) How close are these monos in terms of depth & duration? (0.5 ~ 10% different here)
                    proxs+=[(np.log(all_pls[m1]['depth'])-np.log(all_pls[m2]['depth']))**2/0.25**2+\
                                      (np.log(all_pls[m1]['tdur'])-np.log(all_pls[m2]['tdur']))**2/0.2**2]
                    # 2) Can these monos even possibly produce a "duo" given the other phase coverage?
                    period = abs(all_pls[m1]['tcen']-all_pls[m2]['tcen'])
                    average_dur=np.average([all_pls[m1]['tdur'],all_pls[m2]['tdur']])
                    phase=(lc_time-all_pls[m1]['tcen']-period*0.5)%period-period*0.5
                    Npts_in_tr=np.sum(abs(phase)<0.3*average_dur)
                    Npts_from_known_transits=np.sum(abs(lc_time-all_pls[m1]['tcen'])<0.3*average_dur)+np.sum(abs(lc_time-all_pls[m2]['tcen'])<0.3*average_dur)
                    # Let's multiply the prox_arr from duration/depth with the square of the number of points in transit
                    # Here, if there's ~10% of a transit in the right phase, we get prox_arr==1.0
                    # Here, if there's ~25% of a transit in the right phase, we get prox_arr==6.0
                    proxs[-1]+=(20*(Npts_in_tr/Npts_from_known_transits-1))**2
                    proxs[-1]/=(all_pls[m2]['snr']/all_pls[m1]['snr'])**0.5 #Including SNR factor - higher SNR is favoured
                    #print(m1,m2,all_pls[m1]['depth'],all_pls[m2]['depth'],all_pls[m1]['tdur'],all_pls[m2]['tdur'],Npts_in_tr,Npts_from_known_transits,(all_pls[m2]['snr']/all_pls[m1]['snr'])**0.5,proxs[-1])
            #print("Mono pair searching",m1,proxs)

            #Taking the best-fitting lower-SNR detection which matches:
            proxs=np.array(proxs)
            if np.any(proxs<prox_thresh):
                n2=np.argmin(proxs)
                m2=ordered_monos[n1+1:][n2]
                newm1=deepcopy(all_pls[m1])
                #MATCH with threshold of 2
                for key in all_pls[m1]:
                    if key in all_pls[m2]:
                        if key=='period':
                            #print("tcens = ",all_pls[m1]['tcen'],all_pls[m2]['tcen'])
                            newm1['period']=abs(all_pls[m1]['tcen']-all_pls[m2]['tcen'])
                        elif key in ['snr','snr_r']:
                            newm1[key]=np.hypot(all_pls[m1][key],all_pls[m2][key])
                        elif type(all_pls[m2][key])==float and key!='tcen':
                            #Average of two:
                            #print(key,all_pls[m1][key],all_pls[m2][key],0.5*(all_pls[m1][key]+all_pls[m2][key]))
                            newm1[key]=0.5*(all_pls[m1][key]+all_pls[m2][key])
                newm1['tcen_2']=all_pls[m2]['tcen']
                newm1['orbit_flag']='duo'
                check_pers = newm1['period']/np.arange(1,np.ceil(newm1['period']/10),1.0)
                check_pers_ix=np.tile(False,len(check_pers))
                Npts_from_known_transits=np.sum(abs(lc_time-newm1['tcen'])<0.35*newm1['tdur']) + \
                                         np.sum(abs(lc_time-newm1['tcen_2'])<0.35*newm1['tdur'])
                #print("check pers duos",check_pers,Npts_from_known_transits)
                for nper,per in enumerate(check_pers):
                    phase=(lc_time-newm1['tcen']-per*0.5)%per-per*0.5
                    Npts_in_tr=np.sum(abs(phase)<0.35*newm1['tdur'])
                    check_pers_ix[nper]=Npts_in_tr<1.075*Npts_from_known_transits #Less than 15% of another eclipse is covered
                newm1['period_aliases']=check_pers[check_pers_ix]
                if len(newm1['period_aliases'])>1:
                    newm1['P_min'] = newm1['period_aliases'] if type(newm1['period_aliases'])==float else np.min(newm1['period_aliases'])
                elif len(newm1['period_aliases'])==1:
                    newm1['P_min'] = newm1['period_aliases'][0]
                else:
                    newm1['P_min'] = 999

                #print("period aliases:",newm1['period_aliases'],"P_min:",newm1['P_min'])
                all_pls[m1]=newm1
                all_pls[m2]['orbit_flag']='FP - Confusion with '+m1
                all_pls[m2]['flag']='FP - confusion'
                found+=[m1,m2]
    return all_pls
    
def EB_modelPriors(params,priors):
    lnprior=0.0
    for p in range(len(params)):
        if priors[p,1]=='Gaussian':
            lnprior+=log_gaussian(params[p],float(priors[p,2]),float(priors[p,3]))
        elif priors[p,1]=='Uniform':
            #Outside uniform priors, give extremely harsh restrictions that get worse with distance:
            if params[p]<(float(priors[p,2])-float(priors[p,3])):
                lnprior-=1e3*((float(priors[p,2])-params[p])/float(priors[p,3]))**2
            elif params[p]>(float(priors[p,2])+float(priors[p,3])):
                lnprior-=1e3*((params[p]-float(priors[p,2]))/float(priors[p,3]))**2
            else:
                lnprior-=0.0
    return lnprior

def EBmodel_lnprob(params, x, y, yerr, priors, Ms, tsec=False):
    # Trivial improper prior: uniform in the log.
    lnprior=EB_modelPriors(params,priors)
    llk = log_likelihood_EBmodel(params, x, y, yerr, Ms,tsec=tsec)
    if np.isnan(llk):
        llk=-1e25
    #print("llk:",llk,"prior:",lnprior,"minimise:",-1*(lnprior + llk))
    return lnprior + llk
'''
def log_likelihood_EBmodel(params, x, y, yerr):
    model=EBmodel(params,x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
'''
def log_likelihood_EBmodel(params, x, y, yerr, Ms,tsec=False):
    model=EBmodel(params,x, Ms,tsec=tsec)
    '''
    if abs(np.median(model))>1e100:
        r_1,log_r2_r1,log_sbratio,b,log_light_3,t_zero,log_period,log_sma,log_q,f_c,f_s,ldc_1 = params
        sma=np.exp(log_sma)
        r_1=np.clip(r_1/sma,1e-7,0.5)
        r_2=np.clip(r_1*np.exp(log_r2_r1)/sma,1e-7,0.5)
        print("r1",np.clip(r_1/sma,1e-7,1-1e-7),"|r2",np.clip(r_1*np.exp(log_r2_r1)/sma,1e-7,1-1e-7),
              "|sb",np.exp(log_sbratio),"|b",b,"|incl",np.arccos(abs(b)/sma)*(180/np.pi),"|light_3:",np.exp(log_light_3),
              "|t_zero",t_zero,"|period",np.exp(log_period),"|a",np.exp(log_sma),
              "|q",np.exp(np.clip(log_q,-20,1.0)),"|f_c",f_c,"|f_s",f_s,
              "|ldc_1",ldc_1)
    '''

    inv_sigma2 = 1.0/(yerr**2 + model**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def EBmodel_neg_lnprob(params, x, y, yerr, priors, Ms,tsec=False):
    return -1*EBmodel_lnprob(params, x, y, yerr, priors, Ms,tsec=tsec)

def EBmodel(params, t, Ms, tsec=False):
    if not tsec:
        #given omega directly as parameter:
        r_1,log_r2_r1,log_sbratio,b,log_light_3,t_zero,log_period,log_q,ecc,omega,ldc_1 = params
        incl=np.arccos(abs(b)/sma)
    else:
        #deriving omega from secondary position:
        r_1,log_r2_r1,log_sbratio,b,log_light_3,t_zero,log_period,log_q,ecc,t_sec,ldc_1 = params
        incl=np.arccos(abs(b)/sma)
        omega = np.arccos(np.pi*(((t_sec-t_zero)%period)/np.exp(log_period)-0.5)/(ecc*(1+np.sin(incl)**-1)))
    #Modified these parameters to be:
    # R_1 (in Rsun)
    # R_2/R_1,
    # log of sb ratio
    # b - impact parameter
    # log light 3
    # t_zero
    # log_period
    # sma
    # q - mass ratio
    # f_c - ecos omega
    # f_s - esin omega
    # ldc_1 - limb darkening
    #Using Kepler's laws to derive SMA
    sma=((6.67e-11*(1+np.exp(log_q))*Ms*1.96e30*(np.exp(log_period)*86400)**2)/(4*np.pi**2))**(1/3)/(6.955e8)
    r_1=np.clip(r_1/sma,1e-7,0.75)
    r_2=np.clip(r_1*np.exp(log_r2_r1),1e-7,0.75)
    
    ymodel=ellc.lc(t,r_1,r_2,
                   np.exp(log_sbratio),incl*(180/np.pi),
                   light_3=np.exp(log_light_3),
                   t_zero=t_zero,period=np.exp(log_period),a=sma,
                    f_c=np.sqrt(np.clip(ecc,0.0,1.0-0.5*(r_1+r_2)))*np.cos(omega),
                    f_s=np.sqrt(np.clip(ecc,0.0,1.0-0.5*(r_1+r_2)))*np.sin(omega),
                   q=np.exp(log_q),ldc_1=np.clip(ldc_1,0.0,1.0),ld_1="lin",verbose=0)
    return ymodel
    
def pri_sec_const(time,t_pri,dur_pri,t_sec=None,dur_sec=None):
    #Uses observed times, plus positions of primary and secondary eclipses 
    #  to estimate minimum period and minimum eccentricity of eclipsing binary
    # Requites:
    # - time
    # - t_pri - time of primary
    # - dur_pri - duration of primary (in days)
    # - t_sec - time of secondary
    # - dur_sec - duration of secondary (in days)
    
    dist_from_pri=np.sort(abs(time-t_pri))
    if t_sec is not None:
        dist_from_sec=np.sort(abs(time-t_sec))
    if np.max(np.diff(dist_from_pri))>(0.6*dur_pri):
        #Gaps in lightcurve - finding closest gap from primary
        if t_sec is not None:
            min_per=np.min([dist_from_pri[np.diff(dist_from_pri)>(0.5*dur_pri)][0],
                            dist_from_sec[np.diff(dist_from_sec)>(0.5*dur_pri)][0]])
            max_per=np.max([dist_from_pri[-1]+dur_pri,dist_from_sec[-1]+dur_sec])
            durstep=np.min([dur_pri,dur_sec])*0.25
            cut_time=time[(abs(time-t_pri)<0.5*dur_pri)&(abs(time-t_sec)<0.5*dur_sec)]
        else:
            min_per=dist_from_pri[np.diff(dist_from_pri)>(0.5*dur_pri)][0]
            max_per=dist_from_pri[-1]+dur_pri
            durstep=dur_pri*0.25
            cut_time=time[abs(time-t_pri)<0.5*dur_pri]
    else:
        #No gaps - can return simply the furthest timestamp from primary or secondary and compute min_ecc
        if t_sec is not None:
            min_per=np.max(np.hstack([dist_from_pri,dist_from_sec]))
            min_ecc=1/(2*np.pi)*(abs(t_pri-t_sec)/min_per - 0.5)
        else:
            min_per=np.max(dist_from_pri)
            min_ecc=0.0
        return min_per,min_ecc
    
    
    #Boolean array that will show if period is OK, given pri/sec and gaps, or if it fails:
    per_bool=[]
    psteps=np.arange(min_per-durstep,max_per+durstep,durstep)
    for pstep in psteps:
        phase=(cut_time-t_pri)%pstep
        pbool=(np.min(phase)>(dur_pri*0.4))&(np.max(phase)<(pstep-dur_pri*0.4))
        if pbool and t_sec is not None:
            phase_sec=(t_sec-t_pri)%pstep
            if np.min(abs(phase-phase_sec))<(dur_sec*0.4):
                pbool=False
        per_bool+=[pbool]
    
    #The minimum period is therefore the first per for which a period is ok:
    min_per = psteps[np.array(per_bool)][0]
    if t_sec is not None:
        pri_to_sec=(t_pri-t_sec)%min_per
        
        min_ecc=1/(2*np.pi)*(((t_pri-t_sec)%min_per)/min_per - 0.5)
    else:
        print("No good period here?")
    
def exoplanet_EB_model(lc, objects, Teffs, Rs, Ms, nrep=9,try_sec=False,use_ellc=False):
    with pm.Model() as model:
        return None


def xoEB(lc,planets):
    
    EBs=[pl for pl in planets if planets['pl']['flag']=='EB']
    if len(EBs)==1:
        eb=EBs[0]
    else:
        eb=EBs[np.argmax([planets[eb]['logror'] for eb in EBs])]
    with pm.Model() as model:

        # Systemic parameters
        mean_lc = pm.Normal("mean_lc", mu=0.0, sd=5.0)
        u1 = xo.QuadLimbDark("u1")
        u2 = xo.QuadLimbDark("u2")

        # Parameters describing the primary
        M1 = pm.Lognormal("M1", mu=Ms[0], sigma=abs(Ms[1]+Ms[2]))
        R1 = pm.Lognormal("R1", mu=Rs[0], sigma=abs(Rs[1]+Rs[2]))

        # Secondary ratios
        k = pm.Lognormal("k", mu=0.0, sigma=10.0, testval=np.exp(planets['pl']['logror']))  # radius ratio
        q = pm.Lognormal("q", mu=0.0, sigma=10.0)  # mass ratio
        s = pm.Lognormal("s", mu=np.log(0.5), sigma=10.0)  # surface brightness ratio

        # Prior on flux ratio
        pm.Beta("flux_prior",a=0.5, b=0.5,
            observed=k ** 2 * s)

        pm.Normal(
            "flux_prior",
            mu=lit_flux_ratio[0],
            sigma=lit_flux_ratio[1],
            observed=k ** 2 * s,
        )

        # Parameters describing the orbit
        b = xo.ImpactParameter("b", ror=k, testval=1.5)
        if planets[eb]['orbit_flag']=='mono':
            period = pm.Pareto("period", m=planets[eb]['minP'], alpha=1.0)
            newmask=lc['mask']&(abs(lc['time']-planets[eb]['tcen'])<planets[eb]['tdur']*2.5)
        else:
            period = pm.Lognormal("period", mu=np.log(planets[eb]['period']), sigma=0.1)
            newmask=lc['mask']&(abs((lc['time']-planets[eb]['tcen']-0.5*planets[eb]['period'])%planets[eb]['period'] - \
                                0.5*planets[eb]['period'])<planets[eb]['tdur']*2.5)
        #period = pm.Lognormal("period", mu=np.log(lit_period), sigma=1.0)
        t0 = pm.Normal("t0", mu=planets[eb]['tcen'], sigma=1.0)

        # Parameters describing the eccentricity: ecs = [e * cos(w), e * sin(w)]
        ecs = xo.UnitDisk("ecs", testval=np.array([1e-5, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sqrt(tt.sum(ecs ** 2)))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

        # Build the orbit
        R2 = pm.Deterministic("R2", k * R1)
        M2 = pm.Deterministic("M2", q * M1)
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            ecc=ecc,
            omega=omega,
            b=b,
            r_star=R1,
            m_star=M1,
            m_planet=M2,
        )

        # Track some other orbital elements
        pm.Deterministic("incl", orbit.incl)
        pm.Deterministic("a", orbit.a)

        # Noise model for the light curve
        sigma_lc = pm.InverseGamma(
            "sigma_lc", testval=1.0, **xo.estimate_inverse_gamma_parameters(0.1, 2.0)
        )
        S_tot_lc = pm.InverseGamma(
            "S_tot_lc", testval=2.5, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        ell_lc = pm.InverseGamma(
            "ell_lc", testval=2.0, **xo.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        kernel_lc = xo.gp.terms.SHOTerm(
            S_tot=S_tot_lc, w0=2 * np.pi / ell_lc, Q=1.0 / 3
        )

        # Set up the light curve model
        model_lc = xo.SecondaryEclipseLightCurve(u1, u2, s)

        def get_model_lc(t):
            return (
                mean_lc
                + 1e3 * model_lc.get_light_curve(orbit=orbit, r=R2, t=t, texp=texp)[:, 0]
            )

        # Condition the light curve model on the data
        gp_lc = xo.gp.GP(
            kernel_lc, lc['time'][newmask], lc['flux_err'][newmask] ** 2 + sigma_lc ** 2, mean=get_model_lc
        )
        gp_lc.marginal("obs_lc", observed=lc['flux'][newmask])

        # Optimize the logp
        map_soln = model.test_point

        # Then the LC parameters
        map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b])
        map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b, u1, u2])
        map_soln = xo.optimize(map_soln, [mean_lc, sigma_lc, S_tot_lc, ell_lc, q])
        map_soln = xo.optimize(map_soln, [t0, period])

        # Then all the parameters together
        map_soln = xo.optimize(map_soln)

        model.gp_lc = gp_lc
        model.get_model_lc = get_model_lc

        model.x = lc['time'][newmask]
        model.y = lc['flux'][newmask]

    return model, map_soln
    
    
def minimize_EBmodel(lc, objects, Teffs, Rs, Ms, nrep=9,try_sec=False,use_ellc=False):
    #Running a quick EB model:
    
    multis=[key for key in objects if objects[key]['orbit_flag'] in ['periodic','duo']]
    if len(objects)>1:
        SNRs=np.array([objects[key]['snr'] for key in objects])
        planet=objects[list(objects.keys())[np.argmax(SNRs)]]
    
    
    #secmodel is None
    bestfit,interp=QuickMonoFit(lc,planet['tcen'],planet['tdur'],Rs=Rs[0],Ms=Ms[0],useL2=True)
    '''
    #check periodic secondary at different phase - e.g. sec
    if len(multis)==2:
        per_objs = np.array([[objects[key]['period'],objects[key]['tcen']] for key in multis])
        if (abs(per_objs[0,0]-per_objs[1,0])/0.02)<1:
            epch_diff=(abs(per_objs[0,1]-per_objs[1,1])%per_objs[0,0])/per_objs[0,0]
            if epch_diff>0.08 and epch_diff<0.92:
                #Secondary found with same period at different epoch - 
                secmodel=objects[list(objects.keys())[np.argsort(SNRs)==1]]
                secmodel['min_p']=per_objs[0,0]*0.95
    elif len(multis)==1 and len(objects)==2:
        bestfit,interp=QuickMonoFit(lc,planet['tcen'],planet['tdur'],Rs=Rs[0],Ms=Ms[0],useL2=True)
        #secmodel=SearchForSecondary(lc, interpmodel, bestfit)
        #Secondary position can constrain our eccentricity & omega position considerations

        #If primary-to-secondary distance 
        e_min = abs(np.pi*0.5*(((secmodel['tcen']-bestfit['tcen'])%secmodel['min_p'])/secmodel['min_p']-0.5))
    '''
    
    if not use_ellc:
        return bestfit,None
    else:
        import ellc
        # Getting minimum period:
        per_arr=np.sort(abs(lc['time']-bestfit['tcen']))
        per_arr_jumps=np.where(np.diff(per_arr)>(0.75*bestfit['tdur']))[0]
        if planet['orbit_flag'] in ['periodic', 'duo']:
            min_per= planet['period']*0.95
            init_per=planet['period']
        else:
            min_per= np.max(per_arr) if len(per_arr_jumps)==0 else per_arr[np.min(per_arr_jumps)]
            init_per=bestfit['period']
        #Calculating minimum SMA in Rsun given minimum period (scaled by 0.5 for eccentricity), Mass (scaled by 2 for q) & Radius
        best_res={'fun':np.inf};
        all_models=[]
        init_params_0=np.array([Rs[0],bestfit['logror'],-1.0,0.25,np.log(bestfit['third_light']),bestfit['tcen'],
                               np.log(init_per),-2,0.0,0.0,0.5])
        #priors are Name, [Gaussian OR Uniform], [mu, 2*sd] OR [lower bound, width]
        init_uniform_priors=np.vstack([['R1','Gaussian',Rs[0],(abs(Rs[1])+abs(Rs[2]))],
                                       ['log_R2_R1','Uniform',-1.5,1.5],
                                       ['log_sbratio','Uniform',-2,3.25],
                                       ['b','Uniform',1.5,1.5],
                                       ['log_light_3','Uniform',-3.5,3.5],
                                       ['t_zero','Gaussian',bestfit['tcen'], 0.33*bestfit['tdur']],
                                       ['log_period','Uniform',np.log(min_per)+3.5,3.5],
                                       ['log_q','Uniform',-2,2],
                                       ['ecc','Uniform',0.5,0.5],
                                       ['omega','Uniform',0.0,np.pi],
                                       ['ldc_1','Uniform',0.5,0.5]])
        
        if planet['orbit_flag'] in ['periodic', 'duo']:
            init_uniform_priors[6]=['log_period','Uniform',np.log(min_per)+0.1,0.1]
            
        if secmodel is not None:
            init_uniform_priors[9]=['t_sec','Gaussian',secmodel['tcen'],0.33*secmodel['tdur']]
            use_t_sec=True
        else:
            use_t_sec=False
        if try_sec:
            init_uniform_priors[0,2]=Rs[0]*0.5#Making radius R_2
            init_uniform_priors[0,3]=Rs[0]*0.5
            init_params_0[0]=Rs[0]*np.exp(bestfit['logror'])#Inverting log_ror, log_sbratio and log_q
            init_uniform_priors[1,2]=float(init_uniform_priors[1,2])*-1; init_params_0[1]*=-1
            init_uniform_priors[2,2]=float(init_uniform_priors[2,2])*-1; init_params_0[2]*=-1
            init_uniform_priors[7,2]=float(init_uniform_priors[7,2])*-1; init_params_0[7]*=-1
        newmask=lc['mask']&(abs(lc['time']-bestfit['tcen'])<5)
        newmask[(abs(lc['time']-bestfit['tcen'])<5)]*=CutAnomDiff(lc['flux_flat'][abs(lc['time']-bestfit['tcen'])<5])
        
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        #Doing N initial minimizatio using random parameters near initialised ones:
        for n in range(nrep):
            #Initialising priors and parameters for EB model:
            #radius_1 = R1/sma,radius_2 = R2/sma,log_sbratio = 0.25,
            #incl=90pm20,light_3=,t_zero,period,a,q,f_c,f_s,ldc_1
            #Taking initial parameters as uniform between priors:
            init_params=init_params_0+\
                        np.random.normal(np.tile(0.0,len(init_params_0)),0.15*init_uniform_priors[:,3].astype(float))
            #print("depth:",1.0-np.min(EBmodel(init_params,lc['time'],Ms[0])),"@",lc['time'][np.argmin(EBmodel(init_params,lc['time'],Ms[0]))])
            #print("init_q:",np.exp(init_params[7]))
            res=optimize.minimize(EBmodel_neg_lnprob,init_params,args=(lc['time'][newmask],
                                                                       1.0+lc['flux_flat'][newmask]-np.nanmedian(lc['flux_flat'][newmask]),
                                                                       lc['flux_err'][newmask],
                                                                       init_uniform_priors,Ms[0],use_t_sec),
                                                                       method=methods[nrep%3])
            all_models+=[EBmodel(res['x'],lc['time'],Ms[0])]
            #print("OUT:",res['fun'],res['x'])
            if res['fun']<best_res['fun']:
                best_res=res
        '''
        #Doing N more initial minimizations using random parameters near the best-fit ones:
        for n in range(nrep):
            init_params=np.random.normal(best_res['x'],0.15*init_uniform_priors[:,3].astype(float))
            res=optimize.minimize(EBmodel_neg_lnprob,init_params,
                                  args=(lc['time'][newmask],lc['flux_flat'][newmask]-np.nanmedian(lc['flux_flat'][newmask]),
                                        1.0+lc['flux_err'][newmask],init_uniform_priors,Ms[0]),method=methos[nrep%3])
            if res['fun']<best_res['fun']:
                best_res=res
        '''
        #Organising parameters to become best-fit:
        if use_t_sec:
            #deriving omega used from t_sec, t_pri and e:
            r_1,r2_r1,log_sbratio,b,log_light_3,t_zero,log_period,log_q,ecc,t_sec,ldc_1 = best_res.x
            omega=np.arccos(np.pi*(((t_sec-t_zero)%period)/np.exp(period)-0.5)/(ecc*(1+np.sqrt(1-(b/sma)**2)**-1)))
        else:
            #deriving t_sec in model from t_pri, omega and e:
            r_1,r2_r1,log_sbratio,b,log_light_3,t_zero,log_period,log_q,ecc,omega,ldc_1 = best_res.x
            t_sec = t_zero + np.exp(log_period)*((ecc*(1+np.sqrt(1-(b/sma)**2)**-1))*np.cos(omega)/np.pi + 0.5)
        sma=((6.67e-11*(1+np.exp(log_q))*Ms[0]*1.96e30*(np.exp(log_period)*86400)**2)/(4*np.pi**2))**(1/3),
        newres={"final_EBmodel":EBmodel(best_res['x'],lc['time'],Ms[0]),
                "final_pars":best_res.x,"final_lnprior":EB_modelPriors(best_res.x,init_uniform_priors),
                "final_lnprob":log_likelihood_EBmodel(best_res.x, lc['time'][lc['mask']],
                                                      lc['flux_flat'][lc['mask']]-np.nanmedian(lc['flux_flat'][lc['mask']]),
                                                      lc['flux_err'][lc['mask']],Ms[0]),
                "R_1":r_1,"R_2":r_1*r2_r1,"R2_R1":np.exp(r2_r1),
                "sbratio":np.exp(log_sbratio),"light_3":np.exp(log_light_3),"ldc_1":ldc_1,
                "sma":sma,"sma_R1":sma/(r_1*6.955e8),
                "ecc":ecc,"omega":omega,"t_sec":t_sec,
                "b":b,"incl":np.arccos(b/sma)*(180/np.pi),"tcen":t_zero,"period":np.exp(log_period),
                "M_1":Ms[0],"M_2":np.exp(log_q)*Ms[0],"q":np.exp(log_q),
                "T_1":Teffs[0],"T_2":Teffs[0]*np.exp(log_sbratio)**(1/4),
                "init_EBmodel":EBmodel(init_params_0,lc['time'],Ms[0]),'init_pars':init_params_0}
        newres['uniform_priors']=init_uniform_priors;newres['init_lnprior']=EB_modelPriors(init_params_0,init_uniform_priors)
        newres['init_prob']=log_likelihood_EBmodel(init_params_0, lc['time'][lc['mask']],
                                                  lc['flux_flat'][lc['mask']]-np.nanmedian(lc['flux_flat'][lc['mask']]),
                                                  lc['flux_err'][lc['mask']],Ms[0])
        return newres,all_models
    
def CutAnomDiff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    #Adding a test for the first and last points if they are >3*thresh from median RMS wrt next two points.
    anoms=np.hstack((abs(flux[0]-np.median(flux[1:3]))<(np.median(abs(diffarr[0,:]))*thresh*5),
                     ((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),
                     abs(flux[-1]-np.median(flux[-3:-1]))<(np.median(abs(diffarr[0,:]))*thresh*5)))
    return anoms


def get_interpmodels(Rs,Ms,Teff,lc_time,lc_flux_unit,mission='tess',n_durs=3,gap_thresh=2.0,texp=None):
    #Uses radius, mass and lightcurve duration to create fake transit models to use in monotransit search

    if texp is None:
        texp=np.nanmedian(np.diff(lc_time))
    
    u_star = tools.getLDs(Teff,logg=np.log10(Ms/Rs**2)+4.431,FeH=0.0)[0]
    
    #Computing monotransit minimum P from which to estimate durations:
    cadence=np.nanmedian(np.diff(lc_time))
    jumps=np.hstack((0,np.where(np.diff(lc_time)>gap_thresh)[0],len(lc_time)-1))
    
    #Using the maximum uninterupted patch of lightcurve as the period guess:
    P_guess=np.clip(np.max(lc_time[jumps[1:]]-lc_time[jumps[:-1]]),5.0,250)
    
    #print(jumps,jumps[np.argmax(np.diff(jumps))],jumps[1+np.argmax(np.diff(jumps))],P_guess)

    # Orbit models - for every n_dur over 4, we add longer durations to check:
    per_steps=np.logspace(np.log10(0.4)-0.03*np.clip(n_durs-9,0.0,7.0),np.log10(2.5+0.33*np.clip(n_durs-4,0.0,2.0)),n_durs)
    b_steps=np.linspace(0.88,0,n_durs)
    orbits = xo.orbits.KeplerianOrbit(r_star=Rs,m_star=Ms,period=P_guess*per_steps,t0=np.tile(0.0,n_durs),b=b_steps)
    
    vx, vy, vz = orbits.get_relative_velocity(0.0)
    tdurs=((2*1.1*np.clip(Rs,0.1,10)*np.sqrt(1-b_steps**2))/tt.sqrt(vx**2 + vy**2)).eval().ravel()

    # Compute the model light curve using starry
    interpt=np.linspace(-0.6*np.max(tdurs),0.6*np.max(tdurs),600).astype(np.float64)
    
    ys=xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbits, r=np.tile(0.1*np.clip(Rs,0.1,10),n_durs), 
                                                     t=interpt, texp=texp
                                                     ).eval()/lc_flux_unit
    interpmodels=[]
    for row in range(n_durs):
        interpmodels+=[interp.interp1d(interpt.astype(float).ravel(),ys[:,row].astype(float).ravel(),
                                            bounds_error=False,fill_value=(0.0,0.0),kind = 'cubic')]

    return interpmodels,tdurs


def VetCand(pl_dic,pl,ID,lc,mission,Rs=1.0,Ms=1.0,Teff=5800,
            mono_SNR_thresh=6.5,mono_SNR_r_thresh=5,variable_llk_thresh=5,
            plot=False,file_loc=None,vet_do_fit=True,return_fit_lcs=False,do_cent=True,**kwargs):
    #Best-fit model params for the mono transit:
    if pl_dic['orbit_flag']=='mono' and vet_do_fit:
        #Making sure our lightcurve mask isn't artificially excluding in-transit points:
        in_trans=abs(lc['time']-pl_dic['tcen'])<(0.6*pl_dic['tdur'])
        lc['mask'][in_trans]=np.isfinite(lc['flux'][in_trans])&np.isfinite(lc['flux_err'][in_trans])
        
        monoparams = QuickMonoFit(deepcopy(lc),pl_dic['tcen'],np.clip(pl_dic['tdur'],0.1,2.5),
                                               Rs=Rs,Ms=Ms,Teff=Teff,how='mono',**kwargs)
        if monoparams['tdur']>2:
            #Running with solar values:
            monoparams2 = QuickMonoFit(deepcopy(lc),pl_dic['tcen'],np.clip(pl_dic['tdur'],0.1,2),
                                                   Rs=1.0,Ms=1.0,Teff=Teff,how='mono',**kwargs)
            #print("Star-param-free model llk:",monoparams2['log_lik_mono'],"dur:",monoparams2['tdur'],
            #      "versus:",monoparams['log_lik_mono'],"dur:",monoparams2['tdur'])
            if monoparams2['log_lik_mono']>(monoparams['log_lik_mono']+5):
                monoparams=monoparams2

        #if not bool(monoparams['model_success']):
        #    #Redoing without fitting the polynomial if the fit fails:
        #    monoparams = QuickMonoFit(deepcopy(lc),pl_dic['tcen'],pl_dic['tdur'],
        #                                           Rs=Rs,Ms=Ms,Teff=Teff,how='mono',fit_poly=False)

        #Keeping detection tcen/tdur:
        monoparams['init_tdur']=pl_dic['tdur']
        if 'depth' in pl_dic:
            monoparams['init_depth']=pl_dic['depth']
        monoparams['orbit_flag']='mono'
        
    elif pl_dic['orbit_flag']=='periodic' and vet_do_fit:
        #Making sure our lightcurve mask isn't artificially excluding in-transit points:
        in_trans=abs((lc['time']-pl_dic['tcen']-0.5*pl_dic['period'])%pl_dic['period']-0.5*pl_dic['period'])<(0.6*pl_dic['tdur'])
        lc['mask'][in_trans]=np.isfinite(lc['flux'][in_trans])

        monoparams = QuickMonoFit(deepcopy(lc),pl_dic['tcen'],
                                      pl_dic['tdur'], init_period=pl_dic['period'],how='periodic',
                                      Teff=Teff,Rs=Rs,Ms=Ms)
    if pl_dic['orbit_flag']=='periodic' and (((pl_dic['period_mono']/pl_dic['period'])<0.1)|((pl_dic['period_mono']/pl_dic['period'])>10)):
            #Density discrepancy of 10x, likely not possible on that period
            pl_dic['flag']='discrepant duration'
    
    if plot:
        if pl_dic['orbit_flag']=='mono':
            vetfig=plt.figure(figsize=(11.69,3.25))
            var_ax=vetfig.add_subplot(131)
            ast_ax=vetfig.add_subplot(132)
            cent_ax=vetfig.add_subplot(133)
        elif pl_dic['orbit_flag']=='periodic':
            vetfig=plt.figure(figsize=(8.2,3.25))
            var_ax=vetfig.add_subplot(121)
            cent_ax=vetfig.add_subplot(122)
    else:
        var_ax=None
        ast_ax=None
        cent_ax=None
    #update dic:
    if vet_do_fit:
        pl_dic.update(monoparams)
    pl_dic['flag']='planet'
    #Compares log likelihood of variability fit to that for transit model
    #print("doing variability check")
    if pl_dic['orbit_flag']=='mono':
        #In the Mono case, we will fit both a sin and a step model:
        outs=VariabilityCheck(deepcopy(lc), pl_dic, plot=plot,modeltype='all',return_fit_lcs=return_fit_lcs,
                                        ID=str(ID).zfill(11)+'_'+pl, plot_loc=var_ax, **kwargs)
    else:
        #Only doing the sinusoidal model in the periodic case
        outs=VariabilityCheck(deepcopy(lc), pl_dic, plot=plot,modeltype='sin',return_fit_lcs=return_fit_lcs,
                                        ID=str(ID).zfill(11)+'_'+pl, plot_loc=var_ax, **kwargs)
    varfits=outs[0]
    varfig=outs[1]
    
    if return_fit_lcs:
        print(return_fit_lcs,outs[2])
        pl_dic["variability_vetting_models"]=outs[2]
    for key1 in varfits:
        for key2 in varfits[key1]:
            #print(key1+"_"+key2,varfits[key1][key2],type(varfits[key1][key2]))
            if (type(varfits[key1][key2])==float or type(varfits[key1][key2])==np.float64) and key1+"_"+key2 not in pl_dic:
                pl_dic[key1+"_"+key2]=varfits[key1][key2]

    if pl_dic['snr']<mono_SNR_thresh or pl_dic['snr_r']<(mono_SNR_r_thresh):
        pl_dic['flag']='lowSNR'
    elif 'step_llk_ratio' in pl_dic and pl_dic['step_llk_ratio']>variable_llk_thresh:
        pl_dic['flag']='step'
        print(pl,"step. LogLik=",pl_dic['step_llk_ratio'])
    elif pl_dic['sin_llk_ratio']>variable_llk_thresh:
        #print("Vetted after variability:",pl_dic['snr'],pl_dic['snr_r'],
        #      pl_dic['depth'],pl_dic['stepLogLik'],variable_llk_thresh,varfits['sin']['llk_ratio'])
        #print(pl_dic['flag']) if 'flag' in pl_dic else ''
        #>1 means variability fits the data ~2.7 times better than a transit.
        pl_dic['flag']='variability'
        print(pl,"variability. LogLik=",pl_dic['sin_llk_ratio'])
        
    if pl_dic['orbit_flag']=='mono' and mission.lower()!='k2':
        #Checks to see if dip is due to background asteroid and if that's a better fit than the transit model:
        outs=AsteroidCheck(deepcopy(lc), pl_dic, plot=plot,return_fit_lcs=return_fit_lcs,
                           ID=str(ID).zfill(11)+'_'+pl, plot_loc=ast_ax, **kwargs)
        if outs[0] is not None:
            for col in outs[0]:
                if 'steroid' in col and type(outs[0][col]) in [str,int,float,np.float64]:
                    pl_dic[col] = outs[0][col]
        else:
            pl_dic['asteroid_DeltaBIC']=0.0
            pl_dic['asteroid_snrw']=0.0
            pl_dic['asteroid_snrr']=0.0
            
        astfig=outs[1]
        if return_fit_lcs:
            print(return_fit_lcs,outs[2])
            pl_dic["asteroid_model"]=outs[2]
        if pl_dic['asteroid_DeltaBIC'] is not None and pl_dic['asteroid_DeltaBIC']<-10 and pl_dic['asteroid_snrr']>pl_dic['snr']:
            pl_dic['flag']='asteroid'
            print(pl,"asteroid. DeltaBic=",pl_dic['asteroid_DeltaBIC'])
    if do_cent:
        #Checks to see if dip is combined with centroid
        outs = CentroidCheck(deepcopy(lc), pl_dic, pl_dic['interpmodel'], plot=plot,
                                                            return_fit_lcs=return_fit_lcs,ID=str(ID).zfill(11)+'_'+pl,
                                                            plot_loc=cent_ax, **kwargs)

        if outs is None or outs[0] is None:
            pl_dic['centroid_llk_ratio']=None
        else:
            for col in outs[0]:
                pl_dic[col]=outs[0][col]
            centfig=outs[1]
            if return_fit_lcs:
                print(return_fit_lcs,outs[2])
                pl_dic['centroid_models']=outs[2]
        if pl_dic['centroid_llk_ratio'] is not None and pl_dic['centroid_llk_ratio']<-6:
            pl_dic['flag']='EB'
            print(pl,"EB - centroid. log lik ratio=",pl_dic['centroid_llk_ratio'])

    pl_dic['instrumental_snr_ratio']=CheckInstrumentalNoise(deepcopy(lc),pl_dic)
    if pl_dic['snr']>mono_SNR_thresh and pl_dic['instrumental_snr_ratio']<(mono_SNR_thresh*0.66):
        pl_dic['flag']='instrumental'
        print(pl,"planet SNR / instrumental SNR =",pl_dic['instrumental_snr_ratio'])

    '''if pl_dic['flag'] in ['asteroid','EB','instrumental']:
        monos.remove(pl)'''
    if pl_dic['orbit_flag']=='mono':
        print(str(pl)+" - Checks complete.",
              " SNR:",str(pl_dic['snr'])[:8],
              " SNR_r:",str(pl_dic['snr_r'])[:8],
              " variability:",str(pl_dic['sin_llk_ratio'])[:8],
              " centroid:",str(pl_dic['centroid_llk_ratio'])[:8],
              "| flag:",pl_dic['flag'])
    elif pl_dic['orbit_flag']=='periodic':
        print(pl,"Checks complete.",
              " SNR:",str(pl_dic['snr'])[:8],
              " SNR_r:",str(pl_dic['snr_r'])[:8],
              " variability:",str(pl_dic['sin_llk_ratio'])[:8],
              " centroid:",str(pl_dic['centroid_llk_ratio'])[:8],
              "| flag:",pl_dic['flag'])

    if 'flag' not in pl_dic:
        pl_dic['flag']='planet'
    if plot:
        #Attaching all our subplots and savings
        #vetfig.tight_layout()
        vetfig.subplots_adjust(left = 0.05,right = 0.97,bottom = 0.075,top = 0.925)
        if file_loc is not None:
            vetfig.savefig(file_loc+"/"+tools.id_dic[mission]+str(int(ID)).zfill(11)+'_'+pl+'_vetting.pdf', dpi=400)
            return pl_dic, file_loc+"/"+tools.id_dic[mission]+str(int(ID)).zfill(11)+'_'+pl+'_vetting.pdf'
        else:
            return pl_dic, None
    else:
        return pl_dic, None

def MonoVetting(ID, mission, tcen=None, tdur=None, overwrite=None, do_search=True, do_fit=False, coords=None, lc=None,
                useL2=False,PL_ror_thresh=0.2,variable_llk_thresh=5,file_loc=None, plot=True, **kwargs):
    '''#Here we're going to initialise the Monotransit fitting by searching for other planets/transits/secondaries and filtering out binaries.
    INPUTS:
    - ID
    - mission
    - useL2=False
    - PL_ror_thresh=0.2
    
    **kwargs:
     - StarPars=None, a list of radius, rho, teff and logg which are each lists of values/errors
     - multi_SNR_thresh=6.25,
     - plot=True
     - file_loc - a location to store images/files
     - overwrite=False
     For MonoTransitSearch:
     - mono_BIC_thresh=-10
     - mono_SNR_thresh
     - n_oversamp=75,
     - use_binned=True
     - use_flat=True
     - use_poly=True
     - binsize=1/96.0,
     - Rs=Rstar[0]
     - Ms=rhostar[0]*Rstar[0]**3
     - Teff=Teff[0],
     - plot=plot
     - plot_loc=file_loc+"/"
     - poly_order=4
     '''
    if overwrite is None:
        overwrites={'starpars':False,'lc':False,'monos':False,'multis':False,'vet':False,'fit':False, 'model_plots':False}
    elif overwrite=='all':
        overwrites={'starpars':True,'lc':True,'monos':True,'multis':True,'vet':True,'fit':True, 'model_plots':True}
    else:
        overwrites={}
        for step in ['starpars','lc','monos','multis','vet','fit','model_plots']:
            overwrites[step]=step in overwrite
        if overwrites['starpars']:
            overwrites['fit']=True
        if overwrites['lc']:
            overwrites['monos']=True
            overwrites['multis']=True
        if overwrites['monos'] or overwrites['multis']:
            overwrites['vet']=True
            overwrites['fit']=True
        if overwrites['fit']:
            overwrites['model_plots']=True
        print(overwrite,overwrites)
        
    ID_string=tools.id_dic[mission]+str(ID).zfill(11)
    kwargs['file_loc']=MonoData_savepath+'/'+ID_string if file_loc is not None else file_loc
    
    if 'mono_SNR_thresh' not in kwargs:
        kwargs['mono_SNR_thresh']=6.5
    if 'mono_SNR_r_thresh' not in kwargs:
        kwargs['mono_SNR_r_thresh']=4.75
    
    if not os.path.isdir(MonoData_savepath+'/'+ID_string):
        os.system('mkdir '+MonoData_savepath+'/'+ID_string)
        '''#Special set-up for Theano to compile directly in this directory
        if 'compiledir' not in os.environ["THEANO_FLAGS"]:
            if not os.path.exists(os.path.join(file_loc,".theano_compile_dir")):
                os.mkdir(os.path.join(file_loc,".theano_compile_dir"))
            os.environ["THEANO_FLAGS"]+="compiledir="+os.path.join(file_loc,".theano_compile_dir")
            import theano.tensor as tt
            import pymc3 as pm'''
    
    #Initialising figures
    if plot:
        try:
            import seaborn as sns
            sns.set_style("darkgrid")
        except:
            print("Seaborn not loaded")
        figs={}

    if 'StarPars' not in kwargs or overwrites['starpars']:
        
        radec=SkyCoord(float(coords.split(',')[0])*u.deg,float(coords.split(',')[1])*u.deg) if coords is not None else None
        #loading Rstar,Tess, logg and rho from csvs:
        if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_starpars.csv') or overwrites['starpars']:
            from . import starpars
            #from .stellar import starpars
            #Gets stellar info
            info,_,_=starpars.getStellarInfoFromCsv(ID,mission,radec=radec)
            info.to_csv(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_starpars.csv')
        else:
            print("loading from ",MonoData_savepath+'/'+ID_string+"/"+ID_string+'_starpars.csv')
            info=pd.read_csv(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_starpars.csv', index_col=0, header=0).T.iloc[0]
        if 'ra' in info.index and radec is None:
            radec=SkyCoord(float(info['ra'])*u.deg,float(info['dec'])*u.deg)
        Rstar=[float(info['rad']),float(info['eneg_rad']),float(info['epos_rad'])]
        Teff=[float(info['teff']),float(info['eneg_teff']),float(info['epos_teff'])]
        logg=[float(info['logg']),float(info['eneg_logg']),float(info['epos_logg'])]
        rhostar=[float(info['rho']),float(info['eneg_rho']),float(info['epos_rho'])]
        FeH=0.0 if 'FeH' not in info else float(info['FeH'])
        if 'mass' in info:
            Ms=float(info['mass'])
        else:
            Ms=rhostar[0]*Rstar[0]**3
        print(Rstar,Teff,logg,rhostar,Ms)
        #Rstar, rhostar, Teff, logg, src = starpars.getStellarInfo(ID, hdr, mission, overwrite=overwrite,
        #                                                         fileloc=savenames[1].replace('_mcmc.pickle','_starpars.csv'),
        #                                                         savedf=True)
    else:
        Rstar, rhostar, Teff, logg = kwargs['StarPars']
        Ms=rhostar[0]*Rstar[0]**3
        radec=None
        
    #opening lightcurve:
    if lc is None:
        if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_lc.pickle') or overwrites['lc']:
            #Gets Lightcurve
            lc,hdr=tools.openLightCurve(ID,mission,coor=radec,use_ppt=False,**kwargs)
            pickle.dump(lc,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_lc.pickle','wb'))
            #lc=lcFlatten(lc,winsize=9*tdur,stepsize=0.1*tdur)
        else:
            lc=pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_lc.pickle','rb'))
    
    #####################################
    #  DOING MONOTRANSIT PLANET SEARCH:
    #####################################
    if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle') or overwrites['monos']:
        if do_search and (not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle') or overwrites['monos']):
            #Doing search if we dont have a mono pickle file or we want to overwrite one:
            both_dic, monosearchparams, monofig = MonoTransitSearch(deepcopy(lc),ID,mission,
                                                                    Rs=Rstar[0],Ms=Ms,Teff=Teff[0],
                                                                    plot_loc=MonoData_savepath+'/'+ID_string+"/", plot=plot, **kwargs)
            pickle.dump(monosearchparams,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monosearchpars.pickle','wb'))
            
            if plot:
                figs['mono']=monofig
        elif os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle'):
            both_dic=pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle','rb'))
            if plot and os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_Monotransit_Search.pdf'):
                figs['mono']= MonoData_savepath+'/'+ID_string+"/"+ID_string+'_Monotransit_Search.pdf'
        elif tcen is not None and tdur is not None:
            intr=lc['mask']&(abs(lc['time']-tcen)<0.45*tdur)
            outtr=lc['mask']&(abs(lc['time']-tcen)<1.25*tdur)&(~intr)
            both_dic={'00':{'tcen':tcen,'tdur':tdur,'orbit_flag':'mono','poly_DeltaBIC':0.0,
                            'depth':np.nanmedian(lc['flux'][outtr])-np.nanmedian(lc['flux'][intr]),
                            'P_min':calc_min_P(lc['time'],tcen,tdur)}}
        else:
            raise InputError("Must either specify do_search or include tdur and tcen")
        ###################################
        #    VETTING MONO CANDIDATES:
        ###################################
        #print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag']} for pl in both_dic})
        for pl in both_dic:
            if len(both_dic)>0:
                #Best-fit model params for the mono transit:
                pl_dic, vet_fig = VetCand(both_dic[pl],pl,ID,lc,mission,Rs=Rstar[0],Ms=Ms,Teff=Teff[0],
                                          variable_llk_thresh=variable_llk_thresh,plot=plot,
                                          vet_do_fit=True,**kwargs)
                if plot:
                    figs[pl]=vet_fig
        pickle.dump(both_dic,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle','wb'))

    else:
        both_dic=pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_monos.pickle','rb'))
        if plot and os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_Monotransit_Search.pdf'):
            figs['mono'] = MonoData_savepath+'/'+ID_string+"/"+ID_string+'_Monotransit_Search.pdf'
    
    #print("monos:",{pl:{'tcen':mono_dic[pl]['tcen'],'depth':mono_dic[pl]['depth']} for pl in mono_dic})

    
    ###################################
    #   DOING PERIODIC PLANET SEARCH:
    ###################################
    if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle') or overwrites['multis']:
        if do_search and (not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle') or overwrites['multis']):
            both_dic, perfig = PeriodicPlanetSearch(deepcopy(lc),ID,deepcopy(both_dic),
                                                    plot_loc=MonoData_savepath+'/'+ID_string+"/",plot=plot,
                                                    rhostar=rhostar[0], Mstar=Ms, Rstar=Rstar[0], Teff=Teff[0], **kwargs)
            if plot:
                figs['multi']=perfig
        elif os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle'):
            both_dic=pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle','rb'))
            if plot and os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multi_search.pdf'):
                figs['multi']= MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multi_search.pdf'
        else:
            raise InputError("Must either specify do_search or include tdur and tcen")
        ###################################
        #  VETTING PERIODIC CANDIDATES:
        ###################################
        if np.sum([both_dic[pl]['orbit_flag']=='periodic' for pl in both_dic])>0:
            for pl in [pl for pl in both_dic if both_dic[pl]['orbit_flag']=='periodic']:
                pl_dic, pl_fig = VetCand(both_dic[pl],pl,ID,lc,mission,Rs=Rstar[0],Ms=Ms,Teff=Teff[0],
                                         variable_llk_thresh=variable_llk_thresh,plot=plot,**kwargs)
                if plot:
                    figs[pl]=pl_fig
        pickle.dump(both_dic,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle','wb'))
    else:
        both_dic=pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multis.pickle','rb'))
        if plot and os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multi_search.pdf'):
            figs['multi']= MonoData_savepath+'/'+ID_string+"/"+ID_string+'_multi_search.pdf'
    
    #######################################
    #  IDENTIFYING CONFUSED CANDIDATES:
    #######################################
    if len(both_dic)>0:
        if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_allpls.pickle') or overwrites['vet']:
            #Loading candidates from file:
            # Removing any monos or multis which are confused (e.g. a mono which is in fact in a multi)
            both_dic,monos,multis = CheckPeriodConfusedPlanets(deepcopy(lc), deepcopy(both_dic), mono_multi=False)
            print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'dur':both_dic[pl]['tdur'],
                       'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})

            #Check pairs of monos for potential match and period:
            both_dic = CheckMonoPairs(lc['time'], deepcopy(both_dic),**kwargs)
            #Doing the period confusion again but now including duos, and comparing multis with monos
            both_dic,monos,multis = CheckPeriodConfusedPlanets(deepcopy(lc),deepcopy(both_dic),
                                                               mono_mono=False,multi_multi=True,mono_multi=True)

            monos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='mono']
            duos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='duo']
            multis=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='periodic']

            for pl in both_dic:
                #Asses whether any dips are significant enough:
                if both_dic[pl]['orbit_flag'][:2]=='FP':
                    both_dic[pl]['flag']='FP - confusion'
                elif 'flag' not in both_dic[pl]:
                    both_dic[pl]['flag']='planet'

                if both_dic[pl]['flag'] not in ['EB','asteroid','instrumental','FP - confusion',
                                                'lowSNR/V-shaped','lowSNR','discrepant duration']:
                    #Check if the Depth/Rp suggests we have a very likely EB, we search for a secondary
                    if 'rp_rs' in both_dic[pl] and (both_dic[pl]['rp_rs']>PL_ror_thresh or Rs[0]*both_dic[pl]['rp_rs']>PL_ror_thresh):
                        #Likely EB
                        both_dic[pl]['flag']='EB'
            pickle.dump(both_dic,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_allpls.pickle','wb'))
        elif os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_allpls.pickle'):
            #Loading candidates from file:
            both_dic = pickle.load(open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_allpls.pickle','rb'))
        monos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='mono']
        duos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='duo']
        multis=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='periodic']
        if plot:
            for obj in both_dic:
                initname=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_'+obj
                if os.path.exists(initname+'_variability_check.pdf'):
                    figs[obj]=initname+'_variability_check.pdf'
                elif os.path.exists(initname+'_vetting.pdf'):
                    figs[obj]=initname+'_vetting.pdf'

        # ASSEMBLING CANDIDATES INTO DATAFRAME TABLE:
        df=pd.DataFrame()
        #all_cands_df=pd.read_csv("all_cands.csv")
        complexkeys=[]
        all_keys=np.unique(np.hstack([np.array([key for key in both_dic[obj]]) for obj in both_dic]).ravel())
        for obj in both_dic:
            ser=pd.Series(name=str(ID).zfill(11)+'_'+obj)
            ser['ID']=ID
            ser['obj_id']=obj
            ser['mission']=mission
            for key in all_keys:
                if key in both_dic[obj]:
                    if type(both_dic[obj][key]) not in [float,int,str,np.float64,np.float64] or (type(both_dic[obj][key])=='str' and len(both_dic[obj][key])>100):
                        complexkeys+=[key]
                    else:
                        if key == 'ID':
                            ser[key]=int(both_dic[obj][key])
                        elif type(both_dic[obj][key]) in [str,int]:
                            ser[key]=both_dic[obj][key]
                        elif type(both_dic[obj][key]) in [float,np.float64,np.float64]:
                            ser[key]=np.round(both_dic[obj][key],4)
            df=df.append(ser)
            #if str(ID).zfill(11)+'_'+obj not in all_cands_df.index or overwrite:
            #    all_cands_df=all_cands_df.append(ser[[ind for ind in ser.index if ind not in complexkeys]])
        #Adding stellar info to df:
        df['rstar']=Rstar[0]
        df['rstar_negerr']=Rstar[1]
        df['rstar_poserr']=Rstar[2]
        df['rho']=rhostar[0]
        df['rho_negerr']=rhostar[1]
        df['rho_poserr']=rhostar[2]
        df['logg']=logg[0]
        df['logg_negerr']=logg[1]
        df['logg_poserr']=logg[2]
        df['Teff']=Teff[0]
        df['Teff_negerr']=Teff[1]
        df['Teff_poserr']=Teff[2]
        
        if plot:
            #Chcking if values in df are different:
            new_df=True
            if not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_candidates.csv') or np.any([overwrites[col] for col in overwrites]):
                # Making table a plot for PDF:
                fig=plt.figure(figsize=(11.69,2+0.5*len(both_dic)))
                ax=fig.add_subplot(111)
                fig.patch.set_visible(False)
                ax.axis('off')
                ax.axis('tight')
                cols=['obj_id','ID','orbit_flag','flag','period','tcen','r_pl','depth','tdur','b','snr','snr_r']
                if 'duo' in df['orbit_flag'].values:
                    cols+=['tcen_2']
                #print(df.loc[:,list(cols.keys())].values)
                tab = ax.table(cellText=df[cols].values,
                          colLabels=cols,
                          loc='center')
                tab.auto_set_font_size(False)
                tab.set_fontsize(8)

                tab.auto_set_column_width(col=range(len(cols))) # Provide integer list of columns to adjust

                fig.tight_layout()
                fig.savefig(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_table.pdf')
                figs['tab']=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_table.pdf'
            else:
                figs['tab']=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_table.pdf'
        if not os.path.isfile(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_candidates.csv') or overwrites['vet']:
            df.to_csv(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_candidates.csv')
        #all_cands_df.to_csv("all_cands.csv")
        
        '''
        #Searches for other dips in the lightcurve
        planet_dic_1=SearchForSubsequentTransits(lc, interpmodel, monoparams, Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)
        '''
        #If any of the dips pass our EB threshold, we will run do a single model as an EB here:
        
        print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],
                   'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})
        
        
        # EB MODELLING:
        if np.any([both_dic[obj]['flag']=='EB' for obj in both_dic]):
            '''#Minimising EB model with ELLC:
            eb_dict={obj:both_dic[obj] for obj in both_dic if both_dic[obj]['flag'] not in ['asteroid','instrumental','FP - confusion']}
            EBdic=minimize_EBmodel(lc, eb_dict,Teff,Rstar[0],rhostar[0]*Rstar[0]**3)#lc, planet, Teffs, Rs, Ms, nrep=9,try_sec=False,use_ellc=False

            EBdic['ID']=ID
            EBdic['mission']=mission
            #Save to table here.
            '''
            mod = None
        
        elif np.any([both_dic[obj]['flag']=='planet' for obj in both_dic]):
             #Doing this import here so that Pymc3 gains the seperate compiledir location for theano
            # PLANET MODELLING:
            print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})
            print("Planets to model:",[obj for obj in both_dic if both_dic[obj]['flag']=='planet'])
            if len(glob.glob(MonoData_savepath+'/'+ID_string+"/"+ID_string+'*_model.pickle'))==0 or overwrites['fit']:
                
                if mission=='kepler' and cutDistance not in kwargs and bin_oot not in kwargs:
                    mod=fit.monoModel(ID, mission, lc, {}, savefileloc=MonoData_savepath+'/'+ID_string+"/",
                                      cutDistance=2.0,bin_oot=False)
                else:
                    mod=fit.monoModel(ID, mission, lc, {}, savefileloc=MonoData_savepath+'/'+ID_string+"/")
                #If not, we have a planet.
                #Checking if monoplanet is single, double-with-gap, or periodic.
                multis=[]
                monos=[]
                for obj in both_dic:
                    print(obj, both_dic[obj]['flag']=='planet',both_dic[obj]['orbit_flag'],
                          both_dic[obj]['period'],both_dic[obj]['P_min'])
                    if both_dic[obj]['flag']=='planet' and both_dic[obj]['orbit_flag']=='periodic':
                        #multi case: (including duos with no period gaps here):
                        mod.add_multi(deepcopy(both_dic[obj]),obj)
                    elif both_dic[obj]['flag']=='planet' and both_dic[obj]['orbit_flag']=='mono':
                        #Simple mono case:
                        mod.add_mono(deepcopy(both_dic[obj]),obj)
                    elif both_dic[obj]['flag']=='planet' and both_dic[obj]['orbit_flag']=='duo':
                        if both_dic[obj]['P_min']==both_dic[obj]['period']:
                            #Only two transits, but we can treat it as a multi as the period is solid:
                            duo2multi=deepcopy(both_dic[obj])
                            duo2multi['orbit_flag']='periodic'
                            mod.add_multi(duo2multi,obj)
                        else:
                            #split by gap duo case:
                            mod.add_duo(deepcopy(both_dic[obj]),obj)
                mod.init_starpars(Rstar=Rstar,rhostar=rhostar,Teff=Teff,logg=logg)
                mod.SaveModelToFile()
                #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))
                mod.init_model(useL2=useL2,FeH=FeH,**kwargs)
                mod.SaveModelToFile()
                #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))
            elif len(glob.glob(MonoData_savepath+'/'+ID_string+"/"+ID_string+'*_model.pickle'))>0:
                print("#Loading model from file")
                mod = fit.monoModel(ID, mission, LoadFromFile=True)
            else:
                mod=None
            if mod is not None and do_fit and (not hasattr(mod,'trace') or overwrites['fit']):
                print("Running MCMC")
                mod.RunMcmc(**kwargs)
                mod.SaveModelToFile()
                #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))

            if mod is not None and plot:
                print("Gathering plots, overwrite:",overwrites['model_plots'])
                #Gathering plots if they exist:
                if not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_plot.pdf') or overwrites['model_plots']:
                    mod.Plot(n_samp=1,plot_loc=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_plot.pdf',interactive=False)
                    mod.SaveModelToFile()
                    #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))
                figs['mcmc_mod']=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_plot.pdf'
                if not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_plot.html') or overwrites['model_plots']:
                    mod.Plot(n_samp=1,plot_loc=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_plot.html',interactive=True)
                    mod.SaveModelToFile()
                    #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))
                if hasattr(mod, 'trace'):
                    if not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_periods.pdf') or overwrites['model_plots']:
                        mod.PlotPeriods(plot_loc=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_periods.pdf')
                    figs['mcmc_pers']=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_periods.pdf'
                    if not os.path.exists(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_table.pdf') or overwrites['model_plots']: 
                        mod.PlotTable(plot_loc=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_table.pdf')
                    figs['mcmc_tab']=MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model_table.pdf'
                    mod.SaveModelToFile()
                    #pickle.dump(mod,open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_model.pickle','wb'))

        else:
            #NO EB OR PC candidates - likely low-SNR or FP.
            print("likely low-SNR or FP. Flags=",[both_dic[obj]['flag']=='planet' for obj in both_dic])
            #Save to table here.
            mod=None
    else:
        print("nothing detected")
        mod, both_dic =  None, None
    # CREATING CANDIDATE VETTING REPORT:
    if plot:
        #print(figs)
        #Compiling figures into a multi-page PDF
        from PyPDF2 import PdfFileReader, PdfFileWriter

        output = PdfFileWriter()
        pdfPages=[]
        for figname in figs:
            if figs[figname] is not None:
                #print(figname,type(figs[figname]))
                output.addPage(PdfFileReader(open(figs[figname], "rb")).getPage(0))
        outputStream = open(MonoData_savepath+'/'+ID_string+"/"+ID_string+'_report.pdf', "wb")
        output.write(outputStream)
        outputStream.close()

    return mod, both_dic
        
