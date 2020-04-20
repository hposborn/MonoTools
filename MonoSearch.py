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
from copy import deepcopy
from datetime import datetime
from . import tools
from . import MonoFit
from scipy import optimize
from scipy import interpolate
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt

import scipy.interpolate as interp
import scipy.optimize as optim
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns


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

def QuickMonoFit(lc,it0,dur,Rs=None,Ms=None,Teff=None,useL2=False,fit_poly=True,
                 polyorder=2,ndurs=3.2, how='mono', init_period=None):
    # Performs simple planet fit to monotransit dip given the detection data.
    #Initial depth estimate:
    dur=0.3 if dur/dur!=1.0 else dur #Fixing duration if it's broken/nan.
    
    if 'flux_flat' not in lc:
        lc=lcFlatten(lc,winsize=9*dur,stepsize=0.1*dur)

    if how=='periodic':
        assert init_period is not None
        xinit=(lc['time']-it0-init_period*0.5)%init_period-init_period*0.5
        nearby=(abs(xinit)<ndurs*dur)
        cad = float(int(max(set(list(lc['cadence'][nearby])), key=list(lc['cadence'][nearby]).count)[1:]))/1440

        x = xinit[(abs(xinit)<ndurs*dur)&lc['mask']]+it0 #re-aligning this fake "mono" with the t0 provided
        y = lc['flux_flat'][nearby&lc['mask']][np.argsort(x)]
        y-=np.nanmedian(y)
        yerr=lc['flux_err'][nearby&lc['mask']][np.argsort(x)]
        x=np.sort(x).astype(np.float32)

        oot_flux=np.nanmedian(y[(abs(x-it0)>0.65*dur)])
        int_flux=np.nanmedian(y[(abs(x-it0)<0.35*dur)])
        #print(it0,dur,oot_flux,int_flux,abs(oot_flux-int_flux)/lc['flux_unit'],x,y,yerr)
        fit_poly=False
        
    else:
        #Mono case:
        nearby=abs(lc['time']-it0)<np.clip(dur*ndurs,1,4)
        cad = float(int(max(set(list(lc['cadence'][nearby])), key=list(lc['cadence'][nearby]).count)[1:]))/1440

        x=lc['time'][nearby&lc['mask']].astype(np.float32)
        yerr=lc['flux_err'][nearby&lc['mask']]
        if not fit_poly:
            y=lc['flux_flat'][nearby&lc['mask']]
            y-=np.nanmedian(y)
            oot_flux=np.nanmedian(y[(abs(x-it0)>0.65*dur)])
            int_flux=np.nanmedian(y[(abs(x-it0)<0.35*dur)])
        else:
            y=lc['flux'][nearby&lc['mask']]
            y-=np.nanmedian(y)
            init_poly=np.polyfit(x[abs(x-it0)<0.6]-it0,y[abs(x-it0)<0.6],polyorder)
            oot_flux=np.nanmedian((y-np.polyval(init_poly,x-it0))[abs(x-it0)>0.65*dur])
            int_flux=np.nanmedian((y-np.polyval(init_poly,x-it0))[abs(x-it0)<0.35*dur])
    dep=abs(oot_flux-int_flux)/lc['flux_unit']
    #print(dep,dur,it0,init_poly, [init_poly[nip]>-5*(10.0 ** -np.arange(polyorder+1)[::-1])[nip] and init_poly[nip]<5*(10.0 ** -np.arange(polyorder+1)[::-1])[nip] for nip in range(polyorder+1)])
    
    with pm.Model() as model:
        # Parameters for the stellar properties
        if fit_poly:
            trend = pm.Uniform("trend", upper=np.tile(1,polyorder+1), shape=polyorder+1,
                              lower=np.tile(-1,polyorder+1), testval=init_poly)

            #trend = pm.Normal("trend", mu=np.zeros(polyorder+1), sd=5*(10.0 ** -np.arange(polyorder+1)[::-1]), 
            #                  shape=polyorder+1, testval=np.zeros(polyorder+1))
            #trend = pm.Uniform("trend", upper=np.tile(10,polyorder+1),lower=np.tile(-10,polyorder+1),
            #                   shape=polyorder+1, testval=np.zeros(polyorder+1))
        else:
            mean = pm.Normal("mean", mu=0.0, sd=3*np.nanstd(y))
        
        r_star = Rs if Rs is not None and not np.isnan(Rs) else 1.0
        m_star = Ms if Ms is not None and not np.isnan(Ms) else 1.0
        Ts = Teff if Teff is not None and not np.isnan(Teff) else 5500.0

        u_star = tools.getLDs(Ts)[0]
        #xo.distributions.QuadLimbDark("u_star")
        if how=='periodic' and init_period is not None:
            log_per = pm.Normal("log_per", mu=np.log(init_period),sd=0.4)
        else:
            init_per = abs(18226*((2*np.sqrt((1+dep**0.5)**2-0.41**2))/dur)**-3)
            # Orbital parameters for the planets
            log_per = pm.Uniform("log_per", lower=np.log(dur*5),upper=np.log(3000),
                                 testval=np.clip(np.log(init_per),np.log(dur*6),np.log(3000))
                                )

        tcen = pm.Bound(pm.Normal, lower=it0-0.7*dur, upper=it0+0.7*dur)("tcen", 
                                            mu=it0,sd=0.25*dur,testval=it0)
        
        b = pm.Uniform("b",upper=1.0,lower=0.0)
        log_ror = pm.Uniform("log_ror",lower=-6,upper=-1)
        ror = pm.Deterministic("ror", tt.exp(log_ror))
        #ror, b = xo.distributions.get_joint_radius_impact(min_radius=0.0075, max_radius=0.25,
        #                                                  testval_r=np.sqrt(dep), testval_b=0.41)
        #logror = pm.Deterministic("logror",tt.log(ror))
        
        
        #pm.Potential("ror_prior", -logror) #Prior towards 

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
        
        #Adding a potential to force our transit towards the observed transit duration:
        pm.Potential("tdur_prior", -10*abs(tt.log(tdur/dur)))        
        
        # The 2nd light (not third light as companion light is not modelled) 
        # This quantity is in delta-mag
        if useL2:
            deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
            third_light = pm.Deterministic("third_light", tt.power(2.511,-1*deltamag_contam))#Factor to multiply normalised lightcurve by
        else:
            third_light=0.0

        # Compute the model light curve using starry
        light_curves = (
            xo.LimbDarkLightCurve(u_star).get_light_curve(
                orbit=orbit, r=r_pl/109.1, t=x, texp=cad))*(1+third_light)/lc['flux_unit']
        if fit_poly:
            flux_trend = pm.Deterministic("flux_trend", tt.dot(np.vander(x - it0, polyorder+1), trend))
            transit_light_curve = pm.math.sum(light_curves, axis=-1)
            light_curve = transit_light_curve + flux_trend
        else:
            transit_light_curve = pm.math.sum(light_curves, axis=-1)
            light_curve = transit_light_curve + mean
        
        pm.Deterministic("light_curve", light_curve)

        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)
        
        if fit_poly:
            map_soln = xo.optimize(start=model.test_point,vars=[trend],verbose=False)
            map_soln = xo.optimize(start=map_soln,vars=[trend,log_ror,log_per,tcen],verbose=False)

            # Fit for the maximum a posteriori parameters
        else:
            map_soln = xo.optimize(start=model.test_point,vars=[mean],verbose=False)
            map_soln = xo.optimize(start=map_soln,vars=[mean,log_ror,log_per,tcen],verbose=False)
        
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
        if not func['success']:
            print("model failing. Why?")
            print(model.check_test_point())
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
    interpt=np.linspace(map_soln['tcen']-ndurs*float(dur),map_soln['tcen']+ndurs*float(dur),600)
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
        oot_mask=lc['mask']&(abs(lc['time']-best_fit['tcen'])>0.5)
        binlc=bin_lc_segment(np.column_stack((lc['time'][oot_mask],lc['flux_flat'][oot_mask],lc['flux_err'][oot_mask])),
                             best_fit['tdur'])
        best_fit['cdpp']=np.nanstd(binlc[:,1])
        best_fit['Ntrans']=1
    else:
        phase=(abs(lc['time']-best_fit['tcen']+0.5*best_fit['tdur'])%init_period)
        oot_mask=lc['mask']&(phase>best_fit['tdur'])
        binlc=bin_lc_segment(np.column_stack((lc['time'][oot_mask],lc['flux_flat'][oot_mask],lc['flux_err'][oot_mask])),
                             best_fit['tdur'])
        durobs=0
        for cad in np.unique(lc['cadence']):
            durobs+=np.sum(phase[lc['mask']&(lc['cadence']==cad)]<best_fit['tdur'])*float(int(cad[1:]))/1440
        best_fit['Ntrans']=durobs/best_fit['tdur']
        best_fit['cdpp']=np.nanstd(binlc[:,1])

    best_fit['snr_r']=best_fit['depth']/(best_fit['cdpp']/np.sqrt(best_fit['Ntrans']))
    
    interpmodel=interpolate.interp1d(np.hstack((-10000,interpt-best_fit['tcen'],10000)),
                                     np.hstack((0.0,transit_zoom,0.0)))
    
    return best_fit, interpmodel

def MonoTransitSearch(lc,ID,Rs=None,Ms=None,Teff=None,
                      mono_SNR_thresh=6.5,mono_BIC_thresh=-6,n_durs=5,poly_order=3,
                      n_oversamp=20,binsize=15/1440.0,
                      transit_zoom=3.5,use_flat=False,use_binned=False,use_poly=True,
                      plot=False,plot_loc=None,**kwargs):
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
    '''
    
    #Computing a fine x-range to search:
    search_xrange=[]
    
    Rs=1.0 if Rs is None else float(Rs)
    Ms=1.0 if Ms is None else float(Ms)
    Teff=5800.0 if Teff is None else float(Teff)

    interpmodels,tdurs=get_interpmodels(Rs,Ms,Teff,lc['time'],lc['flux_unit'],n_durs=5,texp=binsize)
    
    print("Searching "+str(ID)+" for monotransits")

    #print("Checking input model matches. flux:",np.nanmedian(uselc[:,0]),"std",np.nanstd(uselc[:,1]),"transit model:",
    #       interpmodels[0](10.0),"depth:",interpmodels[0](0.0))
        
    search_xranges=[]
    #Removing gaps bigger than 2d (with no data)
    for n in range(n_durs):
        search_xranges_n=[]
        for arr in np.array_split(lc['time'][lc['mask']],1+np.where(np.diff(lc['time'][lc['mask']])>tdurs[2])[0]):
            search_xranges_n+=[np.arange(arr[0]+0.33*tdurs[n],arr[-1]-0.33*tdurs[n],tdurs[n]/n_oversamp)]
        search_xranges+=[np.hstack(search_xranges_n)]

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
    
    #For each duration we scan across the lightcurve and compare a transit model to others:
    for n,search_xrange in enumerate(search_xranges):
        tdur=tdurs[n%n_durs]
        
        #flattening and binning as a function of the duration we are searching in order to avoid
        if use_binned:
            #Having may points in a lightcurve makes flattening difficult (and fitting needlessly slow)
            # So let's bin to a fraction of the tdur - say 9 in-transit points.
            lc=lcBin(lc,binsize=tdur/9,use_flat=False)
            if use_flat and not use_poly:
                lc=lcFlatten(lc,winsize=tdur*13,stepsize=tdur*0.333,use_bin=True)
                uselc=np.column_stack((lc['bin_time'],lc['bin_flux_flat'],lc['bin_flux_err']))
            else:
                uselc=np.column_stack((lc['bin_time'],lc['bin_flux'],lc['bin_flux_err']))
        else:
            if use_flat and not use_poly:
                lc=lcFlatten(lc,winsize=tdur*13,stepsize=tdur*0.333)
                uselc=np.column_stack((lc['time'][lc['mask']],lc['flux_flat'][lc['mask']],lc['flux_err'][lc['mask']]))
            else:
                uselc=np.column_stack((lc['time'][lc['mask']],lc['flux'][lc['mask']],lc['flux_err'][lc['mask']]))
                
        #Making depth vary from 0.1 to 1.0
        init_dep_shifts=np.exp(np.random.normal(0.0,n_oversamp*0.01,len(search_xrange)))
        randns=np.random.randint(2,size=(len(search_xrange),2))
        cad=np.nanmedian(np.diff(uselc[:,0]))
        p_transit = 3/(len(uselc[:,0])*cad)
        
        #What is the probability of transit given duration (used in the prior calculation) - duration
        methods=['SLSQP','Nelder-Mead','Powell']
        logmodeldep=np.log(abs(interpmodels[n](0.0)))

        for n_mod,x2s in enumerate(search_xrange):
            #minimise single params - depth
            round_tr=abs(uselc[:,0]-x2s)<(transit_zoom*tdur)
            #Centering x array on epoch to search
            x=uselc[round_tr,0]-x2s
            in_tr=abs(x)<(0.45*tdur)
            if not np.isnan(x[in_tr]).all():
                
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
                        'init_dep':np.exp(init_log_dep),'init_dur':tdur,
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
    signfct=np.where((outparams['sin_llk_ratio']>1.5)&(outparams['poly_llk_ratio']>1.5)&(outparams['poly_DeltaBIC']<mono_BIC_thresh)&(outparams['trans_snr']>(mono_SNR_thresh-0.5)))[0]

    if len(signfct)>0:
        jumps=np.hstack((0,1+np.where(np.diff(signfct)>0.66*(n_durs*0.5)*n_oversamp)[0],len(signfct)))
        min_ixs=[]
        #Looping through clustered regions of "detection" space and finding the maximum value within
        for n_jump in range(len(jumps)-1):
            ix=signfct[jumps[n_jump]:jumps[n_jump+1]]
            if use_poly:
                min_ix=ix[np.argmin(outparams.iloc[ix]['poly_DeltaBIC'])]
                min_ixs+=[[min_ix,outparams.iloc[min_ix]['poly_DeltaBIC']]]
                '''print('nearbys:',outparams[min_ix-1,6],outparams[min_ix-1,0]-outparams[min_ix-1,1],
                                 outparams[min_ix,6],outparams[min_ix,0]-outparams[min_ix,1],
                                 outparams[min_ix+1,6],outparams[min_ix+1,0]-outparams[min_ix+1,1])'''
            else:
                min_ix=ix[np.argmax(outparams.iloc[ix]['poly_llk_ratio'])]
                min_ixs+=[[min_ix,outparams.iloc[min_ix]['poly_llk_ratio']]]
        if use_poly:
            min_ixs=np.array(min_ixs)
            min_ixs=min_ixs[min_ixs[:,1].argsort()]
        else:
            min_ixs=np.array(min_ixs)
            min_ixs=min_ixs[min_ixs[:,1].argsort()[::-1]]

        detns = {}
        lc_std=np.nanstd(uselc[:,1])
        cad=np.nanmedian(np.diff(uselc[:,0]))
        for nix,ix in enumerate(min_ixs):
            detn_row=outparams.iloc[int(ix[0])]
            detns[str(nix).zfill(2)]={}
            detns[str(nix).zfill(2)]['llk_trans']=detn_row['llk_trans']
            detns[str(nix).zfill(2)]['llk_sin']=detn_row['llk_sin']
            detns[str(nix).zfill(2)]['llk_poly']=detn_row['llk_poly']
            detns[str(nix).zfill(2)]['BIC_trans']=detn_row['BIC_trans']
            detns[str(nix).zfill(2)]['BIC_sin']=detn_row['BIC_sin']
            detns[str(nix).zfill(2)]['BIC_poly']=detn_row['BIC_poly']
            detns[str(nix).zfill(2)]['sin_DeltaBIC']=detn_row['sin_DeltaBIC']
            detns[str(nix).zfill(2)]['poly_DeltaBIC']=detn_row['poly_DeltaBIC']
            detns[str(nix).zfill(2)]['tcen']=detn_row['tcen']
            detns[str(nix).zfill(2)]['period']=np.nan
            detns[str(nix).zfill(2)]['period_err']=np.nan
            detns[str(nix).zfill(2)]['DeltaBIC']=ix[1]
            detns[str(nix).zfill(2)]['tdur']=detn_row['init_dur']
            detns[str(nix).zfill(2)]['depth']=detn_row['trans_dep']
            detns[str(nix).zfill(2)]['orbit_flag']='mono'
            detns[str(nix).zfill(2)]['snr']=detn_row['trans_snr']
            #Calculating minimum period:
            abs_times=abs(uselc[:,1]-detn_row['tcen'])
            abs_times=np.sort(abs_times)
            whr=np.where(np.diff(abs_times)>detn_row['init_dur']*0.75)[0]
            if len(whr)>0:
                detns[str(nix).zfill(2)]['P_min']=abs_times[whr[0]]
            else:
                detns[str(nix).zfill(2)]['P_min']=np.max(abs_times)
            if nix>8:
                break
    else:
        detns={}
    if plot:
        fig_loc= PlotMonoSearch(lc,ID,outparams,detns,interpmodels,tdurs,
                             use_flat=use_flat,use_binned=use_binned,use_poly=use_poly,plot_loc=plot_loc)
        return detns, outparams, fig_loc
    else:
        return detns, outparams, None
    
def PlotMonoSearch(lc,ID,monosearchparams,mono_dic,interpmodels,tdurs,
                   use_flat=True,use_binned=True,use_poly=False,transit_zoom=2.5,plot_loc=None,**kwargs):
    if plot_loc is None:
        plot_loc = str(ID).zfill(11)+"_Monotransit_Search.pdf"
    elif plot_loc[-1]=='/':
        plot_loc = plot_loc+str(ID).zfill(11)+"_Monotransit_Search.pdf"
    if use_flat and not use_poly:
        lc=lcFlatten(lc,winsize=np.median(tdurs)*7.5,stepsize=0.2*np.median(tdurs))
        lc=lcBin(lc,30/1440,use_masked=True,use_flat=True)
        flux_key='flux_flat'
    else:
        flux_key='flux'
        lc=lcBin(lc,30/1440,use_masked=True,use_flat=False)
    if use_binned:
        flux_key='bin_'+flux_key

    fig = plt.figure(figsize=(11.69,8.27))
    import seaborn as sns
    sns.set_palette(sns.set_palette("RdBu",14))
    axes=[]
    axes +=[fig.add_subplot(411)]
    axes[0].set_title(str(ID).zfill(7)+" - Monotransit Search")
    
    #rast=True if np.sum(lc['mask']>12000) else False
    axes[0].plot(lc['time'][lc['mask']],lc[flux_key][lc['mask']],'.k',alpha=0.28,markersize=0.75, rasterized=True)
    if use_flat:
        axes[0].plot(lc['bin_time'],lc['bin_flux_flat'],'.k',alpha=0.7,markersize=1.75, rasterized=True)
    else:
        axes[0].plot(lc['bin_time'],lc['bin_flux'],'.k',alpha=0.7,markersize=1.75, rasterized=True)
    axes[0].set_ylim(np.percentile(lc[flux_key][lc['mask']],(0.25,99.75)))
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
    axes[1].legend()
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
            round_tr=lc['mask']&(abs(lc['time']-tcen)<(transit_zoom*tdur))
            x=(lc['time'][round_tr]-tcen)
            y=lc[flux_key][round_tr]
            if use_poly:
                y-=np.nanmedian(lc[flux_key][round_tr&(abs(lc['time']-tcen)>(0.7*tdur))])
            
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
            
            axes[-1].plot(lc['bin_time'][round_tr_bin],lc['bin_flux'][round_tr_bin],
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

def PeriodicPlanetSearch(lc,ID,planets,use_binned=None,use_flat=True,binsize=1/96.0,n_search_loops=5,
                         multi_FAP_thresh=0.00125,multi_SNR_thresh=7.0,plot=False, plot_loc=None,**kwargs):
    #Searches an LC (ideally masked for the monotransiting planet) for other *periodic* planets.
    from transitleastsquares import transitleastsquares
    print("Using TLS on ID="+str(ID)+" to search for multi-transiting planets")
    
    #Using bins if we have a tonne of points (eg >1 sector). If not, we can use unbinned data.
    use_binned=True if use_binned is None and len(lc['flux'])>40000 else False
    
    if use_binned and 'bin_flux' not in lc:
        if abs(np.nanmedian(np.diff(lc['time']))-binsize)/binsize<0.1:
            use_binned=False
            #Lightcurve already has cadence near the target binsize
        else:
            lc=lcBin(lc,binsize=binsize,use_flat=use_flat)
            suffix='_flat'
    prefix='bin_' if use_binned else ''
    if use_flat:
        if 'flux_flat' not in lc:
            lc=lcFlatten(lc)
        suffix='_flat'
    else:
        suffix=''
    
    if plot:
        sns.set_palette("viridis",10)
        fig = plt.figure(figsize=(11.69,8.27))
        
        plt.subplot(312)
        rast=True if np.sum(lc['mask']>12000) else False

        #plt.plot(lc['time'][lc['mask']],lc['flux_flat'][lc['mask']]*lc['flux_unit']+(1.0-np.nanmedian(lc['flux_flat'][lc['mask']])*lc['flux_unit']),',k')
        #if prefix=='bin_':
        #    plt.plot(lc[prefix+'time'],lc[prefix+'flux'+suffix]*lc['flux_unit']+(1.0-np.nanmedian(lc[prefix+'flux'+suffix])*lc['flux_unit']),'.k')
        plt.subplot(313)
        plt.plot(lc['time'][lc['mask']],
                lc['flux_flat'][lc['mask']]*lc['flux_unit']+(1.0-np.nanmedian(lc['flux_flat'][lc['mask']])*lc['flux_unit']),
                 '.k', markersize=0.5,rasterized=rast)
        lc=lcBin(lc,binsize=1/48,use_flat=True)
        plt.plot(lc['bin_time'],
                 lc['bin_flux_flat']*lc['flux_unit']+(1.0-np.nanmedian(lc['bin_flux_flat'])*lc['flux_unit']),
                 '.k', markersize=4.0)

    #Looping through, masking planets, until none are found.
    #{'01':{'period':500,'period_err':100,'FAP':np.nan,'snr':np.nan,'tcen':tcen,'tdur':tdur,'rp_rs':np.nan}}
    if prefix+'mask' in lc:
        anommask=lc[prefix+'mask'][:]
    else:
        anommask=~np.isnan(lc[prefix+'flux'+suffix][:])
    plmask=np.tile(False,np.sum(anommask))
    
    SNR_last_planet=100;init_n_pl=len(planets);n_pl=len(planets);results=[]
    while SNR_last_planet>multi_SNR_thresh and n_pl<(n_search_loops+init_n_pl):
        if len(planets)>1:
            planet_name=str(int(1+np.max([float(key) for key in planets]))).zfill(2)
        else:
            planet_name='00'
        #Making model. Making sure lc is at 1.0 and in relatie flux, not ppt/ppm:
        modx = lc[prefix+'time'][anommask]
        mody = lc[prefix+'flux'+suffix][anommask]*lc['flux_unit']+(1.0-np.nanmedian(lc[prefix+'flux'+suffix][anommask])*lc['flux_unit'])
        if np.sum(plmask)>0:
            mody[plmask] = mody[np.random.choice(np.sum(anommask),np.sum(plmask))]
        model = transitleastsquares(modx, mody)
        results+=[model.power(period_min=0.5,period_max=0.75*(np.nanmax(lc[prefix+'time'])-np.nanmin(lc[prefix+'time'])),
                            use_threads=1,show_progress_bar=False, n_transits_min=3)]

        if 'FAP' in results[-1] and 'snr' in results[-1] and not np.isnan(results[-1]['snr']) and 'transit_times' in results[-1]:
            #Defining transit times as those times where the SNR in transit is consistent with expectation (>-3sigma)
            snr_per_trans_est=np.sqrt(np.sum(results[-1].snr_per_transit>0))
            trans=np.array(results[-1]['transit_times'])[results[-1].snr_per_transit>snr_per_trans_est/2]
            #np.array(results[-1]['transit_times'])[np.any(abs(np.array(results[-1]['transit_times'])[np.newaxis,:]-lc['time'][:,np.newaxis])<(0.16*results[-1]['duration']),axis=0)]

            #print(results[-1].snr,'>',multi_SNR_thresh,results[-1].FAP,'<',multi_FAP_thresh)
        else:
            trans=[]
            #print(results[-1].keys())
            #print(np.max(results[-1].power))
        if 'FAP' in results[-1] and 'snr' in results[-1]:
            '''#recomputing the SNR using the in-transit and round-transit flux & using a threshold 33% lower.
            dur_per=results[-1].duration/results[-1].period
            in_tr=lc[prefix+'flux'+suffix][maskall][(results[-1].folded_phase>0.5-(0.4*dur_per))&(results[-1].folded_phase<0.5+(0.4*dur_per))]
            out_tr=lc[prefix+'flux'+suffix][maskall][((results[-1].folded_phase>0.5-(2.5*dur_per))&(results[-1].folded_phase<0.55-(0.5*dur_per)))|((results[-1].folded_phase>0.5+(0.55*dur_per))&(results[-1].folded_phase<0.5+(2.5*dur_per)))]
            recomp_SNR=(np.nanmedian(out_tr)-np.nanmedian(in_tr))/np.hypot(np.nanstd(in_tr),np.nanstd(out_tr))'''
            #print(planet_name,"FAP:",results[-1].FAP,"SNR:",results[-1].snr,"period:",results[-1].period)
            if (results[-1].FAP<multi_FAP_thresh) and results[-1].snr>multi_SNR_thresh and len(trans)>2:
                SNR_last_planet=results[-1].snr
                planets[planet_name]={'period':results[-1].period, 'period_err':results[-1].period_uncertainty,
                                      'P_min':results[-1].period,
                                      'snr':results[-1].snr,'FAP':results[-1].FAP, 
                                      'tcen':results[-1].T0, 'tdur':results[-1].duration,'rp_rs':results[-1].rp_rs,
                                      'orbit_flag':'periodic',
                                      'depth':1.0-results[-1].depth,
                                      'xmodel':results[-1].model_lightcurve_time,
                                      'ymodel':results[-1].model_lightcurve_model, 'N_trans':len(trans)}
                plt.subplot(311)
                plt.plot([results[-1].period,results[-1].period],[0,1.4*results[-1].snr],'-',
                         linewidth=4.5,alpha=0.4,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det_'+str(n_pl))
                plt.plot(results[-1].periods,results[-1].power,c=sns.color_palette()[n_pl-init_n_pl])
                plt.subplot(312)
                plt.plot(results[-1]['model_lightcurve_time'],results[-1]['model_lightcurve_model'],alpha=0.5,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det_'+str(n_pl),linewidth=4)
                plt.subplot(313)
                plt.plot(results[-1]['model_lightcurve_time'][results[-1]['model_lightcurve_model']<1],
                         results[-1]['model_lightcurve_model'][results[-1]['model_lightcurve_model']<1],
                         alpha=0.75,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det='+str(n_pl),
                         linewidth=4,rasterized=True)

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
                this_pl_masked=(((modx-planets[planet_name]['tcen']+0.6*planets[planet_name]['tdur'])%planets[planet_name]['period'])<(1.2*planets[planet_name]['tdur']))
                plmask=plmask+this_pl_masked#Masking previously-detected transits
                print("pl_mask",np.sum(this_pl_masked)," total:",np.sum(plmask))
            elif results[-1].snr>multi_SNR_thresh:
                # pseudo-fails - we have a high-SNR detection but it's a duo or a mono.
                this_pl_masked=(((modx-results[-1].T0+0.6*results[-1].duration)%results[-1].period)<(1.2*results[-1].duration))
                plmask=plmask+this_pl_masked
                SNR_last_planet=results[-1].snr
            
            else:
                # Fails
                print("detection at ",results[-1].period," with ",len(trans)," transits does not meet SNR ",results[-1].snr,"or FAP",results[-1].FAP)
                SNR_last_planet=0
        n_pl+=1

    if plot:
        if plot_loc is None:
            plot_loc = str(ID).zfill(11)+"_multi_search.pdf"
        elif plot_loc[-1]=='/':
            plot_loc = plot_loc+str(ID).zfill(11)+"_multi_search.pdf"
        plt.subplot(311)
        plt.legend()
        plt.subplot(312)
        plt.legend()
        plt.subplot(313)
        plt.legend()
        if 'jd_base' in lc:
            plt.xlabel("time [BJD-"+str(lc['jd_base'])+"]")
        else:
            plt.xlabel("time")
        plt.suptitle(str(ID).zfill(11)+"-  Multi-transit search")
        plt.tight_layout()
        plt.savefig(plot_loc, dpi=400)
    #TBD - Check against known FP periods (e.g. 0.25d K2 or 14d TESS)?
    if plot:
        return planets, plot_loc
    else:
        return planets, None

def CheckPeriodConfusedPlanets(lc,all_dets):
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
    if len(mono_detns)>1:
        #removing monos which are effectively the same. Does this through tcen/tdur matching.
        for monopl in mono_detns:
            if all_dets[monopl]['orbit_flag'][:2]!='FP':
                other_dets=np.array([[other,all_dets[other]['tcen'],all_dets[other]['tdur']] for other in mono_detns if other !=monopl])
                trans_prox=np.min(abs(all_dets[monopl]['tcen']-other_dets[:,1].astype(float))) #Proximity to a transit
                prox_stats=abs(trans_prox/(0.5*(all_dets[monopl]['tdur']+other_dets[:,2].astype(float))))
                print("Mono-mono compare",monopl,all_dets[monopl]['tcen'],other_dets[:,1],prox_stats)
                if np.min(prox_stats)<0.5:
                    other=other_dets[np.argmin(prox_stats),0]
                    if all_dets[other]['snr']>all_dets[monopl]['snr']:
                        all_dets[monopl]['orbit_flag']='FP - confusion with '+other
                    else:
                        all_dets[other]['orbit_flag']='FP - confusion with '+monopl
    mono_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag']=='mono')&(all_dets[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability'])]
                    
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]
    if len(perdc_detns)>1:
        #removing periodics which are effectively the same. Does this through cadence correlation.
        for perpl in perdc_detns:
            new_trans_arr=((lc['time'][lc['mask']]-all_dets[perpl]['tcen']+0.5*all_dets[perpl]['tdur'])%all_dets[perpl]['period'])<all_dets[perpl]['tdur']

            for perpl2 in perdc_detns:
                if perpl!=perpl2 and all_dets[perpl]['orbit_flag'][:2]!='FP' and all_dets[perpl2]['orbit_flag'][:2]!='FP':
                    new_trans_arr2=((lc['time'][lc['mask']]-all_dets[perpl2]['tcen']+0.5*all_dets[perpl2]['tdur'])%all_dets[perpl2]['period'])<all_dets[perpl2]['tdur']
                    sum_overlap=np.sum(new_trans_arr&new_trans_arr2)
                    prox_arr=np.hypot(sum_overlap/np.sum(new_trans_arr),sum_overlap/np.sum(new_trans_arr2))
                    print("Multi-multi compare",perpl,all_dets[perpl]['period'],perpl2,all_dets[perpl2]['period'],prox_arr)
                    if prox_arr>0.6:
                        #These overlap - taking the highest SNR signal
                        if all_dets[perpl]['snr']>all_dets[perpl2]['snr']:
                            all_dets[perpl2]['orbit_flag']='FP - confusion with '+perpl2
                        else:
                            all_dets[perpl]['orbit_flag']='FP - confusion with '+perpl
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]
    trans_arr=[]
    #print(mono_detns,perdc_detns,len(mono_detns)>0 and len(perdc_detns)>0)
    if len(mono_detns)>0 and len(perdc_detns)>0:
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
                    print("Multi-mono compare",perpl,all_dets[perpl]['tdur'],"|",monopl,all_dets[monopl]['tcen'],all_dets[monopl]['tdur'],prox_stat)
                    if prox_stat>0.33:
                        #These overlap - taking the highest SNR signal
                        print("correlation - ",all_dets[perpl]['snr'],all_dets[monopl]['snr'])
                        if all_dets[perpl]['snr']>=all_dets[monopl]['snr']:
                            all_dets[monopl]['orbit_flag']= 'FP - confusion with '+perpl
                        elif all_dets[perpl]['snr']<all_dets[monopl]['snr']:
                            all_dets[perpl]['orbit_flag']= 'FP - confusion with '+monopl

    mono_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag']=='mono')&(all_dets[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability'])]
    perdc_detns=[pl for pl in all_dets if (all_dets[pl]['orbit_flag'] in ['periodic','duo'])&(all_dets[pl]['orbit_flag']!='variability')]

    return all_dets, mono_detns, perdc_detns


def weighted_avg_and_std(values, errs): 
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=1/errs**2)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=1/errs**2)
    return [average, np.sqrt(variance)/np.sqrt(len(values))]

def lcBin(lc,binsize=1/48,use_flat=True,use_masked=True):
    #Binning lightcurve to e.g. 30-min cadence for planet search
    # Can optionally use the flatted lightcurve
    binlc=[]
    
    #Using flattened lightcurve:
    if use_flat and 'flux_flat' not in lc:
        lc=lcFlatten(lc)
    flux_dic='flux_flat' if use_flat else 'flux'
    
    if np.nanmax(np.diff(lc['time']))>3:
        loop_blocks=np.array_split(np.arange(len(lc['time'])),np.where(np.diff(lc['time'])>3.0)[0])
    else:
        loop_blocks=[np.arange(len(lc['time']))]
    for sh_time in loop_blocks:
        if use_masked:
            lc_segment=np.column_stack((lc['time'][sh_time][lc['mask'][sh_time]],
                                        lc[flux_dic][sh_time][lc['mask'][sh_time]],
                                        lc['flux_err'][sh_time][lc['mask'][sh_time]]))
        else:
            lc_segment=np.column_stack((lc['time'][sh_time],lc[flux_dic][sh_time],lc['flux_err'][sh_time]))
        if binsize>(1.66*np.nanmedian(np.diff(lc['time'][sh_time]))):
            #Only doing the binning if the cadence involved is >> the cadence
            digi=np.digitize(lc_segment[:,0],np.arange(lc_segment[0,0],lc_segment[-1,0],binsize))
            binlc+=[bin_lc_segment(lc_segment, binsize)]
        else:
            binlc+=[lc_segment]
        
    binlc=np.vstack(binlc)
    lc['bin_time']=binlc[:,0]
    lc['bin_'+flux_dic]=binlc[:,1]
    #Need to clip error here as tiny (and large) errors from few points cause problems down the line.
    lc['bin_flux_err']=np.clip(binlc[:,2],0.9*np.nanmedian(binlc[:,2]),20*np.nanmedian(binlc[:,2]))
    return lc

def bin_lc_segment(lc_segment, binsize):
    digi=np.digitize(lc_segment[:,0],np.arange(np.min(lc_segment[:,0]),np.max(lc_segment[:,0]),binsize))
    return np.vstack([[[np.nanmedian(lc_segment[digi==d,0])]+\
                       weighted_avg_and_std(lc_segment[digi==d,1],lc_segment[digi==d,2])] for d in np.unique(digi)])
    
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
                                                                   t=lc['time'], texp=cad*0.98
                                                                  ).eval()]
        elif all_pls[pl]['orbit_flag'] in ['mono','duo']:
            #generating two monos for duo
            per_guess=18226*rhostar*(2*np.sqrt(1-0.4**2)/all_pls[pl]['tdur'])**-3#from circular orbit and b=0.4
            orbit = xo.orbits.KeplerianOrbit(r_star=Rstar, rho_star=rhostar,
                                             period=per_guess, t0=all_pls[pl]['tcen'], b=0.4)
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=np.sqrt(all_pls[pl]['depth']), 
                                                                   t=lc['time'], texp=cad*0.98
                                                                   ).eval()
            light_curve[abs(lc['time']-all_pls[pl]['tcen'])>per_guess*0.4]=0.0
            if all_pls[pl]['orbit_flag'] is 'duo' and 'tcen_2' in all_pls[pl]:
                orbit2 = xo.orbits.KeplerianOrbit(r_star=Rstar, rho_star=rhostar,
                                                 period=per_guess, t0=all_pls[pl]['tcen_2'], b=0.4)
                light_curve2 = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=np.sqrt(all_pls[pl]['depth']), 
                                                                       t=lc['time'], texp=cad*0.98
                                                                       ).eval()
                light_curve2[abs(lc['time']-all_pls[pl]['tcen_2'])>per_guess*0.4]=0.0
                light_curve=light_curve+light_curve2
            light_curves+=[light_curve]
    return np.column_stack(light_curves)



def dopolyfit(win,mask=None,stepcent=0.0,d=3,ni=10,sigclip=3):
    mask=np.tile(True,len(win)) if mask is None else mask
    maskedwin=win[mask]
    
    #initial fit and llk:
    best_base = np.polyfit(maskedwin[:,0]-stepcent,maskedwin[:,1],w=1.0/maskedwin[:,2]**2,deg=d)
    best_offset = (maskedwin[:,1]-np.polyval(best_base,maskedwin[:,0]))**2/maskedwin[:,2]**2
    best_llk=-0.5 * np.sum(best_offset)
    
    #initialising this "random mask"
    randmask=np.tile(True,len(maskedwin))

    for iter in range(ni):
        # If a point's offset to the best model is great than a normally-distributed RV, it gets masked 
        # This should have the effect of cutting most "bad" points,
        #   but also potentially creating a better fit through bootstrapping:
        randmask=abs(np.random.normal(0.0,1.0,len(best_offset)))<best_offset
        new_base = np.polyfit(maskedwin[randmask,0]-stepcent,maskedwin[randmask,1],
                          w=1.0/np.power(maskedwin[randmask,2],2),deg=d)
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        new_offset = (maskedwin[randmask,1]-np.polyval(new_base,maskedwin[randmask,0]))**2/maskedwin[randmask,2]**2
        new_llk=-0.5 * np.sum(new_offset)
        if new_llk>best_llk:
            #If that fit is better than the last one, we update the offsets and the llk:
            best_llk=new_llk
            best_offset=new_offset[:]
            best_base=new_base[:]
    return best_base

def formwindow(dat,cent,size,boxsize,gapthresh=1.0):
    
    win = (dat[:,0]>cent-size/2.)&(dat[:,0]<cent+size/2.)
    box = (dat[:,0]>cent-boxsize/2.)&(dat[:,0]<cent+boxsize/2.)
    if np.sum(win)>0:
        high=dat[win,0][-1]
        low=dat[win,0][0]
        highgap = high < (cent+size/2.)-gapthresh
        lowgap = low > (cent-size/2.)+gapthresh

        if highgap and not lowgap:
            win = (dat[:,0] > high-size)&(dat[:,0] <= high)
        elif lowgap and not highgap:
            win = (dat[:,0] < low+size)&(dat[:,0] >= low)

        win = win&(~box)
    return win, box

def lcFlatten(lc,winsize = 3.5, stepsize = 0.15, polydegree = 2, 
              niter = 10, sigmaclip = 3., gapthreshold = 1.0,
              use_binned=False,use_mask=True,reflect=True):
    '''#Flattens any lightcurve while maintaining in-transit depth.

    Args:
    lc.           # dictionary with time,flux,flux_err, flux_unit (1.0 or 0.001 [ppt]) and mask
    winsize = 2   #days, size of polynomial fitting region
    stepsize = 0.2  #days, size of region within polynomial region to detrend
    polydegree = 3  #degree of polynomial to fit to local curve
    niter = 20      #number of iterations to fit polynomial, clipping points significantly deviant from curve each time.
    sigmaclip = 3.   #significance at which points are clipped (as niter)
    gapthreshold = 1.0  #days, threshold at which a gap in the time series is detected and the local curve is adjusted to not run over it
    '''
    winsize=3.9 if np.isnan(winsize) else winsize
    stepsize=0.15 if np.isnan(stepsize) else stepsize
    
    prefix='bin_' if use_binned else ''
    
    lc[prefix+'flux_flat']=np.zeros(len(lc[prefix+'time']))
    #general setup
    uselc=np.column_stack((lc[prefix+'time'][:],lc[prefix+'flux'][:],lc[prefix+'flux_err'][:]))
    if len(lc['mask'])==len(uselc[:,0]) and use_mask:
        initmask=(lc['mask']&(lc['flux']/lc['flux']==1.0)).astype(int)[:]
    else:
        initmask=np.ones(len(lc['time']))
    uselc=np.column_stack((uselc,initmask[:]))
    uselc[:,1:3]/=lc['flux_unit']
    uselc[:,1]-=np.nanmedian(lc[prefix+'flux'])
    
    jumps=np.hstack((0,np.where(np.diff(uselc[:,0])>winsize*0.8)[0]+1,len(uselc[:,3]) )).astype(int)
    stepcentres=[]
    uselc_w_reflect=[]
    
    for n in range(len(jumps)-1):
        stepcentres+=[np.arange(uselc[jumps[n],0],
                                uselc[np.clip(jumps[n+1],0,len(uselc)-1),0],
                                stepsize) + 0.5*stepsize]
        if reflect:
            partlc=uselc[jumps[n]:jumps[n+1]]
            incad=np.nanmedian(np.diff(partlc[:,0]))
            xx=[np.arange(np.nanmin(partlc[:,0])-winsize*0.4,np.nanmin(partlc[:,0])-incad,incad),
                np.arange(np.nanmax(partlc[:,0])+incad,np.nanmax(partlc[:,0])+winsize*0.4,incad)]
            #Adding the lc, plus a reflected region either side of each part. 
            # Also adding a boolean array to show where the reflected parts are
            refl_t=np.hstack((xx[0],partlc[:,0],xx[1]))
            refl_flux=np.vstack((partlc[:len(xx[0]),1:][::-1],
                                 partlc[:,1:], 
                                 partlc[-1*len(xx[1]):,1:][::-1]  ))
            refl_bool=np.hstack((np.zeros(len(xx[0])),np.tile(1.0,len(partlc[:,0])),np.zeros(len(xx[1]))))
            print(partlc.shape,len(xx[0]),len(xx[1]),refl_t.shape,refl_flux.shape,refl_bool.shape)
            uselc_w_reflect+=[np.column_stack((refl_t,refl_flux,refl_bool))]
    stepcentres=np.hstack(stepcentres)
    if reflect:
        uselc=np.vstack(uselc_w_reflect)
    else:
        uselc=np.column_stack((uselc,np.ones(len(uselc[:,0])) ))
    uselc[:,2]=np.clip(uselc[:,2],np.nanmedian(uselc[:,2])*0.8,100)
    print(len(uselc),np.sum(uselc[:,3]),np.sum(uselc[:,4]))
    #now for each step centre we perform the flattening:
    #actual flattening
    for s,stepcent in enumerate(stepcentres):
        win,box = formwindow(uselc,stepcent,winsize,stepsize,gapthreshold)  #should return window around box not including box
        '''print(np.sum(np.isnan(uselc[win,:3][uselc[win,3].astype(bool)])),
              np.sum(uselc[win,2]==0.0),
              np.sum(uselc[win&uselc[:,3].astype(bool),0]/uselc[win&uselc[:,3].astype(bool),0]!=1.0),
              np.sum(uselc[win&uselc[:,3].astype(bool),1]/uselc[win&uselc[:,3].astype(bool),1]!=1.0),
              np.sum(uselc[win&uselc[:,3].astype(bool),2]/uselc[win&uselc[:,3].astype(bool),2]!=1.0))'''
        newbox=box[uselc[:,4].astype(bool)]
        if np.sum(newbox&initmask)>0 and np.sum(win&uselc[:,3].astype(bool))>0:
            baseline = dopolyfit(uselc[win,:3],mask=uselc[win,3].astype(bool),
                                 stepcent=stepcent,d=polydegree,ni=niter,sigclip=sigmaclip)
            lc[prefix+'flux_flat'][newbox] = lc[prefix+'flux'][newbox] - np.polyval(baseline,lc[prefix+'time'][newbox]-stepcent)*lc['flux_unit']
    return lc
    
def DoEBfit(lc,tc,dur):
    # Performs EB fit to primary/secondary.
    return None
    
def Sinusoid_neg_lnprob(params, x, y, yerr, priors, polyorder):
    return -1*Sinusoid_lnprob(params, x, y, yerr, priors, polyorder=polyorder)

def Sinusoid_lnprob(params, x, y, yerr, priors, polyorder=2):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    for p in np.arange(3):
        #Simple log gaussian prior here:
        lnp=log_gaussian(params[p], priors[p,0], priors[p,1])
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

def Gaussian_neg_lnprob(params, x, y, yerr, priors, interpmodel, order=3):
    return -1*Gaussian_lnprob(params, x, y, yerr, priors, interpmodel, order=order)

def Gaussian_lnprob(params, x, y, yerr, priors, interpmodel, order=3):
    # Trivial improper prior: uniform in the log.
    lnprior=0
    for p in np.arange(3):
        #Simple log gaussian prior here:
        lnp=log_gaussian(params[p], priors[p,0], priors[p,1])
        if lnp<-0.5 and lnp>-20:
            lnprior+=3*lnp
        elif lnp<-20:
            lnprior+=1e6*lnp
    llk = log_likelihood_gaussian_dip(params, x, y, yerr, interpmodel)
    return lnprior + llk

def log_gaussian(x, mu, sig, weight=0.1):
    return -1*weight*np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

def log_likelihood_gaussian_dip(params, x, y, yerr, interpmodel):
    model=dipmodel_gaussian(params,x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

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
        gauss=0.0 if gauss>-0.5 else gauss
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
    xdip = 1.0+params[0]*interpmodel(t)
    ydip = 1.0+params[1]*interpmodel(t)
    xmod = np.polyval(params[2:2+(order+1)],t)*xdip
    ymod = np.polyval(params[2+(order+1):],t)*ydip
    return xmod,ymod


def AsteroidCheck(lc,monoparams,interpmodel,ID,order=3,dur_region=3.5,plot=False,plot_loc=None, **kwargs):
    # Checking lightcure for background flux boost during transit due to presence of bright asteroid
    # Performing two model fits 
    # - one with a gaussian "dip" roughly corresponding to the transit combined with a polynomial trend for out-of-transit
    # - one with only a polynomial trend
    # These are then compared, and the BIC returned to judge model fit
    
    nearTrans=(abs(lc['time']-monoparams['tcen'])<monoparams['tdur']*dur_region)&lc['mask']

    if 'bg_flux' in lc and np.sum(np.isfinite(lc['bg_flux'][nearTrans]))>0:
        # Fits two models - one with a 2D polynomial spline and the interpolated "dip" model from the fit, and one with only a spline
        
        #If there's a big gap, we'll remove the far side of that
        if np.max(np.diff(lc['time'][nearTrans]))>0.4:
            jump_n=np.argmax(np.diff(lc['time'][nearTrans]))
            jump_time=np.average(lc['time'][nearTrans][jump_n:jump_n+1])
            if jump_time < monoparams['tcen']:
                nearTrans=(lc['time']>jump_time)&((lc['time']-monoparams['tcen'])<monoparams['tdur']*dur_region)&lc['mask']
            elif jump_time > monoparams['tcen']:
                nearTrans=((lc['time']-monoparams['tcen'])>(-1*monoparams['tdur']*dur_region))&(lc['time']<jump_time)&lc['mask']
        
        bg_lc=np.column_stack((lc['time'][nearTrans],
                               lc['bg_flux'][nearTrans],
                               np.tile(np.nanstd(lc['bg_flux'][nearTrans]),np.sum(nearTrans))
                              ))
        bg_lc=bg_lc[np.isfinite(np.sum(bg_lc,axis=1))]
        bg_lc[:,0]-=monoparams['tcen']
        outTransit=(abs(bg_lc[:,0])>monoparams['tdur']*0.75)
        inTransit=(abs(bg_lc[:,0])<monoparams['tdur']*0.35)
        #print(bg_lc)
        bg_lc[:,1:]/=np.nanmedian(bg_lc[outTransit,1])
        #print(bg_lc)
        log_height_guess=np.log(2*np.clip((np.nanmedian(bg_lc[inTransit,1])-np.nanmedian(bg_lc[outTransit,1])),0.00001,1000))
        
        priors= np.column_stack(([log_height_guess,np.log(monoparams['tdur']),0.0],[3.0,0.5,0.75*monoparams['tdur']]))
        best_nodip_res={'fun':1e30,'bic':1e9}

        best_dip_res={'fun':1e30,'bic':1e9}
        
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        n=0
        while n<21:
            #Running the minimization 7 times with different initial params to make sure we get the best fit
            
            #non-dip is simple poly fit. Including 10% cut in points to add some randomness over n samples
            
            rand_choice=np.random.random(len(bg_lc))<0.9
            nodip_res={'x':np.polyfit(bg_lc[rand_choice,0],bg_lc[rand_choice,1],order)}
            sigma2 = bg_lc[:,2]**2
            nodip_res['fun'] = -1*(-0.5 * np.sum((bg_lc[:,1] - np.polyval(nodip_res['x'],bg_lc[:,0])) ** 2 / sigma2 + np.log(sigma2)))

            #BIC = 2*(neg_log_likelihood-log_prior) + log(n_points)*n_params 
            nodip_res['bic']=(2*nodip_res['fun'] + np.log(len(bg_lc[:,1]))*len(nodip_res['x']))
            #print('no dip:',nodip_args,nodip_res,nodip_bic)
            if nodip_res['bic']<best_nodip_res['bic']:
                best_nodip_res=nodip_res
            
            #log10(height), log10(dur), tcen
            dip_args= np.hstack(([np.random.normal(log_height_guess,0.25)-(n%4)/2.0,
                                  np.log10(1.5*monoparams['tdur'])+abs(np.random.normal(0.0,0.5)),
                                  np.random.normal(0.0,0.5*monoparams['tdur'])],
                                  np.polyfit(bg_lc[outTransit&rand_choice,0],
                                             bg_lc[outTransit&rand_choice,1],order)))

            dip_res=optim.minimize(Gaussian_neg_lnprob, dip_args,
                                     args=(bg_lc[:,0],
                                           bg_lc[:,1],
                                           bg_lc[:,2],
                                           priors,
                                           interpmodel,
                                           order),method=methods[n%3])
            dip_res['bic']=(2*dip_res.fun + np.log(len(bg_lc[:,1]))*len(dip_args))
            
            #print('dip:',dip_args,dip_res,dip_bic)
            if dip_res['bic']<best_dip_res['bic']:
                best_dip_res=dip_res
                
            #Increasing the polynomial order every odd number if we haven't yet found a good solution:
            if n>7 and n%2==1:
                order+=1
                #Need to expand on priors to include new polynomial orders in this case:
            #print(n,"dip:",dip_res['fun'],dip_res['bic'],dip_args,"nodip:",nodip_res['fun'],nodip_res['bic'],"order:",order)

            if n>=7 and (best_dip_res['fun']<1e30) and (best_nodip_res['fun']<1e30):
                break
            n+=1
            
        if (best_dip_res['fun']<1e30) and (best_nodip_res['fun']<1e30):
            DeltaBIC=best_dip_res['bic']-best_nodip_res['bic']
            #print(ID,"| Ran Asteroid fitting "+str(n)+" times, and DeltaBIC="+str(DeltaBIC),
            #      "| params:",best_dip_res['x'],best_nodip_res['x'])
        
        if plot:
            if plot_loc is not None and type(plot_loc)!=str:
                ax = plot_loc
            else:
                fig = plt.figure(figsize=(8,8))
                ax = plt.add_subplot(111)
                if plot_loc is None:
                    plot_loc=str(ID)+"_asteroid_check.pdf"
                elif plot_loc[-1]=='/':
                    plot_loc=plot_loc+str(ID)+"_asteroid_check.pdf"
                print("Plotting asteroid",ID, np.sum(bg_lc[:,1]/bg_lc[:,1]==1.0),len(bg_lc[:,1]),plot_loc,best_nodip_res)
            if plot:
                #### PLOTTING ###
                ax.scatter(bg_lc[:,0],bg_lc[:,1],s=2,alpha=0.75,rasterized=True)

                ax.plot([-0.5*monoparams['tdur'],-0.5*monoparams['tdur']],[0.0,2.0],':k',alpha=0.6,rasterized=True)
                ax.plot([0.0,0.0],[0.0,2.0],'--k',linewidth=3,alpha=0.8,rasterized=True)
                ax.plot([0.5*monoparams['tdur'],0.5*monoparams['tdur']],[0.0,2.0],':k',alpha=0.6,rasterized=True)
                ax.set_ylabel("Relative background flux")

                if (best_nodip_res['fun']<1e30):
                    ax.plot(bg_lc[:,0],
                             np.polyval(best_nodip_res['x'],bg_lc[:,0]),
                             label='pure trend',alpha=0.75,rasterized=True)
                if (best_dip_res['fun']<1e30):
                    ax.plot(bg_lc[:,0],
                             dipmodel_gaussian(best_dip_res.x,bg_lc[:,0]),
                             label='trend+asteroid',alpha=0.75,rasterized=True)
                try:
                    ax.set_ylim(np.nanmin(bg_lc[:,1]),np.nanmax(bg_lc[:,1]))
                except:
                    b=0
                #plt.ylim(np.percentile(bg_lc[inTransit,1],[0.2,99.8]))
                ax.legend()
                if (best_dip_res['fun']<1e30) and (best_nodip_res['fun']<1e30):
                    ax.set_title(str(ID)+" Asteroid. - "+["pass","fail"][int(DeltaBIC<0)])
                else:
                    ax.set_title(str(ID)+" Asteroid. No fit ???")
                if type(plot_loc)==str:
                    fig.savefig(plot_loc, dpi=400)
                return DeltaBIC, ax

                return DeltaBIC, ax
        elif (best_dip_res['fun']<1e30) and (best_nodip_res['fun']<1e30):
            return DeltaBIC, None
        else:
            return 0.0, None
    else:
        return 0.0, None

def VariabilityCheck(lc,params,ID,plot=False,plot_loc=None,ndurs=2.4, polyorder=1, **kwargs):
    # Checking lightcure for variability flux boost during transit due to presence of bright asteroid
    # Performing variability model fits 
    # the BIC returned to judge against the transit model fit
    
    #assuming QuickMonoFit has been run, we can replicate the exact x/y/yerr used there:
    x = params['monofit_x']-params['tcen']
    round_trans=(abs(x)<ndurs*params['tdur'])
    x=x[round_trans]
    y = params['monofit_y'][round_trans]
    yerr = params['monofit_yerr'][round_trans]
    y_trans = params['monofit_ymodel'][round_trans]
    
    
    outTransit=abs(x)>0.7*params['tdur']
    yspan=np.diff(np.percentile(y,[5,95]))[0]
    priors= np.column_stack(([0.0,np.log(params['tdur']),np.log(yspan)],
                             [0.5*params['tdur'],3.0,4.0]))
    best_sin_res={'fun':1e30,'bic':1e9,'sin_llk':-1e9}

    methods=['L-BFGS-B','Nelder-Mead','Powell']
    n=0
    while n<21:
        #Running the minimization 7 times with different initial params to make sure we get the best fit
        
        rand_choice=np.random.random(len(x))<0.9
        #non-dip is simple poly fit. Including 10% cut in points to add some randomness over n samples
        #log10(height), log10(dur), tcen
        sin_args= np.hstack(([np.random.normal(0.0,0.5*params['tdur']),
                              np.log(params['tdur'])+np.random.normal(0.0,0.5)],
                              np.log(params['depth'])+np.random.normal(0.0,0.5),
                              np.polyfit(x[outTransit&rand_choice],
                                         y[outTransit&rand_choice],polyorder)
                            ))

        sin_res=optim.minimize(Sinusoid_neg_lnprob, sin_args,
                               args=(x[np.argsort(x)],y[np.argsort(x)],yerr[np.argsort(x)],priors,polyorder),
                               method=methods[n%3])
        
        sin_res['bic']=(2*sin_res.fun + np.log(len(x))*len(sin_res.x))
        sin_res['sin_llk']=-1*sin_res.fun
        
        #print('dip:',dip_args,dip_res,dip_bic)
        if sin_res['bic']<best_sin_res['bic']:
            best_sin_res=sin_res
        
        #Increasing the polynomial order every odd number if we haven't yet found a good solution:
        if n>7 and n%2==1:
            polyorder+=1
            #Need to expand on priors to include new polynomial orders in this case:
        #print(n,"dip:",dip_res['fun'],dip_res['bic'],dip_args,"nodip:",nodip_res['fun'],nodip_res['bic'],"order:",order)
        
        if n>=7 and (best_sin_res['fun']<1e30):
            break
        n+=1
    
    best_sin_res['trans_llk']= -0.5 * np.sum((y - y_trans)**2 / yerr**2 + 2*np.log(yerr))
    
    best_sin_res['llk_ratio']=best_sin_res['sin_llk'] - best_sin_res['trans_llk']
    
    #print("sin:", best_sin_res['sin_llk'], "trans:", best_sin_res['trans_llk'], "llk_ratio:", best_sin_res['llk_ratio'])
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
        
        markers, caps, bars = ax.errorbar(x,y,yerr=yerr,fmt='.k',ecolor='#DDDDDD',markersize=2.5,alpha=0.6,rasterized=True)
        [bar.set_alpha(0.2) for bar in bars]
        [cap.set_alpha(0.2) for cap in caps]

        ax.plot([-0.5*params['tdur'],-0.5*params['tdur']],[0.0,2.0],':k',alpha=0.6,zorder=2,rasterized=True)
        ax.plot([0.0,0.0],[0.0,2.0],'--k',linewidth=3,alpha=0.8,zorder=2,rasterized=True)
        ax.plot([0.5*params['tdur'],0.5*params['tdur']],[0.0,2.0],':k',alpha=0.6,zorder=2,rasterized=True)
        if lc['flux_unit']==0.001:
            ax.set_ylabel("Relative flux [ppm]")
        elif lc['flux_unit']==1:
            ax.set_ylabel("Relative flux [ppm]")
        if (best_sin_res['fun']<1e30):
            ax.plot(x[np.argsort(x)],dipmodel_sinusoid(best_sin_res['x'],x[np.argsort(x)]),
                     label='sinusoid',linewidth=3.0,zorder=10,rasterized=True)
        
        ax.plot(x,y_trans,label='transit',linewidth=3.0,zorder=11)
        
        try:
            ax.set_ylim(np.nanmin(y),np.nanmax(y))
        except:
            b=0
        #plt.ylim(np.percentile(bg_lc[inTransit,1],[0.2,99.8]))
        ax.legend()
        if (best_sin_res['fun']<1e30):
            ax.set_title(str(ID)+" Variability - "+["pass","fail"][int(best_sin_res['llk_ratio']>0)])
        else:
            ax.set_title(str(ID)+" Variability. No fit ???")
        if plot_loc is not None and type(plot_loc)==str:
            fig.savefig(plot_loc, dpi=400)
            print("Saved varble plot to",plot_loc)
            return best_sin_res['llk_ratio'], plot_loc
        else:
            return best_sin_res['llk_ratio'], ax
    elif (best_sin_res['fun']<1e30):
        return best_sin_res['llk_ratio'], None
    else:
        return 0.0, None

def CheckInstrumentalNoise(lc,monodic,jd_base=None, **kwargs):
    '''# Using the processed "number of TCEs per cadence" array, we try to use this as a proxy for Instrumental noise in TESS
    # Here we simply use the detected SNR over the instrumental noise SNR as a proxy
    INPUTS:
    - lc
    - monotransit dic
    - jd_base (assumed to be that of TESS)'''
    tces_per_cadence=np.loadtxt(MonoFit.MonoFit_path+'/data/tables/tces_per_cadence.txt')
    if 'jd_base' in lc and jd_base is None:
        jd_base=lc['jd_base']
    elif jd_base is None:
        jd_base=2457000
    tces_per_cadence[:,0]-=(jd_base-2457000)
    #print(jd_base,tces_per_cadence[0,0],tces_per_cadence[-1,0], np.nanmin(lc['time']),np.nanmax(lc['time']))
    tces_per_cadence=tces_per_cadence[(tces_per_cadence[:,0]>np.nanmin(lc['time']))*(tces_per_cadence[:,0]<np.nanmax(lc['time']))]
    inst_snr=1+np.average(tces_per_cadence[abs(tces_per_cadence[:,0]-monodic['tcen'])<monodic['tdur'],1])
    return monodic['snr']/np.clip(inst_snr,1.0,1000)

def GapCull(t0,t,dat,std_thresh=7,boolean=None,time_jump_thresh=0.4):
    #Removes data before/after gaps and jumps in t & y
    #If there's a big gap or a big jump, we'll remove the far side of that
    if boolean is None:
        boolean=np.tile(True,len(t))
    if np.max(np.diff(t[boolean]))>time_jump_thresh:
        jump_n=np.argmax(np.diff(t[boolean]))
        jump_time=np.average(t[boolean][jump_n:jump_n+1])
        #print("TIME JUMP IN CENTROID AT",jump_time)
        if jump_time < t0:
            boolean*=(t>jump_time)
        elif jump_time > t0:
            boolean*=(t<jump_time)
    #data must be iterable list
    for arr in dat:
        noise=np.nanmedian(abs(np.diff(arr[boolean])))
        #5-sigma x centroid jump - need to cut
        if len(arr[boolean])>0 and np.nanmax(np.diff(arr[boolean]))>std_thresh*noise:
            jump_n=np.argmax(np.diff(arr[boolean]))
            jump_time=np.average(t[boolean][jump_n:jump_n+1])
            #print("X JUMP IN CENTROID AT",jump_time)
            if jump_time < t0:
                boolean*=(t>jump_time)
            elif jump_time > t0:
                boolean*=(t<jump_time)
    return boolean

def CentroidCheck(lc,monoparams,interpmodel,ID,order=2,dur_region=3.5, plot=True,plot_loc=None, **kwargs):
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
        if len(x)>0 and len(y)>0:
            xerr=np.std(x)
            x-=np.median(x)
            if len(x[inTransit])>0 and len(x[outTransit])>0:
                xdep_guess=np.median(x[inTransit])-np.median(x[outTransit])
                init_poly_x=np.polyfit(t[outTransit],x[outTransit],order)
            else:
                xdep_guess=0.0
                if len(x[inTransit])>0:
                    init_poly_x=np.polyfit(t,x,order)
                else:
                    init_poly_x=np.zeros(polyorder+1)
                
            y-=np.median(y)
            yerr=np.std(y)
            if len(y[inTransit])>0 and len(y[outTransit])>0:
                ydep_guess=np.median(y[inTransit])-np.median(y[outTransit])
                init_poly_y=np.polyfit(t[outTransit],y[outTransit],order)
            else:
                ydep_guess=0.0
                if len(y[inTransit])>0:
                    init_poly_y=np.polyfit(t,y,order)
                else:
                    init_poly_y=np.zeros(polyorder+1)

        priors= np.column_stack(([0,0],[abs(xerr)*5,abs(yerr)*5]))
        
        best_nodip_res={'fun':1e30,'bic':1e6}
        best_dip_res={'fun':1e30,'bic':1e6}
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        for n in range(7):
            #Doing simple polynomial fits for non-dips. Excluding 10% of data each time to add some randomness
            rand_choice=np.random.choice(len(x),int(0.9*len(x)),replace=False)
            xfit=np.polyfit(t[rand_choice],x[rand_choice],order)
            xsigma2=xerr**2
            xlogprob=-0.5 * np.sum((x - np.polyval(xfit,t)) ** 2 / xsigma2 + np.log(xsigma2))
            yfit=np.polyfit(t[rand_choice],y[rand_choice],order)
            ysigma2=yerr**2
            ylogprob=-0.5 * np.sum((y - np.polyval(yfit,t)) ** 2 / ysigma2 + np.log(ysigma2))
            nodip_res={'fun':-1*(xlogprob+ylogprob)}
            nodip_res['x']=[xfit,yfit]
            nodip_res['bic']=2*nodip_res['fun'] + np.log(np.sum(roundTransit))*(len(xfit)+len(yfit))
            if nodip_res['bic']<best_nodip_res['bic']:
                best_nodip_res=nodip_res

            dip_args= np.hstack(([np.random.normal(xdep_guess,abs(0.25*xdep_guess)),
                                  np.random.normal(ydep_guess,abs(0.25*ydep_guess))],
                                 init_poly_x,init_poly_y ))

            dip_res=optim.minimize(centroid_neg_lnprob, dip_args,
                                     args=(t,x,y,xerr,yerr,
                                           priors,
                                           interpmodel,
                                           order),method=methods[n%3])

            dip_res['bic']=2*dip_res.fun + np.log(np.sum(roundTransit))*len(dip_res['x'])
            if dip_res['bic']<best_dip_res['bic']:
                best_dip_res=dip_res

        #Computing difference in Bayesian Information Criterium - DeltaBIC -  between "dip" and "no dip models"
        DeltaBIC=best_dip_res['bic']-best_nodip_res['bic']
        #print("with centroid:",best_dip_res,"| without:",best_nodip_res)
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
            if best_dip_res['fun']<1e29 and best_nodip_res['fun']<1e29 and len(best_nodip_res['x'])==2:
                ax.plot(t,np.polyval(best_nodip_res['x'][0],t),label='pure trend - x',rasterized=True)
                ax.plot(t,np.polyval(best_nodip_res['x'][1],t),label='pure trend - y',rasterized=True)
                ax.plot(t,dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[0],label='trend+centroid - x',rasterized=True)
                ax.plot(t,dipmodel_centroid(best_dip_res.x,t,interpmodel,order)[1],label='trend+centroid - y',rasterized=True)
                ax.legend()

                ax.set_title(str(ID)+" Centroid  - "+["pass","fail"][int(DeltaBIC<2)])
            else:
                ax.set_title(str(ID)+" Centroid  - No fit ???")
            
            xlim=np.percentile(x,[0.2,99.8])
            ylim=np.percentile(y,[0.2,99.8])
            ax.set_ylim(np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]]))
            ax.set_xlim(np.min(t),np.max(t))

            if plot_loc is not None and type(plot_loc)==str:
                fig.savefig(plot_loc, dpi=400)
                return DeltaBIC, plot_loc
            else:
                return DeltaBIC, ax
        elif not plot:
            return DeltaBIC, None
    else:
        return None

    
def CheckMonoPairs(lc_time, all_pls,prox_thresh=3.0):
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
                            print("tcens = ",all_pls[m1]['tcen'],all_pls[m2]['tcen'])
                            newm1['period']=abs(all_pls[m1]['tcen']-all_pls[m2]['tcen'])
                        elif key=='snr':
                            newm1['snr']=np.hypot(all_pls[m1]['snr'],all_pls[m2]['snr'])
                        elif type(all_pls[m2][key])==float and key!='tcen':
                            #Average of two:
                            #print(key,all_pls[m1][key],all_pls[m2][key],0.5*(all_pls[m1][key]+all_pls[m2][key]))
                            newm1[key]=0.5*(all_pls[m1][key]+all_pls[m2][key])
                newm1['tcen_2']=all_pls[m2]['tcen']
                newm1['orbit_flag']='duo'
                check_pers = newm1['period']/np.arange(1,np.ceil(newm1['period']/10),1.0)
                check_pers_ix=np.tile(False,len(check_pers))
                Npts_from_known_transits=np.sum(abs(lc_time-newm1['tcen'])<0.3*newm1['tdur']) + \
                                         np.sum(abs(lc_time-newm1['tcen_2'])<0.3*newm1['tdur'])
                for nper,per in enumerate(check_pers):
                    phase=(lc_time-newm1['tcen']-per*0.5)%per-per*0.5
                    Npts_in_tr=np.sum(abs(phase)<0.3*newm1['tdur'])
                    check_pers_ix[nper]=Npts_in_tr<1.075*Npts_from_known_transits #Less than 15% of another eclipse is covered
                newm1['period_aliases']=check_pers[check_pers_ix]
                P_mins=check_pers[check_pers_ix]
                newm1['P_min']=P_mins if type(P_mins)==float else np.min(P_mins)
                print("period aliases:",newm1['period_aliases'],"P_min:",newm1['P_min'])
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


def get_interpmodels(Rs,Ms,Teff,lc_time,lc_flux_unit,mission='tess',n_durs=3,texp=None):
    #Uses radius, mass and lightcurve duration to create fake transit models to use in monotransit search
    
    if texp is None:
        texp=np.nanmedian(np.diff(lc_time))
    
    u_star = tools.getLDs(Teff,logg=np.log10(Ms/Rs**2)+4.431,FeH=0.0)[0]
    
    #Computing monotransit minimum P from which to estimate durations:
    cadence=np.nanmedian(np.diff(lc_time))
    jumps=np.hstack((0,np.where(np.diff(lc_time)>5.0)[0],len(lc_time)-1))
    P_guess=np.clip(lc_time[jumps[1+np.argmax(np.diff(jumps))]]-lc_time[jumps[np.argmax(np.diff(jumps))]],5.0,250)

    #print(jumps,jumps[np.argmax(np.diff(jumps))],jumps[1+np.argmax(np.diff(jumps))],P_guess)

    # Orbit models
    per_steps=np.logspace(np.log10(0.4),np.log10(2.5),n_durs)
    b_steps=np.linspace(0.85,0,n_durs)
    orbits = xo.orbits.KeplerianOrbit(r_star=Rs,m_star=Ms,period=P_guess*per_steps,t0=np.tile(0.0,n_durs),b=b_steps)

    vx, vy, vz = orbits.get_relative_velocity(0.0)
    tdurs=((2*1.1*Rs*np.sqrt(1-b_steps**2))/tt.sqrt(vx**2 + vy**2)).eval().ravel()

    # Compute the model light curve using starry
    interpt=np.linspace(-0.6*np.max(tdurs),0.6*np.max(tdurs),600)

    ys=xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbits, r=np.tile(0.1*Rs,n_durs), 
                                                     t=interpt, texp=texp
                                                     ).eval()/lc_flux_unit
    interpmodels=[]
    for row in range(n_durs):
        interpmodels+=[interpolate.interp1d(interpt,ys[:,row],bounds_error=False,fill_value=(0.0,0.0))]

    return interpmodels,tdurs

def MonoVetting(ID, mission, tcen=None, tdur=None, overwrite=False, do_search=True,
                useL2=False,PL_ror_thresh=0.2,variable_llk_thresh=5,file_loc=None,
                plot=False,do_fit=False,re_vet=False,re_fit=False,**kwargs):
    '''#Here we're going to initialise the Monotransit fitting by searching for other planets/transits/secondaries and filtering out binaries.
    INPUTS:
    - ID
    - mission
    - useL2=False
    - PL_ror_thresh=0.2
    - re_vet = redo the vetting stage
    
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
    if file_loc is None:
        #Creating a ID directory in the current directory for the planet fits/docs
        file_loc=tools.id_dic[mission]+str(ID).zfill(11)
    elif file_loc[-1]=='/':
        #If we're given a directory, we'll create a ID directory in there for the planet fits/docs
        file_loc=file_loc+tools.id_dic[mission]+str(ID).zfill(11)
    kwargs['file_loc']=file_loc
    
    mono_SNR_thresh=6.5 if 'mono_SNR_thresh' not in kwargs else kwargs['mono_SNR_thresh']

    if not os.path.isdir(file_loc):
        os.system('mkdir '+file_loc)
    
    #Initialising figures
    if plot:
        figs={}

    if 'StarPars' not in kwargs:
        #loading Rstar,Tess, logg and rho from csvs:
        if not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_starpars.csv') or overwrite:
            from stellar import starpars
            #Gets stellar info
            info,_=starpars.getStellarInfoFromCsv(ID,mission)
            info.to_csv(file_loc+"/"+file_loc.split('/')[-1]+'_starpars.csv')
        else:
            info=pd.read_csv(file_loc+"/"+file_loc.split('/')[-1]+'_starpars.csv', index_col=0, header=0).T.iloc[0]
        
        print(info.index)
        Rstar=[float(info['rad']),float(info['eneg_Rad']),float(info['epos_Rad'])]
        Teff=[float(info['Teff']),float(info['eneg_Teff']),float(info['epos_Teff'])]
        logg=[float(info['logg']),float(info['eneg_logg']),float(info['epos_logg'])]
        rhostar=[float(info['rho'])/1.411,float(info['eneg_rho'])/1.411,float(info['epos_rho'])/1.411]
        if 'mass' in info:
            Ms=float(info['mass'])
        else:
            Ms=rhostar[0]*Rstar[0]**3
        #Rstar, rhostar, Teff, logg, src = starpars.getStellarInfo(ID, hdr, mission, overwrite=overwrite,
        #                                                         fileloc=savenames[1].replace('_mcmc.pickle','_starpars.csv'),
        #                                                         savedf=True)
    else:
        Rstar, rhostar, Teff, logg = StarPars
        
    #opening lightcurve:
    if not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_lc.pickle') or overwrite:
        #Gets Lightcurve
        lc,hdr=tools.openLightCurve(ID,mission,use_ppt=False)
        pickle.dump(lc,open(file_loc+"/"+file_loc.split('/')[-1]+'_lc.pickle','wb'))
        #lc=lcFlatten(lc,winsize=9*tdur,stepsize=0.1*tdur)
    else:
        lc=pickle.load(open(file_loc+"/"+file_loc.split('/')[-1]+'_lc.pickle','rb'))
    
    # DOING MONOTRANSIT PLANET SEARCH:
    if (not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_monos.pickle') or overwrite) and do_search:
        mono_dic, monosearchparams, monofig = MonoTransitSearch(deepcopy(lc),ID,
                                                                Rs=Rstar[0],Ms=Ms,Teff=Teff[0],
                                                                plot_loc=file_loc+"/", plot=plot,**kwargs)
        figs['mono']=monofig
        pickle.dump(mono_dic,open(file_loc+"/"+file_loc.split('/')[-1]+'_monos.pickle','wb'))
    elif do_search:
        mono_dic=pickle.load(open(file_loc+"/"+file_loc.split('/')[-1]+'_monos.pickle','rb'))
        if plot and os.path.isfile(file_loc+"/"+str(ID).zfill(11)+'_Monotransit_Search.pdf'):
            figs['mono'] = file_loc+"/"+str(ID).zfill(11)+'_Monotransit_Search.pdf'
    elif not do_search:
        intr=lc['mask']&(abs(lc['time']-tcen)<0.45*tdur)
        outtr=lc['mask']&(abs(lc['time']-tcen)<1.25*tdur)&(~intr)
        mono_dic={'00':{'tcen':tcen,'tdur':tdur,'orbit_flag':'mono','poly_DeltaBIC':0.0,
                        'depth':np.nanmedian(lc['flux'][outtr])-np.nanmedian(lc['flux'][intr])}}
    #print("monos:",{pl:{'tcen':mono_dic[pl]['tcen'],'depth':mono_dic[pl]['depth']} for pl in mono_dic})
    
    # DOING PERIODIIC PLANET SEARCH:
    if (not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_multis.pickle') or overwrite) and do_search:
        both_dic, perfig = PeriodicPlanetSearch(deepcopy(lc),ID,deepcopy(mono_dic),plot_loc=file_loc+"/",plot=plot, **kwargs)
        figs['multi']=perfig
        pickle.dump(both_dic,open(file_loc+"/"+file_loc.split('/')[-1]+'_multis.pickle','wb'))
    elif do_search:
        both_dic=pickle.load(open(file_loc+"/"+file_loc.split('/')[-1]+'_multis.pickle','rb'))
        if plot and os.path.isfile(file_loc+"/"+str(ID).zfill(11)+'_multi_search.pdf'):
            figs['multi']= file_loc+"/"+str(ID).zfill(11)+'_multi_search.pdf'
    else:
         both_dic=mono_dic
    # VETTING DETECTED CANDIDATES:
    #print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag']} for pl in both_dic})
    if len(both_dic)>0:
        if not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_allpls.pickle') or overwrite or re_vet:
            for pl in both_dic:
                #Best-fit model params for the mono transit:
                if both_dic[pl]['orbit_flag']=='mono':
                    monoparams, interpmodel = QuickMonoFit(deepcopy(lc),both_dic[pl]['tcen'],both_dic[pl]['tdur'],
                                                           Rs=Rstar[0],Ms=Ms,Teff=Teff[0],how='mono')
                    if monoparams['model_success']=='False':
                        #Redoing without fitting the polynomial if the fit fails:
                        monoparams, interpmodel = QuickMonoFit(deepcopy(lc),both_dic[pl]['tcen'],both_dic[pl]['tdur'],
                                                               Rs=Rstar[0],Ms=Ms,Teff=Teff[0],how='mono',fit_poly=False)

                    monoparams['interpmodel']=interpmodel

                    #Keeping detection tcen/tdur:
                    monoparams['init_tdur']=both_dic[pl]['tdur']
                    monoparams['init_depth']=both_dic[pl]['depth']
                    monoparams['orbit_flag']='mono'
                elif both_dic[pl]['orbit_flag']=='periodic':
                    monoparams,interpmodel = QuickMonoFit(deepcopy(lc),both_dic[pl]['tcen'],
                                                  both_dic[pl]['tdur'], init_period=both_dic[pl]['period'],how='periodic',
                                                  Teff=Teff[0],Rs=Rstar[0],Ms=Ms)
                    for col in ['period','vrel']:
                        par = monoparams.pop(col) #removing things which may spoil the periodic detection info (e.g. derived period)
                        both_dic[pl][col+'_mono']=par
                    if (((both_dic[pl]['period_mono']/both_dic[pl]['period'])<0.1)|((both_dic[pl]['period_mono']/both_dic[pl]['period'])>10)):
                            #Density discrepancy of 10x, likely not possible on that period
                            both_dic[pl]['flag']='discrepant duration'

                #update dic:
                both_dic[pl].update(monoparams)
                both_dic[pl]['flag']='planet'
                if both_dic[pl]['snr']<mono_SNR_thresh or both_dic[pl]['snr_r']<(mono_SNR_thresh*0.5):
                    both_dic[pl]['flag']='lowSNR'
                    print(pl,"lowSNR, SNR=",both_dic[pl]['snr'],"SNR_r=",both_dic[pl]['snr_r'],"depth:",both_dic[pl]['depth'],"fit:",both_dic[pl]["model_success"])
                elif both_dic[pl]['b']>0.98 and both_dic[pl]['snr']<(mono_SNR_thresh+2):
                    both_dic[pl]['flag']='lowSNR/V-shaped'
                    print(pl,"lowSNR, SNR=",both_dic[pl]['snr'],"SNR_r=",both_dic[pl]['snr_r'],"b=",both_dic[pl]['b'])
                else:

                    if plot:
                        if both_dic[pl]['orbit_flag']=='mono':
                            vetfig=plt.figure(figsize=(11.69,3.25))
                            var_ax=vetfig.add_subplot(131)
                            ast_ax=vetfig.add_subplot(132)
                            cent_ax=vetfig.add_subplot(133)
                        elif both_dic[pl]['orbit_flag']=='periodic':
                            vetfig=plt.figure(figsize=(8.2,3.25))
                            var_ax=vetfig.add_subplot(121)
                            cent_ax=vetfig.add_subplot(122)

                    #Compares log likelihood of variability fit to that for transit model
                    #print("doing variability check")
                    both_dic[pl]['variableLogLik'],varfig=VariabilityCheck(deepcopy(lc), both_dic[pl], plot=plot,
                                                                    ID=str(ID).zfill(11)+'_'+pl, plot_loc=var_ax, **kwargs)
                    #>1 means variability fits the data ~2.7 times better than a transit.
                    if both_dic[pl]['variableLogLik']>variable_llk_thresh:
                        both_dic[pl]['flag']='variability'
                        print(pl,"variability. LogLik=",both_dic[pl]['variableLogLik'])
                
                    if both_dic[pl]['orbit_flag']=='mono':
                        #Checks to see if dip is due to background asteroid and if that's a better fit than the transit model:
                        both_dic[pl]['asteroidDeltaBIC'], astfig=AsteroidCheck(deepcopy(lc), both_dic[pl], 
                                                                               interpmodel, plot=plot,
                                                                               ID=str(ID).zfill(11)+'_'+pl, 
                                                                               plot_loc=ast_ax, **kwargs)
                        if both_dic[pl]['asteroidDeltaBIC'] is not None and both_dic[pl]['asteroidDeltaBIC']<both_dic[pl]['poly_DeltaBIC']:
                            both_dic[pl]['flag']='asteroid'
                            print(pl,"asteroid. DeltaBic=",both_dic[pl]['asteroidDeltaBIC'])
                    
                    #Checks to see if dip is combined with centroid
                    both_dic[pl]['centroidDeltaBIC'],centfig = CentroidCheck(deepcopy(lc), both_dic[pl], interpmodel, plot=plot,
                                                                   ID=str(ID).zfill(11)+'_'+pl, plot_loc=cent_ax, **kwargs)
                    if both_dic[pl]['centroidDeltaBIC'] is not None and both_dic[pl]['centroidDeltaBIC']<-10:
                        both_dic[pl]['flag']='EB'
                        print(pl,"EB - centroid. DeltaBic=",both_dic[pl]['centroidDeltaBIC'])
                    
                    both_dic[pl]['instrumental_snr_ratio']=CheckInstrumentalNoise(deepcopy(lc),both_dic[pl])
                    if both_dic[pl]['snr']>mono_SNR_thresh and both_dic[pl]['instrumental_snr_ratio']<(mono_SNR_thresh*0.66):
                        both_dic[pl]['flag']='instrumental'
                        print(pl,"planet SNR / instrumental SNR =",both_dic[pl]['instrumental_snr_ratio'])

                    '''if both_dic[pl]['flag'] in ['asteroid','EB','instrumental']:
                        monos.remove(pl)'''
                    if both_dic[pl]['orbit_flag']=='mono':
                        print(str(pl)+" - Checks complete.",
                              " SNR:",str(both_dic[pl]['snr'])[:8],
                              " SNR_r:",str(both_dic[pl]['snr_r'])[:8],
                              " variability:",str(both_dic[pl]['variableLogLik'])[:8],
                              " centroid:",str(both_dic[pl]['centroidDeltaBIC'])[:8],
                              "| flag:",both_dic[pl]['flag'])
                    elif both_dic[pl]['orbit_flag']=='periodic':
                        print(pl,"Checks complete.",
                              " SNR:",str(both_dic[pl]['snr'])[:8],
                              " SNR_r:",str(both_dic[pl]['snr_r'])[:8],
                              " variability:",str(both_dic[pl]['variableLogLik'])[:8],
                              " centroid:",str(both_dic[pl]['centroidDeltaBIC'])[:8],
                              "| flag:",both_dic[pl]['flag'])

                    if plot:
                        #Attaching all our subplots and savings
                        #vetfig.tight_layout()
                        vetfig.subplots_adjust(left = 0.05,right = 0.97,bottom = 0.075,top = 0.925)
                        vetfig.savefig(file_loc+"/"+str(ID).zfill(11)+'_'+pl+'_vetting.pdf', dpi=400)
                        figs[pl]=file_loc+"/"+str(ID).zfill(11)+'_'+pl+'_vetting.pdf'
                if 'flag' not in both_dic[pl]:
                    both_dic[pl]['flag']='planet'

            # Removing any monos or multis which are confused (e.g. a mono which is in fact in a multi)
            both_dic,monos,multis = CheckPeriodConfusedPlanets(deepcopy(lc),deepcopy(both_dic))
            print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})
            #Check pairs of monos for potential match and period:
            both_dic = CheckMonoPairs(lc['time'], deepcopy(both_dic))
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
                    if 'rp_rs' in both_dic[pl] and both_dic[pl]['rp_rs']>PL_ror_thresh:
                        #Likely EB
                        both_dic[pl]['flag']='EB'
            pickle.dump(both_dic,open(file_loc+"/"+file_loc.split('/')[-1]+'_allpls.pickle','wb'))

        else:
            both_dic = pickle.load(open(file_loc+"/"+file_loc.split('/')[-1]+'_allpls.pickle','rb'))
            monos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='mono']
            duos=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='duo']
            multis=[pl for pl in both_dic if both_dic[pl]['orbit_flag']=='periodic']
            if plot:
                for obj in both_dic:
                    initname=file_loc+"/"+str(ID).zfill(11)+'_'+obj
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
                    if type(both_dic[obj][key]) not in [float,int,str,np.float64,np.float32] or (type(both_dic[obj][key])=='str' and len(both_dic[obj][key])>100):
                        complexkeys+=[key]
                    else:
                        if key is 'ID':
                            ser[key]=int(both_dic[obj][key])
                        elif type(both_dic[obj][key]) in [str,int]:
                            ser[key]=both_dic[obj][key]
                        elif type(both_dic[obj][key]) in [float,np.float64,np.float32]:
                            ser[key]=np.round(both_dic[obj][key],4)
            df=df.append(ser)
            #if str(ID).zfill(11)+'_'+obj not in all_cands_df.index or overwrite:
            #    all_cands_df=all_cands_df.append(ser[[ind for ind in ser.index if ind not in complexkeys]])
        print(df.columns)
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
            if os.path.exists(file_loc+"/"+file_loc.split('/')[-1]+'_candidates.csv') and not re_vet:
                new_df=False
                old_df=pd.read_csv(file_loc+"/"+file_loc.split('/')[-1]+'_candidates.csv')
                
                print(old_df.iloc[0],df.iloc[0])
                #print(old_df.values,df.values, old_df.values!=df.values)
                if ((type(old_df.values!=df.values)==bool)&(old_df.values!=df.values)) or ((type(old_df.values!=df.values)==np.ndarray)&(old_df.values!=df.values).all()):
                    new_df=True
                
            if not os.path.isfile(file_loc+"/"+str(ID).zfill(11)+'_table.pdf') or overwrite or new_df or re_vet:
                # Making table a plot for PDF:
                fig=plt.figure(figsize=(11.69,4))
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
                fig.savefig(file_loc+"/"+str(ID).zfill(11)+'_table.pdf')
                figs['tab']=file_loc+"/"+str(ID).zfill(11)+'_table.pdf'
            else:
                figs['tab']=file_loc+"/"+str(ID).zfill(11)+'_table.pdf'
        if not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_candidates.csv') or overwrite or re_vet:
            df.to_csv(file_loc+"/"+file_loc.split('/')[-1]+'_candidates.csv')
        #all_cands_df.to_csv("all_cands.csv")
        
        '''
        #Searches for other dips in the lightcurve
        planet_dic_1=SearchForSubsequentTransits(lc, interpmodel, monoparams, Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)
        '''
        #If any of the dips pass our EB threshold, we will run do a single model as an EB here:
        
        print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],
                   'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})
        
        # CREATING CANDIDATE VETTING REPORT:
        if plot:
            #print(figs)
            #Compiling figures into a multi-page PDF
            from PyPDF2 import PdfFileReader, PdfFileWriter

            output = PdfFileWriter()
            pdfPages=[]
            for figname in figs:
                #print(figname,type(figs[figname]))
                output.addPage(PdfFileReader(open(figs[figname], "rb")).getPage(0))
            outputStream = open(file_loc+"/"+file_loc.split('/')[-1]+'_report.pdf', "wb")
            output.write(outputStream)
            outputStream.close()
        
        # EB MODELLING:
        if np.any([both_dic[obj]['flag']=='EB' for obj in both_dic]):
            #Minimising EB model with ELLC:
            eb_dict={obj:both_dic[obj] for obj in both_dic if both_dic[obj]['flag'] not in ['asteroid','instrumental','FP - confusion']}
            EBdic=minimize_EBmodel(lc, eb_dict,Teff,Rstar[0],rhostar[0]*Rstar[0]**3)#lc, planet, Teffs, Rs, Ms, nrep=9,try_sec=False,use_ellc=False

            EBdic['ID']=ID
            EBdic['mission']=mission
            #Save to table here.
            return EBdic, both_dic

        elif np.any([both_dic[obj]['flag']=='planet' for obj in both_dic]):
            # PLANET MODELLING:
            print({pl:{'tcen':both_dic[pl]['tcen'],'depth':both_dic[pl]['depth'],'period':both_dic[pl]['period'],'orbit_flag':both_dic[pl]['orbit_flag'],'flag':both_dic[pl]['flag']} for pl in both_dic})
            print("Planets to model:",[obj for obj in both_dic if both_dic[obj]['flag']=='planet'])
            if not os.path.isfile(file_loc+"/"+file_loc.split('/')[-1]+'_model.pickle') or overwrite or re_fit:
                mod=MonoFit.monoModel(ID, lc, {},savefileloc=file_loc+'/')
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
                mod.init_model(useL2=useL2,mission=mission,FeH=0.0)
                pickle.dump(mod,open(file_loc+"/"+file_loc.split('/')[-1]+'_model.pickle','wb'))
            else:
                mod = pickle.load(open(file_loc+"/"+file_loc.split('/')[-1]+'_model.pickle','rb'))
            if do_fit:
                mod.RunMcmc()
            return mod, both_dic
        else:
            #NO EB OR PC candidates - likely low-SNR or FP.
            print("likely low-SNR or FP. Flags=",[both_dic[obj]['flag']=='planet' for obj in both_dic])
            #Save to table here.
            return None,both_dic
    else:
        print("nothing detected")
        
if __name__=='__main__':
    import sys
    assert(len(sys.argv)==15)
    ID=int(sys.argv[1])
    mission=sys.argv[2]
    tcen=float(sys.argv[3]) if sys.argv[3]!=0.0 else None
    tdur=float(sys.argv[4]) if sys.argv[4]!=0.0 else None
    overwrite=bool(sys.argv[5])
    do_search=bool(sys.argv[6])
    useL2=bool(sys.argv[7])
    PL_ror_thresh=float(sys.argv[8])
    variable_llk_thresh=float(sys.argv[9])
    file_loc=sys.argv[10]
    plot=bool(sys.argv[11])
    do_fit=bool(sys.argv[12])
    re_vet=bool(sys.argv[13])
    re_fit=bool(sys.argv[14])
    out=MonoVetting(ID,mission,
                    tcen=tcen,tdur=tdur,overwrite=overwrite,do_search=do_search,
                    useL2=useL2,PL_ror_thresh=PL_ror_thresh,variable_llk_thresh=variable_llk_thresh,file_loc=file_loc,
                    plot=plot,do_fit=do_fit,re_vet=re_vet,re_fit=re_fit)
