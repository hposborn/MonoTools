# 
# Let's use this python script to:
# - perform transit "Quick fit"
# - Search for dips in a lightcurve which match that detected
# - Search for secondaries
# - Perform quick/dirty EB modelling
#
import warnings
warnings.filterwarnings("ignore")

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
import matplotlib.gridspec as gridspec


from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.io import fits

import seaborn as sns
import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib').disabled = True


MonoData_tablepath = os.path.join(os.path.dirname(__file__),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join('/'.join(os.path.dirname( __file__ ).split('/')[:-1]),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')

if not os.path.isdir(MonoData_savepath):
    os.mkdir(MonoData_savepath)

#os.environ["CXXFLAGS"]="-fbracket-depth=512" if not "CXXFLAGS" in os.environ else "-fbracket-depth=512,"+os.environ["CXXFLAGS"]
os.environ["CFLAGS"] = "-fbracket-depth=512" if not "CFLAGS" in os.environ else "-fbracket-depth=512,"+os.environ["CFLAGS"]

#creating new hidden directory for theano compilations:
theano_dir=MonoData_savepath+'/.theano_dir_'+str(np.random.randint(8))

theano_pars={'device':'cpu','floatX':'float64',
             'base_compiledir':theano_dir}#,"gcc.cxxflags":"-fbracket-depth=1024"}
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

import pymc as pm
import pymc_ext as pmx

from . import tools
from . import fit
from . import lightcurve
from . import starpars

class target():
    """The core target class which represents an individual star
    """
    def __init__(self, id, mission, radec=None, lc=None, dataloc=None):
        """Initialising `target` class.

        Args:
            id (int): Mission ID of object (i.e. TIC, EPIC or KIC)
            mission (str): Mission (i.e. 'tess', 'k2', 'kepler')
            radec (astropy.SkyCoord, optional): RA & Declination of star. Defaults to None which uses the `get_radec` function of `lightcurve.multilc` to access position from ID
            lc (lightcurve.multilc, optional): Already-initialised lightcurve class. Defaults to None, which gets lightcurve from file
            dataloc (str, optional): Path of folder in which to store data. Defaults to None, which adds an ID-specific folder in `tools.MonoData_savpath`
        """
        if lc is None:
            #Initial
            self.lc = lightcurve.multilc(id, mission, radec=radec, savefileloc=dataloc)
        else:
            assert type(lc)==lightcurve.multilc
            self.lc=lc
        
        self.detns={}
        self.monos=[];self.duos=[];self.multis=[]
        
        self.radec = self.lc.radec #multilc has the capability to search an ID to get radec, so we'll use that
        self.id = id
        self.mission = mission
        if dataloc is None:
            self.dataloc = os.path.join(MonoData_savepath,tools.id_dic[self.mission]+str(int(self.id)).zfill(11))
        else:
            self.dataloc = dataloc
        if not os.path.isdir(self.dataloc):
            os.mkdir(self.dataloc)

    def init_starpars(self,Rstar=None,Teff=None,logg=None,FeH=0.0,rhostar=None,Mstar=None):
        """Adds stellar parameters to model

        Args:
            Rstar (list, optional): Stellar radius in Rsol in format [value, neg_err, pos_err]. Defaults to np.array([1.0,0.08,0.08]).
            Teff (list, optional): Stellar effective Temperature in K in format [value, neg_err, pos_err]. Defaults to np.array([5227,100,100]).
            logg (list, optional): Stellar logg in cgs in format [value, neg_err, pos_err]. Defaults to np.array([4.3,1.0,1.0]).
            FeH (float, optional): Stellar log Metallicity. Defaults to 0.0.
            rhostar (list, optional): Stellar density in rho_sol (1.411gcm^-3) in format [value, neg_err, pos_err]. Defaults to None.
            Mstar (float or list, optional): Stellar mass in Msol either as a float or in format [value, neg_err, pos_err]. Defaults to None.
        """
        #Making sure we have a pd.Series rather than a DF:
        if hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess']:
            self.lc.all_ids['tess']['data']=self.lc.all_ids['tess']['data'].iloc[0] if type(self.lc.all_ids['tess']['data'])==pd.DataFrame else self.lc.all_ids['tess']['data']

        if Rstar is None and hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'rad' in self.lc.all_ids['tess']['data']:
            #Radius info from lightcurve data (TIC)
            if 'eneg_Rad' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Rad'] is not None and self.lc.all_ids['tess']['data']['eneg_Rad']>0:
                Rstar={'val':self.lc.all_ids['tess']['data']['rad'],
                       'err_neg':self.lc.all_ids['tess']['data']['eneg_Rad'],
                       'err_pos':self.lc.all_ids['tess']['data']['epos_Rad']}
                Rstar['av_err']=0.5*(abs(Rstar['err_neg'])+Rstar['err_pos'])
            else:
                Rstar={'val':self.lc.all_ids['tess']['data']['rad'],
                       'av_err':self.lc.all_ids['tess']['data']['e_rad'],
                       'err_neg':self.lc.all_ids['tess']['data']['e_rad'],
                       'err_pos':self.lc.all_ids['tess']['data']['e_rad']}
            if 'eneg_Teff' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Teff'] is not None and self.lc.all_ids['tess']['data']['eneg_Teff']>0:
                Teff={'val':self.lc.all_ids['tess']['data']['Teff'],
                        'err_neg':self.lc.all_ids['tess']['data']['eneg_Teff'],
                        'err_pos':self.lc.all_ids['tess']['data']['epos_Teff']}
                Teff['av_err']=0.5*(abs(Teff['err_neg'])+Teff['err_pos'])
            else:
                Teff={'val':self.lc.all_ids['tess']['data']['Teff'],
                      'av_err':self.lc.all_ids['tess']['data']['e_Teff']}
                Teff['err_neg']=Teff['av_err'];Teff['err_pos']=Teff['av_err']
            if 'eneg_logg' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_logg'] is not None and self.lc.all_ids['tess']['data']['eneg_logg']>0:
                logg={'val':self.lc.all_ids['tess']['data']['logg'],
                      'err_neg':self.lc.all_ids['tess']['data']['eneg_logg'],
                      'err_pos':self.lc.all_ids['tess']['data']['epos_logg']}
                logg['av_err']=0.5*(abs(logg['err_neg'])+logg['err_pos'])
            else:
                logg={'val':self.lc.all_ids['tess']['data']['logg'],
                      'av_err':self.lc.all_ids['tess']['data']['e_logg'],
                      'err_neg':self.lc.all_ids['tess']['data']['e_logg'],
                      'err_pos':self.lc.all_ids['tess']['data']['e_logg']}
        if Rstar is None:
            Rstar={'val':1.0,'av_err':0.25,'err_neg':0.25,'err_pos':0.25}
        if Teff is None:
            Teff={'val':5227,'av_err':100,'err_neg':100,'err_pos':100}
        if logg is None:
            logg={'val':4.3,'av_err':1.0,'err_neg':1.,'err_pos':1.}
        self.FeH=FeH
        self.Rstar=Rstar
        self.Teff=Teff
        self.logg=logg

        if Mstar is not None and type(Mstar)==float:
            self.Mstar = {'val':Mstar,'av_err':0.33*Mstar,'err_neg':0.33*Mstar,'err_pos':0.33*Mstar}
        elif Mstar is not None and (type(Mstar)==list or type(Mstar)==np.ndarray):
            self.Mstar = {'val':Mstar[0]}
            if len(Mstar)==2:
                self.Mstar['av_err']=Mstar[1]
                self.Mstar['err_neg']=Mstar[1]
                self.Mstar['err_pos']=Mstar[1]
            else:
                self.Mstar['err_neg']=Mstar[1]
                self.Mstar['err_pos']=Mstar[2]
                self.Mstar['av_err']=0.5*(abs(self.Mstar['err_neg'])+self.Mstar['err_pos'])
        elif Mstar is not None and type(Mstar)==dict and 'val' in Mstar:
            self.Mstar={'val':Mstar['val']}
            if 'av_err' in Mstar:
                self.Mstar['av_err']=Mstar['av_err']
                self.Mstar['err_neg']=Mstar['av_err']
                self.Mstar['err_pos']=Mstar['av_err']
            elif 'err_neg' in Mstar and 'err_pos' in Mstar:
                self.Mstar['err_neg']=Mstar['err_neg']
                self.Mstar['err_pos']=Mstar['err_pos']
                self.Mstar['av_err']=0.5*(abs(self.Mstar['err_neg'])+self.Mstar['err_pos'])

        elif Mstar is None and hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'mass' in self.lc.all_ids['tess']['data']:
            if 'eneg_mass' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_mass'] is not None and self.lc.all_ids['tess']['data']['eneg_mass']>0:
                self.Mstar={'val':self.lc.all_ids['tess']['data']['mass'],
                       'err_neg':self.lc.all_ids['tess']['data']['eneg_mass'],
                       'err_pos':self.lc.all_ids['tess']['data']['epos_mass']}
                self.Mstar['av_err']=0.5*(abs(self.Mstar['err_neg'])+self.Mstar['err_pos'])
            else:
                self.Mstar={'val':self.lc.all_ids['tess']['data']['mass'],
                       'av_err':self.lc.all_ids['tess']['data']['e_mass'],
                       'err_neg':self.lc.all_ids['tess']['data']['e_mass'],
                       'err_pos':self.lc.all_ids['tess']['data']['e_mass']}

        #Here we only have a mass, radius, logg- Calculating rho two ways (M/R^3 & logg/R), and doing weighted average
        if rhostar is None:
            rho_logg={'val':np.power(10,self.logg['val']-4.43)/self.Rstar['val']}
            rho_logg['err_pos']=np.power(10,self.logg['val']+self.logg['err_pos']-4.43)/(self.Rstar['val']-abs(self.Rstar['err_neg']))-rho_logg['val']
            rho_logg['err_neg']=rho_logg['val']-np.power(10,self.logg['val']-self.logg['err_pos']-4.43)/(self.Rstar['val']+self.Rstar['err_neg'])
            rho_logg['relerr_pos']=rho_logg['err_pos']/rho_logg['val']
            rho_logg['relerr_neg']=rho_logg['err_neg']/rho_logg['val']
            rho_logg['relerr_sum']=rho_logg['relerr_neg']+rho_logg['relerr_pos']
            
            if Mstar is not None:
                rho_MR={'val':self.Mstar['val']/self.Rstar['val']**3}
                rho_MR['err_pos']=(self.Mstar['val']+self.Mstar['err_pos'])/(self.Rstar['val']-abs(self.Rstar['err_neg']))**3 - rho_MR['val']
                rho_MR['err_neg']=rho_MR['val'] - (self.Mstar['val']-abs(self.Mstar['err_neg']))/(self.Rstar['val']+self.Rstar['err_pos'])**3
                rho_MR['relerr_neg']=rho_MR['err_neg']/rho_MR['val']
                rho_MR['relerr_pos']=rho_MR['err_pos']/rho_MR['val']
                rho_MR['relerr_sum']=rho_MR['relerr_neg']+rho_MR['relerr_pos']

                #Weighted sums of two avenues to density:
                total_relerr_sums=(rho_logg['relerr_pos']+rho_logg['relerr_neg']+rho_MR['relerr_pos']+rho_MR['relerr_neg'])
                self.rhostar = {'val':rho_logg['val']*rho_MR['relerr_sum']/total_relerr_sums+\
                                 rho_MR['val']*rho_logg['relerr_sum']/total_relerr_sums}
                self.rhostar['err_neg']=self.rhostar['val']*(rho_logg['relerr_neg']*rho_MR['relerr_sum']/total_relerr_sums+
                                      rho_MR['relerr_neg']*rho_logg['relerr_sum']/total_relerr_sums)
                self.rhostar['err_neg']=self.rhostar['val']*(rho_logg['relerr_pos']*rho_MR['relerr_sum']/total_relerr_sums+
                                      rho_MR['relerr_pos']*rho_logg['relerr_sum']/total_relerr_sums)
            else:
                self.rhostar=rho_logg
        elif type(rhostar) in [np.ndarray,list]:
            self.rhostar = {'val':rhostar[0]}
            if len(rhostar)==2:
                self.rhostar['av_err']=rhostar[1]
                self.rhostar['err_neg']=rhostar[1]
                self.rhostar['err_pos']=rhostar[1]
            else:
                self.rhostar['err_neg']=rhostar[1]
                self.rhostar['err_pos']=rhostar[2]
                self.rhostar['av_err']=0.5*(abs(self.rhostar['err_neg'])+self.rhostar['err_pos'])

            if Mstar is None and hasattr(self,'Mstar') is not None:
                self.Mstar={'val':self.rhostar['val']*self.Rstar['val']**3}
                self.Mstar['err_pos']=self.Mstar['val']-(self.rhostar['val']+self.rhostar['err_pos'])*(self.Rstar['val']+self.Rstar['err_pos'])**3
                self.Mstar['err_neg']=(self.rhostar['val']+abs(self.rhostar['err_neg']))*(self.Rstar['val']+abs(self.Rstar['err_neg']))**3-self.Mstar['val']
                self.Mstar['av_err']=0.5*(self.Mstar['err_pos']+self.Mstar['err_neg'])
        if Mstar is None and not hasattr(self,'Mstar'):
            self.Mstar = {'val':1.0, 'av_err':0.33, 'err_neg':0.33, 'err_pos':0.33}
    
    def search_monos(self,mono_SNR_thresh=6.5,mono_BIC_thresh=-6,n_durs=5,poly_order=3,
                     n_oversamp=20,binsize=15/1440.0,custom_mask=None,
                     transit_zoom=3.5,use_flat=False,use_binned=True,use_poly=True,
                     plot=False, plot_loc=None ,n_max_monos=8, use_stellar_dens=True, **kwargs):
        """AI is creating summary for search_monos

        Args:
            mono_SNR_thresh (float, optional): threshold in sigma to alert mono candidate. Defaults to 6.5.
            mono_BIC_thresh (int, optional): threshold in BIC to alert mono candidate. Defaults to -6.
            n_durs (int, optional): Number of distinct durations to iterate lightcurve search. Defaults to 5.
            poly_order (int, optional): polynomial order to use for a "no transit" fit. Defaults to 3.
            n_oversamp (int, optional): oversampling factor in transit durations from which to match to lightcurve. Defaults to 20.
            binsize (float, optional): size of bins in days. Defaults to 15/1440.0.
            custom_mask (np.ndarray, optional): [description]. Defaults to None.
            transit_zoom (float, optional): size (in transit durations) to cut around transit when minimizing transit model. Defaults to 3.5.
            use_flat (bool, optional): Flattens lightcurve before monotransit search. Defaults to False.
            use_binned (bool, optional): Uses binned (rather than raw) lightcurve. Defaults to True.
            use_poly (bool, optional): co-fits transits with a polynomial (order = poly_order-1) to account for variability. Defaults to True.
            plot (bool, optional): Whether to plot the transit search output. Defaults to False.
            plot_loc (str, optional): Location at which to save plot. Defaults to None.
            n_max_monos (int, optional): Max number of mono candidates to save/alert. Defaults to 8.
            use_stellar_dens (bool, optional): Use stellar density to produce transit templates to search. Defaults to True.

        """
        
        #Computing a fine x-range to search:
        search_xrange=[]
        
        mincad=np.min([float(cad.split('_')[1])/86400 for cad in np.unique(self.lc.cadence_list)])
        self.mono_search_interpmodels, self.mono_search_tdurs = self.get_interpmodels(self.Rstar['val'], self.Mstar['val'], self.Teff['val'], 
                                               n_durs=n_durs, texp=mincad)
        
        #print("Checking input model matches. flux:",np.nanmedian(uselc[:,0]),"std",np.nanstd(uselc[:,1]),"transit model:",
        #       interpmodels[0](10.0),"depth:",interpmodels[0](0.0))
            
        search_xranges=[]
        
        mask = self.lc.mask if custom_mask is None else custom_mask
        #Removing gaps bigger than 2d (with no data)
        for n in range(n_durs):
            search_xranges_n=[]
            if np.max(np.diff(self.lc.time[mask]))<self.mono_search_tdurs[n]:
                lc_regions=[self.lc.time[mask]]
            else:
                lc_regions = np.array_split(self.lc.time[mask],1+np.where(np.diff(self.lc.time[mask])>self.mono_search_tdurs[n])[0])
            for arr in lc_regions:
                search_xranges_n+=[np.arange(arr[0]+0.33*self.mono_search_tdurs[n],arr[-1]-0.33*self.mono_search_tdurs[n],self.mono_search_tdurs[n]/n_oversamp)]
            search_xranges+=[np.hstack(search_xranges_n)]
        
        print(str(self.id)+" - Searching "+str(np.sum([len(xr) for xr in search_xranges]))+" positions with "+str(n_durs)+" durations:",','.join(list(np.round(self.mono_search_tdurs,3).astype(str))))

        #Looping through search and computing chi-sq at each position:
        self.mono_search_timeseries=pd.DataFrame()

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
            tdur=self.mono_search_tdurs[n%n_durs]
            
            #flattening and binning as a function of the duration we are searching in order to avoid
            if use_flat and not use_poly:
                self.lc.flatten(knot_dist=tdur*4)
                if use_binned:
                    #Having may points in a lightcurve makes flattening difficult (and fitting needlessly slow)
                    # So let's bin to a fraction of the tdur - say 9 in-transit points.
                    self.lc.bin(binsize=tdur/9, timeseries_names=['flux','flux_flat'], extramask=custom_mask,overwrite=True)
                    uselc=np.column_stack((self.lc.bin_time,self.lc.bin_flux_flat,self.lc.bin_flux_err))#self.lc.bin_time,self.lc.bin_flux_flat,self.lc.bin_flux_err))
                    uselc=uselc[np.isfinite(uselc[:,1])]
                else:
                    uselc=np.column_stack((self.lc.time,self.lc.flux_flat,self.lc.flux_err))[self.lc.mask]#self.lc.bin_time,self.lc.bin_flux_flat,self.lc.bin_flux_err))
            else:
                if use_binned:
                    #Having may points in a lightcurve makes flattening difficult (and fitting needlessly slow)
                    # So let's bin to a fraction of the tdur - say 9 in-transit points.
                    self.lc.bin(binsize=tdur/9, extramask=custom_mask,overwrite=True)
                    uselc=np.column_stack((self.lc.bin_time,self.lc.bin_flux,self.lc.bin_flux_err))#self.lc.bin_time,self.lc.bin_flux_flat,self.lc.bin_flux_err))
                    uselc=uselc[np.isfinite(uselc[:,1])]
                else:
                    uselc=np.column_stack((self.lc.time,self.lc.flux,self.lc.flux_err))[self.lc.mask]#self.lc.bin_time,self.lc.bin_flux_flat,self.lc.bin_flux_err))
                    
            #Making depth vary from 0.1 to 1.0
            init_dep_shifts=np.exp(np.random.normal(0.0,n_oversamp*0.01,len(search_xrange)))
            randns=np.random.randint(2,size=(len(search_xrange),2))
            cad=np.nanmedian(np.diff(uselc[:,0]))
            p_transit = np.clip(3/(len(uselc[:,0])*cad),0.0,0.05)
            
            #What is the probability of transit given duration (used in the prior calculation) - duration
            methods=['SLSQP','Nelder-Mead','Powell']
            logmodeldep=np.log(abs(self.mono_search_interpmodels[n](0.0)))

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
                    
                    #print(x,y,poly_order)
                    poly_fit=np.polyfit(x,y,poly_order)
                    poly_neg_llik=0.5 * np.sum((y - np.polyval(poly_fit,x))**2 / sigma2 + np.log(sigma2))

                    if use_poly:
                        init_grad=np.polyfit(x[~in_tr],y[~in_tr],1)[0]
                        res_trans=optim.minimize(trans_model_poly_neglnlik,np.hstack((init_log_dep,init_grad)),
                                                args=(x,y,sigma2,
                                                    logmodeldep,self.mono_search_interpmodels[n%n_durs]),
                                                method = methods[randns[n_mod,0]])
                        res_sin=optim.minimize(sin_model_poly_neglnlik, 
                                            np.hstack((init_log_dep,init_grad)),
                                            args=(x,y,sigma2,x2s,tdur),
                                            method = methods[randns[n_mod,1]])
                    else:
                        res_trans=optim.minimize(trans_model_neglnlik,(init_log_dep),
                                                args=(x,y,sigma2,
                                                    logmodeldep,self.mono_search_interpmodels[n%n_durs]),
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
                    self.mono_search_timeseries=self.mono_search_timeseries.append(pd.Series(outdic,name=len(self.mono_search_timeseries)))
            #print(n,len(self.mono_search_timeseries))
        #bar.finish()
        self.mono_search_timeseries=self.mono_search_timeseries.sort_values('tcen')
        #Transit model has to be better than the sin model AND the DeltaBIC w.r.t to the polynomial must be <-10.
        # transit depth must be <0.0,
        # the sum of DeltaBICs has to be < threshold (e.g. -10)
        self.mono_search_timeseries['sin_llk_ratio']=self.mono_search_timeseries['llk_trans'].values-self.mono_search_timeseries['llk_sin'].values
        self.mono_search_timeseries['poly_llk_ratio']=self.mono_search_timeseries['llk_trans'].values-self.mono_search_timeseries['llk_poly'].values
        self.mono_search_timeseries['sin_DeltaBIC']=self.mono_search_timeseries['BIC_trans'].values-self.mono_search_timeseries['BIC_sin'].values
        self.mono_search_timeseries['poly_DeltaBIC']=self.mono_search_timeseries['BIC_trans'].values-self.mono_search_timeseries['BIC_poly'].values
        self.mono_search_timeseries['sum_DeltaBICs']=self.mono_search_timeseries['sin_DeltaBIC']+self.mono_search_timeseries['poly_DeltaBIC']
        self.mono_search_timeseries['mean_DeltaBICs']=0.5*self.mono_search_timeseries['sum_DeltaBICs']
        #if use_poly:
            #In the case of co-fitting for polynomials, BICs fail, but log likelihoods still work.
            #We will use 1.5 (4.5x) over sin and 3 (20x) over polynomial as our threshold:
        #    signfct=np.where((self.mono_search_timeseries['sin_llk_ratio']>1.5)&(self.mono_search_timeseries['poly_llk_ratio']>3)&(self.mono_search_timeseries['trans_dep']>0.0))[0]
        #else:
        #    signfct=np.where((self.mono_search_timeseries['sin_llk_ratio']>1.0)&(self.mono_search_timeseries['poly_DeltaBIC']<mono_BIC_thresh)&(self.mono_search_timeseries['trans_dep']>0.0))[0]
        signfct=(self.mono_search_timeseries['sin_llk_ratio']>1.5)&(self.mono_search_timeseries['poly_llk_ratio']>1.5)&(self.mono_search_timeseries['poly_DeltaBIC']<mono_BIC_thresh)&(self.mono_search_timeseries['trans_snr']>(mono_SNR_thresh-0.5))
        n_sigs=np.sum(signfct)

        if n_sigs>0:
            nix=0
            while n_sigs>0 and nix<=n_max_monos:
                #Getting the best detection:
                signfct_df=self.mono_search_timeseries.loc[signfct]
                
                #Placing the best detection info into our dict:
                detn_row=signfct_df.iloc[np.argmin(signfct_df['poly_DeltaBIC'])]
                self.init_mono(detn_row['tcen'], detn_row['init_dur'], detn_row['trans_dep'], otherinfo=detn_row[[col for col in detn_row.index if col not in ['tcen','init_dur','trans_dep']]])
                
                #Removing the regions around this detection from our array
                away_from_this_detn=abs(self.mono_search_timeseries['tcen']-detn_row['tcen'])>np.where(self.mono_search_timeseries['init_dur']<detn_row['init_dur'],
                                                                    0.66*detn_row['init_dur'], 0.66*self.mono_search_timeseries['init_dur'])
                signfct=signfct&away_from_this_detn
                n_sigs=np.sum(signfct)
                nix+=1
    
    def init_mono(self,tcen,tdur,depth,name=None,otherinfo=None, **kwargs):
        """Initalise Monotransit

        Args:
            tcen (float): transit epoch [d]
            tdur (float): transit duration [d]
            depth (float): transit depth [ratio]
            name (str, optional): Name of the detection - for the `detns` dictionary. Defaults to None, in which case it will take a numerical value
            otherinfo (dict or pd.Series, optional): Dictionary of extra info related to the monotransit detection. Defaults to None.
        """
        otherinfo=pd.Series(otherinfo) if type(otherinfo)==dict else otherinfo
        name=str(len(self.detns)).zfill(2) if name is None else name
        self.detns[name]={'tcen':tcen,'tdur':tdur,'depth':depth,'min_P':self.calc_min_P(tcen,tdur),'orbit_flag':'mono','period':np.nan}
        if otherinfo is not None:
            self.detns[name].update({col:otherinfo[col] for col in otherinfo.index if col not in self.detns[name]})
        self.monos+=[name]

        self.quick_mono_fit(name, ndurs=4.5, **kwargs)
    
    def init_multi(self,tcen,tdur,depth,period,name=None,otherinfo=None,**kwargs):
        """Initalise multi-transiting candidate

        Args:
            tcen (float): transit epoch [d]
            tdur (float): transit duration [d]
            depth (float): transit depth [ratio]
            name (str, optional): Name of the detection - for the `detns` dictionary. Defaults to None, in which case it will take a numerical value
            otherinfo (dict or pd.Series, optional): Dictionary of extra info related to the monotransit detection. Defaults to None.
        """

        otherinfo=pd.Series(otherinfo) if type(otherinfo)==dict else otherinfo
        name=str(len(self.detns)).zfill(2) if name is None else name
        self.detns[name]={'tcen':tcen,'tdur':tdur,'depth':depth,'period':period,'orbit_flag':'multi'}
        if otherinfo is not None:
            self.detns[name].update({col:otherinfo[col] for col in otherinfo.index if col not in self.detns[name]})
        self.multis+=[name]
        
        #Fitting:
        self.quick_mono_fit(name,ndurs=4.5,fluxindex='flux_flat',fit_poly=False,**kwargs)
    
    def init_duo(self,tcen,tcen2,tdur,depth,period,name=None,otherinfo=None,**kwargs):
        """Initalise multi-transiting candidate

        Args:
            tcen (float): transit epoch [d]
            tcen2 (float): second transit epoch [d]
            tdur (float): transit duration [d]
            depth (float): transit depth [ratio]
            name (str, optional): Name of the detection - for the `detns` dictionary. Defaults to None, in which case it will take a numerical value
            otherinfo (dict or pd.Series, optional): Dictionary of extra info related to the monotransit detection. Defaults to None.
        """

        otherinfo=pd.Series(otherinfo) if type(otherinfo)==dict else otherinfo
        name=str(len(self.detns)).zfill(2) if name is None else name
        self.detns[name]={'tcen':tcen,'tcen_2':tcen2,'tdur':tdur,'depth':depth,'period':period,'orbit_flag':'duo'}
        if otherinfo is not None:
            self.detns[name].update({col:otherinfo[col] for col in otherinfo.index if col not in self.detns[name]})
        self.duos+=[name]
        
        #Fitting:
        self.quick_mono_fit(name,ndurs=4.5,fluxindex='flux_flat',fit_poly=False,**kwargs)

    def remove_detn(self,name):
        """Remove candidate detection given the name. 
        This stores the data in "olddetns", but the candidate is removed from multis/monos dicts

        Args:
            name (str): key from the `detns` dict to remove.
        """
        if not hasattr(self,'olddetns'):
            self.olddetns={}
        self.olddetns['x'+name]=self.detns.pop(name)
        if name in self.multis:
            self.multis.remove(name)
        if name in self.monos:
            self.monos.remove(name)
        if name in self.duos:
            self.duos.remove(name)


    def calc_min_P(self,tcen,tdur):
        """For a monotransit, calculate the minimum possible radius given that no other transit was detected for this candidate

        Args:
            tcen (float): Transit epoch
            tdur (float): Transit durtion

        Returns:
            float: Minimum period in days
        """
        abs_times=abs(self.lc.time-tcen)
        abs_times=np.sort(abs_times)
        whr=np.where(np.diff(abs_times)>tdur*0.75)[0]
        if len(whr)>0:
            return abs_times[whr[0]]
        else:
            return np.max(abs_times)

    def create_transit_mask(self):
        #Making sure we mask detected transits:
        if not hasattr(self.lc,'in_trans'):
            setattr(self.lc,'in_trans',np.tile(False,len(self.lc.time)))
        for d in self.detns:
            if self.detns[d]['orbit_flag']=='mono':
                self.lc.in_trans[abs(self.lc.time-self.detns[d]['tcen'])<0.5*self.detns[d]['tdur']]=True
            elif self.detns[d]['orbit_flag']=='duo':
                self.lc.in_trans[abs(self.lc.time-self.detns[d]['tcen'])<0.5*self.detns[d]['tdur']]=True
                self.lc.in_trans[abs(self.lc.time-self.detns[d]['tcen_2'])<0.5*self.detns[d]['tdur']]=True
            if self.detns[d]['orbit_flag']=='mono':
                self.lc.in_trans[abs((self.lc.time-self.detns[d]['tcen']-0.5*self.detns[d]['period'])%self.detns[d]['period']-0.5*self.detns[d]['period'])<0.5*self.detns[d]['tdur']]=True

    def plot_mono_search(self, use_flat=True, use_binned=True, use_poly=False,transit_zoom=2.25,plot_loc=None,**kwargs):
        """Plot the results found by `search_monos`

        Args:
            use_flat (bool, optional): Whether to plot the flattened or raw lightcurve. Defaults to True.
            use_binned (bool, optional): Whether to plot the binned or raw lightcurve. Defaults to True.
            use_poly (bool, optional): Whether to use the local polynomials fits in plotting. Defaults to False.
            transit_zoom (float, optional): Size of window around transit to plot zoom (in transit durations). Defaults to 2.25.
            plot_loc (str, optional): File location to save plot. Defaults to None.
        """

        self.create_transit_mask()
        if use_flat:
            self.lc.flatten(knot_dist = 1.75*np.max(self.mono_search_tdurs))
            self.lc.bin(timeseries=['flux','flux_flat'])
            flux_key='flux_flat'
        else:
            self.lc.bin()
            flux_key='flux'
       
        # Initialising plots:
        fig = plt.figure(figsize=(11.69,8.27))#,constrained_layout=True)
        
        outer = gridspec.GridSpec(2, 1,hspace=0.25)
        gs_timeseries = gridspec.GridSpecFromSubplotSpec(3, 24, subplot_spec = outer[0], wspace=0.05, hspace=0.)
        #fig.add_gridspec(3,6*len(self.monos),wspace=0.07,hspace=0.18)
        gs_zooms = gridspec.GridSpecFromSubplotSpec(3, len(self.monos), subplot_spec = outer[1], wspace=0.05, hspace=0.)
        #fig.add_gridspec(3,len(self.monos),wspace=0.07,hspace=0.18)

        import seaborn as sns
        sns.set_palette(sns.set_palette("RdBu",14))
        axes={}
        
        islands = tools.find_time_regions(self.lc.time)
        
        #Cutting the lightcurve into regions and fitting these onto the first plot row:
        isl_lens=[islands[j][1]-islands[j][0] for j in range(len(islands))]
        from iteround import saferound
        plot_ix = np.hstack((0,np.cumsum(saferound(24*np.array(isl_lens)/np.sum(isl_lens), places=0)))).astype(int)
        print(plot_ix,24,isl_lens)
        ix=(np.isfinite(self.mono_search_timeseries['sum_DeltaBICs']))&(self.mono_search_timeseries['sin_DeltaBIC']<1e8)&(self.mono_search_timeseries['poly_DeltaBIC']<1e8)
        min_bic=np.nanmin(self.mono_search_timeseries.loc[ix,'poly_DeltaBIC'])
        maxylim=np.nanmax([np.nanpercentile(self.mono_search_timeseries.loc[ix,'poly_DeltaBIC'],98),
                           np.nanpercentile(self.mono_search_timeseries.loc[ix,'sin_DeltaBIC'],98)])

        for ni,isle in enumerate(islands):
            axes[str(ni)+'a']=fig.add_subplot(gs_timeseries[:2,plot_ix[ni]:plot_ix[ni+1]])
            #rast=True if np.sum(self.lc.self.lc.mask>12000) else False
            timeix=self.lc.mask*(self.lc.time>isle[0])*(self.lc.time<isle[1])
            bin_timeix=np.isfinite(self.lc.bin_time)*(self.lc.bin_time>isle[0])*(self.lc.bin_time<isle[1])
            axes[str(ni)+'a'].plot(self.lc.time[timeix],getattr(self.lc,flux_key)[timeix],'.k',alpha=0.28,markersize=0.75, rasterized=True)
            if use_flat:
                axes[str(ni)+'a'].plot(self.lc.bin_time[bin_timeix],self.lc.bin_flux_flat[bin_timeix],'.k',alpha=0.7,markersize=1.75, rasterized=True)
            else:
                axes[str(ni)+'a'].plot(self.lc.bin_time[bin_timeix],self.lc.bin_flux[bin_timeix],'.k',alpha=0.7,markersize=1.75, rasterized=True)
            axes[str(ni)+'a'].set_ylim(np.nanpercentile(getattr(self.lc,flux_key)[self.lc.mask],(0.25,99.75)))
            #plt.plot(self.mono_search_timeseries_2['tcen'],self.mono_search_timeseries_2['worstBIC'],'.',c='C5',alpha=0.4)

            axes[str(ni)+'b']=fig.add_subplot(gs_timeseries[2,plot_ix[ni]:plot_ix[ni+1]])
            axes[str(ni)+'b'].plot([isle[0]-0.3,isle[1]+0.3],[-10,-10],'--k',alpha=0.25)
            for nd,dur in enumerate(np.unique(self.mono_search_timeseries['init_dur'])):
                mono_timeix=(self.mono_search_timeseries['tcen']>isle[0])*(self.mono_search_timeseries['tcen']<isle[1])*(self.mono_search_timeseries['init_dur']==dur)
                if nd==0:
                    axes[str(ni)+'b'].plot(self.mono_search_timeseries.loc[mono_timeix,'tcen'],
                            self.mono_search_timeseries.loc[mono_timeix,'poly_DeltaBIC'],
                            c=sns.color_palette()[nd],alpha=0.6,label='Transit - polynomial', rasterized=True)
                    axes[str(ni)+'b'].plot(self.mono_search_timeseries.loc[mono_timeix,'tcen'],
                            self.mono_search_timeseries.loc[mono_timeix,'sin_DeltaBIC'],
                            c=sns.color_palette()[-1*(nd+1)],alpha=0.6,label='Transit - wavelet', rasterized=True)
                else:
                    axes[str(ni)+'b'].plot(self.mono_search_timeseries.loc[mono_timeix,'tcen'],
                            self.mono_search_timeseries.loc[mono_timeix,'poly_DeltaBIC'],
                                c=sns.color_palette()[nd],alpha=0.6, rasterized=True)
                    axes[str(ni)+'b'].plot(self.mono_search_timeseries.loc[mono_timeix,'tcen'],
                            self.mono_search_timeseries.loc[mono_timeix,'sin_DeltaBIC'],
                                c=sns.color_palette()[-1*(nd+1)],alpha=0.6, rasterized=True)
            
            if ni==0:
                axes[str(ni)+'a'].set_ylabel("flux ["+self.lc.flx_system+"]")
                axes[str(ni)+'b'].set_ylabel("Delta BIC")
            else:
                axes[str(ni)+'a'].set_yticklabels([])
                axes[str(ni)+'b'].set_yticklabels([])
            if ni==len(islands)-1:
                axes[str(ni)+'b'].legend(prop={'size': 5})

            axes[str(ni)+'a'].set_xlim(isle[0]-0.3,isle[1]+0.3)
            axes[str(ni)+'b'].set_xlim(isle[0]-0.3,isle[1]+0.3)
            axes[str(ni)+'a'].set_xticks([])
            axes[str(ni)+'a'].set_xticklabels([])
            axes[str(ni)+'b'].set_ylim(maxylim,min_bic)
            from matplotlib.ticker import FormatStrFormatter
            axes[str(ni)+'b'].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        fig.text(0.5, 0.5, "Time [BJD-"+str(self.lc.jd_base)+"]", rotation="horizontal", ha="center",va="center")
        
        if len(self.monos)>0:
            trans_model_mins=[]

            n_poly=int(np.sum([1 for n in range(10) if 'poly_'+str(n) in self.detns[list(self.detns.keys())[0]]]))
            for nm, monopl in enumerate(self.monos):

                tdur=self.detns[monopl]['tdur']
                tcen=self.detns[monopl]['tcen']

                loc_of_trans = str(np.arange(len(islands)).astype(int)[np.array([(tcen>isle[0])*(tcen<isle[1]) for isle in islands])][0])
                axes[loc_of_trans+'b'].text(tcen,np.clip(np.min([self.detns[monopl]['sin_DeltaBIC'],self.detns[monopl]['poly_DeltaBIC']]),
                                            min_bic,1e6),monopl)
                axes['m_'+monopl+'b']=fig.add_subplot(gs_zooms[2,nm:(nm+1)])
                axes['m_'+monopl+'b'].text(tcen+0.1,np.clip(np.min([self.detns[monopl]['sin_DeltaBIC'],self.detns[monopl]['poly_DeltaBIC']]),
                                    min_bic,1e6),monopl)
                for n,dur in enumerate(np.unique(self.mono_search_timeseries['init_dur'])):
                    index=(self.mono_search_timeseries['init_dur']==dur)&(abs(self.mono_search_timeseries['tcen']-tcen)<transit_zoom*tdur)
                    axes['m_'+monopl+'b'].plot(self.mono_search_timeseries.loc[index,'tcen'], 
                                self.mono_search_timeseries.loc[index,'BIC_trans']-self.mono_search_timeseries.loc[index,'BIC_poly'],
                                c=sns.color_palette()[n], alpha=0.6, rasterized=True)
                    axes['m_'+monopl+'b'].plot(self.mono_search_timeseries.loc[index,'tcen'], 
                                self.mono_search_timeseries.loc[index,'BIC_trans']-self.mono_search_timeseries.loc[index,'BIC_sin'],
                                c=sns.color_palette()[-1*n], alpha=0.6, rasterized=True)
                #plt.plot(self.mono_search_timeseries['tcen'],self.mono_search_timeseries['worstBIC'],',k')
                axes['m_'+monopl+'b'].plot([tcen,tcen],[-150,130],'--k',linewidth=1.5,alpha=0.25)
                axes['m_'+monopl+'b'].set_ylim(maxylim,min_bic)
                if nm==0:
                    axes['m_'+monopl+'b'].set_ylabel("Delta BIC")
                else:
                    axes['m_'+monopl+'b'].set_yticks([])
                    axes['m_'+monopl+'b'].set_yticklabels([])
                
                mono_dets=self.mono_search_timeseries.loc[self.mono_search_timeseries['tcen']==self.detns[monopl]['tcen']].iloc[0]
                axes['m_'+monopl+'a']=fig.add_subplot(gs_zooms[:2,nm:(nm+1)])
                
                nmod=np.arange(len(self.mono_search_tdurs))[np.argmin(abs(self.mono_search_tdurs-tdur))]
                round_tr=self.lc.mask&(abs(self.lc.time-tcen)<(transit_zoom*tdur))
                x=(self.lc.time[round_tr]-tcen)
                y=getattr(self.lc,flux_key)[round_tr]
                
                y_offset=np.nanmedian(getattr(self.lc,flux_key)[round_tr&(abs(self.lc.time-tcen)>(0.7*tdur))]) if use_poly else 0
                y-=y_offset
                
                #Plotting polynomial:
                axes['m_'+monopl+'a'].plot(self.lc.time[round_tr],np.polyval([mono_dets['poly_'+str(n)] for n in range(n_poly)],x),'--',
                        c=sns.color_palette()[-4],linewidth=2.0,alpha=0.6, rasterized=True)
                
                #Plotting transit:
                modeldep=abs(self.mono_search_interpmodels[nmod](0.0))
                
                if use_flat and not use_poly:
                    trans_model=(mono_dets['trans_dep']/modeldep)*self.mono_search_interpmodels[nmod](x)
                else:
                    trans_model=mono_dets['trans_grad']*x+(mono_dets['trans_dep']/modeldep)*self.mono_search_interpmodels[nmod](x)

                #Plotting sin wavelet:
                newt=x/(2*tdur)*2*np.pi
                amp=np.exp(-1*np.power(newt*1.25, 2.) / (2 * np.power(np.pi, 2.)))
                if use_flat and not use_poly:
                    sin_model=mono_dets['sin_dep']*(amp*np.sin(newt-np.pi*0.5)-0.1)
                else:
                    sin_model=mono_dets['sin_grad']*x+mono_dets['sin_dep']*(amp*np.sin(newt-np.pi*0.5)-0.1)

                if nm==0:
                    axes['m_'+monopl+'a'].set_ylabel("flux ["+self.lc.flx_system+"]")
                else:
                    axes['m_'+monopl+'a'].set_yticks([])
                    axes['m_'+monopl+'a'].set_yticklabels([])
                axes['m_'+monopl+'a'].set_xticklabels([])
                axes['m_'+monopl+'a'].plot(self.lc.time[round_tr],y,'.k',markersize=1.5,alpha=0.3, rasterized=True)
                round_tr_bin=abs(self.lc.bin_time-tcen)<(transit_zoom*tdur)
                if use_flat:
                    axes['m_'+monopl+'a'].plot(self.lc.bin_time[round_tr_bin],self.lc.bin_flux_flat[round_tr_bin]-y_offset,
                                '.k',alpha=0.7,markersize=2.5, rasterized=True)
                else:
                    axes['m_'+monopl+'a'].plot(self.lc.bin_time[round_tr_bin],self.lc.bin_flux[round_tr_bin]-y_offset,
                                '.k',alpha=0.7,markersize=2.5, rasterized=True)

                axes['m_'+monopl+'a'].plot(self.lc.time[round_tr],trans_model,'-',
                            c=sns.color_palette()[0],linewidth=2.0,alpha=0.85, rasterized=True)
                axes['m_'+monopl+'a'].plot(self.lc.time[round_tr],sin_model,'-.',
                            c=sns.color_palette()[-1],linewidth=2.0,alpha=0.6, rasterized=True)
                
                axes['m_'+monopl+'b'].set_xlim(tcen-tdur*transit_zoom,tcen+tdur*transit_zoom)
                axes['m_'+monopl+'a'].set_xlim(tcen-tdur*transit_zoom,tcen+tdur*transit_zoom)
                from matplotlib.ticker import FormatStrFormatter
                axes['m_'+monopl+'b'].xaxis.set_major_formatter(FormatStrFormatter('%g'))

                trans_model_mins+=[np.min(trans_model)]
            #print(trans_model_mins)
            trans_model_min=np.min(np.array(trans_model_mins))
            for nm,monopl in enumerate(self.monos):
                axes['m_'+monopl+'a'].set_ylim(trans_model_min*1.2,np.nanpercentile(self.lc.bin_flux,97.5))

        #fig.subplots_adjust(wspace=0.06, hspace=0.15)
        #plt.tight_layout()
        if plot_loc is None:
            plot_loc = os.path.join(self.dataloc,tools.id_dic[self.mission]+str(self.id).zfill(11)+"_Monotransit_Search.pdf")
        elif plot_loc[-1]=='/' or os.path.isdir(plot_loc):
            plot_loc = os.path.join(plot_loc,tools.id_dic[self.mission]+str(self.id).zfill(11)+"_Monotransit_Search.pdf")

        #fig.suptitle(str(self.id).zfill(7)+" - Monotransit Search")
        fig.savefig(plot_loc, dpi=400)
        #plt.xlim(1414,1416)
        return plot_loc

    def run_BLS(self,modx,mody,modyerr,dur_shift=0.04,n_durs=6,min_period=1.1,max_period=None):
        """run a Box-Least-Squares search

        Args:
            dur_shift (float, optional): Fractional shift in duration per loop. Defaults to 0.05.
            min_per (float, optional): Minimum period. Defaults to 1.1.
            max_per (float, optional): Maximum period. Defaults to None, which derives a maxp
        """
        from astropy.timeseries import BoxLeastSquares

        max_period=0.66*np.sum(np.diff(modx)[np.diff(modx)<0.4]) if max_period is None else max_period
        time_span=np.max(modx)-np.min(modx)
        
        #The fractional shift in P^(4/3) from the e.g. expected duration and lightcurve timespan:
        frac_p43_shift=86400**(-2/3)*np.sqrt((1+0.025)**2 - 0.41**2) * (3/(np.pi**2*6.67e-11*1409.78*self.Mstar['val']/self.Rstar['val']**3))**(1/3)*dur_shift/time_span
        pers=[min_period]
        while pers[-1]<max_period:
            pers+=[pers[-1]+frac_p43_shift*pers[-1]**(4/3)]
        samp_pers = np.array(pers) 
        samp_durs = np.random.normal(self.Rstar['val'],self.Rstar['av_err'],len(samp_pers)) * np.sqrt((1+0.025)**2 - np.random.random(len(samp_pers))**2) * ((3*samp_pers*86400)/(np.pi**2*6.67e-11*1409.78*np.random.normal(self.Mstar['val'],self.Mstar['av_err'],len(samp_pers))))**(1/3)/86400
        model = BoxLeastSquares(modx, mody, dy=modyerr)

        print(np.percentile(samp_durs,np.linspace(2,98,n_durs)),np.percentile(samp_durs,np.arange(8,98,n_durs)))
        periodogram = model.power(samp_pers, duration=np.percentile(samp_durs,np.linspace(8,98,n_durs)), oversample=1/dur_shift)

        detn = np.argmax(periodogram['power'])

        stats = model.compute_stats(periodogram.period[detn],
                            periodogram.duration[detn],
                            periodogram.transit_time[detn])

        tlslike_periodogram={'power':periodogram['power'],'periods':periodogram['period'],
                             'period':periodogram['period'][detn],
                             'snr':periodogram['depth_snr'][detn],'depth':periodogram['depth'][detn],
                             'T0':periodogram['transit_time'][detn],'duration':periodogram['duration'][detn],
                             'transit_times':stats['transit_times'][stats['per_transit_count']>0],
                             'SDE_raw':periodogram['power'][detn],
                             'depth_mean_even':stats['depth_even'], 'depth_mean_odd':stats['depth_odd'],
                             'llk_per_transit':stats['per_transit_log_likelihood'][stats['per_transit_count']>0],
                             'total_llk':stats['harmonic_delta_log_likelihood'],'n_trans':np.sum(stats['per_transit_count']>0),
                             'pts_per_transit':stats['per_transit_count'][stats['per_transit_count']>0]}
        tlslike_periodogram['model_lightcurve_phase']=(modx-tlslike_periodogram['T0']-0.5*tlslike_periodogram['period'])%tlslike_periodogram['period']-0.5*tlslike_periodogram['period']
        tlslike_periodogram['model_lightcurve_time']=modx[:]
        tlslike_periodogram['model_lightcurve_model']=np.where(abs(tlslike_periodogram['model_lightcurve_phase'])<0.5*tlslike_periodogram['depth'],1.0-tlslike_periodogram['depth'],1.0)
        #'tcen', 'tdur', 'depth', 'period', 'orbit_flag', 'duration_tls', 'snr_tls', 'FAP_tls', 'SDE_tls', 'SDE_raw_tls', 'chi2_min_tls', 'rp_rs_tls', 'odd_even_mismatch_tls', 'transit_count_tls', 'transit_depths_tls', 'transit_times_tls', 'model_lightcurve_time_tls', 'model_lightcurve_model_tls', 'log_lik_mono_monofit', 'model_success_monofit', 'mean_monofit', 'tcen_monofit', 'log_tdur_monofit', 'tdur_monofit', 'b_monofit', 'log_ror_monofit', 'ror_monofit', 'circ_per_monofit', 'r_pl_monofit', 'depth_lc_monofit', 'light_curve_monofit', 'third_light_monofit', 'x_monofit', 'ymodel_monofit', 'y_monofit', 'yerr_monofit', 'depth_monofit', 'depth_err_monofit', 'snr_monofit', 'interpmodel_monofit', 'Ntrans_monofit', 'cdpp_monofit', 'snr_r_monofit'
        #'transit_times', 'per_transit_count', 'per_transit_log_likelihood', 'depth', 'depth_phased', 'depth_half', 'depth_odd', 'depth_even', 'harmonic_amplitude', 'harmonic_delta_log_likelihood
        return tlslike_periodogram, np.column_stack((periodogram['period'],periodogram['power']))


    def run_broken_TLS(self,masked_t,masked_y,masked_yerr,max_period=None,min_period=1.1):
        """Running TLS on non-consecutive timeseries.
        This is performed by searching for short-period planets in individual sectors and then stacking the resulting power spectra together using interpolation.
        For detections found in the stacked power spectrum, a focused TLS is re-run on a small (1%-wide) period window to hone the period.

        Args:
            masked_t (np.ndarray): Time with known transits and anomalies masked
            masked_y (np.ndarray): Flux with known transits and anomalies masked. This must be normalised to median 1.0
            masked_yerr (np.ndarray): Flux error with known transits and anomalies masked
            max_period (float, optional): Maximum period to search. Defaults to 2/3 of the total observation duration.
            min_period (float, optional): Minimum period to search. Defaults to 1.1d.

        Returns:
            dict: TransitLeastSquares results dictionary for the detected object
            np.ndarray: full combined power spectrum with columns of period and power
        """
        from transitleastsquares import transitleastsquares

        #Using 2/3 of the total observed duration (ignoring gaps)
        max_period=0.66*np.sum(np.diff(masked_t)[np.diff(masked_t)<0.4]) if max_period is None else max_period

        time_regions = tools.find_time_regions(masked_t,split_gap_size=7)
        if len(time_regions)>1:
            iresults=[]
            reg_durs = np.array([it[1]-it[0] for it in time_regions])

            for nt,it in enumerate(time_regions):
                ix=(masked_t>it[0])*(masked_t<it[1])
                model = transitleastsquares(masked_t[ix], masked_y[ix], masked_yerr[ix])
                iresults+=[model.power(period_min=min_period,period_max=np.clip(np.min(reg_durs)/3,min_period,max_period),duration_grid_step=1.0625,Rstar=self.Rstar['val'],Mstar=self.Mstar['val'],
                                       use_threads=1,show_progress_bar=False, n_transits_min=3)]
            #print(reg_durs)
            base_arr = np.column_stack((iresults[np.argmax(reg_durs)].periods,iresults[np.argmax(reg_durs)].power))
            for nt,it in enumerate(time_regions):
                if nt!=np.argmax(reg_durs):
                    base_arr=np.column_stack((base_arr,
                                    interp.interp1d(iresults[nt].periods,iresults[nt].power)(np.clip(base_arr[:,0],np.min(iresults[nt].periods),np.max(iresults[nt].periods)))))
            mixed_spec = np.sum(base_arr[:,1:]**2,axis=1)**0.5
            if max_period>np.min(reg_durs)/3:
                model = transitleastsquares(masked_t, masked_y, masked_yerr)
                iresults+=[model.power(period_min=np.min(reg_durs)/3,period_max=max_period,duration_grid_step=1.0625,Rstar=self.Rstar['val'],Mstar=self.Mstar['val'],
                                        use_threads=1,show_progress_bar=False, n_transits_min=3)]
                print(np.min(iresults[-1].periods),np.max(iresults[-1].periods))
                all_periods=np.hstack((base_arr[:,0],iresults[-1].periods))
                all_powers =np.hstack((mixed_spec,iresults[-1].power))
            else:
                all_periods=base_arr[:,0]
                all_powers=mixed_spec
            detper=all_periods[np.argmax(all_powers)]
            if detper<np.min(reg_durs)/3:
                #Re-doing the TLS with a very constrained (1%) fine-point fit:
                result=model.power(period_min=detper*0.99,period_max=detper*1.01,duration_grid_step=1.033,Rstar=self.Rstar['val'],Mstar=self.Mstar['val'],
                                        use_threads=1,show_progress_bar=False, n_transits_min=3)
            else:
                #Using the result from the longer-span TLS run
                result=iresults[-1]

            stackpers=np.hstack((all_periods,result.periods))
            setattr(result,'periods',np.sort(stackpers))
            setattr(result,'power',np.hstack((all_powers,result.power))[np.argsort(stackpers)])
            return result, np.column_stack((all_periods,all_powers))[np.argsort(all_periods),:]
        else:
            model = transitleastsquares(masked_t, masked_y, masked_yerr)
            res=model.power(period_min=min_period,period_max=max_period,duration_grid_step=1.0625,Rstar=self.Rstar['val'],Mstar=self.Mstar['val'],
                               use_threads=1,show_progress_bar=False, n_transits_min=3)
            return res, np.column_stack((res.periods,res.power))



    def search_multi_planets(self, fluxname='bin_flux_flat', binsize=15/1440.0, n_search_loops=5, use_tls=True,
                            multi_FAP_thresh=0.00125, multi_SNR_thresh=7.0, max_period=None,
                            mask_prev_planets=False, do_sample=False, **kwargs):
        """Use transitleastsquares to search for periodic planets.

        Args:
            fluxname (str, optional): Which flux array to use for the detection. Defaults to 'bin_flux_flat'.
            binsize (float, optional): Size of data binning in days. Defaults to 15/1440.0 (i.e. 15mins).
            n_search_loops (int, optional): Maximum number of times to run a TLS search (i.e. multiple detections). Defaults to 5.
            multi_FAP_thresh (float, optional): False Alarm Probability threshold for multi detection. Defaults to 0.00125.
            multi_SNR_thresh (float, optional): Signal to Noise Ratio threshold for multi detection. Defaults to 7.0.
            max_period (float, optional): Maximum period up to which to search. If None, then uses 66% of the sum of all cadences
            mask_prev_planets (bool, optional): Whether to mask those events we found during other searches (e.g. search_mono_planets). Defaults to False.
        """
        #Searches an LC (ideally masked for the monotransiting planet) for other *periodic* planets.
        print("Using "+["BLS","TLS"][int(use_tls)]+" on ID="+str(self.id)+" to search for multi-transiting planets")
        self.multi_power_spectra={}

        if not hasattr(self,'Rstar'):
            self.init_starpars()
        #Max period is half the total observed time NOT half the distance from t[0] to t[-1]
        p_max=0.66*np.sum(np.diff(self.lc.time)[np.diff(self.lc.time)<0.4]) if max_period is None else max_period
        #np.clip(0.75*(np.nanmax(lc[prefix+'time'])-np.nanmin(lc[prefix+'time'])),10,80)
        if 'flat' in fluxname:
            #Setting the window to fit over as 5*maximum duration
            rhostar=1.0 if self.rhostar==0.0 or self.rhostar is None else self.rhostar['val']
            durmax = (p_max/(3125*rhostar))**(1/3)
            
            plmask_0 = np.tile(False,len(self.lc.time))
            if mask_prev_planets:
                self.create_transit_mask()
                #Masking each of those transit events we detected during the MonoTransit search
                plmask_0+=self.lc.in_trans
            
            self.lc.flatten(knot_dist=1.9*durmax,transit_mask=~plmask_0,**kwargs)

        if 'bin' in fluxname:
            lc=self.lc.bin(binsize=binsize,timeseries=[fluxname.replace('bin_','')])

        #Looping through, masking planets, until none are found.
        #{'01':{'period':500,'period_err':100,'FAP':np.nan,'snr':np.nan,'tcen':tcen,'tdur':tdur,'rp_rs':np.nan}}
        if 'bin' not in fluxname:
            time=self.lc.time[:]
            anommask=self.lc.mask[:]
        else:
            time=self.lc.bin_time[:]
            print(fluxname, hasattr(self.lc,fluxname))
            anommask=np.isfinite(getattr(self.lc,fluxname))
        plmask=np.tile(False,len(anommask))
        SNR_last_planet=100;init_n_pl=len(self.detns);n_pl=len(self.detns);self.multi_results=[]
        while SNR_last_planet>multi_SNR_thresh and n_pl<(n_search_loops+init_n_pl):
            planet_name=str(len(self.detns)).zfill(2)
            print("searching for candidate",planet_name)
            assert planet_name not in self.detns
            #Making model. Making sure lc is at 1.0 and in relatie flux, not ppt/ppm:
            modx = time[anommask]
            mody = (getattr(self.lc,fluxname)[anommask] * self.lc.flx_unit) + (1.0-np.nanmedian(getattr(self.lc,fluxname)[anommask])*self.lc.flx_unit)
            modyerr = (getattr(self.lc,fluxname+'_err')[anommask]*self.lc.flx_unit) if hasattr(self.lc,fluxname+'_err') else (getattr(self.lc,'flux_err')[anommask]*self.lc.flx_unit)
            #print(n_pl,len(mody),len(anommask),np.sum(anommask),len(plmask),np.sum(plmask))
            if np.sum(plmask)>0:
                #Filling in the plmask with out-of-transit data:
                mody[plmask[anommask]] = mody[~plmask[anommask]][np.random.choice(np.sum(~plmask[anommask]),np.sum(plmask[anommask]))]
                #print("in-transit points:",np.sum(plmask),", median of all points:",np.nanmedian(mody),", median of randomly selected points",
                #      np.nanmedian(rand_oot_data),", median of in-transit points",np.nanmedian(mody[plmask&anommask]))
            print(np.isnan(mody).sum(),mody)
            if use_tls:
                ires=self.run_broken_TLS(modx,mody,modyerr,p_max,min_period=1.1)
            else:
                ires=self.run_BLS(modx,mody,modyerr,max_period=p_max,min_period=1.1)
            self.multi_results+=[ires[0]]
            self.multi_power_spectra[planet_name]=ires[1]
            #anommask *= tools.cut_anom_diff(mody)
            #print(n_pl,"norm_mask:",np.sum(self.lc.mask),"anoms:",np.sum(anommask),"pl mask",np.sum(plmask),"total len",len(anommask))
            #print(results[-1])
            print(self.multi_results[-1])
            if 'snr' in self.multi_results[-1] and not np.isnan(self.multi_results[-1]['snr']) and 'transit_times' in self.multi_results[-1]:
                if 'snr_per_transit' in self.multi_results[-1]:
                    #Defining transit times as those times where the SNR in transit is consistent with expectation (>-3sigma)
                    snr_per_trans_est=np.sqrt(np.sum(self.multi_results[-1].snr_per_transit>0))
                    trans=np.array(self.multi_results[-1]['transit_times'])[self.multi_results[-1].snr_per_transit>snr_per_trans_est/2]
                elif 'llk_per_transit' in self.multi_results[-1]:
                    per_pt_llk=abs(self.multi_results[-1]['total_llk'])/np.sum(self.multi_results[-1]['pts_per_transit'])
                    print(len(self.multi_results[-1]['transit_times']),len(self.multi_results[-1]['llk_per_transit']),len(self.multi_results[-1]['pts_per_transit']),
                          per_pt_llk,self.multi_results[-1]['llk_per_transit'],(self.multi_results[-1]['pts_per_transit']*per_pt_llk*0.5))
                    trans=self.multi_results[-1]['transit_times'][self.multi_results[-1]['llk_per_transit']>(self.multi_results[-1]['pts_per_transit']*per_pt_llk*0.5)]
            else:
                trans=[]
            print(len(time),len(anommask),len(plmask))
            phase_nr_trans=(time[anommask&(~plmask)]-self.multi_results[-1]['T0']-0.5*self.multi_results[-1]['period'])%self.multi_results[-1]['period']-0.5*self.multi_results[-1]['period']
            if 'snr' in self.multi_results[-1] and np.sum(abs(phase_nr_trans)<0.5*np.clip(self.multi_results[-1]['duration'],0.2,2))>3:
                if self.multi_results[-1]['duration']<0.15 or self.multi_results[-1]['duration']>2:
                    #TLS throws a crazy-short duration... let's take th duration from period & impact parameter
                    dur = np.sqrt((1+np.sqrt(1-self.multi_results[-1]['depth']))**2 - 0.35**2) * ((3*self.multi_results[-1]['period']*86400)/(np.pi**2*6.67e-11*1409.78*self.Mstar['val']/self.Rstar['val']**3))**(1/3)/86400
                else:
                    dur = self.multi_results[-1]['duration']
                if use_tls:
                    oth_inf=pd.Series({key+'_tls':self.multi_results[-1][key] for key in ['duration','snr','FAP','SDE','SDE_raw','chi2_min','rp_rs','odd_even_mismatch','transit_count','transit_depths','transit_times','model_lightcurve_time','model_lightcurve_model']})
                else:
                    oth_inf=pd.Series({key+'_bls':self.multi_results[-1][key] for key in ['duration','snr','SDE_raw','total_llk','llk_per_transit','pts_per_transit','transit_times','model_lightcurve_time','model_lightcurve_model']})

                self.init_multi(tcen=self.multi_results[-1]['T0'],tdur=dur,
                                depth=1-self.multi_results[-1]['depth'],period=self.multi_results[-1]['period'],
                                otherinfo=oth_inf,
                                name=planet_name,
                                sample_model=do_sample)
                SNR=np.max([self.detns[planet_name]['snr_monofit'],self.multi_results[-1]['snr']])
                print(SNR,self.multi_results[-1]['snr'],self.detns[planet_name]['period'],self.detns[planet_name]['tdur_monofit'])
                if SNR>multi_SNR_thresh and len(trans)>2:
                    #Removing planet from future data to be searched
                    this_pl_masked=abs((time-self.detns[planet_name]['tcen']+0.5*self.detns[planet_name]['period'])%self.detns[planet_name]['period']-0.5*self.detns[planet_name]['period'])<(0.6*self.detns[planet_name]['tdur_monofit'])
                    plmask=plmask+this_pl_masked#Masking previously-detected transits
                    #print(np.sum(plmask))
                    #print(n_pl,results[-1].period,plparams['tdur'],np.sum(this_pl_masked),np.sum(plmask))
                    #print(n_pl,"pl_mask",np.sum(this_pl_masked)," total:",np.sum(plmask))
                elif SNR>multi_SNR_thresh:
                    # pseudo-fails - we have a high-SNR detection but it's a duo or a mono.
                    #print(plparams['tcen'],plparams['tdur'],"fails with transits at ",trans,"with durations",plparams['tdur'],"transits. Reserching")
                    this_pl_masked=abs((time-self.detns[planet_name]['tcen']+0.5*self.detns[planet_name]['period'])%self.detns[planet_name]['period']-0.5*self.detns[planet_name]['period'])<(0.6*self.detns[planet_name]['tdur_monofit'])
                    #np.min(abs((time[np.newaxis,:])-np.array(trans)[:,np.newaxis]),axis=0)<(0.7*self.detns[planet_name]['tdur_monofit'])
                    #this_pl_masked=(((lc[prefix+'time']-plparams['tcen']+0.7*plparams['tdur'])%results[-1].period)<(1.4*plparams['tdur']))
                    #print(n_pl,results[-1].period,plparams['tdur'],np.sum(this_pl_masked))
                    plmask = plmask|this_pl_masked
                    SNR_last_planet=SNR
                else:
                    # Fails
                    SNR_last_planet=0
                    self.remove_detn(planet_name)
            n_pl+=1
        
    def plot_multi_search(self,plot_loc=None,plot_extent=0.8):
        """Plot the output of search_multi_planets

        Args:
            plot_loc (str, optional): Location at which to save. Defaults to a file in `self.dataloc`
            plot_extent (float, optional): Extent in time of zoom plots in days. Defaults to 0.8.
        """
        sns.set_palette("viridis",10)
        fig = plt.figure(figsize=(11.69,8.27))
        nplots=len(self.multis)*24
        gs = gridspec.GridSpec(3, nplots,hspace=0.25)

        rast=True if np.sum(self.lc.mask>12000) else False

        self.create_transit_mask()
        self.lc.flatten()
        self.lc.bin(binsize=1/48, timeseries_names=['flux','flux_flat'])

        islands = tools.find_time_regions(self.lc.time,split_gap_size=5)
        #Cutting the lightcurve into regions and fitting these onto the first plot row:
        isl_lens=[islands[j][1]-islands[j][0] for j in range(len(islands))]
        from iteround import saferound
        
        plot_ix = np.hstack((0,np.cumsum(saferound(nplots*np.array(isl_lens)/np.sum(isl_lens), places=0)))).astype(int)
        axes={}
        max_all_depths = np.max([self.detns[mult]['depth_monofit']*self.lc.flx_unit for mult in self.multis])
        
        all_in_trans=np.tile(False,len(self.lc.time))
        if len(self.multis)>0:
            for n_m,mult in enumerate(self.multis):
                phase=(self.lc.time-self.detns[mult]['tcen']-0.5*self.detns[mult]['period'])%self.detns[mult]['period']-0.5*self.detns[mult]['period']
                all_in_trans+=abs(phase)<0.55*self.detns[mult]['tdur_monofit']
        self.lc.flatten(in_trans_mask=~all_in_trans)
        self.lc.bin(timeseries=['flux_flat'])

        for ni,isle in enumerate(islands):
            axes['timeseries_'+str(ni)]=fig.add_subplot(gs[1,plot_ix[ni]:plot_ix[ni+1]])
            #rast=True if np.sum(self.lc.self.lc.mask>12000) else False
            timeix=self.lc.mask*(self.lc.time>isle[0])*(self.lc.time<isle[1])
            bin_timeix=np.isfinite(self.lc.bin_time)*(self.lc.bin_time>isle[0])*(self.lc.bin_time<isle[1])
            axes['timeseries_'+str(ni)].plot(self.lc.time[timeix],self.lc.flux_flat[timeix],'.k',alpha=0.28,markersize=0.75, rasterized=rast)
            axes['timeseries_'+str(ni)].plot(self.lc.bin_time[bin_timeix],self.lc.bin_flux_flat[bin_timeix],'.',alpha=0.8, rasterized=rast)
            axes['timeseries_'+str(ni)].set_xlim(isle[0]-0.3,isle[1]+0.3)
            print(1.5*np.nanpercentile(self.lc.flux_flat[self.lc.mask],[0.25,99.75]))
            axes['timeseries_'+str(ni)].set_ylim(1.5*np.nanpercentile(self.lc.flux_flat[self.lc.mask],[0.1,99.9]))
            #axes['timeseries_'+str(ni)].set_ylim(-3.5*np.nanstd(self.lc.flux_flat[self.lc.mask])-max_all_depths, 3.5*np.nanstd(self.lc.flux_flat[self.lc.mask]))

        axes['spec']=fig.add_subplot(gs[0,:])
        max_all_time = np.max([np.max(self.multi_power_spectra[mult][:,1]) for mult in self.multi_power_spectra])
        if len(self.multis)>0:
            for n_m,mult in enumerate(self.multis):
                axes['spec'].plot([self.detns[mult]['period'],self.detns[mult]['period']],[-1,1.1*max_all_time],':',
                                  linewidth=4.5,alpha=0.6,c=sns.color_palette()[n_m+2],label=mult+'/det_'+str(n_m))
                axes['spec'].plot([2*self.detns[mult]['period'],2*self.detns[mult]['period']],[-1,0.66*max_all_time],'--',
                                  linewidth=1.5,alpha=0.25,c=sns.color_palette()[n_m+2])
                axes['spec'].plot([0.5*self.detns[mult]['period'],0.5*self.detns[mult]['period']],[-1,0.66*max_all_time],'--',
                                  linewidth=1.5,alpha=0.25,c=sns.color_palette()[n_m+2])

                axes['spec'].plot(self.multi_power_spectra[mult][:,0],
                                  (2*len(self.multis)-2-2*n_m)+self.multi_power_spectra[mult][:,1],c=sns.color_palette()[n_m+2],alpha=0.6,linewidth=0.8)

                phase=(self.lc.time-self.detns[mult]['tcen']-0.5*self.detns[mult]['period'])%self.detns[mult]['period']-0.5*self.detns[mult]['period']
                axes['zoom_'+mult]=fig.add_subplot(gs[2,int(n_m*24):int((n_m+1)*24)])
                if n_m==0:
                    axes['zoom_'+mult].set_ylabel("Flux ["+self.lc.flx_system+"]")
                else:
                    axes['zoom_'+mult].set_yticklabels([])
                for ni,isle in enumerate(islands):
                    
                    if 'model_lightcurve_time_tls' not in self.detns[mult]:
                        phase = (self.lc.time-self.detns[mult]['tcen']-0.5*self.detns[mult]['period'])%self.detns[mult]['period']-0.5*self.detns[mult]['period']
                        plotmodix=(self.lc.time>isle[0])*(self.lc.time<isle[1])*abs(phase)<0.65*self.detns[mult]['tdur']
                        mod = np.where(abs(phase)<0.5*self.detns[mult]['tdur'],1000*(self.detns[mult]['depth']-1),0.0)
                        axes['timeseries_'+str(ni)].plot(self.lc.time[plotmodix],mod[plotmodix],
                                                         '-', alpha=0.4, linewidth=1,c=sns.color_palette()[2+n_m],label=mult+'/det='+str(n_m), rasterized=rast)
                    else:
                        plotmodix=(self.detns[mult]['model_lightcurve_time_tls']>isle[0])*(self.detns[mult]['model_lightcurve_time_tls']<isle[1])*(self.detns[mult]['model_lightcurve_model_tls']<1.0)
                        axes['timeseries_'+str(ni)].plot(self.detns[mult]['model_lightcurve_time_tls'][plotmodix],
                                                        (self.detns[mult]['model_lightcurve_model_tls'][plotmodix]-np.nanmedian(self.detns[mult]['model_lightcurve_model_tls']))/self.lc.flx_unit,
                                                        '-', alpha=0.4, linewidth=1,c=sns.color_palette()[2+n_m],label=mult+'/det='+str(n_m), rasterized=rast)
                    trans_in_isle=np.arange(np.ceil((isle[0]-self.detns[mult]['tcen'])/self.detns[mult]['period']),
                                            np.floor((isle[1]-self.detns[mult]['tcen'])/self.detns[mult]['period'])+1,1)
                    axes['timeseries_'+str(ni)].plot(self.detns[mult]['tcen']+self.detns[mult]['period']*trans_in_isle,
                                                     np.tile(-1.1*self.detns[mult]['depth_monofit'] - 4*np.nanstd(self.lc.bin_flux_flat),len(trans_in_isle)),
                                                     "^",markersize=8,alpha=0.6,c=sns.color_palette()[2+n_m],rasterized=rast)
                    if ni>0:
                        axes['timeseries_'+str(ni)].set_yticklabels([])
                    axes['timeseries_'+str(ni)].set_ylim(2*np.nanpercentile(self.lc.bin_flux_flat,[0.05,99.95]))
                #print("subplot ",3, len(multis), len(multis)*2+1+n_m)
                bin_phase=tools.bin_lc_segment(np.column_stack((np.sort(phase[(abs(phase)<1.2)*self.lc.mask]),
                                                                self.lc.flux[(abs(phase)<1.2)*self.lc.mask][np.argsort(phase[(abs(phase)<1.2)*self.lc.mask])],
                                                                self.lc.flux_err[(abs(phase)<1.2)*self.lc.mask][np.argsort(phase[(abs(phase)<1.2)*self.lc.mask])])),binsize=self.detns[mult]['tdur_monofit']*0.15)
                #time_shift=0.4*np.nanstd(bin_phase[:,1])*(self.lc.time[abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])] - \
                #                                                                                self.detns[mult]['tcen'])/self.detns[mult]['period']
                #plt.scatter(phase[abs(phase)<1.2],time_shift+self.lc.flux[abs(phase)<1.2],
                axes['zoom_'+mult].scatter(phase[self.lc.mask&(abs(phase)<1.2)],self.lc.flux[self.lc.mask&(abs(phase)<1.2)],
                            s=0.5,c='k',alpha=0.25)
                axes['zoom_'+mult].errorbar(bin_phase[:,0],bin_phase[:,1],yerr=bin_phase[:,2],fmt='.',markersize=4,
                            c=sns.color_palette()[n_m+2],zorder=10)
                axes['zoom_'+mult].plot(np.sort(phase[abs(phase)<1.2]),
                        self.detns[mult]['interpmodel_monofit'](phase[abs(phase)<1.2][np.argsort(phase[abs(phase)<1.2])])/self.lc.flx_unit,
                        c=sns.color_palette()[n_m+2],alpha=0.4,linewidth=4.)
                #plt.ylim(np.nanmin(bin_phase[:,1])-2*np.nanstd(bin_phase[:,1]),
                #         np.nanmax(bin_phase[:,1])+2*np.nanstd(bin_phase[:,1]))
                axes['zoom_'+mult].set_title(mult+'/det='+str(n_m))
                axes['zoom_'+mult].set_xlim(-0.5*plot_extent,0.5*plot_extent)
                axes['zoom_'+mult].set_ylim(-3*np.nanstd(self.lc.flux_flat[self.lc.mask])-self.detns[mult]['depth_monofit']/self.lc.flx_unit,
                                            3*np.nanstd(self.lc.flux_flat[self.lc.mask]))

        #Even if we don't have a detection, there will still be a final power spectrum in multi_power_spectra
        last_index=list(self.multi_power_spectra.keys())[-1]
        #axes['spec'].plot(self.multi_power_spectra[last_index][:,0],self.multi_power_spectra[last_index][:,1],label='no detection')
        axes['spec'].legend()
        axes['spec'].set_xlabel('Period [d]')
        axes['spec'].set_ylabel('Power')
        axes['spec'].set_ylim(-1,1.1*max_all_time+(2*len(self.multis)-2))
        axes['spec'].set_xlim(0,np.max(self.multi_power_spectra[last_index][:,0]))

        if plot_loc is None:
            plot_loc = os.path.join(self.dataloc,tools.id_dic[self.mission]+str(self.id).zfill(11)+"_Multi_Search.pdf")
        elif plot_loc[-1]=='/' or os.path.isdir(plot_loc):
            plot_loc = os.path.join(plot_loc,tools.id_dic[self.mission]+str(self.id).zfill(11)+"_Multi_Search.pdf")
        axes['timeseries_'+str(ni)].legend(prop={'size': 5})
        #plt.plot(results[-1]['model_lightcurve_time'],results[-1]['model_lightcurve_model'],
        #        alpha=0.5,c=sns.color_palette()[n_pl-init_n_pl],label=planet_name+'/det_'+str(n_pl),linewidth=4)
        #plt.legend(prop={'size': 5})
        if hasattr(self.lc,'jd_base'):
            fig.text(0.5, 0.36, "Time [BJD-"+str(self.lc.jd_base)+"]", rotation="horizontal", ha="center",va="center")
        else:
            fig.text(0.5, 0.36,"Time", rotation="horizontal", ha="center",va="center")
        
        fig.savefig(plot_loc, dpi=400)

    def quick_mono_fit(self, planet, useL2=False, fit_poly=True, tdur_prior='loguniform', sample_model=True, 
                     polyorder=3, ndurs=3.3, fluxindex='flux', mask=None, **kwargs):
        """Performs simple planet fit to monotransit dip given the detection data.

        Args:
            planet (str): Candidate identifying to fit
            useL2 (bool, optional): Use diluted second light. Defaults to False.
            fit_poly (bool, optional): Fit a transit dip AND a polynomial trend around transit. Defaults to True.
            tdur_prior (str, optional): Prior function to use on tdur - 'lognormal','loguniform','normal' or 'uniform'. Defaults to 'loguniform'.
            sample_model (bool, optional): Whether to run a quick HMC sampling. Defaults to False.
            polyorder (int, optional): Degree of polynomial fit. Defaults to 2.
            ndurs (float, optional): Number of transit durations to fit each side. Defaults to 3.3
            fluxindex (str, optional): Which array in self.lc to use for the fitting. Defaults to 'flux_flat'.
            mask (np.ndarray, optional): Addidtional flux mask to use. Defaults to None (in which can self.lc.mask is used)
        """

        #Initial depth estimate:
        dur=0.3 if not np.isfinite(self.detns[planet]['tdur']) else self.detns[planet]['tdur'] #Fixing duration if it's broken/nan.
        winsize=np.clip(dur*ndurs,0.75,3.5)
        
        timeindex='bin_time' if 'bin_' in fluxindex else 'time'
        fluxerrindex='bin_flux_err' if 'bin_' in fluxindex else 'flux_err'
        
        if self.detns[planet]['orbit_flag']=='multi':
            assert np.isfinite(self.detns[planet]['period'])
            init_p=self.detns[planet]['period']
            fit_poly=False
            init_poly=None
            n_trans=np.ptp(np.round((getattr(self.lc,timeindex)-self.detns[planet]['tcen'])/init_p))
        elif self.detns[planet]['orbit_flag']=='duo':
            assert 'tcen_2' in self.detns[planet]
            init_p=abs(self.detns[planet]['tcen_2']-self.detns[planet]['tcen'])
            fit_poly=False
            init_poly=None
        elif self.detns[planet]['orbit_flag']=='mono':
            init_p=9999
        phase=(getattr(self.lc,timeindex)-self.detns[planet]['tcen']-init_p*0.5)%init_p-init_p*0.5
        nearby=(abs(phase)<winsize)

        cad = np.nanmedian(np.diff(getattr(self.lc,timeindex)))
        
        if fluxindex=='flux_flat' and not hasattr(self.lc,'flux_flat'):
            #Reflattening and masking transit:
            self.lc.flatten(knot_dist=1.7*dur, transit_mask=abs(phase)>dur*0.5)
        
        mask=np.isfinite(getattr(self.lc,fluxindex)) if 'bin' in fluxindex else self.lc.mask

        assert np.sum(nearby&mask)>0
        x = getattr(self.lc,timeindex)[nearby&mask]
        y = getattr(self.lc,fluxindex)[nearby&mask][np.argsort(x)]*self.lc.flx_unit
        y-=np.nanmedian(y)
        yerr=getattr(self.lc,fluxerrindex)[nearby&mask][np.argsort(x)]*self.lc.flx_unit
        x=np.sort(x).astype(np.float64)
        if not fit_poly or np.sum(abs(phase[nearby&mask])<0.6)==0.0:
            oot_flux=np.nanmedian(y[(abs(phase[nearby&mask])>0.65*dur)])
            int_flux=np.nanmedian(y[(abs(phase[nearby&mask])<0.35*dur)])
            init_poly=None
        else:
            init_poly=np.polyfit(x[abs(phase[nearby&mask])>0.55*dur]-self.detns[planet]['tcen'],y[abs(phase[nearby&mask])>0.55*dur],polyorder-1)
            oot_flux=np.nanmedian((y-np.polyval(init_poly,x-self.detns[planet]['tcen']))[abs(phase[nearby&mask])>0.65*dur])
            int_flux=np.nanmedian((y-np.polyval(init_poly,x-self.detns[planet]['tcen']))[abs(phase[nearby&mask])<0.35*dur])
        dep=abs(oot_flux-int_flux)
        print(dep,dur,init_poly)
        print(x[abs(phase[nearby&mask])>0.55*dur],y[abs(phase[nearby&mask])>0.55*dur])
        
        with pm.Model() as model:
            # Parameters for the stellar properties
            if fit_poly:
                trend = pm.Normal("trend", mu=0, sd=np.exp(np.arange(1-polyorder,1,1.0)), shape=polyorder, testval=init_poly)
                flux_trend = pm.Deterministic("flux_trend", pm.math.dot(np.vander(x-self.detns[planet]['tcen'], polyorder), trend))
                #trend = pm.Uniform("trend", upper=np.tile(1,polyorder+1), shape=polyorder+1,
                #                  lower=np.tile(-1,polyorder+1), testval=init_poly)
                #trend = pm.Normal("trend", mu=np.zeros(polyorder+1), sd=5*(10.0 ** -np.arange(polyorder+1)[::-1]), 
                #                  shape=polyorder+1, testval=np.zeros(polyorder+1))
                #trend = pm.Uniform("trend", upper=np.tile(10,polyorder+1),lower=np.tile(-10,polyorder+1),
                #                   shape=polyorder+1, testval=np.zeros(polyorder+1))
            else:
                mean = pm.Normal("mean", mu=0.0, sd=3*np.nanstd(y))
                flux_trend = mean

            u_star = tools.get_lds(self.Teff['val'])[0]
            #xo.distributions.QuadLimbDark("u_star")
            rhostar=self.Mstar['val']/self.Rstar['val']**3

            #init_per = abs(18226*rhostar*((2*np.sqrt((1+dep**0.5)**2-0.41**2))/dur)**-3) if ('period' not in self.detns[planet] or not np.isfinite(self.detns[planet]['period'])) else self.detns[planet]['period']
            #print(rhostar,init_per,(2*np.sqrt((1+dep**0.5)**2-0.41**2)))
            #log_per = pm.Uniform("log_per", lower=np.log(dur*5),upper=np.log(1000),
            #                    testval=np.clip(np.log(init_per),np.log(dur*6),np.log(1000))
            #                    )
            #per = pm.Deterministic("per",pm.math.exp(log_per))


            # Orbital parameters for the planets
            log_ror = pm.Uniform("log_ror",lower=-6,upper=-0.5,testval=np.clip(0.5*np.log(dep),-6,-0.5))
            ror = pm.Deterministic("ror", pm.math.exp(log_ror))
            b = xo.distributions.ImpactParameter("b", ror=ror, testval=0.41)

            tcen = pm.Bound(pm.Normal, lower=self.detns[planet]['tcen']-0.7*dur, upper=self.detns[planet]['tcen']+0.7*dur)("tcen", mu=self.detns[planet]['tcen'], sd=0.25*dur, testval=np.random.normal(self.detns[planet]['tcen'],0.1*dur))
            if self.detns[planet]['orbit_flag']=='duo':
                tcen2 = pm.Bound(pm.Normal, lower=-0.7*dur, upper=0.7*dur)("tcen2", mu=self.detns[planet]['tcen_2'], sd=0.25*dur, testval=np.random.normal(self.detns[planet]['tcen_2'],0.1*dur))
            elif self.detns[planet]['orbit_flag']=='multi':
                period = pm.Bound(pm.Normal, lower=(-0.7*dur)/n_trans, upper=(0.7*dur)/n_trans)("period", mu=init_p, sd=0.25*dur/n_trans, testval=init_p)
            if not self.detns[planet]['orbit_flag']=='multi':
                if tdur_prior=='loguniform':
                    log_tdur = pm.Uniform("log_tdur", lower=np.log(5*cad), upper=np.log(2))
                    tdur = pm.Deterministic("tdur",pm.math.exp(log_tdur))
                elif tdur_prior=='lognormal':
                    log_tdur = pm.Bound(pm.Normal, lower=np.log(5*cad), upper=np.log(2))("log_tdur", mu=np.log(dur), sd=0.33, testval=np.log(dur))
                    tdur = pm.Deterministic("tdur",pm.math.exp(log_tdur))
                elif tdur_prior=='uniform':
                    tdur = pm.Uniform("tdur", lower=5*cad, upper=2, testval=dur)
                elif tdur_prior=='normal':
                    tdur = pm.Bound(pm.Normal, lower=5*cad, upper=2)("tdur", mu=dur, sd=0.33*dur, testval=dur)                

            if self.detns[planet]['orbit_flag']!='multi':
                circ_per = pm.Deterministic("circ_per",(np.pi**2*6.67e-11*rhostar*1409.78)*((tdur*86400)/pm.math.sqrt((1+ror)**2 - b**2))**3/(3*86400))

            #ror, b = xo.distributions.get_joint_radius_impact(min_radius=0.0075, max_radius=0.25,
            #                                                  testval_r=np.sqrt(dep), testval_b=0.41)
            #logror = pm.Deterministic("logror",pm.math.log(ror))
            
            
            #pm.Potential("ror_prior", -logror) #Prior towards larger logror

            r_pl = pm.Deterministic("r_pl", ror*self.Rstar['val']*109.1)

            #period = pm.Deterministic("period", pm.math.exp(log_per))

            # Orbit model
            if self.detns[planet]['orbit_flag']!='multi':
                orbit = xo.orbits.KeplerianOrbit(r_star=self.Rstar['val'],m_star=self.Mstar['val'],
                                                 period=circ_per, t0=tcen, b=b)
            elif self.detns[planet]['orbit_flag']=='multi':
                orbit = xo.orbits.KeplerianOrbit(r_star=self.Rstar['val'], m_star=self.Mstar['val'],
                                                 period=period, t0=tcen, b=b)
                #Deriving transit duration:
                vels=orbit.get_relative_velocity(tcen)
                tdur = pm.Deterministic("tdur",(2*self.Rstar['val']*pm.math.sqrt( (1+ror)**2 - b**2))/pm.math.sqrt(vels[0]**2+vels[1]**2))

            #vx, vy, _ = orbit.get_relative_velocity(tcen)
            #vrel=pm.Deterministic("vrel",pm.math.sqrt(vx**2 + vy**2)/self.Rstar['val'])
            
            #tdur=pm.Deterministic("tdur",(2*pm.math.sqrt(1-b**2))/vrel)
            #correcting for grazing transits by multiplying b by 1-rp/rs
            #tdur=pm.Deterministic("tdur",(2*pm.math.sqrt((1+ror)**2-b**2))/vrel)
            
            #if force_tdur:
            #    Adding a potential to force our transit towards the observed transit duration:
            #    pm.Potential("tdur_prior", -0.05*len(x)*abs(pm.math.log(tdur/dur)))        
            
            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                third_light = pm.Deterministic("third_light", pm.math.power(2.511,-1*deltamag_contam)) #Factor to multiply normalised lightcurve by
            else:
                third_light = 0.0

            # Compute the model light curve using starry
            if self.detns[planet]['orbit_flag']=='duo':
                #Doing lightcurves for both transits and then summing:
                light_curves = pm.math.sum(pm.math.stack([xo.LimbDarkLightCurve(u_star).get_light_curve(
                                                        orbit=orbit, r=r_pl/109.1, t=x, texp=cad),
                                                xo.LimbDarkLightCurve(u_star).get_light_curve(
                                                        orbit=orbit, r=r_pl/109.1, t=x-tcen2, texp=cad)]),axis=-1)*(1+third_light)
                pm.math.printing.Print("duo light_curves")(light_curves)
            else:
                light_curves = pm.math.sum(xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit, r=r_pl/109.1, t=x, texp=cad),axis=-1)*(1+third_light)
            #transit_light_curve = pm.math.sum(light_curves, axis=-1)
            depth_lc = pm.Deterministic("depth_lc",pm.math.min(light_curves))
            
            light_curve = pm.Deterministic("light_curve", light_curves + flux_trend)
            pm.math.printing.Print("lc")(light_curves)
            pm.math.printing.Print("trend")(flux_trend)
            pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)
            print(model.check_test_point())
            if fit_poly:
                map_soln = pmx.optimize(start=model.test_point,vars=[trend],verbose=False)
                print(map_soln['trend'],map_soln['log_ror'],map_soln['b'],map_soln['tcen'],map_soln['tdur'],map_soln['depth_lc'],map_soln['circ_per'])
                map_soln = pmx.optimize(start=map_soln,vars=[trend,log_ror,tdur,b,tcen],verbose=False)
                print(map_soln['trend'],map_soln['log_ror'],map_soln['b'],map_soln['tcen'],map_soln['tdur'],map_soln['depth_lc'],map_soln['circ_per'])

                # Fit for the maximum a posteriori parameters
            else:
                map_soln = pmx.optimize(start=model.test_point,vars=[mean],verbose=False)
                map_soln = pmx.optimize(start=map_soln,vars=[mean,log_ror,tdur,b,tcen],verbose=True)
            
            map_soln, func = pmx.optimize(start=map_soln,verbose=False,return_info=True)
        
        #Reconstructing best-fit model into a dict:
        best_fit={'log_lik_mono':-1*func['fun'],'model_success':str(func['success'])}
        
        if sample_model:
            with model:
                trace = pm.sample(tune=1200, draws=500, chains=3, cores=1, regularization_steps=20,
                                   start=map_soln, target_accept=0.9)
            
            for col in trace.varnames:
                if 'interval__' not in col:
                    if trace[col].size<2:
                        best_fit[col]=np.nanmedian(trace[col])
                        best_fit[col+"_sd"]=np.nanstd(trace[col])
                        best_fit[col+"_samps"]=trace[col]
                    else:
                        best_fit[col]=np.nanmedian(trace[col],axis=0)
                        best_fit[col+"_sd"]=np.nanstd(trace[col],axis=0)
                        best_fit[col+"_samps"]=trace[col]
        else:
            for col in map_soln:
                if 'interval__' not in col:
                    if map_soln[col].size==1:
                        best_fit[col]=float(map_soln[col])
                    else:
                        best_fit[col]=map_soln[col].astype(float)


        #print(func)
        interpt=np.linspace(best_fit['tcen']-winsize,best_fit['tcen']+winsize,600)
        if 'third_light' not in best_fit:
            best_fit['third_light']=np.array(0.0)
        
        transit_zoom = (xo.LimbDarkLightCurve(u_star).get_light_curve(
                            orbit=xo.orbits.KeplerianOrbit(r_star=self.Rstar['val'],m_star=self.Mstar['val'],
                                                        period=best_fit['circ_per'],t0=best_fit['tcen'], b=best_fit['b']),
                                                        r=best_fit['r_pl']/109.1, t=interpt, texp=cad
                                                        )*(1+best_fit['third_light'])
                    ).eval().ravel()

        
        #print({bf:type(best_fit[bf]) for bf in best_fit})
        #print(best_fit["vrel"],best_fit["b"],map_soln["tdur"],best_fit["tcen"])
        if np.isnan(best_fit["tdur"]):
            best_fit['tdur']=dur
        #Adding depth:
        best_fit['x']=x
        best_fit['ymodel']=map_soln['light_curve']/self.lc.flx_unit
        best_fit['y']=y
        best_fit['yerr']=yerr
        best_fit['depth']=np.max(transit_zoom)-np.min(transit_zoom)
        #err = std / sqrt(n_pts in transit)
        best_fit['depth_err']=np.nanstd(y[abs(x-best_fit['tcen'])<0.475*best_fit["tdur"]])/\
                            np.sqrt(np.sum(abs(x-best_fit['tcen'])<0.475*best_fit["tdur"]))
        best_fit['snr']=best_fit['depth']/(best_fit['depth_err'])
        
        
        best_fit['interpmodel']=interp.interp1d(np.hstack((-10000,interpt-best_fit['tcen'],10000)),
                                        np.hstack((0.0,transit_zoom,0.0)))

        best_fit.update(self.get_snr_red(best_fit['tcen'], best_fit['tdur'], planet, best_fit['depth'], 
                                         tcen_2=self.detns[planet]['tcen_2'] if 'tcen_2' in self.detns[planet] else None, 
                                         period=self.detns[planet]['period'] if 'period' in self.detns[planet] else None))

        for col in best_fit:
            self.detns[planet][col+"_monofit"] = best_fit[col]
        '''
        print("time stuff:", best_fit['tcen'],interpt[0],interpt[-1],
            "\nfit stuff:",best_fit['r_pl'],best_fit['b'], best_fit['logror'],best_fit['tdur'],(1+best_fit['third_light'])/lc['flux_unit'],
            "\ndepth stuff:",best_fit['depth'],best_fit['depth_err'],lc['flux_unit'],np.min(transit_zoom),np.min(map_soln['light_curve']))'''
    
    def get_snr_red(self, tcen, tdur, planet, depth, tcen_2=None, period=None):
        """Calculating std in typical bin with width of the transit duration, to compute SNR_red

        Args:
            tcen (float): Transit epoch
            tdur (float): Transut duration
            planet (str): string for the candidate idenfier
            depth (float): Depth (as a ratio) of the 
            tcen_2 (float, optional): Second transit time centre (for duo transiting planets). Defaults to None.
            period (float, optional): Orbital period. Defaults to None.

        Returns:
            dict: Dictionary with 'N_trans' (number of transits), 'cdpp' (scatter at the transit duration) and 'snr_r' (S/N of the transit w.r.t. red noise at the transit duration)
        """
        print(tcen, tdur, planet, depth)
        outdic={}
        if planet in self.monos:
            oot_mask=self.lc.mask*(abs(self.lc.time-tcen)>0.5)*(abs(self.lc.time-tcen)<25)
            outdic['Ntrans']=1
        elif planet in self.duos:
            iper=abs(tcen_2 - tcen)
            oot_mask=self.lc.mask*(abs((self.lc.time-tcen-0.5*iper)%iper-0.5*iper)>0.5)
            outdic['Ntrans']=2
        elif planet in self.multis:
            oot_mask=self.lc.mask*(abs((self.lc.time-tcen-0.5*period)%period-0.5*period)>0.5)
            durobs = np.sum([float(cad.split('_')[1])/86400 for cad in self.lc.cadence[~oot_mask]])
            outdic['Ntrans']=durobs/tdur
        print(oot_mask,np.sum(oot_mask),self.lc.flx_unit)
        binlc=tools.bin_lc_segment(np.column_stack((self.lc.time[oot_mask],self.lc.flux[oot_mask]*self.lc.flx_unit,
                                                    self.lc.flux_err[oot_mask]*self.lc.flx_unit)),tdur)
        outdic['cdpp'] = 1.05*np.nanmedian(abs(np.diff(binlc[:,1])))#np.nanstd(binlc[:,1])
        outdic['snr_r']=depth/(outdic['cdpp']/np.sqrt(outdic['Ntrans']))

        return outdic
    
    def plot_vet(self):
        return None

    def vet(self,planet,tests=['all'], variable_llk_thresh=-6,centroid_llk_thresh=-6, **kwargs):
        """Vet candidate

        Args:
            planet (str): String ID for planet candidate
            tests (list, optional): list of tests to perform. Can be one of:
                     - 'multidur'; 
                     - 'model_variability'; Compares transit model fit to a variable (sinusoid) model fit
                     - 'model_step'; Compares transit model fit to a step model fit
                     - 'check_centroid'; Checks if the transit flux model is also significant in x/y centroids
                     - 'check_instrum'; Compares planet SNR to SNR of all candidates at that timestamp
                     - 'all'; performs all above tests
            variable_llk_thresh (int, optional): [description]. Defaults to -6.
            centroid_llk_thresh (int, optional): [description]. Defaults to -6.
        """
        if 'all' in tests:
            tests=['check_snr_r','multidur','model_step','model_variability','check_asteroid','check_centroid','check_instrum']

        if planet in self.multis and 'multidur' in tests and planet not in self.fps:
            if ((self.detns[planet]['period_monofit']/self.detns[planet]['period'])<0.1)|((self.detns[planet]['period_monofit']/self.detns[planet]['period'])>10):
                #Density discrepancy of 10x, likely not possible on that period
                self.detns[planet]['flag']='discrepant duration'

        if 'model_variability' in tests and planet not in self.fps:
            self.model_variability_fp(planet, how='sin', thresh=variable_llk_thresh, **kwargs)

        if planet in self.monos and 'model_step' in tests and planet not in self.fps:
            #In the Mono case, we will fit both a sin and a step model:
            self.model_variability_fp(planet, how='step', **kwargs)

        if planet in self.monos and self.mission.lower()!='k2' and planet not in self.fps:
            #Checks to see if dip is due to background asteroid and if that's a better fit than the transit model:
            self.model_asteroid_fp(planet, **kwargs)

        if 'check_centroid' in tests and planet not in self.fps:
            #Checks to see if dip is combined with centroid
            self.model_centroid_shift(planet, centroid_llk_thresh, **kwargs)

        if self.mission=='tess' and 'check_instrum' in tests and planet not in self.fps:
            self.check_instrum_noise(planet)
    
    def dipmodel_step(self,params,x):
        #model for a simple step function either side of two polynomials
        #params = [tcen],[poly_before],[poly_after]
        npolys=int(0.5*(len(params)-1))
        assert len(params)%2==1, "must have odd number of parameters, as both sets of polynomials must be equal length, plus the time of step"
        return np.sum([np.polyval( params[1:1+npolys], x)*(x<params[0]),
                        np.polyval( params[-npolys:], x)*(x>=params[0])],axis=0)
    
    def dipmodel_sinusoid(self,params,x):
        #Sinusoidal model aligned with dip.
        # tcen, log(dur), log(dep), [n x polynomials]
        newt=(x-params[0])/(4.5*np.exp(params[1]))*2*np.pi-np.pi*0.5
        return np.polyval(params[3:],x) + np.exp(params[2])*(np.sin(newt))
    
    def dipmodel_polynomial(self,params,x):
        #Polynomial model
        #params = [polynomials]
        return np.polyval(params,x)

    def dipmodel_centroid(self,params,t,interpmodel,order):
        #params=xdep, ydep, xpolyvals, ypolyvals
        xdip = params[0]*interpmodel(t)
        ydip = params[1]*interpmodel(t)
        xmod = np.polyval(params[2:2+(order+1)],t)+xdip
        ymod = np.polyval(params[2+(order+1):],t)+ydip
        return xmod,ymod

    def dipmodel_gaussian(self, params,x):
        dip=1.0+np.exp(params[0])*np.exp(-1*np.power(x - params[2], 2.) / (2 * np.power((0.075*np.exp(params[1])), 2.)))
        mod = np.polyval(params[3:],x)*dip
        return mod    
    
    def log_likelihood(self, params,x,y,yerr,model):
        ymodel=model(params,x)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - ymodel) ** 2 / sigma2)
    
    def log_priors(self,params,priors,weight=0.1):
        lnprior=0
        for p in range(len(params)):
            if priors[p][0]=='norm':
                lnprior+=self.log_gaussian(params[p],priors[p][1],priors[p][2])
            elif priors[p][0]=='uniform':
                lnprior+=-250*(int((params[p]>priors[p][0])&(params[p]>priors[p][1]))-1)
        return lnprior

    def log_gaussian(self,x, mu, sig, weight=0.1):
        return -1*weight*np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

    def neg_log_prob(self,params,priors,x,y,yerr,model):
        lnprior=self.log_priors(params,priors)
        llk = self.log_likelihood(params, x, y, yerr,model)
        return -1*(lnprior + llk)
    
    def optimize_model(self,params,priors,x,y,yerr,model,method='L-BFGS-B'):
        mod_res=optim.minimize(self.neg_log_prob,params,args=(priors,x,y,yerr,model),method=method)
        mod_res['llk']=self.log_likelihood(mod_res['x'],x,y,yerr,model)
        mod_res['bic']=np.log(len(x))*len(mod_res['x']) - 2 * mod_res['llk']
        return mod_res
    # self.dipmodel_gaussian(params,x)
    # self.dipmodel_centroid(params,t,interpmodel,order)
    # self.dipmodel_sinusoid(params,x)
    # self.dipmodel_step(params,x,npolys)
    # np.polyval(params,x)

    def model_variability_fp(self,planet):
        #modelling: sin, step and polynomial
        x=self.detns[planet]['x_monofit']-self.detns[planet]['tcen_monofit']
        y=self.detns[planet]['y_monofit']
        yerr=self.detns[planet]['yerr_monofit']
        tdur=self.detns[planet]['tdur_monofit']
        depth=self.detns[planet]['depth_monofit']
        rms=np.nanstd(self.detns[planet]['y_monofit'])

        outTransit=abs(x)>0.55*tdur
        best_mod_res={}
        best_mod_res['sin']={'fun':1e30,'bic':1e9,'sin_llk':-1e9}
        best_mod_res['step']={'fun':1e30,'bic':1e9,'sin_llk':-1e9}
        best_mod_res['poly']={'fun':1e30,'bic':1e9,'sin_llk':-1e9}
        
        basesinpriors=[['norm',0.0,0.5],
                    ['norm',np.log(tdur),3.0],
                   ['norm',np.log(np.percentile(y,95)-np.percentile(y,5)),4]]
        basesteppriors=[['norm',0.0,0.5*tdur]]
        methods=['L-BFGS-B','Nelder-Mead','Powell']
        n=0
        while n<21:
            if np.sum(outTransit)>20:
                rand_choice=np.random.random(len(x))<0.95
            else:
                rand_choice=np.tile(True,len(x))
            step_guess=np.random.normal(0.0,0.5*tdur)
            if np.sum(x<step_guess)==0:
                step_guess=x[1+int(n/7)+1]
            elif np.sum(x>step_guess)==0:
                step_guess=x[-1*(1+int(n/7))-1]
            npolystep=1+int(n/7)
            polysteppriors=[]
            for n1 in range(2):
                for n2 in range(npolystep+1):
                    polysteppriors+=[['norm',0.0,10**(0.25-(n2*(0.25-np.log10(rms))/(npolystep+1)))]]
            
            stepparams=np.hstack(([step_guess,
                                   np.polyfit(x[(x<=step_guess)&rand_choice],
                                              y[(x<=step_guess)&rand_choice],npolystep),
                                   np.polyfit(x[(x>step_guess)&rand_choice],
                                              y[(x>step_guess)&rand_choice],npolystep)
                                   ]))
            mod_res_step=self.optimize_model(stepparams,basesteppriors+polysteppriors,x,y,yerr,
                                             self.dipmodel_step,method=methods[n%3])
            npolysin=1+int(n/7)
            sinparams=np.hstack(([np.random.normal(0.0,0.5*tdur),
                                   np.log(tdur)+np.random.normal(0.0,0.5)],
                                   np.log(depth)+np.random.normal(0.0,0.5),
                                   np.polyfit(x[outTransit&rand_choice],
                                              y[outTransit&rand_choice],npolysin)))
            polysinpriors=[]
            for n2 in range(npolysin+1):
                polysinpriors+=[['norm',0.0,10**(0.25-(n2*(0.25-np.log10(rms))/(npolysin+1)))]]

            mod_res_sin=self.optimize_model(sinparams,basesinpriors+polysinpriors,x,y,yerr,
                                             self.dipmodel_sinusoid,method=methods[n%3])
            polypriors=[]
            npoly=1+int(n/5)
            for n2 in range(npoly+1):
                polypriors+=[['norm',0.0,10**(0.25-(n2*(0.25-np.log10(rms))/(npoly+1)))]]

            polyparams=np.polyfit(x[rand_choice],
                                  y[rand_choice],npoly)
            mod_res_poly=self.optimize_model(polyparams,polypriors,x,y,yerr,
                                             self.dipmodel_polynomial,method=methods[n%3])

            if mod_res_step['bic']<best_mod_res['step']['bic']:
                best_mod_res['step']=mod_res_step
                best_mod_res['step']['npolys']=1+int(n/7)
            if mod_res_sin['bic']<best_mod_res['sin']['bic']:
                best_mod_res['sin']=mod_res_sin
                best_mod_res['sin']['npolys']=1+int(n/7)
            if mod_res_poly['bic']<best_mod_res['poly']['bic']:
                best_mod_res['poly']=mod_res_poly
                best_mod_res['poly']['npolys']=2+int(n/5)
            n+=1

        #Adding to the dictionary
        self.detns[planet]['variability_res']=best_mod_res

    def model_asteroid_fp(self,planet,dur_region=3.5):
        
        tdur=self.detns[planet]['tdur_monofit']
        tcen=self.detns[planet]['tcen_monofit']
        nearish_region=np.max([4.5,tdur*np.clip(dur_region,2/tdur,5/tdur)]) #For the background fit, we'll take a region 9d long
        nearishTrans=(abs(self.lc.time-tcen)<nearish_region)&self.lc.mask.astype(bool)
        if not hasattr(self.lc,"bg_flux") or np.sum(np.isfinite(self.lc.bg_flux[nearishTrans]))==0:
            return None
        cad=np.nanmedian(np.diff(self.lc.time[nearishTrans]))
        #Checking for a "jump" in background flux, which we can remove:
        if np.max(np.diff(self.lc.time[nearishTrans]))>0.4:
            jump_n=np.argmax(np.diff(self.lc.time[nearishTrans]))
            jump_time=0.5*(self.lc.time[nearishTrans][jump_n]+self.lc.time[nearishTrans][jump_n+1])
            if jump_time < tcen:
                nearishTrans=(self.lc.time>jump_time)&((self.lc.time-tcen)<nearish_region)&self.lc.mask
            elif jump_time > tcen:
                nearishTrans=((self.lc.time-tcen)>(-1*nearish_region))&(self.lc.time<jump_time)&self.lc.mask
        #Cutting nans/infs
        nearishTrans[nearishTrans]=np.isfinite(self.lc.bg_flux[nearishTrans])

        x=self.lc.time[nearishTrans]-tcen
        y=self.lc.bg_flux[nearishTrans]
        y/=np.nanmedian(y)
        rms=np.nanmedian(abs(np.diff(y)))
        #yerr=self.lc.bg_flux_err[nearishTrans]/np.nanmedian(self.lc.bg_flux[nearishTrans]) if hasattr(self.lc,"bg_flux_err") else 
        yerr=np.tile(rms,np.sum(nearishTrans))
        outTransit=(abs(x)>tdur*0.75)
        inTransit=(abs(x)<tdur*0.35)
        log_height_guess=np.log(2*np.clip((np.nanmedian(y[inTransit])-np.nanmedian(y[outTransit])),
                                        0.00001,1000) )
        basedippriors= [['norm',log_height_guess,3.0],
                        ['norm',np.log(np.clip(1.4*tdur,6*cad,3.0)),0.25],
                        ['norm',0.0,0.75*tdur]]
        best_mod_res={}
        best_mod_res['nodip']={'fun':1e30,'bic':1e9,'llk':-1e20}
        best_mod_res['dip']={'fun':1e30,'bic':1e9,'llk':-1e20}
        methods=['L-BFGS-B', 'Nelder-Mead', 'Powell']
        n=0
        while n<20:
            polyorder=1+int(n/5)
            rand_choice=np.random.random(len(y))<0.9
            
            #Gaussian dip case:
            dip_args= np.hstack(([np.random.normal(log_height_guess,0.25)-(n%4)/2.0,
                                  np.log10(1.5*tdur)+abs(np.random.normal(0.0,0.5)),
                                  np.random.normal(0.0,0.5*tdur)],
                                  np.polyfit(x[outTransit&rand_choice],
                                             y[outTransit&rand_choice],
                                             polyorder)))
            polydippriors=[]
            for n1 in range(polyorder+1):
                polydippriors+=[['norm',0.0,10**(0.25-(n1*(0.25-np.log10(rms))/(polyorder+1)))]]

            this_dip_res=self.optimize_model(dip_args,basedippriors+polydippriors,x,y,yerr,
                                             self.dipmodel_gaussian,method=methods[n%3])
            if this_dip_res['bic']<best_mod_res['dip']['bic']:
                best_mod_res['dip']=this_dip_res
                best_mod_res['dip']['npoly']=polyorder
            
            #Simple polynomial fit for the no-dip case:
            nodip_args= np.polyfit(x[rand_choice],y[rand_choice],polyorder)
            polynodippriors=[]
            for n1 in range(polyorder+1):
                polynodippriors+=[['norm',0.0,10**(0.25-(n1*(0.25-np.log10(rms))/(polyorder+1)))]]
            this_nodip_res=self.optimize_model(nodip_args,polynodippriors,x,y,yerr,
                                               self.dipmodel_polynomial,method=methods[n%3])
            if this_nodip_res['bic']<best_mod_res['nodip']['bic']:
                best_mod_res['nodip']=this_nodip_res
                best_mod_res['nodip']['npoly']=polyorder

            n+=1
        self.detns[planet]['bg_res']=best_mod_res
        self.detns[planet]['bg_res']['bg_fitx']=x
        self.detns[planet]['bg_res']['bg_fity']=y
        self.detns[planet]['bg_res']['bg_fityerr']=yerr
        

    def model_centroid_shift(self,planet,llk_thresh):
        return None

    def check_instrum_noise(self,planet,update=False):
        '''# Using the processed "number of TCEs per cadence" array, we try to use this as a proxy for Instrumental noise in TESS
        # Here we simply use the detected SNR over the instrumental noise SNR as a proxy
        INPUTS:
        - lc
        - monotransit dic
        - jd_base (assumed to be that of TESS)'''

        if update:
            tools.update_tce_timeseres()

        tcen=self.detns[planet]['tcen_monofit']+self.lc.jd_base
        tdur=self.detns[planet]['tdur_monofit']
        tcefile=pd.read_csv(MonoData_tablepath+"/tess_tce_fractions.csv.gz",index_col=0,compression="gzip")
        tcefile_nearby=tcefile.loc[abs(tcefile.index.values-tcen)<5]
        in_trans_av=np.nanmedian(tcefile_nearby.loc[abs(tcefile_nearby.index.values-tcen)<np.clip(0.4*tdur,0.076,2),'TCE_fraction'].values)
        out_trans_av=np.nanmedian(tcefile_nearby.loc[abs(tcefile_nearby.index.values-tcen)>0.75*tdur,'TCE_fraction'].values)
        rms=np.nanmedian(abs(np.diff(tcefile_nearby['TCE_fraction'].values)))
        self.detns[planet]['excess_tce_noise']=np.clip((in_trans_av-out_trans_av)/rms,0,100)
        