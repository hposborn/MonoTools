import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iteround import saferound

from astropy.io import fits
from astropy.io import ascii
from scipy.signal import savgol_filter

import os

import astropy.units as u
from astropy.units import cds
from astropy import constants as c

from astropy import units
from astropy.coordinates.sky_coordinate import SkyCoord

import pickle
import cloudpickle
import os.path
from datetime import datetime

import glob

import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

MonoData_tablepath = os.path.join(os.path.dirname( __file__ ),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname( __file__ ),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')
if not os.path.isdir(MonoData_savepath):
    os.mkdir(MonoData_savepath)

from . import tools
from . import search
from . import lightcurve
#from . import tools
#from .stellar import starpars
#from . import MonoSearch

#creating new hidden directory for theano compilations:
#setting float type:
floattype=np.float64

import pymc as pm
import pymc_ext as pmx
from celerite2.pymc import terms as pymc_terms
import celerite2.pymc
import arviz as az
from pytensor import tensor

class monoModel():
    """The core MonoTools model class

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    #The default monoModel class. This is what we will use to build a Pymc3 model

    def __init__(self, ID, mission, lc=None, rvs=None, planets=None, overwrite=False, savefileloc=None, **kwargs):
        """Initialises MonoTools fit model

        Args:
            ID (int): Mission ID of object (i.e. TIC, EPIC or KIC)
            mission (str): Mission (i.e. 'tess', 'k2', 'kepler')
            lc (dict, optional): light curve dictionary with keys 'time','flux','flux_err',etc. Defaults to None
            rvs (dict, optional): RV dictionary with keys 'time','rv','rv_err',etc. Defaults to None.
            planets (dict, optional): Planet parameter dictionary, with entries (and dictionaries) corresponding to each planet. Defaults to None.
            overwrite (bool, optional): Defaults to False.
            savefileloc (str, optional): File save location (will be automatically generated). Defaults to None.
        """
        #Initialising default modelling parameters. These can also be updated in init_model()
        self.overwrite=overwrite
        self.defaults={'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'use_GP':True,           # use_GP - bool - Fit a GP for the transit photometry
                       'train_GP':True,         # train_GP - bool - Train the GP using out-of-transit photometry
                       'local_spline':False,    # local_spline - bool - For mono/duo/trio transits, include a local spline around transit to incorporate extra correlated noise.
                       'n_spline_pts':6,        # n_spline_pts - int - Number of spline breakpoints for the around transit bspline
                       'spline_order':3,        # spline_order - int - Order for spline splitting. Default = 3 (cubic)
                       'fit_no_flatten':False,  # fit_no_flatten - bool - If no GP, by default we spline-flatten the lightcurve. Use fit_no_flatten to turn this off
                       'constrain_LD':True,     # constrain_LD - bool - Use constrained LDs from model or unconstrained?
                       'ld_mult':3.,            # ld_mult - float - How much to multiply theoretical LD param uncertainties
                       'use_L2':False,           # use_L2 - bool - Fit for "second light" (i.e. a binary or planet+blend)
                       'FeH':0.0,               # FeH - float - Stellar FeH
                       'load_from_file':False,  # load_from_file - bool - Load previous model?
                       'cut_distance':3.75,     # cut_distance - float - cut out points further than cut_distance*Tdur. 0.0 means no cutting
                       'mask_distance': 0.666,       #Distance, in transit durations, from set transits, to "mask" as in-transit data when e.g. flattening.
                       'force_match_input':None,# force_match_input - Float/None add potential with this the sigma between the input and the output logror and logdur to force MCMC to match the input duration & maximise logror [e.g. 0.1 = match to 1-sigma=10%]
                       'debug':False,           # debug - bool - print debug statements?
                       'fit_params':['logror','b','tdur', 't0'], # fit_params - list of strings - fit these parameters. Options: ['logror', 'b' or 'tdur', 'ecc', 'omega']
                       'marginal_params':['per','ecc','omega'], # marginal_params - list of strings - marginalise over these parameters. Options: ['per', 'b' ´or 'tdur', 'ecc', 'omega','logror']
                       'interpolate_v_prior':True, # Whether to use interpolation to produce transit velocity prior
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'per_index':-8/3,        # per_index - float - period prior index e.g. P^{index}. -8/3 in to Kipping 2018
                       'mono_model_type':'split_per_gaps', # mono_model_type - str - What type of monotransit model? 'param_per_gap' or 'split_per_gaps'
                       'mutual_incl_sigma':None,# mutual_incl_sigma - float - Mutual inclination standard deviation. Normally not used but for coplanar resonant systems can help constrain b.
                       'derive_K':True,         # If we have RVs, do we derive K for each alias or fit for a single K param
                       'pred_all':False,        # Do we predict all time array, or only a cut-down version?
                       'use_multinest':False,   # use_multinest - bool - currently not supported
                       'use_pymc3':True,        # use_pymc3 - bool
                       'bin_all':False,         # bin_all - bool - Bin all points to 10mins (to speed up certain)
                       'bin_all_size':10/1440., # bin_all_size - float - Bin size if binning all points in minutes (default to 10mins)
                       'bin_oot':True,          # bin_oot - bool - Bin points outside the cut_distance to 30mins
                       'model_t03_ttv':False,   # model_t03_ttv - bool - Whether to model the third transit as a seperate parameter, otherwise it is constrained with the other params
                       'timing_sigma':0.1,      # timing_sigma - float - the sigma (as a function of transit duration) to use when setting transit times. Default=0.1*t_D
                       'periodic_kernel':None}  # periodic_kernel - dict - info (period, period_err, logamp, logamp_err) to create a periodic Celerite kernel on top of the GP used already. Default is None (i.e. no periodic kernel)
        for param in self.defaults:
            if not hasattr(self,param) or self.overwrite:
                if param in kwargs:
                    setattr(self,param,kwargs[param])
                else:
                    setattr(self,param,self.defaults[param])

        self.id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
                     'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}
        ID=ID.replace('_','') if type(ID)==str and '_' in ID else ID
        ID=ID.replace(' ','') if type(ID)==str and ' ' in ID else ID
        self.ID=ID

        self.mission=mission

        #Initalising save locations
        if self.load_from_file and not self.overwrite:
            #Catching the case where the file doesnt exist:
            success = self.load_model_from_file(loadfile=savefileloc)
            self.load_from_file = success
        
        if lc is None or type(lc) is not lightcurve.multilc:
            lc = lightcurve.multilc(ID,mission)
        self.lc=lc
        setattr(lc,'near_trans', {'all':np.tile(False, len(lc.time))} if not hasattr(lc,'near_trans') else lc.near_trans)
        setattr(lc,'in_trans', {'all':np.tile(False, len(lc.time))} if not hasattr(lc,'in_trans') else lc.in_trans)

        if not np.all(np.array([hasattr(self,starpar) for starpar in ['rhostar','Rstar','logg','Mstar']])):
            self.init_starpars()
        
        assert ID is not None and mission is not None and lc is not None

        if not self.load_from_file:
            if rvs is not None:
                self.add_rvs(rvs)
            #If we don;t have a past model to load, we load the lightcurve and, if a "planets" dict was passes, initialise those:
            self.planets={};self.rvplanets={}
            self.multis=[];self.monos=[];self.duos=[];self.trios=[]

            if planets is not None:
                for pl in self.planets:
                    self.add_planet(self, planets[pl]['orbit_flag'], planets[pl], pl)
            self.savefileloc=savefileloc

    def load_model_from_file(self, loadfile=None):
        """Load a monoModel object direct from file.

        Args:
            loadfile (str, optional): File to load from, otherwise it takes the default location using `get_savename`. Defaults to None.

        Returns:
            bool: Whether the load is successful
        """
        if loadfile is None:
            self.get_savename(how='load')
            loadfile=self.savenames[0]+'_model.pickle'
            if self.debug: print(self.savenames)

        if os.path.exists(loadfile.replace('_model.pickle','_trace.nc')):
            #New version using cloudpickle
            pick=cloudpickle.load(open(loadfile,'rb'))
            for key in pick:
                setattr(self,key,pick[key])
            del pick
            setattr(self,'trace',az.InferenceData.from_netcdf(loadfile.replace('_model.pickle','_trace.nc')))
        elif os.path.exists(loadfile):
            #Loading old version using pickle from pickled dictionary
            pick=pickle.load(open(loadfile,'rb'))
            assert not isinstance(pick, monoModel)
            #print(In this case, unpickle your object separately)
            for key in pick:
                setattr(self,key,pick[key])
            del pick
            return True
        else:
            return False

    def save_model_to_file(self, savefile=None, limit_size=False):
        """Save a monoModel object direct to file.

        Args:
            savefile (str, optional): File location to save to, otherwise it takes the default location using `get_savename`. Defaults to None.
            limit_size (bool, optional): If we want to limit size this function can delete unuseful hyperparameters before saving. Defaults to False.
        """
        if savefile is None:
            if not hasattr(self,'savenames'):
                self.get_savename(how='save')
            savefile=self.savenames[0]+'_model.pickle'
        if hasattr(self,'trace'):
            try:
                self.trace.to_netcdf(self.savenames[0]+'_trace.nc')
            except:
                try:
                    #Stacking/unstacking removes Multitrace objects:
                    self.trace.to_netcdf(self.savenames[0]+'_trace.nc')
                except:
                    print("Still a save error after unstacking")
        excl_types=[az.InferenceData]
        cloudpickle.dump({attr:getattr(self,attr) for attr in self.__dict__ if type(getattr(self,attr)) not in excl_types},open(savefile,'wb'))

        # #Loading from pickled dictionary
        # saving={}
        # if limit_size and hasattr(self,'trace'):
        #     #We cannot afford to store full arrays of GP predictions and transit models
        #     # But first we need to turn the predicted arrays into percentiles now for plotting:
        #     if self.use_GP:
        #         self.init_gp_to_plot()
        #     self.init_trans_to_plot()

        #     #And let's clip gp and lightcurves and pseudo-variables from the trace:
        #     remvars=[var for var in self.trace.posterior if (('gp_' in var or '_gp' in var or 'light_curve' in var) and np.product(self.trace.posterior[var].shape)>6*len(self.trace.posterior['Rs'])) or '__' in var]
        #     for key in remvars:
        #         #Permanently deleting these values from the trace.
        #         self.trace.remove_values(key)
        #     #medvars=[var for var in self.trace.posterior if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
        # n_bytes = 2**31
        # max_bytes = 2**31-1

        # bytes_out = pickle.dumps(self.__dict__)
        # #bytes_out = pickle.dumps(self)
        # with open(savefile, 'wb') as f_out:
        #     for idx in range(0, len(bytes_out), max_bytes):
        #         f_out.write(bytes_out[idx:idx+max_bytes])
        # del saving
        #pick=pickle.dump(self.__dict__,open(loadfile,'wb'))

    def drop_planet(self, name):
        """Removes planet from saved planet properties dict (mod.planets)

        Args:
            name (str): Name of planet within planets dictionary
        """
        if not hasattr(self,'deleted'):
            self.deleted={}
        if name in self.planets:
            self.deleted[name]=self.planets.pop(name)
        if name in self.monos:
            _=self.monos.remove(name)
        if name in self.multis:
            _=self.multis.remove(name)
        if name in self.duos:
            _=self.duos.remove(name)
        if name in self.trios:
            _=self.trios.remove(name)

    def add_planet(self, pltype, pl_dic, name, **kwargs):
        """Adds any planet type to planet properties dict

        Args:
            pltype (str): Type of planet - i.e. 'mono', 'duo', 'multi' or 'rvplanet'
            pl_dic (dict): Dictionary of planet properties which requires:
                depth: transit depth in unitless flux ratio (NOT ppm or ppt)
                tdur: transit duration in same units as time array (i.e. days)
                tcen: transit epoch in same units as time array (i.e. TJD)
                period: (if multi or rvplanet) transit period in same units as time array (i.e. days)
                period_err: (if rvplanet) transit period error in same units as time array (i.e. days)
                tcen_2: (if duo) second transit epoch in same units as time array (i.e. TJD)
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
        """
        if 'log_ror' not in pl_dic:
            if 'ror' in pl_dic:
                pl_dic['log_ror']=np.log(pl_dic['ror'])
            elif 'depth' in pl_dic:
                assert pl_dic['depth']<0.25 #Depth must be a ratio (not in mmags)
                pl_dic['ror']=pl_dic['depth']**0.5
                pl_dic['log_ror']=np.log(pl_dic['ror'])
        if 'ror' not in pl_dic:
            pl_dic['ror']=np.exp(pl_dic['log_ror'])

        if 'r_pl' not in pl_dic and hasattr(self,'Rstar'):
            pl_dic['r_pl']=pl_dic['ror']*self.Rstar[0]*109.2

        #Adding dict as planet:
        if pltype=='multi':
            self.add_multi(pl_dic, name,**kwargs)
        elif pltype=='rvplanet':
            self.add_rvplanet(pl_dic, name,**kwargs)
        else:
            if 'period_err' not in pl_dic:
                pl_dic['period_err']=999
            if pltype=='duo':
                self.add_duo(pl_dic, name,**kwargs)
            elif pltype=='trio':
                self.add_trio(pl_dic, name,**kwargs)
            elif pltype=='mono':
                if 'period' not in pl_dic:
                    pl_dic['period']=999
                self.add_mono(pl_dic, name,**kwargs)

    def add_rvplanet(self, pl_dic, name,**kwargs):
        """Adds non-transiting planet seen only in RVs to planet properties dict

        Args:
            pl_dic (dict): Dictionary of planet properties which requires:
                tcen: transit epoch in same units as time array (i.e. TJD)
                tcen_err: transit epoch error (optional)
                period: transit period in same units as time array (i.e. days)
                period_err: transit period error in same units as time array (i.e. days)
                K: RV semi-amplitude in m/s
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
        """
        assert name not in list(self.planets.keys())+list(self.rvplanets.keys())
        #
        # Dictionary requires: period, period_err, tcen, tcen_err, semi-amplitude K
        #
        # As these rv planets need to be treated seperately, you can specify the following (usually global) parameters:
        # - assume_circ=False (default is the same as the other planets, i.e. False)
        # - ecc_prior='auto' (default is the same as the other planets)
        # - Note that tcen is the expected time of transit (not necessarily the t0 expected for pure RV curves).

        assert 'tcen' in pl_dic and 'K' in pl_dic and 'period' in pl_dic and 'period_err' in pl_dic
        pl_dic['logK']=np.log(pl_dic['K'])

        #Adding error on logK:
        if 'K_err' not in pl_dic:
            pl_dic['logK_err']=1.5
        else:
            pl_dic['logK_err']=pl_dic['K_err']/pl_dic['K']

        #Adding error on tcen:
        if 'tcen_err' not in pl_dic:
            pl_dic['tcen_err']=pl_dic['period']
        for attr in ['assume_circ','ecc_prior','derive_K']:
            pl_dic[attr]=getattr(self,attr) if attr not in pl_dic else pl_dic[attr]

        self.rvplanets[name]=pl_dic

    def add_multi(self, pl_dic, name, update_per=False, **kwargs):
        """Adds a transiting planet with multiple, consecutive transits to planet properties dict

        Args:
            pl_dic (dict): Dictionary of planet properties which requires:
                tcen: transit epoch in same units as time array (i.e. TJD)
                tcen_err: transit epoch error (optional)
                period: transit period in same units as time array (i.e. days)
                period_err: transit period error in same units as time array (i.e. days)
                K: RV semi-amplitude in m/s
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
        """
        assert name not in self.planets
        #Adds planet with multiple eclipses
        if not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.5*pl_dic['tdur']/pl_dic['period']

        if 'ror' not in pl_dic:
            assert pl_dic['depth']<0.25 #Depth must be a ratio (not in mmags)
            pl_dic['ror']=np.sqrt(pl_dic['depth']) if hasattr(self,'depth') else 0.025

        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            #Estimating b from simple geometry:

            pl_dic['b']=np.clip((1+pl_dic['ror'])**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*pl_dic['period']*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        if update_per:
            if not hasattr(self.lc,'flux_flat'):
                try:
                    self.lc.flatten()
                except:
                    setattr(self.lc,'flux_flat',self.lc.flux[:])
            pl_dic['period']=tools.update_period_w_tls(self.lc.time[self.lc.mask], self.lc.flux_flat[self.lc.mask], pl_dic['period'])
        
        phase=(self.lc.time-pl_dic['tcen']-0.5*pl_dic['period'])%pl_dic['period']-0.5*pl_dic['period']
        self.lc.near_trans[name] = abs(phase)<self.cut_distance*pl_dic['tdur']
        self.lc.near_trans['all']+= self.lc.near_trans[name][:]
        self.lc.in_trans[name] = abs(phase)<self.mask_distance*pl_dic['tdur']
        self.lc.in_trans['all']+= self.lc.in_trans[name][:]

        self.planets[name]=pl_dic
        self.multis+=[name]

    def add_mono(self, pl_dic, name, gap_prob_thresh=1e-10, prob_index=-5/3, gap_width_thresh=0.5, max_fraction_increase=0.09,**kwargs):
        """Adds a transiting planet with a single transit to planet properties dict

        Args:
            pl_dic (dict): Dictionary of planet properties which requires:
                tcen: transit epoch in same units as time array (i.e. TJD)
                tcen_err: transit epoch error (optional)
                period: transit period in same units as time array (i.e. days)
                period_err: transit period error in same units as time array (i.e. days)
                K: RV semi-amplitude in m/s
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
            gap_prob_thresh (float,optional): Threshold in prior probability, below which we remove a "gap" from the period space.
            gap_width_thresh (float,optional): Threshold in observation width (in units of transit duration), below which we merge gaps and ignore the observations
                                               i.e. if there is a 2-hour observation which splits the probability space up into gaps, then we can ignore it for a 4-hour transit if gap_width_thresh>0.5
            max_fraction_increase (float,optional): In the "split_per_gaps" case where we break large gaps up into smaller chunks, this is the maximum gap width as a ratio of the gap start period. Defaults to 0.09 (9%)
        """
        #Adds planet with single eclipses

        #Adding the planet to the lightcurve mask arrays first (as compute_period_gaps performs flattening).
        self.lc.near_trans[name] = abs(self.lc.time-pl_dic['tcen'])<self.cut_distance*pl_dic['tdur']
        self.lc.near_trans['all'] += self.lc.near_trans[name][:]
        self.lc.in_trans[name] = abs(self.lc.time-pl_dic['tcen'])<self.mask_distance*pl_dic['tdur']
        self.lc.in_trans['all'] += self.lc.in_trans[name][:]

        min_P=np.min([self.planets[pl]['period'] for pl in self.multis]) if len(self.multis)>0 else None

        #Calculating whether there are period gaps:
        assert name not in self.planets
        p_gaps,rms_series=self.compute_period_gaps(pl_dic['tcen'], tdur=pl_dic['tdur'], depth=pl_dic['depth'], 
                                                   gap_width_thresh=gap_width_thresh, min_P=min_P,**kwargs)
        pl_dic['per_gaps']={'gap_starts':p_gaps[:,0],'gap_ends':p_gaps[:,1],
                           'gap_widths':p_gaps[:,1]-p_gaps[:,0],'gap_probs':prob_index*(p_gaps[:,1]**(prob_index)-p_gaps[:,0]**(prob_index))}
        pl_dic['per_gaps']['gap_probs']/=np.sum(pl_dic['per_gaps']['gap_probs'])
        
        # Removing gaps which have negligible prior probability:
        prob_thresh_ix = pl_dic['per_gaps']['gap_probs']>gap_prob_thresh
        for col in pl_dic['per_gaps']:
            pl_dic['per_gaps'][col]=pl_dic['per_gaps'][col][prob_thresh_ix]

        if self.mono_model_type=="split_per_gaps":
            # In this case, we split the allowed period distribution into N bins (where N<100) and compute the implied probability as for a duo
            mod_per_gaps={'gap_starts':[],'gap_ends':[],'gap_widths':[]}
            
            for gp in range(len(pl_dic['per_gaps']['gap_widths'])):
                if pl_dic['per_gaps']['gap_widths'][gp]/pl_dic['per_gaps']['gap_starts'][gp]>max_fraction_increase:
                    startendarr=np.exp(np.linspace(np.log(pl_dic['per_gaps']['gap_starts'][gp]),np.log(pl_dic['per_gaps']['gap_ends'][gp]),int(np.ceil(np.exp(np.log(pl_dic['per_gaps']['gap_widths'][gp]/pl_dic['per_gaps']['gap_starts'][gp])-np.log(max_fraction_increase)))+1)))
                    print(np.exp(np.log(pl_dic['per_gaps']['gap_widths'][gp]/pl_dic['per_gaps']['gap_starts'][gp])-np.log(max_fraction_increase)),startendarr)
                    #Splitting into N gaps:
                    mod_per_gaps['gap_starts']+=list(startendarr[:-1])
                    mod_per_gaps['gap_ends']+=list(startendarr[1:])
                    mod_per_gaps['gap_widths']+=list(np.diff(startendarr))
                else:
                    mod_per_gaps['gap_starts']+=[pl_dic['per_gaps']['gap_starts'][gp]]
                    mod_per_gaps['gap_ends']+=[pl_dic['per_gaps']['gap_ends'][gp]]
                    mod_per_gaps['gap_widths']+=[pl_dic['per_gaps']['gap_widths'][gp]]
            mod_per_gaps['gap_starts']=np.array(mod_per_gaps['gap_starts'])
            mod_per_gaps['gap_ends']=np.array(mod_per_gaps['gap_ends'])
            mod_per_gaps['gap_widths']=np.array(mod_per_gaps['gap_widths'])
            mod_per_gaps['gap_probs']=prob_index*(mod_per_gaps['gap_ends']**(prob_index)-mod_per_gaps['gap_starts']**(prob_index))
            mod_per_gaps['gap_probs']/=np.sum(mod_per_gaps['gap_probs'])
            pl_dic['per_gaps']=mod_per_gaps

        pl_dic['P_min']=pl_dic['per_gaps']['gap_starts'][0]
        pl_dic['rms_series']=rms_series
        if 'log_ror' not in pl_dic:
            if 'ror' in pl_dic:
                pl_dic['log_ror']=np.log(pl_dic['ror'])
            elif 'depth' in pl_dic:
                assert pl_dic['depth']<0.25 #Depth must be a ratio (not in mmags)
                pl_dic['ror']=pl_dic['depth']**0.5
                pl_dic['log_ror']=np.log(pl_dic['ror'])
        pl_dic['ngaps']=len(mod_per_gaps['gap_starts'])

        if 'b' not in pl_dic and 'depth' in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            assert pl_dic['depth']<0.25 #Depth must be a ratio (not in mmags)
            ror=np.sqrt(pl_dic['depth']) if hasattr(self,'depth') else 0.0
            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+pl_dic['ror'])**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*np.min(pl_dic['per_gaps']['gap_starts'])*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        self.planets[name]=pl_dic
        self.monos+=[name]
        #self.calc_gap_edge_likelihoods(name)

    def compute_rms_series(self,tdur,split_gap_size=2.0,n_steps_per_dur=7,**kwargs):
        """Computing an RMS time series for the lightcurve by binning

        Args:
            tdur (float): transit duration
            split_gap_size (float, optional): Duration at which to cut the lightcurve and compute in loops. Defaults to 2.0.
            n_steps_per_dur (int, optional): number of steps with which to cut up each duration. Odd numbers work most uniformly. Defaults to 7.

        Returns:
            array: 2-column array of bin times, and bin RMS
        """

        if not hasattr(self.lc,'flux_flat') or len(self.lc.flux_flat)!=len(self.lc.flux_err):
            self.lc.flatten(transit_mask=~self.lc.in_trans['all'],**kwargs)

        rms_series=np.zeros((len(self.lc.time)))
        binsize=(1/n_steps_per_dur)*tdur
        if np.nanmax(np.diff(self.lc.time))>split_gap_size:
            loop_blocks=np.array_split(np.arange(len(self.lc.time)),np.where(np.diff(self.lc.time)>split_gap_size)[0])
        else:
            loop_blocks=[np.arange(len(self.lc.time))]
        rms_series_sh=[]
        bins=[]
        for sh_time in loop_blocks:
            thesebins=np.arange(np.nanmin(self.lc.time[sh_time])-tdur,
                                np.nanmax(self.lc.time[sh_time])+tdur+binsize, binsize)
            theserms=np.zeros_like(thesebins)
            for n,b in enumerate(thesebins):
                ix=(abs(b-self.lc.time[sh_time])<(0.5*tdur))*self.lc.mask[sh_time]
                if np.sum(ix)>1:
                    theserms[n]=tools.weighted_avg_and_std(self.lc.flux_flat[sh_time][ix],
                                                           self.lc.flux_err[sh_time][ix])[1]
                else:
                    theserms[n]=np.nan
            bins+=[thesebins]
            rms_series_sh+=[np.array(theserms)]

            '''
            lc_segment=np.column_stack((self.lc.time[sh_time],self.lc.flux_flat[sh_time],
                                        self.lc.flux_err[sh_time],self.lc.mask[sh_time].astype(int)))
            digi=np.digitize(lc_segment[:,0],
                             np.arange(np.min(lc_segment[:,0])-0.5*binsize,np.max(lc_segment[:,0])+0.5*binsize,binsize))
            digi=np.hstack((digi,0.0))
            unq_digi=np.unique(digi[:-1][lc_segment[:,3]==1.0])
            digis=np.vstack(([digi[:-1]+n for n in np.arange(0,n_steps_per_dur,1)-int(np.floor(n_steps_per_dur*0.5))]))
            rms_series_sh=np.tile(1000.,digis.shape)

            for d in unq_digi:
                rms_series_sh[digis==d]=tools.weighted_avg_and_std(lc_segment[(digi[:-1]==d)&(lc_segment[:,3]==1.0),1],
                                                                   lc_segment[(digi[:-1]==d)&(lc_segment[:,3]==1.0),2])[1]

            rms_series_sh[digis==-1]=1000.
            rms_series[sh_time] = np.sqrt(np.sum(rms_series_sh**2,axis=0))
            '''
        return np.column_stack((np.hstack(bins),np.hstack(rms_series_sh)))

    def compute_period_gaps(self,tcen,tdur,depth,max_per=1250,SNR_thresh=4, gap_width_thresh=0.5, min_P=None,**kwargs):
        """Compute regions of period space which are not covered by photometry (i.e. find the gaps)
                e.g. Given the time array, the t0 of transit, and the fact that another transit is not observed,
                we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
                In this case, a list of periods is returned, with all points within 0.5dur to be cut

        Args:
            tcen (float): transit epoch
            tdur (float): transit duration (in days)
            depth (float): transit depth (in ratio, i.e. NOT ppt/ppm)
            max_per (int, optional): Maximum period bound. Defaults to 1250.
            SNR_thresh (int, optional): [description]. Defaults to 4.
            gap_width_thresh (float, optional): Width of photometric timeseries (in transit durations) below which we ignore)
            min_P (float, optional): Minimum period from external sources (i.e. internal planets)

        Returns:
            gap_start_ends (array): Period gap start and ends, with each gap start/end forming a tuple entry to the array
            rmsseries (array): Array of binned times and RMS scatter for each time at the implied duration, as calculated by `compute_rms_series`
        """

        rmsseries = self.compute_rms_series(tdur,**kwargs)

        dist_from_t0=abs(tcen-rmsseries[:,0])

        #Here we only want the parts of the timeseries where we could have detected a transit (if there was one):
        dist_from_t0=dist_from_t0[(((depth/self.lc.flx_unit)/rmsseries[:,1])>SNR_thresh)*(~np.isnan(rmsseries[:,1]))]
        dist_from_t0=np.sort(dist_from_t0)
        if min_P is not None and min_P>0:
            #Cutting all points below our pre-set minimum period (if we have one)
            dist_from_t0=np.hstack((min_P,dist_from_t0[dist_from_t0>min_P]))
        gaps=np.where(np.diff(dist_from_t0)>(0.9*tdur))[0]
        if len(gaps)>0:
            #Looping from minimum distance from transit to gap, to maximum distance from transit to end-of-lc
            checkpers=np.arange(dist_from_t0[gaps[0]]-tdur,np.max(dist_from_t0)+tdur,tdur*0.166)
            checkpers_ix=self.check_periods_have_gaps(checkpers,tdur,tcen,**kwargs).astype(int) #Seeing if each period has data coverage

            #Creating an array of tuples which form start->end of specific gaps:
            starts=checkpers[:-1][np.diff(checkpers_ix)==1.0]
            #Because the above array ends beyond the max lc extent, we need to add the max period to this array:
            ends=np.hstack((checkpers[1:][np.diff(checkpers_ix)==-1.0],max_per))
            #print(starts,ends)
            gap_start_ends=np.array([(starts[n],ends[n]) for n in range(len(starts))])
        else:
            gap_start_ends=np.array([(np.max(dist_from_t0),max_per)])
        return gap_start_ends,rmsseries
    
    def make_phase(self,time,tcens,per):
        if len(tcens)==1 or (len(tcens)==3 and tcens[2] is None):
            return (time-tcens[0]-per*0.5)%per-per*0.5
        else:
            # Trio - getting "phase" such that time is a polynomial and each transit matches the tcen perfectly. For TTV cases.
            # Simply phase-folding for each tcen and then taking the phase in three regions - where each is closest to the transit time.
            tcens = np.array(tcens)[:,None][np.argmin(np.column_stack([abs(time-tc) for tc in tcens]),axis=1)][:,0]
            #print(ix, ix.shape)
            #print(ix,np.column_stack([(time-tc-per*0.5)%per-per*0.5 for tc in tcens])[ix,:])
            return np.hstack([(time-tcens-per*0.5)%per-per*0.5])
            
    def check_periods_have_gaps(self,pers,tdur,tcen,tcen_2=None,tcen_3=None,match_trans_thresh=2.5,coverage_thresh=0.15,**kwargs):
        """Checking a list of potential periods and seeing if period are observed by counting the number of points in-transit

        Args:
            pers (list): List of potential periods (e.g. for each Duotransit alias)
            tdur (float): Transit duration (days)
            tcen (float): Epoch of first transit (days)
            tcen_2 (float, optional): Epoch of second transit (days). Defaults to None.
            match_trans_thresh (float, optional): Whether to check the known three transits to see if they fit all 3 transit times. Defaults to 2.5
            coverage_thresh (float, optional): Threshhold below which we say no transit was observed. Defaults to 0.15.

        Returns:
            array: Boolean array specifying which of the input period list is observed.
        """
        #
        trans=abs(self.lc.time[self.lc.mask]-tcen)<0.45*tdur
        if np.sum(trans)==0:
            trans=abs(self.lc.time[self.lc.mask]-tcen)<0.5*tdur
        if self.debug: print(np.sum(trans),"points in transit")
        #Adding up in-transit cadences to give days in transit:
        days_in_known_transits = [np.sum(np.array([cad.split('_')[1] for cad in self.lc.cadence[self.lc.mask][trans]]).astype(float))/86400]
        if tcen_2 is not None:
            trans2=abs(self.lc.time[self.lc.mask]-tcen_2)<0.45*tdur
            days_in_known_transits += [np.sum(np.array([cad.split('_')[1] for cad in self.lc.cadence[self.lc.mask][trans2]]).astype(float))/86400]
            coverage_thresh*=0.5 #Two transits already in number count, so to compensate we must decrease the thresh
        if tcen_3 is not None:
            assert tcen_2 is not None, "Must have both tcen_2 and tcen_3 if speciffying tcen_3"
            trans3=abs(self.lc.time[self.lc.mask]-tcen_3)<0.45*tdur
            
            days_in_known_transits += [np.sum(np.array([cad.split('_')[1] for cad in self.lc.cadence[self.lc.mask][trans3]]).astype(float))/86400]
            coverage_thresh*=0.66 #Three transits in n_pts count, so to compensate we must decrease the thresh

        check_pers_ix=[]
        #Looping through periods
        
        for per in pers:
            #WE NEED A WAY TO MAKE THIS FOLLOW POTENTIAL CHANGES IN P FROM T0 TO T1 to T2... POLYNOMIAL?
            #print(tcen,tcen_2,tcen_3,per,(tcen_2-tcen-per*0.5)%per-per*0.5,(tcen_3-tcen-per*0.5)%per-per*0.5,0.75*tdur)
            phase=self.make_phase(self.lc.time[self.lc.mask],[tcen,tcen_2,tcen_3],per)
            intr=abs(phase)<0.45*tdur
            #We first need to check whether it matches with all known transits
            #print(np.sum(trans2&intr),np.sum(trans3&intr))
            if tcen_3 is not None and abs((tcen_2-tcen-per*0.5)%per-per*0.5)>match_trans_thresh*tdur and abs((tcen_3-tcen-per*0.5)%per-per*0.5)>match_trans_thresh*tdur:
                # and (np.sum(trans2&intr)<0.75*days_in_known_transits[1] or np.sum(trans3&intr)<0.75*days_in_known_transits[2]):
                #Either second or third transit does not match with this period... Adding zero to list.
                #check_pers_ix+=[False]#
                days_in_tr=np.sum([float(self.lc.cadence[ncad].split('_')[1])/86400 for ncad in np.arange(len(self.lc.cadence))[self.lc.mask][intr]])
                check_pers_ix+=[days_in_tr<(1.0+coverage_thresh)*np.sum(days_in_known_transits)]
                #print(per,"NO",abs((tcen_2-tcen-per*0.5)%per-per*0.5),match_trans_thresh*tdur,abs((tcen_3-tcen-per*0.5)%per-per*0.5),match_trans_thresh*tdur)
            else:
                #Here we need to add up the cadences in transit (and not simply count the points) to check coverage:
                days_in_tr=np.sum([float(self.lc.cadence[ncad].split('_')[1])/86400 for ncad in np.arange(len(self.lc.cadence))[self.lc.mask][intr]])
                #print(days_in_tr,(1.0+coverage_thresh),np.sum(days_in_known_transits))
                check_pers_ix+=[days_in_tr<(1.0+coverage_thresh)*np.sum(days_in_known_transits)]
                #Less than 15% of another eclipse is covered
                #print(per,"OK",np.sum(intr),days_in_known_transits,np.sum(days_in_known_transits),days_in_tr)
        return np.array(check_pers_ix)

    def compute_period_aliases(self,pl_dic,dur=0.5,**kwargs):
        """Calculating Duotransit period aliases
             Given the time array, the t0 of transit, and the fact that two transits are observed,
              we want to calculate a distribution of periods, and then remove those which are impossible/observed

        Args:
            pl_dic (dict): Planet properties dictionary for the selected Duotransit, as described in `add_duo` or `add_trio`
            dur (float, optional): Transit duration [days]. Defaults to 0.5.

        Returns:
            dict: Updated planet properties dictionary with `period_aliases` term
        """
        # Given the time array, the t0 of transit, and the fact that two transits are observed,
        #   we want to calculate a distribution of impossible periods to remove from the period alias list
        #finding the longest unbroken observation for P_min guess
        #P_min = np.max(np.hstack((self.compute_period_gaps(pl_dic['tcen'],dur=pl_dic['tdur']),
        #                          self.compute_period_gaps(pl_dic['tcen_2'],dur=pl_dic['tdur']))))
        #print(P_min,np.ceil(pl_dic['period']/P_min),np.ceil(pl_dic['period']/P_min))
        check_pers_ints = np.arange(1,np.ceil(pl_dic['period']/10),1.0)
        if 'tcen_3' in pl_dic:
            #Also need to check that the implied periods match the third period
            check_pers_ix = self.check_periods_have_gaps(pl_dic['period']/check_pers_ints,pl_dic['tdur'],pl_dic['tcen'],tcen_2=pl_dic['tcen_2'],tcen_3=pl_dic['tcen_3'],**kwargs)
        else:
            check_pers_ix = self.check_periods_have_gaps(pl_dic['period']/check_pers_ints,pl_dic['tdur'],pl_dic['tcen'],tcen_2=pl_dic['tcen_2'],**kwargs)

        pl_dic['period_int_aliases']=check_pers_ints[check_pers_ix]
        if len(pl_dic['period_int_aliases'])==0:
            print("problem in computing Duotransit aliases")
        else:
            print(pl_dic)
            pl_dic['period_aliases']=pl_dic['period']/pl_dic['period_int_aliases']
            pl_dic['P_min']=np.min(pl_dic['period_aliases'])
        return pl_dic

    def calc_gap_edge_likelihoods(self,mono,n_check=100,**kwargs):
        """Calculate the effect on likelihood of a transit model for those transits which occur at the "edges" of photometric data coverage.
            e.g. In the case that we are not creating transit models for each period gap, we want to calculate how the
                "edges" of those gaps affect the log probability.Effectively we'll calculate the likelihood of the edges 
                of the gaps w.r.t the initial-fit transit model. This will then become a 1D (e.g. linear) polynomial 
                which sets the logprior at the edges of each monotransit gap.
            The saved monotransit dictionary is updated within the model.
                
        Args:
            mono (str): Name of the selected Monotransit in the `mod.planets`
            n_check (int, optional): Number of positions to check across each edge. Defaults to 100.
        """
        from scipy import interpolate
        from scipy.stats import mode
        starts=[]
        ends=[]
        tzoom=np.linspace(self.planets[mono]['tcen']-1*self.planets[mono]['tdur'],
                          self.planets[mono]['tcen']+1*self.planets[mono]['tdur'],300)
        for ngap in range(self.planets[mono]['ngaps']):
            #Taking the min and max period for this gap:
            pmin=self.planets[mono]['per_gaps']['gap_starts'][ngap]
            pmax=self.planets[mono]['per_gaps']['gap_ends'][ngap]
            start_gaps=np.linspace(pmin,pmin+0.75*self.planets[mono]['tdur'],int(n_check*0.5))
            mid_gap=0.5*(pmin+pmax)
            end_gaps=np.linspace(pmax-0.75*self.planets[mono]['tdur'],pmax,int(n_check*0.5))
            #Cropping the lightcurve to only keep points next to the gaps (and not in-transit)
            round_gaps=(
                (abs((self.lc.time-self.planets[mono]['tcen']-0.5*pmin)%pmin-0.5*pmin)<5*self.planets[mono]['tdur']) + \
                (abs((self.lc.time-self.planets[mono]['tcen']-0.5*pmax)%pmax-0.5*pmax)<5*self.planets[mono]['tdur'])) * \
                       (abs(self.lc.time-self.planets[mono]['tcen'])>4*self.planets[mono]['tdur'])
            if 'interpmodel' in self.planets[mono]:
                bfmodel=self.planets['interpmodel']
            else:
                #Using exoplanet to generate a lightcurve given the initial info:
                init_b=0.41
                orbit = xo.orbits.KeplerianOrbit(r_star=self.Rstar[0], rho_star=self.rhostar[0],
                                                 t0=self.planets[mono]['tcen'],
                                                 period=18226*self.rhostar[0]*(self.Rstar[0]*2*np.sqrt((1+self.planets[mono]['depth']**0.5)**2-init_b**2)/self.planets[mono]['tdur'])**(-3))
                light_curve = xo.LimbDarkLightCurve(np.array([0.25,0.5])).get_light_curve(orbit=orbit,
                                                                            r=self.planets[mono]['depth']**0.5,
                                                                            t=tzoom).eval()

                bfmodel=interpolate.interp1d(np.hstack((-1000,tzoom-self.planets[mono]['tcen'],1000)),
                                        np.hstack((0.0,light_curve.T[0],0.0)))

            sigma2 = lc['flux_err'][round_gaps] ** 2
            pers=np.hstack((start_gaps,end_gaps,mid_gap))[None,:]
            phases=(self.lc.time[round_gaps,None]-self.planets[mono]['t0']-0.5*pers)%pers-0.5*pers
            #Calculating delta logliks (where the final "mid_gap" should be the furthest point from data, i.e. max loglik
            logliks=-0.5 * np.sum((self.lc.flux_flat[round_gaps,None]*self.lc.flx_unit - bfmodel(phases)) ** 2 / sigma2[:,None] + np.log(sigma2[:,None]),axis=0)
            logliks-=logliks[-1]
            #Adding the polynomial fits to the 'per_gaps' dict:
            starts+=[np.polyfit(start_gaps[logliks[:int(n_check*0.5)]<0]-pmin,logliks[:int(n_check*0.5)][logliks[:int(n_check*0.5)]<0],1)]
            if ngap<(self.planets[mono]['ngaps']-1):
                ends+=[np.polyfit(end_gaps[logliks[int(n_check*0.5):-1]<0]-pmax,logliks[int(n_check*0.5):-1][logliks[int(n_check*0.5):-1]<0],1)]
            else:
                ends+=[np.array([0.0,0.0])]
        self.planets[mono]['per_gaps']['start_loglik_polyvals']=np.vstack(starts)
        self.planets[mono]['per_gaps']['end_loglik_polyvals']=np.vstack(ends)

    def add_trio(self, pl_dic,name,maxint=24,**kwargs):
        """add_trio Adds a transiting planet with two non-consecutive transits to planet properties dict

        Args:
            pl_dic (dict): Dictionary of planet properties which requires:
                tcen: transit epoch in same units as time array (i.e. TJD)
                tcen_2: second transit epoch in same units as time array (i.e. TJD)
                tcen_3: third transit epoch in same units as time array (i.e. TJD)
                period: (optional) transit period in same units as time array (i.e. days)
                period_err: (optional) transit period error in same units as time array (i.e. days)
                K: RV semi-amplitude in m/s
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
            maxint (int): Maximum integer to which to search for a close period for the three transits
        """
        assert name not in self.planets
        #Adds planet with two eclipses and unknown period between these
        tcens=[pl_dic['tcen'],pl_dic['tcen_2'],pl_dic['tcen_3']]
        pl_dic['tcen']=np.sort(tcens)[0]
        pl_dic['tcen_2']=np.sort(tcens)[1]
        pl_dic['tcen_3']=np.sort(tcens)[-1]
        pairs=[pl_dic['tcen_2'] - pl_dic['tcen'],pl_dic['tcen_3']-pl_dic['tcen_2']]
        pl_dic['maxperiod_pair']=np.argmin(pairs)
        pl_dic['maxperiod']=np.min(pairs)
        #Specifically need tcen and tcen_2 as paired across the max period. tcen_3 is then least useful.
        pratios21=abs(pl_dic['tcen_2']-pl_dic['tcen'])/abs(pl_dic['tcen_3']-pl_dic['tcen'])
        explore_pratio=np.vstack([[n,m,n/m,(pratios21-(n/m))**2*np.sqrt(n*m)] for n in range(1,maxint) for m in range(1,maxint) if n<m])
        assert np.min(explore_pratio[:,3])<1.e-4, "Period implied by three transits must be close to integer ratio but"+str(explore_pratio[np.argmin(explore_pratio[:,3]),:2])+" has a weighted distance of"+str(np.min(explore_pratio[:,3]))+"which is larger than 1e-4"
        pl_dic['p_ratio_21']=explore_pratio[np.argmin(explore_pratio[:,3]),:2]
        pl_dic['p_ratio_32']=[explore_pratio[np.argmin(explore_pratio[:,3]),1]-explore_pratio[np.argmin(explore_pratio[:,3]),0],explore_pratio[np.argmin(explore_pratio[:,3]),1]]
        
        # In the case that the max implied period is not actually possible (e.g. a 2/11 ratio) we need the new max period.
        pl_dic['maxperiod']=pl_dic['maxperiod']/pl_dic['p_ratio_21'][0] if pl_dic['p_ratio_21'][0]<pl_dic['p_ratio_32'][0] else pl_dic['maxperiod']/pl_dic['p_ratio_32'][0] 
        
        if 'period' not in pl_dic:
            pl_dic['period']=pl_dic['maxperiod']

        if 'period_err' not in pl_dic or not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.1666*pl_dic['tdur']
        #Calculating P_min and the integer steps
        print(pl_dic)
        pl_dic=self.compute_period_aliases(pl_dic,**kwargs)
        pl_dic['npers']=len(pl_dic['period_int_aliases'])

        pl_dic['ror']=np.sqrt(pl_dic['depth']) if not hasattr(pl_dic,'ror') else 0.01

        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0

            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+pl_dic['ror'])**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*np.median(pl_dic['period_aliases'])*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5

        for per in pl_dic['period_aliases']:
            phase=(self.lc.time-pl_dic['tcen']-0.5*per)%per-0.5*per
            self.lc.near_trans[name]=abs(phase)<self.cut_distance*pl_dic['tdur']
            self.lc.near_trans['all']+=self.lc.near_trans[name][:]
        self.lc.in_trans[name]=(abs(self.lc.time-pl_dic['tcen'])<self.mask_distance*pl_dic['tdur'])|(abs(self.lc.time-pl_dic['tcen_2'])<self.mask_distance*pl_dic['tdur'])
        self.lc.in_trans['all']+=self.lc.in_trans[name][:]

        self.planets[name]=pl_dic
        self.trios+=[name]

    def add_duo(self, pl_dic, name,**kwargs):
        """add_duo Adds a transiting planet with two non-consecutive transits to planet properties dict

        Args:
            pl_dic (dict): Dictionary of planet properties which requires:
                tcen: transit epoch in same units as time array (i.e. TJD)
                tcen_2: second transit epoch in same units as time array (i.e. TJD)
                period: (optional) transit period in same units as time array (i.e. days)
                period_err: (optional) transit period error in same units as time array (i.e. days)
                K: RV semi-amplitude in m/s
            name (str): Planet name (i.e. '01', 'b', or 'Malcolm')
        """
        assert name not in self.planets
        #Adds planet with two eclipses and unknown period between these
        if 'period' not in pl_dic:
            pl_dic['period']=abs(pl_dic['tcen_2'] - pl_dic['tcen'])
        else:
            assert pl_dic['period']==abs(pl_dic['tcen']-pl_dic['tcen_2'])
        if 'period_err' not in pl_dic or not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.1666*pl_dic['tdur']
        tcens=np.array([pl_dic['tcen'],pl_dic['tcen_2']])
        pl_dic['tcen']=np.max(tcens)
        pl_dic['tcen_2']=np.min(tcens)
        #Calculating P_min and the integer steps
        pl_dic=self.compute_period_aliases(pl_dic,**kwargs)
        pl_dic['npers']=len(pl_dic['period_int_aliases'])

        pl_dic['ror']=np.sqrt(pl_dic['depth']) if not hasattr(pl_dic,'ror') else 0.01

        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0

            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+pl_dic['ror'])**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*np.median(pl_dic['period_aliases'])*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5

        for per in pl_dic['period_aliases']:
            phase=(self.lc.time-pl_dic['tcen']-0.5*per)%per-0.5*per
            self.lc.near_trans[name]=abs(phase)<self.cut_distance*pl_dic['tdur']
            self.lc.near_trans['all']+=self.lc.near_trans[name][:]
        self.lc.in_trans[name]=(abs(self.lc.time-pl_dic['tcen'])<self.mask_distance*pl_dic['tdur'])|(abs(self.lc.time-pl_dic['tcen_2'])<self.mask_distance*pl_dic['tdur'])
        self.lc.in_trans['all']+=self.lc.in_trans[name][:]

        self.planets[name]=pl_dic
        self.duos+=[name]
    
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
        if Rstar is None and hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'rad' in self.lc.all_ids['tess']['data']:
            #Radius info from lightcurve data (TIC)
            if 'eneg_Rad' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Rad'] is not None and self.lc.all_ids['tess']['data']['eneg_Rad']>0:
                Rstar=self.lc.all_ids['tess']['data'][['rad','eneg_Rad','epos_Rad']].values
            else:
                Rstar=self.lc.all_ids['tess']['data'][['rad','e_rad','e_rad']].values
        if Teff is None and hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'rad' in self.lc.all_ids['tess']['data']:
            if 'eneg_Teff' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Teff'] is not None and self.lc.all_ids['tess']['data']['eneg_Teff']>0:
                Teff=self.lc.all_ids['tess']['data'][['Teff','eneg_Teff','epos_Teff']].values
            else:
                Teff=self.lc.all_ids['tess']['data'][['Teff','e_Teff','e_Teff']].values
        if logg is None:
            if 'eneg_logg' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_logg'] is not None and self.lc.all_ids['tess']['data']['eneg_logg']>0:
                logg=self.lc.all_ids['tess']['data'][['logg','eneg_logg','epos_logg']].values
            else:
                logg=self.lc.all_ids['tess']['data'][['logg','e_logg','e_logg']].values
        
        #Still None at this point means no TIC data...
        if Rstar is None:
            Rstar=np.array([1.0,0.08,0.08])
        if Teff is None:
            Teff=np.array([5227,100,100])
        if logg is None:
            logg=np.array([4.3,1.0,1.0])

        self.Rstar=np.array(Rstar).astype(float)
        self.Teff=np.array(Teff).astype(float)
        self.logg=np.array(logg).astype(float)
        self.FeH=FeH

        if Mstar is not None:
            self.Mstar = Mstar if type(Mstar)==float else float(Mstar[0])
        #Here we only have a mass, radius, logg- Calculating rho two ways (M/R^3 & logg/R), and doing weighted average
        if rhostar is None:
            rho_logg=[np.power(10,self.logg[0]-4.43)/self.Rstar[0]]
            rho_logg+=[np.power(10,self.logg[0]+self.logg[1]-4.43)/(self.Rstar[0]-self.Rstar[1])/rho_logg[0]-1.0,
                       1.0-np.power(10,self.logg[0]-self.logg[2]-4.43)/(self.Rstar[0]+self.Rstar[2])/rho_logg[0]]
            if Mstar is not None:
                rho_MR=[Mstar[0]/self.Rstar[0]**3]
                rho_MR+=[(Mstar[0]+Mstar[1])/(self.Rstar[0]-abs(self.Rstar[1]))**3/rho_MR[0]-1.0,
                         1.0-(Mstar[0]-abs(Mstar[2]))/(self.Rstar[0]+self.Rstar[2])**3/rho_MR[0]]
                print(rho_MR)
                #Weighted sums of two avenues to density:
                rhostar=[rho_logg[0]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                         rho_MR[0]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])]
                rhostar+=[rhostar[0]*(rho_logg[1]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                                      rho_MR[1]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])),
                          rhostar[0]*(rho_logg[2]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                                      rho_MR[2]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2]))]
                self.Mstar=Mstar
            else:
                rhostar=[rho_logg[0],rho_logg[0]*rho_logg[1],rho_logg[0]*rho_logg[2]]

            self.rhostar=np.array(rhostar).astype(float)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3
        else:
            self.rhostar=np.array(rhostar).astype(float)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3


    def get_savename(self, how='load',overwrite=None):
        """Adds unique savename prefixes to class (self.savenames) with two formats:
        '[savefileloc]/[T/K]IC[11-number ID]_[20YY-MM-DD]_[n]...'
        '[savefileloc]/[T/K]IC[11-number ID]_[n]...'

        Args:
            how (str, optional): 'load' or 'save'. Defaults to 'load'.
            overwrite (bool, optional): if how='save', whether to overwrite past save or not. Defaults to None.
        """
        if overwrite is None and hasattr(self,'overwrite'):
            overwrite=self.overwrite
        else:
            overwrite=True

        if not hasattr(self,'savefileloc') or self.savefileloc is None or overwrite:
            self.savefileloc=os.path.join(MonoData_savepath,self.id_dic[self.mission]+str(self.ID).zfill(11))
        if not os.path.isdir(self.savefileloc):
            os.system('mkdir '+self.savefileloc)
        if self.debug: print(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"*model.pickle"))
        pickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"*model.pickle"))
        if self.debug: print(pickles,[len(p) for p in pickles])
        pickles=[p for p in pickles if len(p.split('/')[-1].split('_'))==4]
        if self.debug: print(pickles)
        if how == 'load' and len(pickles)>1:
            #finding most recent pickle:
            date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
            if self.debug: print([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles])
            datepickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_*model.pickle"))
            if len(datepickles)>1:
                if self.debug: print([int(nmdp.split('_')[2]) for nmdp in datepickles])
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            elif len(datepickles)==1:
                nsim=0
            elif len(datepickles)==0:
                print("problem - no saved mcmc files in correct format")
        elif how == 'load' and len(pickles)==1:
            date=pickles[0].split('_')[1]
            nsim=pickles[0].split('_')[2]
        else:
            #Either pickles is empty (no file to load) or we want to save a fresh file:
            #Finding unique
            date=datetime.now().strftime("%Y-%m-%d")
            datepickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_*"))
            if len(datepickles)==0:
                nsim=0
            elif overwrite:
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            else:
                #Finding next unused number with this date:
                nsim=1+np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
        self.savenames=[os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_"+str(int(nsim))), os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11))]

    def add_rvs(self, rv_dic, n_poly_trend=2, overwrite=False, **kwargs):
        """Add a dictionary of rvs with arrays of:

        Args:
            rv_dic (dict): Dictionary of radial velocity info.
                           Necessary values: "time", "rv", "rv_err"
                           Optional values: "rv_unit" (assumes m/s),
                                            "tele_index" (unique telescope id for each RV),
                                            "jd_base" (assumes same as lc),
                                            "jitter_min" (default=0.5)
                                            "logjitter_mean" (default=0.0)
                                            "logjitter_sd" (default=0.4, or ~40%)
            n_poly_trend (int, optional): [description]. Defaults to 2.
            overwrite (bool, optional): [description]. Defaults to False.
        """
        # necessary: "time", "rv", "rv_err"
        # optional: e.g.: "rv_unit" (assumes m/s), "tele_index" (unique telescope id for each RV), "jd_base" (assumes same as lc)
        #
        # PLEASE NOTE - Due to the fact that, unlike transits, the RVs of each planet affect *all* observed RV data
        # It is not yet possible to isolate individual planet contributions (as it is with transits) and treat seperately
        # Therefore marginalising over multiple planets with ambiguous periods is not yet possible.
        # However, this should work for multi-transiting planets (known periods) with single outer companions.
        #
        # TBD - activity detrending, multiple RV sources, trends, etc

        if 'jd_base' in rv_dic and rv_dic['jd_base']!=self.lc.jd_base:
            rv_dic['time']+=(rv_dic['jd_base']-self.lc.jd_base)
            rv_dic['jd_base']=self.lc.jd_base
        else:
            rv_dic['jd_base']=self.lc.jd_base
        if rv_dic['rv_unit']=='kms' or rv_dic['rv_unit']==1000:
            #We want
            rv_dic['rv']*=1000
            rv_dic['rv_err']*=1000
            rv_dic['rv_unit']='ms'
        elif 'rv_unit' not in rv_dic or rv_dic['rv_unit']!='ms':
            print("Assuming RV unit is in m/s")
        if 'tele_index' not in rv_dic or len(rv_dic['tele_index'])!=len(rv_dic['time']):
            print("Assuming all one telescope (HARPS).")
            if 'tele_index' in rv_dic and type(rv_dic['tele_index'])==str:
                rv_dic['tele_index']=np.tile(rv_dic['tele_index'],len(rv_dic['time']))
            else:
                rv_dic['tele_index']=np.tile('h',len(rv_dic['time']))
        rv_dic['scopes'] = np.unique(rv_dic['tele_index'])
        #Building an array of ones and zeros to use later
        rv_dic['tele_index_arr']=np.zeros((len(rv_dic['time']),len(rv_dic['scopes'])))
        for ns in range(len(rv_dic['scopes'])):
            rv_dic['tele_index_arr'][:,ns]+=(rv_dic['tele_index']==rv_dic['scopes'][ns]).astype(int)

        rv_dic['jitter_min'] = 0.5 if 'jitter_min' not in rv_dic else rv_dic['jitter_min']
        rv_dic['logjitter_mean'] = 0.0 if 'logjitter_mean' not in rv_dic else rv_dic['logjitter_mean']
        rv_dic['logjitter_sd'] = 0.4 if 'logjitter_sd' not in rv_dic else rv_dic['logjitter_sd']

        self.rvs={}
        for key in rv_dic:
            if type(rv_dic)==np.ndarray and type(rv_dic)[0] in [float,np.float32,np.float64]:
                self.rvs[key]=rv_dic[key][:].astype(floattype)
            else:
                self.rvs[key]=rv_dic[key]
        self.rv_tref = np.round(np.nanmedian(rv_dic['time']),-1)
        #Setting polynomial trend
        self.rv_npoly = n_poly_trend
        
        self.rvs['init_perscope_offset'] = np.nansum(self.rvs['rv'][:,None]*self.rvs['tele_index_arr'],axis=0)/np.sum(self.rvs['tele_index_arr']>0,axis=0)
        self.rvs['init_perscope_weightederr'] = np.nansum(self.rvs['rv_err'][:,None]*self.rvs['tele_index_arr'],axis=0)/np.sum(self.rvs['tele_index_arr']>0,axis=0)/np.sqrt(np.sum(self.rvs['tele_index_arr']>0,axis=0))
        self.rvs['init_std'] = np.nanstd(self.rvs['rv']-np.sum(self.rvs['init_perscope_offset'][None,:]*self.rvs['tele_index_arr'],axis=1))

        assert len(self.trios+self.duos+self.monos)<2 #Cannot fit more than one planet with uncertain orbits with RVs (currently)

    def init_lc(self, oot_binsize=1/48, **kwargs):
        """Initialise light curve. This can be done either after or before model initialisation.
              This function creates transit maskes and flattens/bins the light curve in ways to avoid influencing the transit.
        """
        
        #Replacing all timeseries with binned timeseries:
        if self.bin_all:
            print("BIN ALL")
            print(np.min(np.diff(self.lc.time)))
            if hasattr(self,'in_trans'):
                delattr(self, 'in_trans')
            self.lc.remove_binned_arrs()
            print([ts for ts in self.lc.timeseries if type(getattr(self.lc,ts))==np.ndarray and type(getattr(self.lc,ts)[0]) in (float,np.float64)])
            print(type(self.lc.bg_flux))
            self.lc.bin(timeseries=[ts for ts in self.lc.timeseries if type(getattr(self.lc,ts))==np.ndarray and type(getattr(self.lc,ts)[0]) in (float,np.float32,np.float64)],
                        use_masked=False, binsize=self.bin_all_size)
            
            for key in self.lc.timeseries:
                if 'bin' not in key and "bin_"+key in self.lc.timeseries:
                    setattr(self.lc, key, getattr(self.lc, "bin_"+key))
                    print(key)
                elif 'bin' not in key:
                    print(key,"not binned? Removing.")
                    if 'mask' in key:
                        setattr(self.lc,key,np.tile(True,len(self.lc.bin_time)))
            self.lc.remove_binned_arrs()
            self.lc.make_mask(overwrite=True)
            print("BIN ALL")
            print(np.min(np.diff(self.lc.time)))


        #step=0.133*np.min([self.init_soln['tdur_'+pl] for pl in self.planets]) if hasattr(self,'init_soln') else 0.133*np.min([self.planets[pl]['tdur'] for pl in self.planets])
        #win=6.5*np.max([self.init_soln['tdur_'+pl] for pl in self.planets]) if hasattr(self,'init_soln') else 6.5*np.max([self.planets[pl]['tdur'] for pl in self.planets])
        
        self.lc.near_trans = {'all':np.tile(False, len(self.lc.time))}
        self.lc.in_trans   = {'all':np.tile(False, len(self.lc.time))}
        for pl in self.planets:
            if pl in self.multis:
                t0 = self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                p = self.init_soln['per_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['period']
                phase=(self.lc.time-t0-0.5*p)%p-0.5*p
            elif pl in self.trios:
                p=np.max(self.init_soln['per_'+pl]) if hasattr(self,'init_soln') else np.max(self.planets[pl]['period_aliases'])
                if self.model_t03_ttv:
                    t0s= [self.init_soln['t0_'+pl],self.init_soln['t0_2_'+pl],self.init_soln['t0_3_'+pl]] if hasattr(self,'init_soln') else [self.planets[pl]['tcen'],self.planets[pl]['tcen_2'],self.planets[pl]['tcen_3']]
                    phase=self.make_phase(self.lc.time,t0s,p)
                else:
                    t0= self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen_2']
                    p=abs(self.init_soln['t0_2_'+pl]-self.init_soln['t0_'+pl]) if hasattr(self,'init_soln') else abs(self.planets[pl]['tcen_2']-self.planets[pl]['tcen'])
                    phase=(self.lc.time-t0-0.5*p)%p-0.5*p
            elif pl in self.duos:
                t0= self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                p=abs(self.init_soln['t0_2_'+pl]-self.init_soln['t0_'+pl]) if hasattr(self,'init_soln') else abs(self.planets[pl]['tcen_2']-self.planets[pl]['tcen'])
                phase=(self.lc.time-t0-0.5*p)%p-0.5*p
            elif pl in self.monos:
                t0= self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                phase=abs(self.lc.time-t0)
            dur = self.init_soln['tdur_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tdur']
            self.lc.near_trans[pl] = abs(phase)<self.cut_distance*dur
            self.lc.near_trans['all'] += self.lc.near_trans[pl][:]
            self.lc.in_trans[pl] = abs(phase)<self.mask_distance*dur
            self.lc.in_trans['all'] += self.lc.in_trans[pl][:]

        if not self.fit_no_flatten and not self.use_GP:
            self.lc.flatten(transit_mask=~self.lc.in_trans['all'],**kwargs)

        if self.cut_distance>0 or not self.use_GP:
            if self.bin_oot:
                #Creating a pseudo-binned dataset where out-of-transit LC is binned to 30mins but near-transit is not.
                oot_binsize=1/12 if self.mission.lower()=='kepler' else oot_binsize
                print(self.lc.near_trans['all'])
                self.lc.OOTbin(near_transit_mask=self.lc.near_trans['all'],use_flat=(not self.use_GP or self.fit_no_flatten),
                               binsize=oot_binsize)
                self.model_time=self.lc.ootbin_time[:]
                self.model_flux=self.lc.ootbin_flux[:] if self.use_GP or self.fit_no_flatten else self.lc.ootbin_flux_flat[:]
                self.model_flux_err=self.lc.ootbin_flux_err[:]
                self.model_cadence=self.lc.ootbin_cadence[:]
                self.model_near_trans=self.lc.ootbin_near_trans[:]
                self.model_in_trans=self.lc.ootbin_in_trans[:]
            elif not self.bin_oot:
                self.model_time=self.lc.time[self.lc.mask*self.lc.near_trans['all']][:]
                self.model_flux=self.lc.flux[self.lc.mask*self.lc.near_trans['all']][:] if self.use_GP or self.fit_no_flatten else self.lc.flux_flat[self.lc.mask*self.lc.near_trans['all']][:]
                self.model_flux_err=self.lc.flux_err[self.lc.mask*self.lc.near_trans['all']][:]
                self.model_cadence=self.lc.cadence[self.lc.mask*self.lc.near_trans['all']][:]
                self.model_in_trans=self.lc.in_trans['all'][self.lc.mask*self.lc.near_trans['all']][:]
                self.model_near_trans=np.tile(True,len(self.model_time))
                '''self.lc_near_trans={}
                lc_len=len(self.lc.time)
                for key in self.lc:
                    if type(self.lc[key])==np.ndarray and len(self.lc[key])==lc_len:
                        self.lc_near_trans[key]=self.lc[key][self.lc.near_trans]
                    else:
                        self.lc_near_trans[key]=self.lc[key]
                '''
            if self.debug: print(np.sum(self.lc.near_trans['all']&self.lc.mask),"points in new lightcurve, compared to ",np.sum(self.lc.mask)," in original mask, leaving ",np.sum(self.lc.near_trans['all']),"points in the lc")
                
        else:
            self.model_time=self.lc.time[self.lc.mask][:]
            self.model_flux=self.lc.flux[self.lc.mask][:] if self.use_GP or self.fit_no_flatten else self.lc.flux_flat[self.lc.mask][:]
            self.model_flux_err=self.lc.flux_err[self.lc.mask][:]
            self.model_cadence=self.lc.cadence[self.lc.mask][:]
            self.model_in_trans=self.lc.in_trans['all'][self.lc.mask][:]
        
        self.cads_long=np.unique(self.lc.cadence[self.lc.mask])
        self.cads_short=np.unique(["_".join(cad.split("_")[:2]) for cad in self.cads_long])
        #In the case of different cadence/missions, we need to separate their respective errors to fit two logs2
        self.lc.cadence_index=(np.char.find(self.lc.cadence[:,None], self.cads_short[None,:])+1).astype(bool)
        self.model_cadence_index=(np.char.find(self.model_cadence[:,None], self.cads_short[None,:])+1).astype(bool)


    def init_model(self, overwrite=False, **kwargs):
        """Initalise fitting model

        Args:
            overwrite (bool, optional): whether to overwrite. Defaults to False.
        Kwargs:
            assume_circ (int, optional): Assume circular orbits (no ecc & omega)? Defaults to False
            use_GP (bool, optional): Use GPs to co-fit light curve? Defaults to True
            train_GP (bool, optional): Train the lightcurve GP on out-of-transit data before sampling? Defaults to True
            constrain_LD (bool, optional): Use constrained LDs from model or unconstrained? Defaults to True
            ld_mult (float, optional): How much to multiply theoretical LD param uncertainties. Defaults to 3.
            use_L2 (bool, optional): Fit for "second light" (i.e. a binary or planet+blend). Defaults to False
            FeH (float, optional): Stellar FeH. Defaults to 0.0
            load_from_file (bool, optional): Load previous model? Defaults to False
            cut_distance (float, optional): cut out points further than cut_distance*Tdur. 0.0 means no cutting. Defaults to 3.75
            mask_distance (float, optional): Distance, in transit durations, from set transits, to "mask" as in-transit data when e.g. flattening. 
                                        Defaults to 0.666
            force_match_input (float OR None, optional): Float/None add potential with this the sigma between the input and the output logror and logdur 
                                               to force MCMC to match the input duration & maximise logror [e.g. 0.1 = match to 1-sigma=10%]. Defaults to None
            debug (int, optional): print debug statements? Defaults to False
            fit_params (list, optional): fit these parameters. Options: ['logror', 'b' or 'tdur', 'ecc', 'omega']. 
                                         Defaults to ['logror','b','tdur', 't0']
            marginal_params (list, optional): marginalise over these parameters. Options: ['per', 'b' ´or 'tdur', 'ecc', 'omega','logror'].
                                              Defaults to ['per','ecc','omega']
            interpolate_v_prior (bool, optional): Whether to use interpolation to produce transit velocity prior. Defaults to True
            ecc_prior (str, optional): 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity. Defaults to 'auto'
            per_index (float, optional): period prior index e.g. P^{index}. -8/3 in to Kipping 2018. Defaults to -8/3
            derive_K (bool, optional): If we have RVs, do we derive K for each alias or fit for a single K param. Defaults to True
            use_multinest (bool, optional): Use Multinest sampling [NOT SUPPORTED YET]. Defaults to False
            use_pymc3 (bool, optional): Use PyMC3 sampling? Defaults to True
            bin_oot (bool, optional): bool - Bin points outside the cut_distance to 30mins. Defaults to True
        """

        #Adding settings to class - not updating if we already initialised the model with a non-default value:
        self.overwrite=overwrite
        for param in self.defaults:
            if not hasattr(self,param) or param in kwargs:
                setattr(self,param,kwargs[param])
            else:
                setattr(self,param,self.defaults[param])
        self.fit_params=self.fit_params+['ecc'] if self.assume_circ and 'ecc' not in self.fit_params else self.fit_params
        self.fit_params=self.fit_params+['omega'] if self.assume_circ and 'omega' not in self.fit_params else self.fit_params
        self.marginal_params=self.marginal_params+['K'] if hasattr(self,'rvs') and self.derive_K else self.marginal_params
        assert self.use_multinest^self.use_pymc3 #Must have either use_multinest or use_pymc3, though use_multinest doesn't work
        assert not (self.assume_circ and self.interpolate_v_prior and (len(self.monos)+len(self.duos)+len(self.trios)>0)) #Cannot interpolate_v_prior and assume circular unless we only have multiplanets
        assert not ((len(self.trios+self.duos+self.monos)>1)*hasattr(self,'rvs')) #Cannot fit more than one planet with uncertain orbits with RVs (currently)

        n_pl=len(self.planets)
        assert n_pl>0

        if self.debug: print(len(self.planets),'planets |','monos:',self.monos,'multis:',self.multis,'duos:',self.duos, "use GP=",self.use_GP)

        ######################################
        #   Masking out-of-transit flux:
        ######################################
        # To speed up computation, here we loop through each planet and add the region around each transit to the data to keep

        self.init_lc(**kwargs)
        ######################################
        #   Creating flux & telescope index func:
        ######################################

        if not hasattr(self,'model_tele_index'):
            # Here we're making an index for which telescope (kepler vs tess) did the observations,
            #  then we multiply the output n_time array by the n_time x 2 index and sum along the 2nd axis
            self.model_tele_index=(np.char.find(self.model_cadence[:,None], np.array(['ts_','k1_','k2_','co_','ch_'])[None,:])+1).astype(bool)
            self.model_tele_index=np.column_stack((self.model_tele_index[:,0],
                                                   self.model_tele_index[:,1]+self.model_tele_index[:,2],
                                                   self.model_tele_index[:,3:]))

        if self.use_GP:
            self.init_GP(**kwargs)
        ######################################
        #   Initialising sampling models:
        ######################################

        if self.use_pymc3:
            self.init_pymc()
        elif self.use_multinest:
            self.run_multinest(**kwargs)

    def init_GP(self, n_draws=900, max_len_lc=25000, use_binned=False, overwrite=False, n_chains=4, cores=4, **kwargs):
        """Function to train GPs on out-of-transit photometry

        Args:
            n_draws (int, optional): Number of draws from sample. Defaults to 900.
            max_len_lc (int, optional): Maximum length of lightcurve to use (limiting to a few 1000 helps with compute time). Defaults to 25000.
            uselc (bool, optional): Specify lightcurve to use. Defaults to None, which takes the `mod.lc` light curve.
        """

        #Only rerunning the initialis function if overwrite is True:
        if hasattr(self,'gp') and self.gp!={} and hasattr(self,'gp_init_soln') and not overwrite:
            return None
        
        self.gp={}

        print("initialising the GP")

        prefix='bin_' if use_binned else ''
        mask=(~self.lc.in_trans['all'])&self.lc.mask&np.isfinite(self.lc.flux)

        #Also cutting exceptionally steep/sharp bins based on differences
        maxcad=np.max([float(cad.split("_")[1])/86400 for cad in self.lc.cadence_list])
        thresh=1.75
        for binsize in np.geomspace(3*maxcad,0.5,7)[::-1]:
            #1 Find steps where the implied gradient is much larger than the typical error for a run of binsizes, starting at the largests
            self.lc.bin(binsize=binsize,overwrite=True)
            large_grads=np.where(np.diff(self.lc.bin_flux)/np.diff(self.lc.bin_time)>thresh*np.nanmedian(self.lc.bin_flux_err)/maxcad)[0]
            #print("extra points masked due to "+str(thresh)+"-sig bin differences at binsize:",binsize,"=",len(large_grads))
            if len(large_grads)>0:
                for nclip in large_grads:
                    mask*=(self.lc.time<self.lc.bin_time[nclip]-binsize)|(self.lc.time>self.lc.bin_time[1+nclip]+binsize)

        self.lc.bin()
        
        if len(self.lc.time[(~self.lc.in_trans['all'])&self.lc.mask])>max_len_lc:
            mask[mask]*=np.arange(0,np.sum(mask),1)<max_len_lc

        with pm.Model() as gp_train_model:
            #####################################################
            #     Training GP kernel on out-of-transit data
            #####################################################
            phot_mean=pm.Normal("phot_mean",mu=np.median(self.lc.flux[mask]),
                                  sigma=np.std(self.lc.flux[mask]))

            self.log_flux_std=np.array([np.log(np.nanmedian(abs(np.diff(self.lc.flux[self.lc.mask*self.lc.cadence_index[:,n]])))) for n in range(len(self.cads_short))]).ravel().astype(floattype)
            if self.debug: print(self.log_flux_std)
            if self.debug: print(self.cads_long)
            if self.debug: print(self.cads_long,np.unique(self.lc.cadence),self.log_flux_std,np.sum(mask))
            
            logs2 = pm.Normal("logs2", mu = 1.5+self.log_flux_std,
                              sigma = np.tile(1.0,len(self.log_flux_std)), initval=3+self.log_flux_std,
                              shape=len(self.log_flux_std))

            if hasattr(self,'periodic_kernel') and self.periodic_kernel is not None:
                #Building a periodic kernel with amplitude modified by a third kernel term (i.e. allowing amplitude to vary with time)

                periodic_w0=pm.Normal("periodic_w0",mu=(2*np.pi)/self.periodic_kernel['period'],sigma=(2*np.pi)/self.periodic_kernel['period_err'])
                periodic_power=pm.Normal("periodic_logpower",mu=self.periodic_kernel['logamp'],sigma=self.periodic_kernel['logamp_err'])
                if "periodic_Q" not in self.periodic_kernel:
                    periodic_logQ=pm.Normal("periodic_kernel",mu=2,sigma=2)
                ampl_mult_logc=pm.Normal("ampl_mult_logc",mu=3,sigma=2,initval=5)
                ampl_mult_loga=pm.Normal("ampl_mult_loga",mu=-1,sigma=2,initval=-1)
                ampl_mult_kernel=pymc_terms.RealTerm(a=pm.math.exp(ampl_mult_loga),c=pm.math.exp(ampl_mult_logc))
                periodic_kernel = pymc_terms.SHOTerm(S0=pm.math.exp(periodic_power)/(periodic_w0**4), w0=periodic_w0, Q=pm.math.exp(periodic_logQ))
                
                phot_w0, phot_sigma = tools.iteratively_determine_GP_params(gp_train_model,time=self.lc.time[mask],flux=self.lc.flux[mask], flux_err=self.lc.flux_err[mask],
                                                                            tdurs=[self.planets[pl]['tdur'] for pl in self.planets], debug=self.debug)
                optvars=[logs2, phot_sigma, phot_w0, phot_mean,periodic_w0,periodic_power,ampl_mult_logc,ampl_mult_loga]
                kernel = pymc_terms.SHOTerm(sigma=phot_sigma, w0=phot_w0, Q=1/np.sqrt(2))
                self.gp['train'] = celerite2.pymc.GaussianProcess(kernel+ampl_mult_kernel*periodic_kernel,self.lc.time[mask].astype(floattype),
                                                            diag=self.lc.flux_err[mask].astype(floattype)**2 + \
                                        pm.math.dot(self.lc.cadence_index[mask,:].astype(floattype),pm.math.exp(logs2)), quiet=True)
            elif hasattr(self,'rotation_kernel') and self.rotation_kernel is not None:
                #Building a purely rotational kernel
                rotation_period=pm.Normal("rotation_period",mu=self.rotation_kernel['period'],sigma=self.rotation_kernel['period_err'])
                rotation_logamp=pm.Normal("rotation_logamp",mu=self.rotation_kernel['logamp'],sigma=self.rotation_kernel['sigma_logamp'])
                if 'logQ0' in self.rotation_kernel and 'sigma_logQ0' in self.rotation_kernel:
                    rotation_logQ0=pm.Normal("rotation_logQ0",mu=self.rotation_kernel['logQ0'],sigma=self.rotation_kernel['sigma_logQ0'])
                else:
                    rotation_logQ0=pm.Normal("rotation_logQ0",mu=1.0,sigma=5)
                if 'logdeltaQ0' in self.rotation_kernel and 'sigma_logdeltaQ0' in self.rotation_kernel:
                    rotation_logdeltaQ=pm.Normal("rotation_logdeltaQ", mu=self.rotation_kernel['logdeltaQ0'], sigma=self.rotation_kernel['sigma_logdeltaQ0'])
                else:
                    rotation_logdeltaQ=pm.Normal("rotation_logdeltaQ", mu=2.,sigma=10.)
                rotation_mix=pm.Uniform("rotation_mix",lower=0,upper=1.0)
                #'sigma', 'Q0', 'dQ', and 'f'
                optvars=[phot_mean,rotation_logamp,rotation_period,rotation_logQ0,rotation_logdeltaQ,rotation_mix]
                rotational_kernel = pymc_terms.RotationTerm(sigma=pm.math.exp(rotation_logamp), period=rotation_period, 
                                                                Q0=pm.math.exp(rotation_logQ0), dQ=pm.math.exp(rotation_logdeltaQ), f=rotation_mix)

                self.gp['train'] = celerite2.pymc.GaussianProcess(rotational_kernel,self.lc.time[mask].astype(floattype))
                                    #                         diag=self.lc.flux_err[mask].astype(floattype)**2 + \
                                    #  pm.math.dot(self.lc.cadence_index[mask,:].astype(floattype),pm.math.exp(logs2)), quiet=True)
            else:
                phot_w0, phot_sigma = tools.iteratively_determine_GP_params(gp_train_model,time=self.lc.time[mask],flux=self.lc.flux[mask], flux_err=self.lc.flux_err[mask],
                                                                        tdurs=[self.planets[pl]['tdur'] for pl in self.planets], debug=self.debug)

                kernel = pymc_terms.SHOTerm(sigma=phot_sigma, w0=phot_w0, Q=1/np.sqrt(2))
                optvars=[logs2, phot_sigma, phot_w0, phot_mean]
                self.gp['train'] = celerite2.pymc.GaussianProcess(kernel,self.lc.time[mask].astype(floattype))
                                    #                         diag=self.lc.flux_err[mask].astype(floattype)**2 + \
                                    #  pm.math.dot(self.lc.cadence_index[mask,:].astype(floattype),pm.math.exp(logs2)), quiet=True)
            #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sigma=10)
            #max_cad = np.nanmax([np.nanmedian(np.diff(self.lc.time[mask&(self.lc.cadence_index[mask,n])])) for n in range(len(self.cads_short))])
            
            #self.gp['train'].log_likelihood(self.lc.flux[mask].astype(floattype) - phot_mean)
            self.gp['train'].compute(self.lc.time[mask].astype(floattype), 
                                     yerr=np.sqrt(self.lc.flux_err[mask].astype(floattype) ** 2 + pm.math.dot(self.lc.cadence_index[mask,:].astype(floattype),pm.math.exp(logs2))**2), 
                                     quiet=True)

            loglik=self.gp['train'].marginal("loglik",observed=self.lc.flux[mask].astype(floattype))

            self.gp_init_soln = pmx.optimize(start=None, vars=optvars)
            if self.debug: print("sampling init GP", int(n_draws*0.66),"times with",len(self.lc.flux[mask]),"-point lightcurve")
            self.gp_init_trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=self.gp_init_soln, chains=n_chains,cores=cores)# **kwargs)

    def init_interpolated_Mp_prior(self):
        """Initialise a 2D interpolated prior for the mass of a planet given the radius
        """
        MRarray=np.genfromtxt(os.path.join(MonoData_tablepath,"LogMePriorFromRe.txt"))
        self.interpolated_mu = xo.interp.RegularGridInterpolator([np.hstack((0.01,MRarray[:,0])).astype(np.float64)],
                                                                  np.hstack((np.log(0.1),MRarray[:,1])).astype(np.float64)[:, None])
        self.interpolated_sigma = xo.interp.RegularGridInterpolator([np.hstack((0.01,MRarray[:,0])).astype(np.float64)],
                                                                     np.hstack((1.25,MRarray[:,2])).astype(np.float64)[:, None])

    def init_interpolated_v_prior(self):
        """Initialise the interpolated functions for log prob vs log velocity and marginalised eccentricity vs log velocity
        """
        import gzip
        import io

        #Four potential sources of data:
        interp_locs={'kipping':"kip", 'vaneylen':"vve",'flat':"flat",'apogee':'apo'}
        interp_locs['auto']='vve' if len(self.planets)>1 else 'kip'
        f_emarg=gzip.open(os.path.join(MonoData_tablepath,
                                                  "emarg_array_"+interp_locs[self.ecc_prior.lower()]+".txt.gz"), "rb")
        emarg_arr  = np.genfromtxt(io.BytesIO(f_emarg.read()))
        #pd.read_csv(os.path.join(MonoData_tablepath,"emarg_array_"+interp_locs[self.ecc_prior.lower()]+".csv"),
        #                          index_col=0)
        emarg_arr  = np.nan_to_num(emarg_arr,1.025)
        #Because the minimum eccentricity in this array is 0.12, we'll add a row of 0.0s below this:

        self.interpolator_eccmarg = xo.interp.RegularGridInterpolator([emarg_arr[1:,0], np.hstack((-0.01,emarg_arr[0,1:]))],
                                                                      np.column_stack((np.tile(1.025,len(emarg_arr[1:,0])),
                                                                                       emarg_arr[1:,1:]))[:,:],
                                                                      nout=1)
        f_logprob=gzip.open(os.path.join(MonoData_tablepath,
                                         "logprob_array_"+interp_locs[self.ecc_prior.lower()]+".txt.gz"),"rb")
        logprob_arr=np.genfromtxt(io.BytesIO(f_logprob.read()))

        #np.genfromtxt(os.path.join(MonoData_tablepath,"logprob_array_"+interp_locs[self.ecc_prior.lower()]+".txt"))
        #logprob_arr = pd.read_csv(os.path.join(MonoData_tablepath,"logprob_array_"+interp_locs[self.ecc_prior.lower()]+".csv"),
        #                          index_col=0)
        #Rewriting infinite or nan values as <-300:
        self.interpolator_logprob = xo.interp.RegularGridInterpolator([logprob_arr[1:,0], 
                                                                       np.hstack([0,logprob_arr[0,1:]])],
                                                                       np.column_stack((np.tile(-310,len(logprob_arr[1:,0])),logprob_arr[1:,1:])), nout=1)

    def init_pymc(self,ld_mult=1.5):
        """Initialise the PyMC3 sampler
        """
        ######################################
        #       Selecting lightcurve:
        ######################################
        
        if self.interpolate_v_prior:
            #Setting up interpolation functions for v_priors here:
            self.init_interpolated_v_prior()

        if hasattr(self,'rvs'):
            self.init_interpolated_Mp_prior()

        print("initialised priors")
        start=None

        with pm.Model() as model:
            if self.debug: print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos,"trios:",self.trios)

            ######################################
            #   Intialising Stellar Params:
            ######################################
            #Using log rho because otherwise the distribution is not normal:
            logrho_S = pm.TruncatedNormal("logrho_S", mu=np.log(self.rhostar[0]),
                                 sigma=np.average(abs(self.rhostar[1:]/self.rhostar[0])),
                                 upper=np.log(self.rhostar[0])+3,lower=-6,
                                 initval=np.log(self.rhostar[0]))
            rho_S = pm.Deterministic("rho_S",pm.math.exp(logrho_S)) #Converting from rho_sun into g/cm3
            Rs = pm.Normal("Rs", mu=self.Rstar[0], sigma=np.average(abs(self.Rstar[1:])), initval=self.Rstar[0])
            Ms = pm.Deterministic("Ms",(rho_S)*Rs**3)

            # The 2nd light (not third light as companion light is not modelled)
            # This quantity is in delta-mag
            unq_missions = np.unique([cad.split('_')[0] for cad in self.cads_short])
            if self.use_L2:
                deltamag_contam = {mis:pm.Uniform("deltamag_contam_"+mis, lower=-10.0, upper=10.0) for mis in unq_missions}
                mult = {mis:pm.Deterministic("mult_"+mis,(1+pm.math.power(2.511,-1*deltamag_contam[mis]))) for mis in unq_missions} #Factor to multiply normalised lightcurve by
            else:
                mult = {mis:1.0 for mis in unq_missions}

            ######################################
            #     Initialising dictionaries
            ######################################
            pers={};t0s={};logrors={};rors={};rpls={};logmassests={};bs={};dist_in_transits={};a_Rs={};tdurs={};vels={};logvels={};incls={}
            self.n_margs={}
            if not self.assume_circ:
                eccs={};omegas={}
            if len(self.monos+self.duos+self.trios)>0:
                max_eccs={};min_eccs={}
                if 'b' not in self.fit_params:
                    b_priors={}
            if len(self.duos)>0 or len(self.trios)>0:
                t0_2s={}
            if len(self.trios)>0:
                t0_3s={}
            if len(self.monos)>0:
                mono_uniform_index_period={}
                per_meds={} #median period from each bin
            if hasattr(self,'rvs'):
                Ks={};normalised_rv_models={}
                logMp_wrt_normals={}; logKs={}
                Mps={};rhos={};model_rvs={};rvlogliks={};marg_rv_models={};rvorbits={};Krv_priors={}

            ######################################
            #     Initialising All params
            ######################################

            for pl in self.planets:
                if pl not in self.duos+self.trios:
                    t0s[pl] = pm.TruncatedNormal("t0_"+pl,mu=self.planets[pl]['tcen'],
                                            sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                            upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.33,
                                            lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.33,
                                            initval=self.planets[pl]['tcen'])

                #############################################
                #     Initialising specific duo/mono terms
                #############################################

                if pl in self.monos:
                    self.n_margs[pl]=self.planets[pl]['ngaps']
                    ind_min=np.power(self.planets[pl]['per_gaps']['gap_ends']/self.planets[pl]['per_gaps']['gap_starts'],self.per_index)
                    per_meds[pl]=np.power(((1-ind_min)*0.5+ind_min),self.per_index)*self.planets[pl]['per_gaps']['gap_starts']
                    if self.mono_model_type=="param_per_gap":
                        if 'log_per' in self.planets[pl]:
                            testindex=[]
                            for ngap in np.arange(len(self.planets[pl]['per_gaps']['gap_ends'])):
                                if np.exp(self.planets[pl]['log_per'])>self.planets[pl]['per_gaps'][ngap,0] and np.exp(self.planets[pl]['log_per'])<self.planets[pl]['per_gaps']['gap_ends'][ngap]:
                                    testindex+=[((np.exp(self.planets[pl]['log_per'])/self.planets[pl]['per_gaps']['gap_starts'][ngap])**self.per_index - ind_min[ngap])/(1-ind_min[ngap])]
                                else:
                                    testindex+=[np.clip(np.random.normal(0.5,0.25),0.00001,0.99999)]
                            mono_uniform_index_period[pl]=pmx.UnitUniform("mono_uniform_index_"+str(pl),
                                                            shape=len(self.planets[pl]['per_gaps']['gap_starts']),
                                                            initval=testindex)
                        else:
                            mono_uniform_index_period[pl]=pmx.UnitUniform("mono_uniform_index_"+str(pl),
                                                            shape=len(self.planets[pl]['per_gaps']['gap_starts']))
                        pers[pl]=pm.Deterministic("per_"+str(pl), pm.math.power(((1-ind_min)*mono_uniform_index_period[pl]+ind_min),1/self.per_index)*self.planets[pl]['per_gaps']['gap_starts'])
                    elif self.mono_model_type=="split_per_gaps":
                        # In this case, we split the allowed period distribution into N bins (where N<100) and compute the implied probability as for a duo
                        mono_uniform_index_period[pl]=pmx.UnitUniform("mono_uniform_index_"+str(pl)) #Single index param, not for each gap
                        pers[pl]=pm.Deterministic("per_"+str(pl), pm.math.power(((1-ind_min)*mono_uniform_index_period[pl]+ind_min),1/self.per_index)*self.planets[pl]['per_gaps']['gap_starts'])
                    
                elif pl in self.duos:
                    self.n_margs[pl]=self.planets[pl]['npers']
                    t0s[pl] = pm.TruncatedNormal("t0_"+pl,
                                            upper=self.planets[pl]['tcen_2']+self.planets[pl]['tdur']*0.33,
                                            lower=self.planets[pl]['tcen_2']-self.planets[pl]['tdur']*0.33,
                                            mu=self.planets[pl]['tcen_2'],sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                            initval=self.planets[pl]['tcen_2'])
                    t0_2s[pl] = pm.TruncatedNormal("t0_2_"+pl,mu=self.planets[pl]['tcen'],
                                                    sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                                    upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.5,
                                                    lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.5,
                                                    initval=self.planets[pl]['tcen'])
                    pers[pl]=pm.Deterministic("per_"+pl, tensor.basic.tile(pm.math.abs(t0s[pl] - t0_2s[pl]),self.n_margs[pl])/self.planets[pl]['period_int_aliases'])
                elif pl in self.trios:
                    t0s[pl] = pm.TruncatedNormal("t0_"+pl,mu=self.planets[pl]['tcen_3'],
                                            sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                        upper=self.planets[pl]['tcen_3']+self.planets[pl]['tdur']*0.33,
                                        lower=self.planets[pl]['tcen_3']-self.planets[pl]['tdur']*0.33,
                                        initval=self.planets[pl]['tcen_3'])
                    self.n_margs[pl]=self.planets[pl]['npers']
                    #Setting the tcen and tcen_2 as the max distance.
                    #
                    t0_3s[pl] = pm.TruncatedNormal("t0_3_"+pl,mu=self.planets[pl]['tcen'],
                                                    sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                                   upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.5,
                                                   lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.5,
                                                   initval=self.planets[pl]['tcen'])
                    if self.model_t03_ttv:
                        t0_2s[pl] = pm.TruncatedNormal("t0_2_"+pl,mu=self.planets[pl]['tcen_2'],
                                                        sigma=self.planets[pl]['tdur']*self.timing_sigma*self.planets[pl]['tdur'],
                                                        upper=self.planets[pl]['tcen_2']+self.planets[pl]['tdur']*0.5,
                                                        lower=self.planets[pl]['tcen_2']-self.planets[pl]['tdur']*0.5,
                                                        initval=self.planets[pl]['tcen_2'])
                        
                        # last_mid_weight=(mod.planets[pl]['p_ratio_32'][0]/mod.planets[pl]['p_ratio_21'][1])
                        # last_mid=(t0_c - t0_2_c)/mod.planets[pl]['p_ratio_32'][0]
                        # last_mid_x_weight=last_mid_weight*last_mid
                        # print(t0_c,t0_2_c,last_mid,last_mid_weight,last_mid_x_weight)
                        # mid_first_weight=(mod.planets[pl]['p_ratio_21'][0]/mod.planets[pl]['p_ratio_32'][1])
                        # mid_first=(t0_2_c - t0_3_c)/mod.planets[pl]['p_ratio_21'][0]
                        # mid_first_x_weight=mid_first_weight*mid_first
                        # print(t0_2_c,t0_3_c,mid_first,mid_first_weight,mid_first_x_weight)
                        # weighted_av=abs(last_mid_x_weight)+abs(mid_first_x_weight)
                        # tile=np.tile(weighted_av,mod.n_margs[pl])
                        # aliases=tile/mod.planets[pl]['period_int_aliases']
                        #Setting the most recent transit as golden, and then using an average of the two observed transits weighted by distance to fit a period.
                        pers[pl]=pm.Deterministic("per_"+pl, tensor.basic.tile(pm.math.abs((self.planets[pl]['p_ratio_32'][0]/self.planets[pl]['p_ratio_21'][1])*(t0s[pl] - t0_2s[pl])/self.planets[pl]['p_ratio_32'][0] + \
                                                                             (self.planets[pl]['p_ratio_21'][0]/self.planets[pl]['p_ratio_32'][1])*(t0_2s[pl] - t0_3s[pl])/self.planets[pl]['p_ratio_21'][0]),self.n_margs[pl])/self.planets[pl]['period_int_aliases'])
                    else:
                        t0_2s[pl] = pm.Deterministic("t0_2_"+pl,t0s[pl]-(t0s[pl] - t0_3s[pl])*self.planets[pl]['p_ratio_32'][0]/self.planets[pl]['p_ratio_32'][1])
                        pers[pl] = pm.Deterministic("per_"+pl, tensor.basic.tile(pm.math.abs(t0s[pl] - t0_3s[pl])/np.min([self.planets[pl]['p_ratio_32'][0],self.planets[pl]['p_ratio_32'][0]])/self.planets[pl]['p_ratio_32'][1],self.n_margs[pl])/self.planets[pl]['period_int_aliases'])

                elif pl in self.multis:
                    self.n_margs[pl]=1
                    pers[pl] = pm.Normal("per_"+pl,
                                         mu=self.planets[pl]['period'],
                                         sigma=np.clip(self.planets[pl]['period_err']*0.25,0.005,0.02*self.planets[pl]['period']),
                                         initval=self.planets[pl]['period'])

                #############################################
                #     Initialising shared planet params
                #############################################

                if 'logror' in self.marginal_params:
                    logrors[pl]=pm.TruncatedNormal("logror_"+pl,
                                                    mu=np.tile(np.log(self.planets[pl]['ror']),self.n_margs[pl]),
                                                    sigma=np.tile(1.0,self.n_margs[pl]),
                                                    lower=np.log(0.001), upper=np.log(0.25+int(self.use_L2)),
                                                    initval=np.tile(np.log(self.planets[pl]['ror']),self.n_margs[pl]),
                                                    shape=self.n_margs[pl])
                else:
                    logrors[pl]=pm.TruncatedNormal("logror_"+pl,mu=np.log(self.planets[pl]['ror']), sigma=0.75, 
                                                    lower=np.log(0.001), upper=np.log(0.25+int(self.use_L2)),
                                                    initval=np.log(self.planets[pl]['ror']))
                rors[pl]=pm.Deterministic("ror_"+pl,pm.math.exp(logrors[pl]))
                rpls[pl]=pm.Deterministic("rpl_"+pl,109.2*rors[pl]*Rs)
                #Estimating mass using simple polynomial:
                logmassests[pl]=pm.Deterministic("logmassest_"+pl, 5.75402469 - (rpls[pl]<=12.2)*(rpls[pl]>=1.58)*(4.67363091 -0.38348534*rpls[pl]) - \
                                                             (rpls[pl]<1.58)*(5.81943841-3.81604756*pm.math.log(rpls[pl])))

                if not self.assume_circ:
                    #Marginalising over, so one value for each period:
                    if pl not in self.multis and not self.interpolate_v_prior and ('ecc' in self.marginal_params or 'omega' in self.marginal_params):
                        if self.ecc_prior.lower()=='kipping' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = xo.distributions.eccentricity.kipping13("ecc_"+pl,shape=self.n_margs[pl])
                            #eccs[pl] = BoundedBeta("ecc_"+pl, alpha=0.867,beta=3.03,
                            #                             initval=0.05,shape=self.n_margs[pl])
                        elif self.ecc_prior.lower()=='vaneylen' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = xo.distributions.eccentricity.vaneylen19("ecc_"+pl,shape=self.n_margs[pl])
                            #pm.Bound(pm.Weibull, lower=1e-5,
                            #                         upper=1-1e-5)("ecc_"+pl,alpha=0.049,beta=2,initval=0.05,
                            #                                       shape=self.n_margs[pl])
                        elif self.ecc_prior.lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5,
                                                       shape=self.n_margs[pl])

                        omegas[pl] = pmx.angle("omega_"+pl,
                                                                 shape=self.n_margs[pl])

                    elif not self.interpolate_v_prior or pl in self.multis:
                        #Fitting for a single ecc and omega (not marginalising or doing the v interpolation)
                        if self.ecc_prior.lower()=='kipping' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = xo.distributions.eccentricity.kipping13("ecc_"+pl, shape=self.n_margs[pl])
                            #BoundedBeta("ecc_"+pl, alpha=0.867,beta=3.03,initval=0.05)
                        elif self.ecc_prior.lower()=='vaneylen' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = xo.distributions.eccentricity.vaneylen19("ecc_"+pl,shape=self.n_margs[pl])
                            #pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("ecc_"+pl,alpha= 0.049,beta=2, initval=0.05)
                        elif self.ecc_prior.lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5)

                        omegas[pl] = pmx.angle("omega_"+pl)
                if 'b' in self.fit_params or pl in self.multis:
                    if 'logror' in self.marginal_params and pl not in self.multis:
                        # The Espinoza (2018) parameterization for the joint radius ratio and
                        bs[pl] = xo.distributions.ImpactParameter("b_"+pl,ror=rors[pl],shape=self.n_margs[pl])
                    else:
                        bs[pl] = xo.distributions.ImpactParameter("b_"+pl, ror=rors[pl], initval=self.planets[pl]['b'])
                if 'tdur' in self.fit_params and pl not in self.multis:
                    tdursigma=0.25*self.planets[pl]['tdur'] if 'tdur_err' not in self.planets[pl] else self.planets[pl]['tdur_err']
                    tdurs[pl] = pm.TruncatedNormal("tdur_"+pl,
                                                   mu=self.planets[pl]['tdur'],
                                                   sigma=tdursigma,
                                                   lower=0.33*self.planets[pl]['tdur'],
                                                   upper=3*self.planets[pl]['tdur'],
                                                   initval=self.planets[pl]['tdur'])
                if 'b' not in self.fit_params and pl not in self.multis and not self.interpolate_v_prior:
                    #If we're not fitting for b we need to extrapolate b from a period and estimate a prior on b
                    if self.assume_circ:
                        bsqs[pl] = pm.Deterministic("bsq_"+pl, (1+rors[pl])**2 - \
                                                     (tdurs[pl]*86400)**2 * \
                                                     ((3*pers[pl]*86400) / (np.pi**2*6.67e-11*rho_S*1409.78))**(-2/3)
                                                    )
                    else:
                         bsqs[pl] = pm.Deterministic("bsq_"+pl, (1+rors[pl])**2 - \
                                                     (tdurs[pl]*86400)**2 * \
                                                     ((3*pers[pl]*86400)/(np.pi**2*6.67e-11*rho_S*1409.78))**(-2/3) * \
                                                     (1+eccs[pl]*pm.math.cos(omegas[pl]-np.pi/2))/(1-eccs[pl]**2)
                                                    )
                    bs[pl] = pm.Deterministic("b_"+pl,pm.math.clip(bsqs[pl], 1e-5, 100)**0.5)
                    # Combining together prior from db/dtdur (which is needed to renormalise to constant b) ~ P**(-2/3)/b
                    # And custom prior which reduces the logprob of all models with bsq<0 (i.e. unphysical) by 5-25
                if 'b' in self.fit_params and 'tdur' in self.fit_params and pl not in self.multis:
                    #We fit for both duration and b, so we can derive velocity (v/v_circ) directly:
                    vels[pl]=pm.Deterministic("vel_"+pl, pm.math.sqrt((1+rors[pl])**2 - bs[pl]**2)/(tdurs[pl]*86400) * ((3*pers[pl]*86400)/(np.pi**2*6.67e-11*rho_S*1409.78))**(1/3))
                    logvels[pl]= pm.Deterministic("logvel_"+pl, pm.math.log(vels[pl]))

                    #Minimum eccentricity (and the associated omega) are then derived from vel, but are one of a range of values
                    min_eccs[pl] = pm.Deterministic("min_ecc_"+pl,pm.math.clip(pm.math.abs(2/(1 + vels[pl]**2) - 1), 1e-4, 1.0-1e-4))
                    omegas[pl] = pm.Deterministic("omega_"+pl,np.pi-0.5*np.pi*(logvels[pl]/pm.math.abs(logvels[pl])) )

                ######################################
                #         Initialising RVs
                ######################################

                #Circular a_Rs
                a_Rs[pl]=pm.Deterministic("a_Rs_"+pl,((6.67e-11*(rho_S*1409.78)*(86400*pers[pl])**2)/(3*np.pi))**(1/3))
                incls[pl] = pm.Deterministic("incl_"+pl,pm.math.arccos(bs[pl]/a_Rs[pl])*180/np.pi) #incl in degrees.
                if hasattr(self,'rvs'):
                    # Using density directly to connect radius (well-constrained) to mass (likely poorly constrained).
                    # Testval uses mass assuming std of RVs is from planet mass
                    #logrhos[pl] = pm.Bound(pm.Normal, lower=np.log(0.01), upper=np.log(3))("logrho_"+pl,mu=0.0,sigma=1,
                    #                         initval=np.clip(np.log(0.22*np.std(self.rvs['rv']) * \
                    #                                                (self.planets[pl]['ror'] * self.Rstar[0]*109.1)**(-3)),
                    #                                         np.log(0.01),np.log(3)))
                    #print("K est.",0.22*np.std(self.rvs['rv']),"Rp est:",self.planets[pl]['ror'] * self.Rstar[0]*109.1,"Rp^3 est:",(self.planets[pl]['ror'] * self.Rstar[0]*109.1)**(-3))
                    #pm.math.printing.Print("logrhos")(logrhos[pl])
                    #logMp_wrt_normals[pl]=pm.Normal("logmass_wrt_normal_"+pl,mu=0.0,sigma=1.0)
                    #Calculating the mass using a prior derived from the MR distribution:
                    #logMps[pl] = pm.Deterministic("logMp_"+pl, (logMp_wrt_normals[pl] * \
                    #                              self.interpolated_sigma.evaluate(rpls[pl].dimshuffle(0,'x')) + \
                    #                              self.interpolated_mu.evaluate(rpls[pl].dimshuffle(0,'x'))).T[0])
                    #logMps[pl] = pm.Deterministic("logMp_"+pl, logrhos[pl] + 3*pm.math.log(rpls[pl]))
                    #pm.math.printing.Print("logMps")(logMps[pl])
                    #Mps[pl]=pm.Deterministic("Mp_"+pl, pm.math.exp(logMps[pl]))
                    #pm.math.printing.Print("Mps")(Mps[pl])

                    #sin(incl) = pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2)

                    #Data-driven prior:
                    if not self.derive_K or pl in self.multis:
                        if 'K' in self.planets[pl]:
                            logKs[pl] = pm.Normal("logK_"+pl, mu=np.log(self.planets[pl]['K']), sigma=0.5)
                        else:
                            logKs[pl] = pm.Normal("logK_"+pl, mu=np.log(np.std(self.rvs['rv'])/np.sqrt(len(self.planets))), sigma=0.5)
                        Ks[pl] = pm.Deterministic("K_"+pl,pm.math.exp(logKs[pl]))
                        if pl in self.trios+self.duos+self.monos and self.interpolate_v_prior:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,pm.math.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-min_eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        elif not self.assume_circ:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,pm.math.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        else:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,pm.math.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1.9884e30*Ms)**(2/3)/5.972e24)
                        rhos[pl]=pm.Deterministic("rho_"+pl,Mps[pl]/rpls[pl]**3)

                    #if pl in self.duos+self.monos and self.interpolate_v_prior:
                    #    #Using minimum eccentricity as this is most likely.
                    #    #37768.355 = Me*Msun^-2/3
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3) * \
                    #                            tensor.basic.tile(5.972e24*Mps[pl]*(1.9884e30*Ms)**(-2/3),self.n_margs[pl]) * \
                    #                            sin_incls[pl]*(1-min_eccs[pl]**2)**-0.5)
                    #    pm.math.printing.Print("Ks")(Ks[pl])
                    #elif not self.assume_circ:
                    #    #Using global ecc parameter
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3)* \
                    #                            5.972e24*Mps[pl]*sin_incls[pl]*(1.9884e30*Ms)**(-2/3)*(1-eccs[pl]**2)**-0.5)
                    #else:
                    #    #Without eccentricity
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3)* \
                    #                            5.972e24*Mps[pl]*sin_incls[pl]*(1.9884e30*Ms)**(-2/3))
            print("initialised planet info")
            #############################################
            #     Initialising RV_only planet params
            #############################################

            if len(self.rvplanets)>0:
                for pl in self.rvplanets:
                    self.n_margs[pl]=1
                    #Making sure tcen can't drift far onto other repeating tcens
                    t0s[pl] = pm.TruncatedNormal("t0_"+pl, mu=self.rvplanets[pl]['period'], sigma=self.rvplanets[pl]['period_err'],
                                                 upper=self.rvplanets[pl]['tcen']+0.55*self.rvplanets[pl]['period'],
                                                 lower=self.rvplanets[pl]['tcen']-0.55*self.rvplanets[pl]['period'])
                    pers[pl] = pm.Normal("per_"+pl, mu=self.rvplanets[pl]['period'], sigma=self.rvplanets[pl]['period_err'])
                    logKs[pl] = pm.Normal("logK_"+pl, mu=self.rvplanets[pl]['logK'], sigma=self.rvplanets[pl]['logK_err'])
                    Ks[pl] = pm.Deterministic("K_"+pl, pm.math.exp(logKs[pl]))

                    if not self.rvplanets[pl]['assume_circ']:
                        if self.rvplanets[pl]['ecc_prior'].lower()=='kipping' or (self.rvplanets[pl]['ecc_prior'].lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = pm.Beta("ecc_"+pl, alpha=0.867,beta=3.03, initval=0.05)
                        elif self.rvplanets[pl]['ecc_prior'].lower()=='vaneylen' or (self.rvplanets[pl]['ecc_prior'].lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = pm.Weibull("ecc_"+pl,alpha=0.049,beta=2,initval=0.05)
                        elif self.rvplanets[pl]['ecc_prior'].lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5)

                        omegas[pl] = pmx.angle("omega_"+pl)
                    else:
                        eccs[pl] = pm.Deterministic("ecc_"+pl,pm.math.constant(0.0))
                        omegas[pl] = pm.Deterministic("omega_"+pl,pm.math.constant(0.0))
                    Mps[pl]=pm.Deterministic("Mp_"+pl,pm.math.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                      (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24) #This is Mpsini

            ######################################
            #     Initialising Limb Darkening
            ######################################
            # Here we either constrain the LD params given the stellar info, OR we let exoplanet fit them
            # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.

            #Single mission
            if np.any([c[:2]=='ts' for c in self.cads_short]) and self.constrain_LD:
                ld_dists=self.get_lds(n_samples=1200,mission='tess')
                u_star_tess = pm.TruncatedNormal("u_star_tess",
                                                mu=np.nanmedian(ld_dists,axis=0),
                                                sigma=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.1,1.0), shape=2, 
                                                lower=0.0, upper=1.0,initval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[:2]=='ts' for c in self.cads_short]) and not self.constrain_LD:
                u_star_tess = xo.distributions.QuadLimbDark("u_star_tess", initval=np.array([0.3, 0.2]))
            if np.any([(c[:2]=='k1')|(c[:2]=='k2') for c in self.cads_short]) and self.constrain_LD:
                ld_dists=self.get_lds(n_samples=3000,mission='kepler')
                if self.debug: print("LDs",ld_dists)
                u_star_kep = pm.TruncatedNormal("u_star_kep", mu=np.nanmedian(ld_dists,axis=0),
                                                sigma=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.1,1.0), 
                                                lower=0.0, upper=1.0, shape=2, initval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([(c[:2]=='k1')|(c[:2]=='k2') for c in self.cads_short]) and not self.constrain_LD:
                u_star_kep = xo.distributions.QuadLimbDark("u_star_kep", initval=np.array([0.3, 0.2]))
            if np.any([c[:2]=='co' for c in self.cads_short]) and self.constrain_LD:
                ld_dists=self.get_lds(n_samples=1200,mission='corot')
                u_star_corot = pm.TruncatedNormal("u_star_corot", mu=np.nanmedian(ld_dists,axis=0),
                                                sigma=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.1,1.0), shape=2, 
                                                lower=0.0, upper=1.0, initval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[:2]=='co' for c in self.cads_short]) and not self.constrain_LD:
                u_star_corot = xo.distributions.QuadLimbDark("u_star_corot", initval=np.array([0.3, 0.2]))
            if np.any([c[:2]=='ch' for c in self.cads_short]) and self.constrain_LD:
                ld_dists=self.get_lds(n_samples=1200,mission='cheops')
                u_star_cheops = pm.TruncatedNormal("u_star_cheops",
                                                    mu=np.nanmedian(ld_dists,axis=0),
                                                    sigma=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.1,1.0), shape=2, 
                                                    lower=0.0, upper=1.0, initval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[:2]=='ch' for c in self.cads_short]) and not self.constrain_LD:
                u_star_cheops = xo.distributions.QuadLimbDark("u_star_cheops", initval=np.array([0.3, 0.2]))

            if not hasattr(self,'log_flux_std'):
                self.log_flux_std=np.array([np.log(np.nanmedian(abs(np.diff(self.lc.flux[(~self.lc.in_trans['all'])&(self.lc.cadence_index[:,nc])])))) for nc in range(len(self.cads_short))]).ravel().astype(floattype)
            if self.debug: print(self.log_flux_std,np.sum(~self.model_in_trans),"/",len(~self.model_in_trans))

            logs2 = pm.Normal("logs2", mu = self.log_flux_std-1,
                              sigma = np.tile(2.0,len(self.log_flux_std)), shape=len(self.log_flux_std))
            ######################################
            #     Initialising RV background
            ######################################

            if hasattr(self,'rvs'):
                #One offset for each telescope:
                rv_offsets = pm.Normal("rv_offsets",
                             mu=self.rvs['init_perscope_offset'],
                             sigma=self.rvs['init_perscope_weightederr'],
                             shape=len(self.rvs['scopes']))
                #pm.math.printing.Print("rv_offsets")(rv_offsets)
                #Now doing the polynomials with a vander
                if self.rv_npoly>2:
                    #We have encapsulated the offset into rv_offsets, so here we form a poly with rv_npoly-1 terms
                    rv_polys = pm.Normal("rv_polys",mu=0,
                                         sigma=self.rvs['init_std']*(10.0**-np.arange(self.rv_npoly)[::-1])[:-1],
                                         shape=self.rv_npoly-1,initval=np.zeros(self.rv_npoly-1))
                    rv_trend = pm.Deterministic("rv_trend", pm.math.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1) + \
                                         pm.math.dot(np.vander(self.rvs['time']-self.rv_tref,self.rv_npoly)[:,:-1],rv_polys))
                    #pm.math.printing.Print("rv_trend")(rv_trend)
                elif self.rv_npoly==2:
                    #We have encapsulated the offset into rv_offsets, so here we just want a single-param trend term
                    rv_polys = pm.Normal("rv_polys", mu=0, sigma=0.1*np.nanstd(self.rvs['rv']), initval=0.0)
                    #pm.math.printing.Print("trend")(rv_polys*(self.rvs['time']-self.rv_tref))
                    #pm.math.printing.Print("offset")(rv_offsets* self.rvs['tele_index_arr'])
                    rv_trend = pm.Deterministic("rv_trend", pm.math.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1) + \
                                                            rv_polys*(self.rvs['time']-self.rv_tref))
                else:
                    #No trend - simply an offset
                    rv_trend = pm.Deterministic("rv_trend", pm.math.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1))
                #pm.math.sum(rv_offsets*tele_index_arr,axis=1)+pm.math.dot(np.vander(time,3),np.hstack((trend,0)))
                #rv_mean = pm.Normal("rv_mean", mu=np.nanmedian(self.rvs['rv']),sigma=2.5*np.nanstd(self.rvs['rv']))
                rv_logs2 = pm.Normal("rv_logs2", mu = 0.0, sigma=2.5, initval=0.5)

            if self.use_GP:
                ######################################
                #     Initialising GP kernel
                ######################################
                if self.debug: print(np.isnan(self.model_time),np.isnan(self.model_flux),np.isnan(self.model_flux_err))
                if self.train_GP:
                    #Using histograms from the output of the previous GP training as priors for the true model.
                    vars=[var for var in self.gp_init_trace.posterior if '__' not in var and np.product(self.gp_init_trace.posterior[var].shape)<5*len(self.gp_init_trace.posterior.chain)*len(self.gp_init_trace.posterior.draw)]
                    ext_gp_init_trace=az.extract(self.gp_init_trace.posterior,var_names=vars)

                    minmaxs={var:np.percentile(ext_gp_init_trace[var].values,[0.5,99.5]).astype(floattype) for var in vars}
                    hists={var:np.histogram(ext_gp_init_trace[var].values,np.linspace(minmaxs[var][0],minmaxs[var][1],101))[0] for var in vars}
                gpvars=[]
                if hasattr(self, 'periodic_kernel') and self.periodic_kernel is not None:
                    if self.train_GP:
                        #Taking trained values from out-of-transit to use as inputs to GP:
                        periodic_w0=pm.Interpolated("periodic_w0", x_points=np.linspace(minmaxs["periodic_w0"][0],minmaxs["periodic_w0"][1],201)[1::2],pdf_points=hists["periodic_w0"])
                        periodic_logpower=pm.Interpolated("periodic_logpower", x_points=np.linspace(minmaxs["periodic_logpower"][0],minmaxs["periodic_logpower"][1],201)[1::2],pdf_points=hists["periodic_logpower"])
                        ampl_mult_logc=pm.Interpolated("ampl_mult_logc", x_points=np.linspace(minmaxs["ampl_mult_logc"][0],minmaxs["ampl_mult_logc"][1],201)[1::2],pdf_points=hists["ampl_mult_logc"])
                        ampl_mult_loga=pm.Interpolated("ampl_mult_loga", x_points=np.linspace(minmaxs["ampl_mult_loga"][0],minmaxs["ampl_mult_loga"][1],201)[1::2],pdf_points=hists["ampl_mult_loga"])
                        gpvars+=[periodic_w0,periodic_logpower,ampl_mult_logc,ampl_mult_loga]
                        #Normal kernal
                        if 'w0' in minmaxs:
                            phot_w0=pm.Interpolated("phot_w0", x_points=np.linspace(minmaxs["w0"][0],minmaxs["w0"][1],201)[1::2],pdf_points=hists["w0"])
                            gpvars+=[phot_w0]
                        elif 'log_w0' in minmaxs:
                            phot_log_w0=pm.Interpolated("phot_log_w0", x_points=np.linspace(minmaxs["log_w0"][0],minmaxs["log_w0"][1],201)[1::2],pdf_points=hists["log_w0"])
                            phot_w0=pm.Deterministic("phot_w0",pm.math.exp(phot_log_w0))
                            gpvars+=[phot_log_w0]
                        if 'sigma' in minmaxs:
                            phot_sigma=pm.Interpolated("phot_sigma", x_points=np.linspace(minmaxs["sigma"][0],minmaxs["sigma"][1],201)[1::2],pdf_points=hists["sigma"])
                            gpvars+=[phot_sigma]
                        elif 'log_sigma' in minmaxs:
                            phot_log_sigma=pm.Interpolated("phot_log_sigma", x_points=np.linspace(minmaxs["log_sigma"][0],minmaxs["log_sigma"][1],201)[1::2],pdf_points=hists["log_sigma"])
                            phot_sigma=pm.Deterministic("phot_sigma",pm.math.exp(phot_sigma))
                            gpvars+=[phot_log_sigma]

                    else:
                        #Building a periodic kernel with amplitude modified by a third kernel term (i.e. allowing amplitude to vary with time)
                        periodic_w0=pm.Normal("periodic_w0",mu=(2*np.pi)/self.periodic_kernel['period'],sigma=(2*np.pi)/self.periodic_kernel['period_err'])
                        periodic_power=pm.Normal("periodic_logpower",mu=self.periodic_kernel['logamp'],sigma=self.periodic_kernel['logamp_err'])
                        if "periodic_Q" not in self.periodic_kernel:
                            periodic_logQ=pm.Normal("periodic_kernel",mu=2,sigma=2)
                            gpvars+=[periodic_logQ]
                        ampl_mult_logc=pm.Normal("ampl_mult_logc",mu=3,sigma=2,initval=5)
                        ampl_mult_loga=pm.Normal("ampl_mult_loga",mu=-1,sigma=2,initval=-1)
                        #Normal kernel:
                        phot_w0, phot_sigma = tools.iteratively_determine_GP_params(model,time=self.model_time,flux=self.model_flux, flux_err=self.model_flux_err,
                                                        tdurs=[self.planets[pl]['tdur'] for pl in self.planets], debug=self.debug)
                        gpvars+=[periodic_w0,periodic_power,ampl_mult_logc,ampl_mult_loga,phot_w0, phot_sigma]

                    kernel = pymc_terms.SHOTerm(sigma=phot_sigma, w0=phot_w0, Q=1/np.sqrt(2))
                    ampl_mult_kernel=pymc_terms.RealTerm(a=pm.math.exp(ampl_mult_loga),c=pm.math.exp(ampl_mult_logc))
                    periodic_kernel = pymc_terms.SHOTerm(S0=pm.math.exp(periodic_power)/(periodic_w0**4), w0=periodic_w0, Q=pm.math.exp(periodic_logQ))
                elif hasattr(self, 'rotation_kernel') and self.rotation_kernel is not None:
                    #Building a periodic kernel with amplitude modified by a third kernel term (i.e. allowing amplitude to vary with time)
                    if self.train_GP:
                        rotation_period=pm.Interpolated("rotation_period", x_points=np.linspace(minmaxs["rotation_period"][0],minmaxs["rotation_period"][1],201)[1::2],pdf_points=hists["rotation_period"])
                        rotation_logamp=pm.Interpolated("rotation_logamp", x_points=np.linspace(minmaxs["rotation_logamp"][0],minmaxs["rotation_logamp"][1],201)[1::2],pdf_points=hists["rotation_logamp"])
                        rotation_logQ0=pm.Interpolated("rotation_logQ0", x_points=np.linspace(minmaxs["rotation_logQ0"][0],minmaxs["rotation_logQ0"][1],201)[1::2],pdf_points=hists["rotation_logQ0"])
                        rotation_logdeltaQ=pm.Interpolated("rotation_logdeltaQ", x_points=np.linspace(minmaxs["rotation_logdeltaQ"][0],minmaxs["rotation_logdeltaQ"][1],201)[1::2],pdf_points=hists["rotation_logdeltaQ"])
                        rotation_mix=pm.Interpolated("rotation_mix", x_points=np.linspace(minmaxs["rotation_mix"][0],minmaxs["rotation_mix"][1],201)[1::2],pdf_points=hists["rotation_mix"])
                    else:
                        rotation_period=pm.Normal("rotation_period",mu=self.rotation_kernel['period'],sigma=self.rotation_kernel['period_err'])
                        rotation_logamp=pm.Normal("rotation_logamp",mu=self.rotation_kernel['logamp'],sigma=self.rotation_kernel['sigma_logamp'])
                        if 'logQ0' in self.rotation_kernel and 'sigma_logQ0' in self.rotation_kernel:
                            rotation_logQ0=pm.Normal("rotation_logQ0",mu=self.rotation_kernel['logQ0'],sigma=self.rotation_kernel['sigma_logQ0'])
                        else:
                            rotation_logQ0=pm.Normal("rotation_logQ0",mu=1.0,sigma=5)
                        if 'logdeltaQ0' in self.rotation_kernel and 'sigma_logdeltaQ' in self.rotation_kernel:
                            rotation_logdeltaQ=pm.Normal("rotation_logdeltaQ", mu=self.rotation_kernel['logdeltaQ0'], sigma=self.rotation_kernel['sigma_logdeltaQ'])
                        else:
                            rotation_logdeltaQ=pm.Normal("rotation_logdeltaQ", mu=2.,sigma=10.)
                        rotation_mix=pm.Uniform("rotation_mix",lower=0,upper=1.0)
                    gpvars+=[rotation_period,rotation_logamp,rotation_logQ0,rotation_logdeltaQ,rotation_mix]
                    #'sigma', 'Q0', 'dQ', and 'f'
                    rotational_kernel = pymc_terms.RotationTerm(sigma=pm.math.exp(rotation_logamp), period=rotation_period, 
                                                                  Q0=pm.math.exp(rotation_logQ0), dQ=pm.math.exp(rotation_logdeltaQ), f=rotation_mix)
                else:
                    if self.train_GP:
                        #Taking trained values from out-of-transit to use as inputs to GP:
                        if 'w0' in minmaxs:
                            phot_w0=pm.Interpolated("phot_w0", x_points=np.linspace(minmaxs["w0"][0],minmaxs["w0"][1],201)[1::2],pdf_points=hists["w0"])
                            gpvars+=[phot_w0]
                        elif 'log_w0' in minmaxs:
                            phot_log_w0=pm.Interpolated("phot_log_w0", x_points=np.linspace(minmaxs["log_w0"][0],minmaxs["log_w0"][1],201)[1::2],pdf_points=hists["log_w0"])
                            phot_w0=pm.Deterministic("phot_w0",pm.math.exp(phot_log_w0))
                            gpvars+=[phot_log_w0]
                        if 'sigma' in minmaxs:
                            phot_sigma=pm.Interpolated("phot_sigma", x_points=np.linspace(minmaxs["sigma"][0],minmaxs["sigma"][1],201)[1::2],pdf_points=hists["sigma"])
                            gpvars+=[phot_sigma]
                        elif 'log_sigma' in minmaxs:
                            phot_log_sigma=pm.Interpolated("phot_log_sigma", x_points=np.linspace(minmaxs["log_sigma"][0],minmaxs["log_sigma"][1],201)[1::2],pdf_points=hists["log_sigma"])
                            phot_sigma=pm.Deterministic("phot_sigma",pm.math.exp(phot_sigma))
                            gpvars+=[phot_log_sigma]
                        
                    else:
                        # Transit jitter & GP parameters
                        #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sigma=10)
                        phot_w0, phot_sigma = tools.iteratively_determine_GP_params(model,time=self.model_time,flux=self.model_flux, flux_err=self.model_flux_err,
                                                        tdurs=[self.planets[pl]['tdur'] for pl in self.planets], debug=self.debug)
                    kernel = pymc_terms.SHOTerm(sigma=phot_sigma, w0=phot_w0, Q=1/np.sqrt(2))

                # GP model for the light curve
                
                if hasattr(self,'periodic_kernel') and self.periodic_kernel is not None:
                    self.gp['use'] = celerite2.pymc.GaussianProcess(kernel+ampl_mult_kernel*periodic_kernel,self.model_time.astype(floattype),
                                                                diag=self.model_flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.model_cadence_index,pm.math.exp(logs2)), quiet=True)
                    if self.pred_all:
                        self.gp['all'] = celerite2.pymc.GaussianProcess(kernel+ampl_mult_kernel*periodic_kernel, self.lc.time.astype(floattype),
                                                                diag = self.lc.flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.lc.cadence_index,pm.math.exp(logs2)), quiet=True)
                elif hasattr(self,'rotation_kernel') and self.rotation_kernel is not None:
                    self.gp['use'] = celerite2.pymc.GaussianProcess(rotational_kernel,self.model_time.astype(floattype),
                                                                diag=self.model_flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.model_cadence_index,pm.math.exp(logs2)), quiet=True)
                    if self.pred_all:
                        self.gp['all'] = celerite2.pymc.GaussianProcess(rotational_kernel, self.lc.time.astype(floattype),
                                                                diag = self.lc.flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.lc.cadence_index,pm.math.exp(logs2)), quiet=True)
                else:
                    self.gp['use'] = celerite2.pymc.GaussianProcess(kernel,self.model_time.astype(floattype),
                                                                diag=self.model_flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.model_cadence_index,pm.math.exp(logs2)), quiet=True)
                    if self.pred_all:
                        self.gp['all'] = celerite2.pymc.GaussianProcess(kernel, self.lc.time.astype(floattype),
                                                                diag = self.lc.flux_err.astype(floattype)**2 + \
                                                                pm.math.dot(self.lc.cadence_index,pm.math.exp(logs2)), quiet=True)
                if self.mutual_incl_sigma is not None and len(self.planets)>3:
                    #Including mutual inclination prior (for high-order multi systems only)
                    av_incl = pm.Deterministic("av_incl",pm.math.mean([incls[pl] for pl in self.planets]))
                    sd_incl = pm.Deterministic("sd_incl",pm.math.std([incls[pl] for pl in self.planets]))
                    mut_incl_prior = pm.Potential("mut_incl_prior", pm.math.exp(-0.5* ((sd_incl - self.mutual_incl_sigma)/self.mutual_incl_sigma)**2))
                phot_mean=pm.Normal("phot_mean",mu=np.median(self.model_flux),  sigma=2*np.std(self.model_flux))
            elif self.local_spline:
                self.spline_params={}
                from patsy import dmatrix
                from scipy.interpolate import BSpline
                for pl in self.planets:
                    self.spline_params['spline_model_'+pl]=np.zeros(len(self.model_time))
                    #Looping over planets - adding polynomial spline across each transit to model unflattened systematics.
                    if pl not in self.multis:
                        #knots= np.linspace(self.planets[pl]['tcen']-((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.planets[pl]['tcen']+((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.n_spline_pts+self.spline_order+1)
                        self.spline_params['spline_knots_'+pl+'_0']=np.quantile(self.model_time[abs(self.model_time-self.planets[pl]['tcen'])<0.75*self.planets[pl]['tdur']]-self.planets[pl]['tcen'],np.linspace(0,1,self.n_spline_pts+4))*1.5+self.planets[pl]['tcen']
                        #np.linspace(self.planets[pl]['tcen']+((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.planets[pl]['tcen']-((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.n_spline_pts+self.spline_order+1)
                        #
                        #self.spline_params['timeix_'+pl+'_0'] = abs(self.model_time-self.planets[pl]['tcen'])<1.5*self.planets[pl]['tdur']
                        self.spline_params['spline_B_'+pl+'_0'] = BSpline.design_matrix(self.model_time, 
                                                                                   np.hstack([np.tile(np.min(self.model_time),self.spline_order+1), 
                                                                                              self.spline_params['spline_knots_'+pl+'_0'], 
                                                                                              np.tile(np.max(self.model_time),self.spline_order+1)]), k=self.spline_order).toarray()
                        #np.asarray(dmatrix("bs(time, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                        #                                                       {"time": self.model_time.astype(np.float64), "knots": self.spline_params['spline_knots_'+pl+'_0'].astype(np.float64)}), order="F")
                        nrby=(abs(self.model_time-self.planets[pl]['tcen'])>0.6*self.planets[pl]['tdur'])&(abs(self.model_time-self.planets[pl]['tcen'])<5*self.planets[pl]['tdur'])
                        sigma=np.nanmedian(abs(np.diff(self.model_flux[nrby])))#*np.sqrt(np.nanmedian(self.model_time[nrby])/np.nanmedian(self.spline_params['spline_knots_'+pl+'_0']))
                        print(self.spline_params['spline_knots_'+pl+'_0'],self.spline_params['spline_B_'+pl+'_0'].shape,sd)

                        self.spline_params['splines_'+pl+'_0'] = pm.Normal('splines_'+pl+'_0', mu=0, sigma=sd, shape=self.n_spline_pts,initval=np.random.normal(np.zeros(self.n_spline_pts),np.tile(sd,self.n_spline_pts)))
                        self.spline_params['spline_model_'+pl+'_0'] = pm.math.dot(self.spline_params['spline_B_'+pl+'_0'], pm.math.concatenate([pm.math.zeros(self.spline_order+1),self.spline_params['splines_'+pl+'_0'],pm.math.zeros(self.spline_order+1)]))
                        if pl in self.monos:
                            self.spline_params['spline_model_'+pl] = pm.Deterministic("spline_model_"+pl,self.spline_params['spline_model_'+pl+'_0'])
                    if pl in self.trios or pl in self.duos:
                        #self.spline_params['timeix_'+pl+'_1'] = abs(self.model_time-self.planets[pl]['tcen_2'])<1.5*self.planets[pl]['tdur']
                        self.spline_params['spline_knots_'+pl+'_1']=np.quantile(self.model_time[abs(self.model_time-self.planets[pl]['tcen_2'])<0.5*self.planets[pl]['tdur']],np.linspace(0,1,self.n_spline_pts+4))
                        #self.spline_params['spline_knots_'+pl+'_1']=np.linspace(self.planets[pl]['tcen_2']-((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.planets[pl]['tcen']+((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.n_spline_pts+self.spline_order+1)
                        #np.quantile(self.model_time[abs(self.model_time-self.planets[pl]['tcen_2'])<0.5*self.planets[pl]['tdur']],np.linspace(0,1,self.n_spline_pts))
                        self.spline_params['spline_B_'+pl+'_1']= BSpline.design_matrix(self.model_time, np.hstack([np.tile(np.min(self.model_time),self.spline_order+1), self.spline_params['spline_knots_'+pl+'_1'], np.tile(np.max(self.model_time),self.spline_order+1)]), k=self.spline_order).toarray()
                        #dmatrix("bs(time, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                        #                                            {"time": self.model_time, "knots": self.spline_params['spline_knots_'+pl+'_1']})
                        nrby=(abs(self.model_time-self.planets[pl]['tcen_2'])>0.6*self.planets[pl]['tdur'])&(abs(self.model_time-self.planets[pl]['tcen_2'])<5*self.planets[pl]['tdur'])
                        sigma=np.nanmedian(abs(np.diff(self.model_flux[nrby])))*np.sqrt(np.nanmedian(self.model_time[nrby])/np.nanmedian(self.spline_params['spline_knots_'+pl+'_1']))
                        self.spline_params['splines_'+pl+'_1'] = pm.Normal('splines_'+pl+'_1', mu=0, sigma=sd, shape=self.n_spline_pts, initval=np.random.normal(np.zeros(self.n_spline_pts),np.tile(sd,self.n_spline_pts)))
                        self.spline_params['spline_model_'+pl+'_1'] = pm.math.dot(self.spline_params['spline_B_'+pl+'_1'], pm.math.concatenate([pm.math.zeros(self.spline_order+1),self.spline_params['splines_'+pl+'_1'],pm.math.zeros(self.spline_order+1)]))
                        if pl in self.duos:
                            self.spline_params['spline_model_'+pl]=pm.Deterministic("spline_model_"+pl,self.spline_params['spline_model_'+pl+'_1']+self.spline_params['spline_model_'+pl+'_0'])
                    if pl in self.trios:
                        #self.spline_params['timeix_'+pl+'_2'] = abs(self.model_time-self.planets[pl]['tcen_3'])<1.5*self.planets[pl]['tdur']
                        self.spline_params['spline_knots_'+pl+'_2']=np.quantile(self.model_time[abs(self.model_time-self.planets[pl]['tcen_3'])<0.5*self.planets[pl]['tdur']],np.linspace(0,1,self.n_spline_pts+4))
                        #self.spline_params['spline_knots_'+pl+'_2']=np.linspace(self.planets[pl]['tcen']-((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.planets[pl]['tcen_3']+((self.n_spline_pts)/(self.n_spline_pts-self.spline_order-1))*0.5*self.planets[pl]['tdur'],self.n_spline_pts+self.spline_order+1)
                        #np.quantile(self.model_time[abs(self.model_time-self.planets[pl]['tcen_3'])<0.5*self.planets[pl]['tdur']],np.linspace(0,1,self.n_spline_pts))
                        self.spline_params['spline_B_'+pl+'_2']= BSpline.design_matrix(self.model_time, np.hstack([np.tile(np.min(self.model_time),self.spline_order+1), self.spline_params['spline_knots_'+pl+'_2'], np.tile(np.max(self.model_time),self.spline_order+1)]), k=self.spline_order).toarray()
                        #dmatrix("bs(time, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                        #                                            {"time": self.model_time, "knots": self.spline_params['spline_knots_'+pl+'_2']},)
                        nrby=(abs(self.model_time-self.planets[pl]['tcen_3'])>0.6*self.planets[pl]['tdur'])&(abs(self.model_time-self.planets[pl]['tcen_3'])<5*self.planets[pl]['tdur'])
                        sigma=np.nanmedian(abs(np.diff(self.model_flux[nrby])))*np.sqrt(np.nanmedian(self.model_time[nrby])/np.nanmedian(self.spline_params['spline_knots_'+pl+'_2']))
                        self.spline_params['splines_'+pl+'_2'] = pm.Normal('splines_'+pl+'_2', mu=0, sigma=sd, shape=self.n_spline_pts, initval=np.random.normal(np.zeros(self.n_spline_pts),np.tile(sd,self.n_spline_pts)))
                        self.spline_params['spline_model_'+pl+'_2'] = pm.math.dot(self.spline_params['spline_B_'+pl+'_2'], pm.math.concatenate([pm.math.zeros(self.spline_order+1),self.spline_params['splines_'+pl+'_2'],pm.math.zeros(self.spline_order+1)]))
                        #self.spline_params['spline_model_'+pl]=pm.math.sum(pm.math.stack(self.spline_params['spline_model_'+pl+'_0'],self.spline_params['spline_model_'+pl+'_2'],self.spline_params['spline_model_'+pl+'_2']),axis=0)
                        if pl in self.duos:
                            self.spline_params['spline_model_'+pl]=pm.Deterministic("spline_model_"+pl,self.spline_params['spline_model_'+pl+'_2']+self.spline_params['spline_model_'+pl+'_1']+self.spline_params['spline_model_'+pl+'_0'])
                phot_mean=pm.Normal("phot_mean",mu=np.median(self.model_flux),  sigma=2*np.std(self.model_flux))
            else:
                phot_mean=pm.Normal("phot_mean",mu=np.median(self.model_flux),  sigma=2*np.std(self.model_flux))
            ################################################
            #  Creating function to generate transit models
            ################################################
            def gen_lc(i_orbit, i_rpl, n_pl, mask=None,prefix='',make_deterministic=False,pred_all=False):
                """Short flexible method to create stacked lightcurves for any cadence and orbit type:
                        # This function is needed because we may have
                        #   -  1) multiple cadences and
                        #   -  2) multiple telescopes (and therefore limb darkening coefficients)

                Args:
                    i_orbit (xo.orbits.KeplerianOrbit): Planetary orbits (in units of solar radii) to create lightcurve
                    i_rpl (pymc3 variable): Planetary radii (in units of solar radii) to create lightcurve
                    n_pl (int): Number of planets/orbits to generate lightcurves
                    mask (array, optional): Specific mask to the light curve]. Defaults to None.
                    prefix (str, optional): prefix to the PyMC3 variable name. Defaults to ''.
                    make_deterministic (bool, optional): Add the output as a deterministic PyMC3 variable (may be memory intensive). Defaults to False.

                Returns:
                    array OR pymc3 variable: 1D lightcurve
                """
                # 
                trans_pred=[]
                if pred_all:
                    mask = self.lc.mask if mask is None else mask
                    pred_time = self.lc.time
                else:
                    mask = np.isfinite(self.model_time) if mask is None else mask
                    pred_time = self.model_time
                cad_index=[]

                if n_pl>1:
                    r=tensor.basic.tile(i_rpl,n_pl)
                else:
                    r=i_rpl

                for nc,cad in enumerate(self.cads_short):
                    cadmask=mask&self.model_cadence_index[:,nc]
                    #Taking the texp from the in-transit points (not the potentially binned out-of-transit regions)
                    texp=np.nanmedian(np.diff(pred_time[cadmask*self.lc.in_trans['all']])) if pred_all else np.nanmedian(np.diff(pred_time[cadmask*self.model_in_trans]))

                    #print(self.lc['tele_index'][mask,0].astype(bool),len(self.lc['tele_index'][mask,0]),cadmask[mask],len(cadmask[mask]))
                    miss=cad.lower().split('_')[0]
                    cad_index+=[cadmask]
                    #Have three transits and ttvs - need to modify the input time vector such that 
                    if miss=='ts':
                        #Taking the "telescope" index, and adding those points with the matching cadences to the cadmask
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=pred_time.astype(floattype),texp=texp
                                                                 )/(self.lc.flx_unit*mult['ts'])]
                    elif miss=='k1':
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=pred_time.astype(floattype),texp=texp
                                                                 )/(self.lc.flx_unit*mult['k1'])]
                    elif miss=='k2':
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=pred_time.astype(floattype),texp=texp
                                                                 )/(self.lc.flx_unit*mult['k2'])]
                    elif miss=='co':
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_corot).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=pred_time.astype(floattype),texp=texp
                                                                 )/(self.lc.flx_unit*mult['co'])]
                    elif miss=='ch':
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_cheops).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=pred_time.astype(floattype),texp=texp
                                                                 )/(self.lc.flx_unit*mult['ch'])]
                    if self.debug: print(miss)
                #pm.math.printing.Print("trans_pred")(trans_pred)
                # transit arrays (ntime x n_pls x 2) * telescope index (ntime x n_pls x 2), summed over dimension 2
                if n_pl>1 and make_deterministic:

                    return pm.Deterministic(prefix+"light_curves",
                                        pm.math.sum(pm.math.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               pm.math.stack(cad_index).dimshuffle(1,'x',0),axis=2))
                elif n_pl==1 and make_deterministic:
                    return pm.Deterministic(prefix+"light_curves",
                                        pm.math.sum(pm.math.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               pm.math.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2)))
                elif n_pl>1 and not make_deterministic:
                    return pm.math.sum(pm.math.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                  pm.math.stack(cad_index).dimshuffle(1,'x',0),axis=2)

                elif n_pl==1 and not make_deterministic:
                    return pm.math.sum(pm.math.stack(trans_pred,axis=2).dimshuffle(0,1,2) * pm.math.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2))

            def create_orbit(pl, Rs, rho_S, pers, t0s, bs, n_marg=1, eccs=None, omegas=None):
                """AI is creating summary for create_orbit

                Args:
                    pl (str): Planet name as seen in `mod.planets` dict
                    Rs (pymc3 variable): Solar radius (in Rsun)
                    rho_S (pymc3 variable): Solar density (in rho_sun)
                    pers (pymc3 variable): Periods
                    t0s (pymc3 variable): Epochs
                    bs (pymc3 variable): Impact parameters
                    n_marg (int, optional): Number of periods to marginalise over. Defaults to 1.
                    eccs (pymc3 variable, optional): Orbital eccentricity. Defaults to None.
                    omegas (pymc3 variable, optional): Orbital argument of periasteron (omega). Defaults to None.

                Returns:
                    xo.orbits.KeplerianOrbit: Exoplanet Keplerian orbit object initialised to model the lightcurve
                """
                #Setting up Exoplanet orbit
                if pl in self.multis or self.interpolate_v_prior:
                    #Single orbit expected:
                    i_t0s=t0s;i_pers=pers;i_bs=bs
                    if not self.assume_circ:
                        i_eccs=eccs;i_omegas=omegas
                else:
                    #Multiple orbits expected
                    i_t0s=tensor.basic.tile(t0s,n_marg)
                    i_pers=tensor.basic.tile(pers,n_marg)
                    i_bs=tensor.basic.tile(bs,n_marg) if 'b' not in self.marginal_params else bs
                    if not self.assume_circ:
                        i_eccs=tensor.basic.tile(eccs,n_marg) if 'ecc' not in self.marginal_params else eccs
                        i_omegas=tensor.basic.tile(omegas,n_marg) if 'omega' not in self.marginal_params else omegas
                if self.assume_circ:
                    return xo.orbits.KeplerianOrbit(r_star=Rs, rho_star=rho_S*1.40978, period=i_pers,t0=i_t0s,b=i_bs)
                else:
                    return xo.orbits.KeplerianOrbit(r_star=Rs, rho_star=rho_S*1.40978, period=i_pers,t0=i_t0s,b=i_bs,
                                                    ecc=i_eccs,omega=i_omegas)

            #####################################################
            #  Multiplanet lightcurve model & derived parameters
            #####################################################
            orbits={}
            light_curves={}
            vx={};vy={};vz={}
            logpriors={};rvlogliks={};lclogliks={};rvjitters={}
            if len(self.trios+self.duos+self.monos)>0:
                #Initialising priors:
                per_priors={};b_priors={};geom_ecc_priors={};ecc_lim_priors={}
                edge_priors={};v_priors={};gap_width_priors={}
            if self.force_match_input is not None:
                match_input_potentials={}

            for pl in self.rvplanets:
                rvorbits[pl] = xo.orbits.KeplerianOrbit(period=pers[pl], t0=t0s[pl], ecc=eccs[pl], omega=omegas[pl])
                model_rvs[pl] = pm.Deterministic("model_rv_"+pl,
                                                 rvorbits[pl].get_radial_velocity(self.rvs['time'], K=pm.math.exp(logKs[pl])))

            for pl in self.multis+self.trios+self.duos+self.monos:
                #Making orbit and lightcurve(s)
                if self.assume_circ:
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl], t0s[pl], bs[pl], n_marg=self.n_margs[pl])
                    light_curves[pl] = gen_lc(orbits[pl], rpls[pl]/109.2, self.n_margs[pl], mask=None,
                                              prefix=pl+'_', make_deterministic=True)
                elif self.interpolate_v_prior and pl in self.trios+self.duos+self.monos:
                    #  We only need to create one orbit if we're not marginalising over N periods
                    #      (i.e. when we only have the lightcurve and we're interpolating a velocity prior)
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl][tensor.math.argmin(min_eccs[pl])], t0s[pl], bs[pl], n_marg=1,
                                              omegas=omegas[pl][tensor.math.argmin(min_eccs[pl])], eccs=pm.math.min(min_eccs[pl]))
                    light_curves[pl] = gen_lc(orbits[pl], rpls[pl]/109.2, 1, mask=None,
                                              prefix=pl+'_', make_deterministic=True)
                else:
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl], t0s[pl], bs[pl], n_marg=self.n_margs[pl],
                                              eccs=eccs[pl], omegas=omegas[pl])
                    light_curves[pl] = gen_lc(orbits[pl], rpls[pl]/109.2, 1, mask=None,
                                              prefix=pl+'_', make_deterministic=pl in self.multis)

                if hasattr(self,'rvs'):
                    if pl in self.trios+self.duos+self.monos and self.interpolate_v_prior:
                        #In this case, we need to create N orbits but only one lightcurve (from the min eccentricity)
                        if self.debug:
                            pm.math.printing.Print("min_eccs[pl]")(min_eccs[pl])
                            pm.math.printing.Print("omegas[pl]")(omegas[pl])
                            pm.math.printing.Print("bs[pl]")(bs[pl])
                            pm.math.printing.Print("t0s[pl]")(t0s[pl])
                            pm.math.printing.Print("pers[pl]")(pers[pl])

                        rvorbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl], t0s[pl], bs[pl], n_marg=self.n_margs[pl],
                                                    eccs=min_eccs[pl], omegas=omegas[pl])
                    else:
                        rvorbits[pl] = orbits[pl]

                #Doing extra deterministic variables:
                if not self.interpolate_v_prior or pl in self.multis:
                    dist_in_transits[pl]=pm.Deterministic("dist_in_transit_"+str(pl),
                                                          orbits[pl].get_relative_position(t0s[pl])[2])
                if pl not in vels:
                    vx[pl], vy[pl], vz[pl] = orbits[pl].get_relative_velocity(t0s[pl])
                    vels[pl] = pm.Deterministic("vel_"+pl,pm.math.sqrt(vx[pl]**2 + vy[pl]**2))
                if pl not in logvels:
                    logvels[pl]= pm.Deterministic("logvel_"+pl,pm.math.log(vels[pl]))
                if pl not in a_Rs:
                    a_Rs[pl] = pm.Deterministic("a_Rs_"+pl, orbits[pl].a/Rs)
                if pl not in tdurs:
                    #if 'tdur' in self.marginal_params:
                    tdurs[pl]=pm.Deterministic("tdur_"+pl,
                            (2*Rs*pm.math.sqrt( (1+rors[pl])**2 - bs[pl]**2)) / vels[pl] )
                #else:
                #    vx, vy, vz=orbits[pl].get_relative_velocity(t0s[pl])



                ################################################
                #                   Priors:
                ################################################
                #Force model to match expected/input depth_duration with sigmoid (not used in default)
                if self.force_match_input is not None:
                    match_input_potentials[pl]=pm.math.sum(pm.math.exp( -(tdurs[pl]**2 + self.planets[pl]['tdur']**2) / (2*(self.force_match_input*self.planets[multi]['tdur'])**2) )) + \
                                     pm.math.sum(pm.math.exp( -(logrors[pl]**2 + self.planets[pl]['log_ror']**2) / (2*(self.force_match_input*self.planets[pl]['log_ror'])**2) ))
                    pm.Potential("all_match_input_potentials",
                                 pm.math.sum([match_input_potentials[i] for i in match_input_potentials]))

                ################################################
                #       Generating RVs for each submodels:
                ################################################
                if hasattr(self,'rvs'):
                    #new_rverr = ((1+pm.math.exp(rv_logs2))*self.rvs['rv_err'].astype(floattype))
                    sum_log_rverr = pm.math.sum(-len(self.rvs['rv'])/2 * pm.math.log(2*np.pi*((1+pm.math.exp(rv_logs2))*self.rvs['rv_err'].astype(floattype))**2))
                    #model_rvs[pl] = pm.Deterministic('model_rv_'+pl, tensor.basic.tile(Ks[pl].dimshuffle('x',0),(len(self.rvs['time']),1)))
                    if pl in self.trios+self.duos+self.monos and not hasattr(model,'nonmarg_rvs') and self.derive_K:
                        #Deriving the best-fit K from the data:
                        sinf, cosf = rvorbits[pl]._get_true_anomaly(self.rvs['time'])
                        #pm.math.printing.Print("cosf")(cosf)
                        #pm.math.printing.Print("sinf")(sinf)
                        normalised_rv_models[pl]=(rvorbits[pl].cos_omega * cosf - rvorbits[pl].sin_omega * sinf + \
                                                  rvorbits[pl].ecc * rvorbits[pl].cos_omega)
                        #pm.math.printing.Print("normalised_rv_models")(normalised_rv_models[pl])
                        #The combination of all RV component that do not require marginalisation:
                        pm.math.printing.Print("sinfs")(sinf)
                        pm.math.printing.Print("cosfs")(cosf)
                        pm.math.printing.Print("eccs")(rvorbits[pl].ecc)
                        pm.math.printing.Print("shape_rv_trend")(rv_trend.shape)
                        pm.math.printing.Print("shape_normalised_rvs")(normalised_rv_models[pl].shape)
                        
                        #pm.math.printing.Print("sum_normalised_rvs")(pm.math.sum(normalised_rv_models[pl]**2,axis=0))
                        if (len(self.multis)+len(self.rvplanets))>1:
                            for ipl in self.multis+list(self.rvplanets.keys()):
                                pm.math.printing.Print("multi_rvplanets_"+ipl)(model_rvs[ipl])
                                pm.math.printing.Print("shape_multi_rvplanets_"+ipl)(model_rvs[ipl].shape)
                            pm.math.printing.Print("stacked_rvplanets_"+ipl)(pm.math.stack([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())]))
                            pm.math.printing.Print("shape_stacked_rvplanets_"+ipl)(pm.math.stack([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())]).shape)
                            nonmarg_rvs = pm.Deterministic("nonmarg_rvs", (rv_trend + pm.math.sum(pm.math.stack([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())]),axis=0)))
                        elif (len(self.multis)+len(self.rvplanets))==1:
                            onlypl=self.multis+list(self.rvplanets.keys())
                            nonmarg_rvs = pm.Deterministic("nonmarg_rvs",(rv_trend+model_rvs[onlypl[0]]))
                        else:
                            nonmarg_rvs = pm.Deterministic("nonmarg_rvs",rv_trend)
                        #Mono or duo. Removing multi orbit if we have one:
                        if self.debug:
                            pm.math.printing.Print("nonmarg_rvs")(nonmarg_rvs)
                            pm.math.printing.Print("self.rvs['rv'] - nonmarg_rvs")(self.rvs['rv'] - nonmarg_rvs)
                            pm.math.printing.Print("tensor.basic.tile(self.rvs['rv'] - nonmarg_rvs,(self.n_margs[pl],1))")(tensor.basic.tile(self.rvs['rv'] - nonmarg_rvs,(self.n_margs[pl],1)))
                            pm.math.printing.Print("normalised_rv_models[pl].T")(normalised_rv_models[pl].T)
                        Ks[pl] = pm.Deterministic("K_"+pl, pm.math.clip(pm.math.batched_tensordot(tensor.basic.tile(self.rvs['rv'] - nonmarg_rvs,(self.n_margs[pl],1)), normalised_rv_models[pl].T, axes=1) / pm.math.sum(normalised_rv_models[pl]**2,axis=0),0.05,1e5))
                        pm.math.printing.Print("pers")(pers[pl])
                        pm.math.printing.Print("Ks")(Ks[pl])

                        model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'], K=Ks[pl]))
                        pm.math.printing.Print("Ks")(Ks[pl].shape)
                        pm.math.printing.Print("model_rvs[pl]")(model_rvs[pl].shape)
                        if pl in self.trios+self.duos+self.monos and self.interpolate_v_prior:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-min_eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        elif not self.assume_circ:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        else:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 pm.math.sqrt(1-(bs[pl]/a_Rs[pl])**2)*(1.9884e30*Ms)**(2/3)/5.972e24)
                        rhos[pl]=pm.Deterministic("rho_"+pl,Mps[pl]/rpls[pl]**3)

                    else:
                        pm.math.printing.Print("pers"+pl)(pers[pl])
                        pm.math.printing.Print("logKs"+pl)(logKs[pl])
                        #if pl in self.duos+self.monos:
                        model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'],
                                                                                                      K=pm.math.exp(logKs[pl])))
                        #else:
                        #    model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'],tensor.basic.tile(Ks[pl].dimshuffle('x',0),(len(self.rvs['time']),1))))


                if pl in self.trios+self.duos+self.monos:
                    #Need the minimum period to normalise
                    per_priors[pl] = pm.Deterministic("per_prior_"+pl,
                                                      self.per_index * pm.math.log(pers[pl]/self.planets[pl]['P_min']))
                    if len(self.multis)>0:
                        #the expected outer gravitational influence of all other planets 
                        # - from aphelion position ((1+ecc)*a) plus hill-sphere radius (a*(Mp/(3*Ms))**(1/3))
                        max_eccs[pl] = pm.Deterministic("max_ecc_"+pl,
                                                        1 - pm.math.max([(1+eccs[i]+(pm.math.exp(logmassests[i]-12.7156)/(3*Ms))**(1/3))*a_Rs[i] for i in self.multis])/a_Rs[pl])
                    else:
                        max_eccs[pl] = pm.Deterministic("max_ecc_"+pl,1 - 2/a_Rs[pl])
                    if not self.assume_circ and not self.interpolate_v_prior:
                        #A correction to the period prior from the increased geometric probability of high-ecc planets:
                        geom_ecc_priors[pl]=pm.Deterministic("geom_ecc_prior_"+pl,
                                                             -1*pm.math.log(dist_in_transits[pl]/a_Rs[pl]))

                        #A sigmoid prior rejecting orbits that either star-graze or cross the orbits of inner planets
                        ecc_lim_priors[pl]=pm.Deterministic("star_ecc_lim_prior_"+pl,
                                            (500 / (1 + pm.math.exp(-30*(max_eccs[pl] - eccs[pl])))-500))
                    else:
                        #These are incorporated into the interpolated velocity prior:
                        geom_ecc_priors[pl]=pm.math.zeros(self.n_margs[pl])
                        ecc_lim_priors[pl]=pm.math.zeros(self.n_margs[pl])
                    if 'b' not in self.fit_params:
                        #Require prior on b to account for the fact that high impact parameters less likely
                        b_priors[pl]=pm.Deterministic("b_prior_"+pl, pm.math.log(pm.math.max(bs[pl])/bs[pl]) - \
                                                                      5/2*pm.math.log(pers[pl]/pm.math.max(pers[pl])) + \
                                                                      pm.math.switch(pm.math.lt(bsqs[pl],0),bsqs[pl]*40-15,0))
                        #pm.math.log( (pm.math.max(bs[pl])/bs[pl]) * (pers[pl]/pm.math.max(pers[pl]))**(-5/2) ) + pm.math.switch(pm.math.lt(bsqs[pl],0),bsqs[pl]*40-15,0)
                    else:
                        b_priors[pl]=pm.math.zeros(self.n_margs[pl])

                    if self.interpolate_v_prior:
                        #Prior on derived velocity implied by period (coming from geometry and eccentricity)
                        v_priors[pl]=pm.Deterministic("v_prior_"+pl,self.interpolator_logprob.evaluate(pm.math.stack([logvels[pl],pm.math.clip(max_eccs[pl],0.0,0.999)],axis=1)))
                        edge_priors[pl]=pm.math.zeros(self.n_margs[pl])

                        '''#Prior here mimics presence of data edges to likelihood, but applied as a prior
                        #Assumes data is at 0 and planet transit is at rp/rs**2
                        if pl in self.monos:
                            edge_ix=self.lc.mask[:,None] * \
                                (abs((self.lc.time[:,None]-t0s[pl]-0.5*pers[pl])%pers[pl]-0.5*pers[pl])<tdurs[pl]*0.5) * \
                                (abs(self.lc.time[:,None]-t0s[pl])>tdurs[pl]*0.5)
                        elif pl in self.duos:
                            edge_ix=self.lc.mask[:,None] * \
                            (abs((self.lc.time[:,None]-t0s[pl]-0.5*pers[pl])%pers[pl]-0.5*pers[pl])<tdurs[pl]*0.5) * \
                            (abs(self.lc.time[:,None]-t0_2s[pl])>tdurs[pl]*0.55) * \
                            (abs(self.lc.time[:,None]-t0s[pl])>tdurs[pl]*0.55) * \
                            self.lc.mask[:,None]

                        #depth^2/errs^2 = ror^4/errs^2
                        edge_priors[pl]=pm.Deterministic("edge_prior_"+pl,
                                       -pm.math.sum(rors[pl]**4/(self.lc.flux_unit*self.lc.flux_err[:,None])**2*edge_ix,axis=0)
                                                        )
                        '''
                    else:
                        v_priors[pl]=pm.math.zeros(self.n_margs[pl])
                        edge_priors[pl]=pm.math.zeros(self.n_margs[pl])

                    '''
                    #We want models to prefer high-K solutions over flat lines, so we include a weak prior on log(K)
                    if hasattr(self,'rvs') and self.derive_K:
                        Krv_priors[pl] = pm.Deterministic("Krv_prior_"+pl, 0.25*pm.math.log(Ks[pl]))
                    elif hasattr(self,'rvs') and not self.derive_K:
                        Krv_priors[pl] = pm.Deterministic("Krv_prior_"+pl, 0.25*tensor.basic.tile(pm.math.log(Ks[pl]),self.n_margs[pl]) )
                    else:
                        Krv_priors[pl] = pm.math.zeros(self.n_margs[pl])
                    '''

                    if pl in self.monos:
                        #For monotransits, there is a specific term required from the width of the gap (not necessary for duos)
                        gap_width_priors[pl] = pm.Deterministic("gap_width_prior_"+pl,
                                                                pm.math._shared(self.planets[pl]['per_gaps']['gap_probs']))
                    else:
                        gap_width_priors[pl] = pm.math.zeros(self.n_margs[pl])

                    #pm.math.printing.Print("per_priors")(per_priors[pl])
                    #pm.math.printing.Print("geom_ecc_priors")(geom_ecc_priors[pl])
                    #pm.math.printing.Print("ecc_lim_priors")(ecc_lim_priors[pl])
                    #pm.math.printing.Print("b_priors")(b_priors[pl])
                    #pm.math.printing.Print("v_priors")(v_priors[pl])
                    #pm.math.printing.Print("edge_priors")(edge_priors[pl])
                    #pm.math.printing.Print("gap_width_priors")(gap_width_priors[pl])
                    #pm.math.printing.Print("Krv_priors")(Krv_priors[pl])
                    #Summing up for total log prior for each alias/gap:
                    logpriors[pl]=pm.Deterministic("logprior_"+pl, per_priors[pl] + geom_ecc_priors[pl] + ecc_lim_priors[pl] + \
                                           b_priors[pl] + v_priors[pl] + edge_priors[pl] + gap_width_priors[pl])#+Krv_priors[pl]

                    if pl in self.trios+self.duos+self.monos and hasattr(self,'rvs'):
                        if not hasattr(model,'nonmarg_rvs'):
                            if (len(self.multis)+len(self.rvplanets))>1:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs", (rv_trend + pm.math.sum([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())],axis=1)))
                            elif (len(self.multis)+len(self.rvplanets))==1:
                                onlypl=self.multis+list(self.rvplanets.keys())
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",(rv_trend+model_rvs[onlypl[0]]))
                            else:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",rv_trend)
                        #Due to overfitting (and underestimation of errorbars), we are going to calculate the "jitter" necessary to make the RV model match within 1-sigma. 
                        #This jitter, compared againsts expected log10(jitter) distribution of 0.0±0.5 then produces the major difference in log_lik between models
                        rv_logliks = (self.rvs['rv'][:,None] - (nonmarg_rvs.dimshuffle(0,'x') + model_rvs[pl]))**2/self.rvs['rv_err'].astype(floattype)[:,None]**2
                        #rvjitters[pl]=pm.Deterministic("rv_jitters_"+pl,pm.math.clip(pm.math.sqrt(pm.math.mean(rv_logliks,axis=0))-np.average(self.rvs['rv_err'].astype(floattype)),self.rvs['jitter_min'],1e4))
                        logmass_sigma = pm.Deterministic("logmass_sd_"+pl, (rpls[pl]<=8)*(0.07904372*rpls[pl]+0.24318296) + (rpls[pl]>8)*(0-0.02313261*rpls[pl]+1.06765343))
                        rvlogliks[pl]=pm.Deterministic("rv_loglik_"+pl, tensor.basic.tile(sum_log_rverr,self.n_margs[pl]) - \
                                                                        (pm.math.log(Mps[pl])-logmassests[pl])**2/((self.rvs['jitter_min']/Ks[pl])**2 + logmass_sd**2) - \
                                                                        pm.math.sum((self.rvs['rv'][:,None] - (nonmarg_rvs.dimshuffle(0,'x') + model_rvs[pl]))**2/(self.rvs['jitter_min']**2 + self.rvs['rv_err'][:,None].astype(floattype)**2),axis=0))
                                                                        #-(pm.math.log(rvjitters[pl])-self.rvs['logjitter_mean'])**2/self.rvs['logjitter_sd']**2 - \
                    elif hasattr(self,'rvs'):
                        rvlogliks[pl] = pm.Deterministic("rv_loglik_"+pl, sum_log_rverr - pm.math.sum((self.rvs['rv'] - (model_rvs[pl] + rv_trend))**2/((1+pm.math.exp(rv_logs2))*self.rvs['rv_err'].astype(floattype))**2))

                        #pm.math.printing.Print("rvlogprobs")(rvlogliks[pl])
                        '''model_rvs_i=[]
                        irvlogliks=[]
                        for i in range(self.n_margs[pl]):
                            model_rvs_i+=[rvorbits[pl].get_radial_velocity(self.rvs['time'],Ks[pl])]
                            imodel = rv_mean + model_rvs_i[-1] + pm.math.sum([model_rvs[ipl] for ipl in self.multis],axis=1)
                            irvlogliks+=[sum_log_rverr - pm.math.sum(-(self.rvs['rv']-imodel)**2/(new_rverr2))]
                        rvlogliks[pl] = pm.Deterministic('rvloglik_'+pl, pm.math.stack(irvlogliks,axis=-1))
                        model_rvs[pl] = pm.Deterministic('model_rv_'+pl, pm.math.stack(model_rvs_i,axis=-1))'''
                    else:
                        rvlogliks[pl]=0.0

            if not self.use_GP:
                #Calculating some extra info to speed up the loglik calculation
                new_yerr_sq = self.model_flux_err.astype(floattype)**2 + \
                              pm.math.sum(self.model_cadence_index*pm.math.exp(logs2).dimshuffle('x',0),axis=1)
                new_yerr = new_yerr_sq**0.5
                sum_log_new_yerr = pm.math.sum(-len(self.model_flux)/2 * pm.math.log(2*np.pi*(new_yerr_sq)))

            stacked_marg_lc={};resids={};logprobs={};logprob_sums={};logprob_margs={};per_marg_avs={};ecc_marg_avs={}
            print("Intiialised everything. Optimizing")

            for pl in self.multis+self.trios+self.duos+self.monos+list(self.rvplanets.keys()):
                if pl in self.multis:
                    #No marginalisation needed for multi-transit candidates, or in the case where we interpolate v_priors
                    stacked_marg_lc[pl]=light_curves[pl]
                    if hasattr(self,'rvs'):
                        marg_rv_models[pl] = pm.Deterministic('marg_rv_model_'+pl, model_rvs[pl])
                elif pl in self.rvplanets:
                    marg_rv_models[pl] = pm.Deterministic('marg_rv_model_'+pl, model_rvs[pl])
                else:
                    if self.n_margs[pl]>1 and not self.interpolate_v_prior:
                        ilogliks=[]
                        resids[pl]={}
                        ################################################
                        #      Compute likelihood for each submodel:
                        ################################################

                        for n in range(self.n_margs[pl]):
                            if self.local_spline:
                                resids[pl][n] = self.model_flux.astype(floattype) - self.spline_params['spline_model_'+pl] - \
                                        (light_curves[pl][:,n] + pm.math.sum([ilc for ilc in stacked_marg_lc],axis=1) + \
                                        phot_mean.dimshuffle('x'))
                            else:
                                resids[pl][n] = self.model_flux.astype(floattype) - \
                                        (light_curves[pl][:,n] + pm.math.sum([ilc for ilc in stacked_marg_lc],axis=1) + \
                                        phot_mean.dimshuffle('x'))
                            if self.debug:
                                pm.math.printing.Print("rawflux_"+str(n))(pm.math._shared(self.model_flux))
                                pm.math.printing.Print("models_"+str(n))(iter_models[pl]['lcs'][:,n])
                                pm.math.printing.Print("resids_"+str(n))(resids[pl][n])
                                pm.math.printing.Print("resids_max_"+str(n))(pm.math.max(resids[pl][n]))
                                pm.math.printing.Print("resids_min_"+str(n))(pm.math.min(resids[pl][n]))
                            if self.use_GP:
                                ilogliks+=[self.gp['use'].log_likelihood(y=resids[pl][n])]
                            else:
                                ilogliks+=[sum_log_new_yerr - pm.math.sum((resids[pl][n])**2/(new_yerr_sq))]
                                #Saving models:

                        lclogliks[pl] = pm.Deterministic('lcloglik_'+pl, pm.math.stack(ilogliks))
                    elif self.interpolate_v_prior:
                        #Assume there is no loglikelihood difference (all difference comes from prior)
                        lclogliks[pl]=pm.math.zeros(self.n_margs[pl])

                    logprobs[pl] = pm.Deterministic('logprob_'+pl, lclogliks[pl] + logpriors[pl] + rvlogliks[pl])
                    #logprob_sums[pl] = pm.Deterministic('logprob_sum_'+pl
                    #Do we want to add this or not..?
                    # If yes, it may cause the fit to converge in local minima for one period
                    # If no, it may let the fit wander to locations bad for *all* models
                    logprob_sums[pl] = pm.Potential("logprob_sum_potential_"+pl,pm.math.logsumexp(logprobs[pl]))
                    #Normalised probability between each submodel (i.e. period):
                    logprob_margs[pl] = pm.Deterministic('logprob_marg_'+pl, logprobs[pl] - logprob_sums[pl])

                    ################################################
                    #       Compute marginalised parameters:
                    ################################################

                    if 'ecc' in self.marginal_params and not self.interpolate_v_prior:
                        pm.Deterministic('ecc_marg_'+pl,pm.math.sum(eccs[pl]*pm.math.exp(logprob_margs[pl])))
                        pm.Deterministic('omega_marg_'+pl,pm.math.sum(omegas[pl]*pm.math.exp(logprob_margs[pl])))
                    elif self.interpolate_v_prior:
                        #Getting double-marginalised eccentricity (across omega space given v and then period space)
                        #pm.math.printing.Print("input_coords")(pm.math.stack([logvels[pl],max_eccs[pl]],axis=-1))
                        eccs[pl] = pm.Deterministic('ecc_'+pl, self.interpolator_eccmarg.evaluate(pm.math.stack([logvels[pl],max_eccs[pl]],axis=-1)).T)
                        #pm.math.printing.Print("eccs[pl]")(eccs[pl])
                        ecc_marg_avs[pl]=pm.Deterministic('ecc_marg_'+pl,pm.math.sum(eccs[pl]*pm.math.exp(logprob_margs[pl])))
                        pm.Deterministic('ecc_marg_sd_'+pl,pm.math.sum(pm.math.exp(logprob_margs[pl])*(eccs[pl]-ecc_marg_avs[pl])**2)/(1-1/self.n_margs[pl]))
                    if 'tdur' not in self.fit_params:
                        pm.Deterministic('tdur_marg_'+pl,pm.math.sum(tdurs[pl]*pm.math.exp(logprob_margs[pl])))
                    if 'b' not in self.fit_params:
                        pm.Deterministic('b_marg_'+pl,pm.math.sum(bs[pl]*pm.math.exp(logprob_margs[pl])))
                    if 'logror' not in self.fit_params:
                        pm.Deterministic('logror_marg_'+pl,pm.math.sum(logrors[pl]*pm.math.exp(logprob_margs[pl])))
                    pm.Deterministic('vel_marg_'+pl,pm.math.sum(vels[pl]*pm.math.exp(logprob_margs[pl])))
                    per_marg_avs[pl] = pm.Deterministic('per_marg_mean_'+pl,pm.math.sum(pers[pl]*pm.math.exp(logprob_margs[pl])))
                    pm.Deterministic('per_marg_sd_'+pl,pm.math.sqrt(pm.math.sum(pm.math.exp(logprob_margs[pl])*(pers[pl]-per_marg_avs[pl])**2)/(1-1/self.n_margs[pl])))
                    if not self.interpolate_v_prior:
                        stacked_marg_lc[pl] = pm.Deterministic('marg_light_curve_'+pl,
                                              pm.math.sum(light_curves[pl] * pm.math.exp(logprob_margs[pl]).dimshuffle('x',0),axis=1))
                    else:
                        stacked_marg_lc[pl] = light_curves[pl]
                    if hasattr(self,'rvs'):
                        marg_rv_models[pl] = pm.Deterministic('marg_rv_model_'+pl,
                                              pm.math.sum(model_rvs[pl] * pm.math.exp(logprob_margs[pl]).dimshuffle('x',0),axis=1))
                        #pm.math.printing.Print("lclogliks")(lclogliks[pl])
                        #pm.math.printing.Print("logpriors")(logpriors[pl])
                        #pm.math.printing.Print("rvlogliks")(rvlogliks[pl])
                        #pm.math.printing.Print("logprobs[pl]")(logprobs[pl])
                        #pm.math.printing.Print("logprobmargs")(logprob_margs[pl])
                        #pm.math.printing.Print("exp(logprobmargs)")(pm.math.exp(logprob_margs[pl]).dimshuffle('x',0))
                        #pm.math.printing.Print("marg_rv_models")(marg_rv_models[pl])
                        pm.Deterministic('K_marg_'+pl,pm.math.sum(Ks[pl]*pm.math.exp(logprob_margs[pl])))
                        pm.Deterministic('Mp_marg_'+pl,pm.math.sum(Mps[pl]*pm.math.exp(logprob_margs[pl])))
                        pm.Deterministic('rho_marg_'+pl,pm.math.sum(rhos[pl]*pm.math.exp(logprob_margs[pl])))

            ################################################
            #     Compute combined model & log likelihood
            ################################################
            marg_all_lc_model = pm.Deterministic("marg_all_lc_model",
                                                    pm.math.sum([stacked_marg_lc[pl] for pl in self.planets],axis=0))

            if hasattr(self,'rvs'):
                if (len(self.planets)+len(self.rvplanets))>1:
                    marg_all_rv_model = pm.Deterministic("marg_all_rv_model",
                            pm.math.sum([marg_rv_models[pl] for pl in list(self.planets.keys())+list(self.rvplanets.keys())],axis=0))
                else:
                    rvkey=list(self.planets.keys())+list(self.rvplanets.keys())
                    marg_all_rv_model = pm.Deterministic("marg_all_rv_model", marg_rv_models[rvkey[0]])
                margrvloglik = pm.Normal("margrvloglik", mu=marg_all_rv_model + rv_trend, sigma=self.rvs['rv_err'].astype(floattype), observed=self.rvs['rv'])
            if self.use_GP:
                self.gp['use'].compute(self.model_time.astype(floattype),
                                       diag=self.model_flux_err.astype(floattype)**2 + \
                                       pm.math.dot(self.model_cadence_index.astype(floattype),pm.math.exp(logs2)), quiet=True)

                total_llk = pm.Deterministic("total_llk",self.gp['use'].log_likelihood(self.model_flux - \
                                                                                       (marg_all_lc_model + phot_mean)))
                gp_pred = pm.Deterministic("gp_pred", self.gp['use'].predict(self.model_flux - (marg_all_lc_model + phot_mean),
                                                                             t=self.model_time,
                                                                             return_var=False, include_mean=False))
                pm.Potential("llk_gp", total_llk)
                #pm.Normal("all_obs",mu=(marg_all_lc_model + gp_pred + mean),sigma=new_yerr,
                #          observed=self.lc.flux[self.lc.near_trans].astype(floattype))
            else:
                marglcloglik=pm.Normal("marglcloglik",mu=(marg_all_lc_model + phot_mean), sigma=new_yerr,
                                       observed=self.model_flux.astype(floattype))
                #pm.math.printing.Print("marglcloglik")(marglcloglik)

            #all_loglik = pm.Normal("all_loglik", mu=pm.math.concatenate([(marg_all_rv_model+rv_trend).flatten(),
            #                                                  (marg_all_lc_model[lc['mask']] + mean).flatten()]),
            #                                                  sigma=pm.math.concatenate([new_rverr.flatten(),new_yerr.flatten()]),
            #                                                  observed=np.hstack([self.rvs['rv'],lc['flux'][lc['mask']]]))


            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.initial_point
            if self.debug: print("optimizing model",model.initial_point)

            ################################################
            #   Creating initial model optimisation menu:
            ################################################
            #Setting up optimization depending on what planet models we have:
            step_initialise=True
            if step_initialise:
                initvars1=[logs2]
                initvars2=[logs2]#P,t0
                initvars3=[logrho_S,logs2]
                initvars4=[logs2]#r,b,P
                for pl in self.planets:
                    initvars1+=[logrors[pl]];initvars2+=[logrors[pl]];initvars4+=[logrors[pl]]
                    initvars2+=[t0s[pl]]
                    if 'b' in self.fit_params or pl in self.multis:
                        initvars4+=[bs[pl]]
                    if 'tdur' in self.fit_params and pl not in self.multis:
                        initvars1+=[tdurs[pl]];initvars4+=[tdurs[pl]]
                    if pl in self.multis:
                        initvars1+=[bs[pl]]
                        initvars2+=[pers[pl]];initvars4+=[pers[pl]]
                    if pl in self.monos:
                        initvars2+=[mono_uniform_index_period[pl]];initvars4+=[mono_uniform_index_period[pl]]
                    if pl in self.duos+self.trios:
                        initvars2+=[t0_2s[pl]]
                    if pl in self.trios and self.model_t03_ttv:
                        initvars2+=[t0_3s[pl]]
                    if not self.assume_circ and (not self.interpolate_v_prior or pl in self.multis):
                        initvars3+=[eccs[pl], omegas[pl]]
                    if hasattr(self,'rvs') and not self.derive_K:
                        initvars1+=[logKs[pl]]
                        initvars3+=[logKs[pl]]
                        #initvars0+=[logMp_wrt_normals[pl]]
                        #initvars2+=[logMp_wrt_normals[pl]]
                        #initvars3+=[logMp_wrt_normals[pl]]

                if hasattr(self,'rvs'):
                    if self.rv_npoly>1:
                        initvars3+=[rv_polys]
                    if len(self.rvplanets)>0:
                        for pl in self.rvplanets:
                            initvars3+=[logKs[pl]]
                    if not self.derive_K:
                        for pl in self.planets:
                            initvars3+=[logKs[pl]]
                    initvars3+=[rv_logs2,Ms]

                if self.use_GP:
                    initvars3+=gpvars
                    initvars3+=[logs2]
                else:
                    if self.local_spline:
                        initvars3+=[self.spline_params['splines_'+pl+'_'+str(int(n))] for pl in self.planets for n in range(3) if 'splines_'+pl+'_'+str(int(n)) in self.spline_params]
                    initvars3+=[phot_mean]
                initvars5=initvars2+initvars3+[logs2,Rs,Ms]
                if np.any([c.split('_')[0]=='ts' for c in self.cads_long]):
                    initvars5+=[u_star_tess]
                if np.any([c.split('_')[0][0]=='k' for c in self.cads_long]):
                    initvars5+=[u_star_kep]
                if np.any([c.split('_')[0]=='co' for c in self.cads_long]):
                    initvars5+=[u_star_corot]
                if np.any([c.split('_')[0]=='ch' for c in self.cads_long]):
                    initvars5+=[u_star_cheops]
                
                ################################################
                #                  Optimising:
                ################################################

                if self.debug: print("before",model.check_initial_point())
                #pm.find_MAP(start=start)
                if self.debug: print("before",model.check_initial_point())
                map_soln = pmx.optimize(vars=initvars1)
                map_soln = pmx.optimize(start=map_soln, vars=initvars2)
                map_soln = pmx.optimize(start=map_soln, vars=initvars3)
                map_soln = pmx.optimize(start=map_soln, vars=initvars4)
                #Doing everything except the marginalised periods:
                map_soln = pmx.optimize(start=map_soln, vars=initvars5)
                map_soln = pmx.optimize(start=map_soln)
                #map_soln = pmx.optimize(start=map_soln, vars=initvars1)
                #map_soln = pmx.optimize(start=start, vars=[logs2])

                if self.debug: print("after",model.check_initial_point())
            else:
                if self.debug: print("before",model.check_initial_point())
                map_soln = pmx.optimize()

            self.model = model
            self.init_soln = map_soln

    def sample_model(self, n_draws=500, n_burn_in=None, overwrite=False, continue_sampling=False, n_chains=4, cores=4, **kwargs):
        """Run PyMC3 sampler

        Args:
            n_draws (int, optional): Number of independent samples to draw from each chain. Defaults to 500.
            n_burn_in (int, optional): Number of steps to 'burn in' the sampler. Defaults to None in which case it becomes 2/3 the number of draws
            overwrite (bool, optional): Overwrite past stored data in this model? Defaults to False.
            continue_sampling (bool, optional): Continue sampling from a previous sampler? Defaults to False.
            n_chains (int, optional): Number of chains to run. Defaults to 4.
        """
        #if not hasattr(self,'trace') and self.use_GP:
        #    #Adding a step to re-do the lightcurve flattening using the new transt durations in the non-GP case
        #    self.init_lc()

        if not overwrite:
            self.load_pickle()
            if hasattr(self,'trace') and self.debug:
                print("LOADED MCMC")

        if not (hasattr(self,'trace') or hasattr(self,'trace_df')) or overwrite or continue_sampling:
            if not hasattr(self,'init_soln'):
                self.init_model()
            #Running sampler:
            try:
                np.random.seed(int(self.ID))
            except:
                np.random.seed(len(self.ID))

            with self.model:
                n_burn_in=np.clip(int(n_draws*0.66),125,15000) if n_burn_in is None else n_burn_in
                if self.debug: print(type(self.init_soln))
                if self.debug: print(self.init_soln.keys())
                if hasattr(self,'trace') and continue_sampling:
                    print("Using already-generated MCMC trace as start point for new trace")
                    self.trace = pm.sample(tune=n_burn_in, draws=n_draws, chains=n_chains, trace=self.trace, compute_convergence_checks=False, cores=cores)#, **kwargs)
                else:
                    self.trace = pm.sample(tune=n_burn_in, draws=n_draws, start=self.init_soln, chains=n_chains, compute_convergence_checks=False, cores=cores)#, **kwargs)
            #Saving both the class and a pandas dataframe of output data.
            self.save_model_to_file()
            _=self.make_table(save=True)
        elif not (hasattr(self,'trace') or hasattr(self,'trace_df')):
            print("Trace or trace df exists...")

    '''
    def Table(self):
        """AI is creating summary for Table
        """
        if load_from_file and not self.overwrite and os.path.exists(self.savenames[0]+'_results.txt'):
            with open(self.savenames[0]+'_results.txt', 'r', encoding='UTF-8') as file:
                restable = file.read()
        else:
            restable=self.to_latex_table(trace, ID, mission=mission, varnames=None,order='columns',
                                       savename=self.savenames[0]+'_results.txt', overwrite=False,
                                       savefileloc=None, tracemask=tracemask)
        '''

    def init_gp_to_plot(self, n_samp=7, max_gp_len=12000, interp=True, newgp=False, overwrite=False,**kwargs):
        """Initialise the GP model for plotting.

            As it is memory-intensive to store predicted GP samples for each datapoint in the light curve during sampling, 
            this is not done by default by MonoTools. Instead, the GPs are re-computed after-the-fact from the sampled 
            hyperparameters, enabling plotting. 
            This re-computation is typically done on limited shorter segments of lightcurve.

            The result is the `gp_to_plot` array, which is a dictionary of predicted GP flux percentiles computed for each point in the time series.

        Args:
            n_samp (int, optional): Number of samples to produce. Defaults to 7.
            max_gp_len (int, optional): Maximum length of photometry to compute a GP on. Defaults to 12000.
            interp (bool, optional): Whether to interpolate the binned out-of-transit GP to the fine time grid (only possible with self.bin_oot is used)
            newgp (bool, optional): Whether to initialise a new GP using the sampled kernel hyperparameters to re-predict the fine time grid
        """
        if self.bin_oot:
            assert interp^newgp, "Have to either interpolate or use a new GP if the out-of-transit data is binned"
        n_samp = 7 if n_samp is None else n_samp
        print("Initalising GP models for plotting with n_samp=",n_samp)
        if newgp:
            from celerite2.pymc import terms as pymc_terms
            import celerite2.pymc
        elif interp:
            from scipy import interpolate
        if not hasattr(self,'lc_regions'):
            self.init_plot(plot_type='lc',**kwargs)
        gp_pred=[]
        gp_sigma=[]
        self.gp_to_plot={'n_samp':n_samp}
        if hasattr(self,'trace'):
            #Using the output of the model trace
            medvars=[var for var in self.trace.posterior if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
            self.meds={}
            for mv in medvars:
                if len(self.trace.posterior[mv].shape)>1:
                    self.meds[mv]=np.median(self.trace.posterior[mv],axis=0)
                elif len(self.trace.posterior[mv].shape)==1:
                    self.meds[mv]=np.median(self.trace.posterior[mv])
        else:
            self.meds=self.init_soln

        limit_mask_bool={}

        # TBD HERE

        if n_samp==1:
            #Creating the median model:
            if interp:
                #assert self.bin_oot
                smooth_func=interpolate.interp1d(np.hstack((np.min(self.lc.time)-0.5,self.model_time,np.max(self.lc.time)+0.5)),
                                                 np.hstack((0,self.init_soln['gp_pred'],0)),kind='slinear')
                print("successfully interpolated GP means")
                self.gp_to_plot['gp_pred']=smooth_func(self.lc.time)+self.init_soln['phot_mean']
                # We're bullshitting the GP SD here by using the lightcurve standard deviation and then boosting it where we dont have good data in the GP...
                # As we're doing a 2D minimum distance matrix, we need to split it up if the lightcurve is very long

                nchunks=int(np.ceil(2+np.log10(len(self.lc.time))))
                timechunks=np.percentile(np.hstack((np.min(self.lc.time)-0.25,self.lc.time,np.max(self.lc.time)-0.25)),
                                         np.linspace(0,100,nchunks+1)) #Splitting using a percentile, this way every time point is in the self.lc.time array
                if np.all([np.any((self.lc.time>timechunks[tc])&(self.lc.time<=timechunks[tc+1])) for tc in range(nchunks)]):
                    min_dist_to_lc=np.hstack([np.min(abs(self.lc.time[(self.lc.time>timechunks[tc])&(self.lc.time<=timechunks[tc+1]),None]-self.model_time[None,(self.model_time>timechunks[tc])&(self.model_time<=timechunks[tc+1])]),axis=1) for tc in range(nchunks)])
                else:
                    print("NO TIME HERE?",timechunks[tc])
                self.gp_to_plot['gp_sd'] = np.tile(np.nanmedian(abs(np.diff(self.lc.bin_flux))),len(self.lc.time))*(np.clip(86400/1800*min_dist_to_lc,1.0,25)**0.33)
            elif newgp:
                for key in self.lc_regions['limits']:
                    #Only creating out-of-transit GP for the binned (e.g. 30min) data
                    cutBools = tools.cut_lc(self.lc.time[self.lc_regions[key]['ix']],max_gp_len,
                                           transit_mask=~self.lc.in_trans['all'][self.lc_regions[key]['ix']])

                    limit_mask_bool[n]={}
                    for nc,c in enumerate(cutBools):
                        limit_mask_bool[n][nc]=np.tile(False,len(self.lc.time))
                        limit_mask_bool[n][nc][self.lc_regions[key]['ix']][c]=self.lc['limit_mask'][n][self.lc_regions[key]['ix']][c]
                        i_kernel = pymc_terms.SHOTerm(S0=self.meds['phot_S0'], w0=self.meds['phot_w0'], Q=1/np.sqrt(2))
                        i_gp = celerite2.pymc.GaussianProcess(i_kernel, mean=self.meds['phot_mean'])

                        i_gp.compute(self.lc.time[limit_mask_bool[n][nc]].astype(floattype),
                                    diag = np.sqrt(self.lc.flux_err[limit_mask_bool[n][nc]]**2 + \
                                    np.dot(self.lc.flux_err_index[limit_mask_bool[n][nc]],np.exp(self.meds['logs2']))))
                        #llk=i_gp.log_likelihood(mod.lc['flux'][mod.lc['mask']][self.lc_regions[key]['ix']][c]-mod.trans_to_plot['all']['med'][mod.lc['mask']][self.lc_regions[key]['ix']][c]-mod.meds['mean'])
                        #print(llk.eval())
                        i_gp_pred, i_gp_var= i_gp.predict(self.lc.flux[limit_mask_bool[n][nc]] - \
                                                        self.trans_to_plot['all']['med'][limit_mask_bool[n][nc]],
                                                        t=self.lc.time[self.lc_regions[key]['ix']][c].astype(floattype),
                                                        return_var=True, return_cov=False, include_mean=False)
                        gp_pred+=[i_gp_pred]
                        gp_sd+=[np.sqrt(i_gp_var)]
                ''''
                gp_pred=[];gp_sigma=[]
                for n in np.arange(len(self.lc['limits'])):
                    with self.model:
                        pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc.time[self.lc_regions[key]['ix']],
                                                                return_var=True,return_cov=False),self.meds)
                    gp_pred+=[pred]
                    gp_sd+=[np.sqrt(var)]
                    print(n,len(self.lc.time[self.lc_regions[key]['ix']]),'->',len(gp_sd[-1]),len(gp_pred[-1]))
                '''
                self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
                self.gp_to_plot['gp_sd']=np.hstack(gp_sd)
                '''
                with self.model:
                    pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc.time.astype(floattype),
                                                                    return_var=True,return_cov=False),self.meds)
                self.gp_to_plot['gp_pred']=pred
                self.gp_to_plot['gp_sd']=np.sqrt(var)
                '''
        elif n_samp>1:
            assert hasattr(self,'trace')
            if interp:
                #assert self.bin_oot
                stacktime=np.hstack((self.lc.time[0]-1,self.model_time,self.lc.time[-1]+1))
                preds=[]
                for i in np.random.choice(len(self.trace.posterior['phot_mean']),int(np.clip(10*n_samp,1,len(self.trace.posterior['phot_mean']))),replace=False):
                    smooth_func=interpolate.interp1d(stacktime, np.hstack((0,self.trace.posterior['gp_pred'][:,i],0)), kind='slinear')
                    preds+=[smooth_func(self.lc.time)+self.trace.posterior['phot_mean'].values[i]]
                prcnts=np.nanpercentile(np.column_stack(preds),[15.8655254, 50., 84.1344746],axis=1)
                self.gp_to_plot['gp_pred']=prcnts[1]
                self.gp_to_plot['gp_sd']=0.5*(prcnts[2]-prcnts[0])

            elif newgp:

                #Doing multiple samples and making percentiles:
                for key in self.lc_regions:
                    #Need to break up the lightcurve even further to avoid GP burning memory:
                    cutBools = tools.cut_lc(self.lc.time[self.lc_regions[key]['ix']],max_gp_len,
                                        transit_mask=~self.lc.in_trans['all'][self.lc_regions[key]['ix']])
                    i_kernel = pymc_terms.SHOTerm(S0=self.meds['phot_S0'], w0=self.meds['phot_w0'], Q=1/np.sqrt(2))
                    i_gp = celerite2.pymc.GaussianProcess(i_kernel, mean=self.meds['phot_mean'])
                    limit_mask_bool[n]={}
                    for nc,c in enumerate(cutBools):
                        limit_mask_bool[n][nc]=np.tile(False,len(self.lc.time))
                        limit_mask_bool[n][nc][self.lc_regions[key]['ix']][c]=self.lc_regions[key]['ix'][self.lc_regions[key]['ix']][c]
                        i_gp_pred=[]
                        i_gp_var=[]
                        for i in np.random.choice(len(self.trace),n_samp,replace=False):
                            sample=self.trace.posterior[i]
                            #print(np.exp(sample['logs2']))
                            i_gp.kernel = pymc_terms.SHOTerm(S0=sample['phot_S0'], w0=sample['phot_w0'], Q=1/np.sqrt(2))
                            i_gp.mean = sample['mean']
                            i_gp.recompute(self.lc.time[limit_mask_bool[n][nc]],
                                        np.sqrt(self.lc.flux_err[limit_mask_bool[n][nc]]**2 + \
                                        np.dot(self.lc.flux_err_index[limit_mask_bool[n][nc]], np.exp(sample['logs2']))))
                            marg_lc=np.tile(0.0,len(self.lc.time))
                            if hasattr(self,'pseudo_binlc') and len(self.trans_to_plot['all']['med'])==len(self.pseudo_binlc['time']):
                                marg_lc[self.lc.near_trans['all']]=sample['marg_all_lc_model'][self.pseudo_binlc['near_trans']]
                            elif hasattr(self,'lc_near_trans') and len(self.trans_to_plot['all']['med'])==len(self.lc_near_trans['time']):
                                marg_lc[self.lc.near_trans['all']]=sample['marg_all_lc_model'][key1][key2]
                            elif len(self.trans_to_plot['all']['med'])==len(self.lc.time):
                                marg_lc[self.lc.near_trans['all']]=sample['marg_all_lc_model'][key1][key2][self.lc.near_trans['all']]

                            #marg_lc[self.lc.near_trans['all']]=sample['marg_all_lc_model'][self.lc.near_trans['all']]
                            ii_gp_pred, ii_gp_var= i_gp.predict(self.lc.flux[limit_mask_bool[n][nc]] - marg_lc[limit_mask_bool[n][nc]],
                                                                t=self.lc.time[self.lc_regions[key]['ix']][c].astype(floattype),
                                                                return_var=True, return_cov=False, include_mean=False)

                            i_gp_pred+=[ii_gp_pred]
                            i_gp_var+=[ii_gp_var]
                        av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
                        gp_pred+=[av]
                        gp_sd+=[std]
                self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
                self.gp_to_plot['gp_sd']=np.hstack(gp_sd)

    def init_trans_to_plot(self,n_samp=None,**kwargs):
        """Initialising the transit models to plot
           
        The result is the `trans_to_plot` array, which is a dictionary of predicted transit flux model percentiles computed for each point in the time series.

        Args:
            n_samp (int, optional): Number of samples to use from the MCMC trace to generate the models & percentiles. Defaults to None.
        """
        n_samp=len(self.trace.posterior['phot_mean']) if n_samp is None else n_samp
        print("Initalising Transit models for plotting with n_samp=",n_samp)
        if not hasattr(self,'lc_regions'):
            self.init_plot(plot_type='lc',**kwargs)
        self.trans_to_plot={'model':{'allpl':{}},
                            'all':{'allpl':{}},
                            'n_samp':n_samp}
        percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}

        if hasattr(self,'trace') and 'marg_all_lc_model' in self.trace.posterior:
            ext_lightcurves=az.extract(self.trace.posterior,var_names=['marg_all_lc_model']+[pl+"_light_curves" for pl in self.planets])

            prcnt=np.percentile(ext_lightcurves['marg_all_lc_model'],list(percentiles.values()),axis=1)
            self.trans_to_plot['model']['allpl']={list(percentiles.keys())[n]:prcnt[n] for n in range(5)}
        elif 'marg_all_lc_model' in self.init_soln:
            self.trans_to_plot['model']['allpl']['med']=self.init_soln['marg_all_lc_model']
        else:
            print("marg_all_lc_model not in any optimised models")
        for pl in self.planets:
            if hasattr(self,'trace') and pl+"_light_curves" in ext_lightcurves:
                prcnt = np.percentile(ext_lightcurves[pl+"_light_curves"], list(percentiles.values()), axis=1)
                self.trans_to_plot['model'][pl] = {list(percentiles.keys())[n]:prcnt[n] for n in range(5)}
            elif hasattr(self,'init_soln') and pl+"_light_curves" in self.init_soln:
                self.trans_to_plot['model'][pl] = {'med':self.init_soln[pl+"_light_curves"]}
        '''
            self.trans_to_plot_i[pl]={}
            if pl in self.multis or self.interpolate_v_prior:
                if hasattr(self,'trace') and 'light_curve_'+pl in ext_lightcurves:
                    if len(ext_lightcurves['mask_light_curves'].shape)>2:
                        prcnt = np.percentile(ext_lightcurves['multi_mask_light_curves'][:,:,self.multis.index(pl)],
                                                  (5,16,50,84,95),axis=0)
                    else:
                        prcnt = np.percentile(ext_lightcurves['multi_mask_light_curves'], (5,16,50,84,95), axis=0)

                elif 'multi_mask_light_curves' in self.init_soln:
                    if len(self.init_soln['multi_mask_light_curves'].shape)==1:
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves']
                    else:
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves'][:,self.multis.index(pl)]
                else:
                    print('multi_mask_light_curves not in any optimised models')
            elif pl in self.duos or self.monos and not self.interpolate_v_prior:
                if hasattr(self,'trace') and 'marg_light_curve_'+pl in ext_lightcurves:
                    prcnt=np.percentile(ext_lightcurves['marg_light_curve_'+pl],(5,16,50,84,95),axis=0)
                    nms=['-2sig','-1sig','med','+1sig','+2sig']
                    self.trans_to_plot_i[pl] = {nms[n]:prcnt[n] for n in range(5)}
                elif 'marg_light_curve_'+pl in self.init_soln:
                    self.trans_to_plot_i[pl]['med'] = self.init_soln['marg_light_curve_'+pl]
                else:
                    print('marg_light_curve_'+pl+' not in any optimised models')
        '''

        #Adding zeros to other regions where we dont have transits (not in the out of transit mask):
        for key1 in self.trans_to_plot['model']:
            self.trans_to_plot['all'][key1]={}
            for key2 in self.trans_to_plot['model'][key1]:
                if hasattr(self,'model_near_trans'):
                    self.trans_to_plot['all'][key1][key2]=np.zeros(len(self.lc.time))
                    self.trans_to_plot['all'][key1][key2][self.lc.near_trans['all']*self.lc.mask]=self.trans_to_plot['model'][key1][key2][self.model_near_trans]
                else:
                    self.trans_to_plot['all'][key1][key2]=np.zeros(len(self.lc.time))
                    self.trans_to_plot['all'][key1][key2][self.lc.mask]=self.trans_to_plot['model'][key1][key2][:]

        if len(self.planets)==1 and list(self.planets.keys())[0] not in self.trans_to_plot['model']:
            self.trans_to_plot['model'][list(self.planets.keys())[0]] = self.trans_to_plot['model']['allpl']
            self.trans_to_plot['all'][list(self.planets.keys())[0]] = self.trans_to_plot['all']['allpl']
    
    def init_spline_to_plot(self,n_samp=None,**kwargs):
        """Initialising the transit models to plot
           
        The result is the `trans_to_plot` array, which is a dictionary of predicted transit flux model percentiles computed for each point in the time series.

        Args:
            n_samp (int, optional): Number of samples to use from the MCMC trace to generate the models & percentiles. Defaults to None.
        """
        ext_info=az.extract(self.trace.posterior,var_names=['phot_mean']+['spline_model_'+pl for pl in self.monos+self.duos+self.trios])

        n_samp=len(ext_info['phot_mean']) if n_samp is None else n_samp
        if not hasattr(self,'lc_regions'):
            self.init_plot(plot_type='lc',**kwargs)
        self.spline_to_plot={'model':{'allpl':{}},
                             'all':{'allpl':{}},
                             'n_samp':n_samp}
        percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}

        if hasattr(self,'trace'):
            prcnt=np.percentile(np.sum(np.dstack([ext_info['spline_model_'+pl] for pl in self.monos+self.duos+self.trios]),axis=2),list(percentiles.values()),axis=0)
            self.spline_to_plot['model']['allpl']={list(percentiles.keys())[n]:prcnt[n] for n in range(5)}
        elif 'marg_all_lc_model' in self.init_soln:
            self.spline_to_plot['model']['allpl']['med']=np.sum(np.vstack([self.init_soln['spline_model_'+pl] for pl in self.monos+self.duos+self.trios]),axis=0)
        else:
            print("spline models not in any optimised models")

        #Adding zeros to other regions where we dont have transits (not in the out of transit mask):
        for key1 in self.spline_to_plot['model']:
            self.spline_to_plot['all'][key1]={}
            for key2 in self.spline_to_plot['model'][key1]:
                if hasattr(self,'model_near_trans'):
                    self.spline_to_plot['all'][key1][key2]=np.zeros(len(self.lc.time))
                    self.spline_to_plot['all'][key1][key2][self.lc.near_trans['all']*self.lc.mask]=self.spline_to_plot['model'][key1][key2][self.model_near_trans]
                else:
                    self.spline_to_plot['all'][key1][key2]=np.zeros(len(self.lc.time))
                    self.spline_to_plot['all'][key1][key2][self.lc.mask]=self.spline_to_plot['model'][key1][key2][:]

    def init_rvs_to_plot(self, n_samp=400, plot_alias='all'):
        """Initialise RV models to plot.

        The result is the `rvs_to_plot` array.
        This is formed of two hierarchical dictionaries - one for each RV x point (`rvs_to_plot["x"]`), and one for a fine grid of time points (`rvs_to_plot["t"]`)
        Then, each contains the following dictionaries of percentiles (keys=['-2sig','-1sig','med','+1sig','+2sig']) computed for each x and t point in the time series:
         - the RV polynomial trend (e.g. `rvs_to_plot["x"]["trend"]`)
         - a summed RV model (e.g. `rvs_to_plot["x"]["all"]`)
         - combined RV models with the trend (e.g. `rvs_to_plot["x"]["all+trend"]`)
         - For each individual planet dictionaries with either:
             - marginalised RV model given the multiple perid aliases (for Duo/Monotransiting planets, e.g. `rvs_to_plot["x"]["marg"]`)
             - Individual RV models (for rv planets/multi-transiting planets, e.g. `rvs_to_plot["x"][0]`)

        Args:
            n_samp (int, optional): Number of MCMC samples to use to generate RV models. Defaults to 300
            plot_alias (str, optional): How to plot aliases - either 'all' or 'best'. Defaults to 'all'.
        """
        #Going from the outputted samples back through exoplanet model to create times on a fine grid

        
        all_pls_in_rvs=list(self.planets.keys())+list(self.rvplanets.keys())

        self.rvs_to_plot={'t':{pl:{} for pl in all_pls_in_rvs},
                          'x':{pl:{} for pl in all_pls_in_rvs}}
        self.rvs_to_plot['n_samp']=n_samp

        self.rvs_to_plot['t']['time']=np.arange(np.min(self.rvs['time'])-5,np.max(self.rvs['time'])+5,0.5)
        if hasattr(self,'trace'):
            vars=['Rs','rho_S','rv_trend','rv_polys','rv_offsets']
            for pl in self.planets:
                vars+=['t0_'+pl,'b_'+pl,'K_'+pl,'per_'+pl,'logprob_marg_'+pl,'marg_rv_model_'+pl,'model_rv_'+pl]
                if not self.assume_circ:
                    vars+=['omega_'+pl,'ecc_'+pl]
            ext_rv_trace = az.extract(self.trace.posterior,var_names=vars,num_samples=n_samp)
            samples=[ext_rv_trace]
        else:
            samples=[self.init_soln]
        all_rv_ts_i={pl:[] for pl in all_pls_in_rvs}
        marg_rv_ts_i={pl:[] for pl in all_pls_in_rvs}
        trends_i=[]
        for i, sample in enumerate(samples):
            with self.model:
                for pl in all_pls_in_rvs:
                    #Generating RV curves on the fly given a sample by re-initialising exoplanet orbits
                    if pl in self.multis:
                        if self.assume_circ:
                            rvs = xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                              rho_star=sample['rho_S']*1.40978,
                                                              period=sample['per_'+pl],t0=sample['t0_'+pl],b=sample['b_'+pl]
                                                  ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]).eval()
                        else:
                            rvs = xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                              rho_star=sample['rho_S']*1.40978,
                                                              period=sample['per_'+pl],t0=sample['t0_'+pl],b=sample['b_'+pl],
                                                              ecc=sample['ecc_'+pl],omega=sample['omega_'+pl]
                                                   ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]).eval()
                        marg_rv_ts_i[pl]+=[rvs]
                    elif pl in self.rvplanets:

                        rvs = xo.orbits.KeplerianOrbit(period=sample['per_'+pl], t0=sample['t0_'+pl],
                                                                        ecc=sample['ecc_'+pl], omega=sample['omega_'+pl]
                                               ).get_radial_velocity(self.rvs_to_plot['t']['time'], K=sample['K_'+pl]).eval()
                        marg_rv_ts_i[pl]+=[rvs]

                    else:
                        if self.interpolate_v_prior:
                            rvs = xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978,
                                                    period=sample['per_'+pl],
                                                    t0=tensor.basic.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tensor.basic.tile(sample['b_'+pl],self.n_margs[pl]),
                                                    ecc=sample['min_ecc_'+pl],omega=sample['omega_'+pl]
                                              ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]).eval()
                        elif not self.assume_circ:
                            rvs = xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978,
                                                    period=sample['per_'+pl],
                                                    t0=tensor.basic.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tensor.basic.tile(sample['b_'+pl],self.n_margs[pl]),
                                                    ecc=tensor.basic.tile(sample['ecc_'+pl],self.n_margs[pl]),
                                                    omega=tensor.basic.tile(sample['omega_'+pl],self.n_margs[pl])
                                             ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]).eval()
                        elif self.assume_circ:
                            rvs = xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978,
                                                    period=sample['per_'+pl],
                                                    t0=tensor.basic.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tensor.basic.tile(sample['b_'+pl],self.n_margs[pl])
                                              ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]).eval()
                        all_rv_ts_i[pl]+=[rvs]
                        marg_rv_ts_i[pl]+=[np.sum(rvs*np.exp(sample['logprob_marg_'+pl]),axis=1)]
                if self.rv_npoly>2:
                    trends_i+=[np.dot(np.vander(self.rvs_to_plot['t']['time']-self.rv_tref,self.rv_npoly)[:,:-1],sample['rv_polys'])]
                elif self.rv_npoly==2:
                    trends_i+=[(self.rvs_to_plot['t']['time']-self.rv_tref)*sample['rv_polys']]
                else:
                    trends_i+=[np.tile(0.0,len(self.rvs_to_plot['t']['time']))]
        #print(trends_i)
        nms=['-2sig','-1sig','med','+1sig','+2sig']
        percentiles=(2.2750132, 15.8655254, 50., 84.1344746, 97.7249868)
        if hasattr(self,'trace'):
            #Taking best-fit model:
            for pl in all_pls_in_rvs:
                xprcnts      = np.percentile(ext_rv_trace["marg_rv_model_"+pl], percentiles, axis=0)
                xtrendprcnts = np.percentile(ext_rv_trace["marg_rv_model_"+pl]+ext_rv_trace["rv_trend"], percentiles, axis=0)

                self.rvs_to_plot['x'][pl]['marg'] = {nms[n]:xprcnts[n] for n in range(5)}
                self.rvs_to_plot['x'][pl]['marg+trend'] = {nms[n]:xtrendprcnts[n] for n in range(5)}
                tprcnts = np.percentile(np.vstack(marg_rv_ts_i[pl]), percentiles, axis=0)
                ttrendprcnts = np.percentile(np.vstack(marg_rv_ts_i[pl])+np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t'][pl]['marg'] = {nms[n]:tprcnts[n] for n in range(5)}
                self.rvs_to_plot['t'][pl]['marg+trend'] = {nms[n]:ttrendprcnts[n] for n in range(5)}
                if pl in self.trios+self.duos+self.monos:
                    alltrvs = np.dstack(all_rv_ts_i[pl])
                    for i in range(self.n_margs[pl]):
                        xiprcnts=np.percentile(ext_rv_trace["model_rv_"+pl][:,:,i], percentiles, axis=0)
                        self.rvs_to_plot['x'][pl][i]={nms[n]:xiprcnts[n] for n in range(5)}
                        tiprcnts=np.percentile(alltrvs[:,i,:], percentiles, axis=1)
                        self.rvs_to_plot['t'][pl][i]={nms[n]:tiprcnts[n] for n in range(5)}
            #print(self.rvs_to_plot)
            if len(all_pls_in_rvs)>1:
                iprcnts = np.percentile(ext_rv_trace["rv_trend"],percentiles, axis=0)
                self.rvs_to_plot['x']["trend+offset"] = {nms[n]:iprcnts[n] for n in range(5)}
                iprcnts = np.percentile(ext_rv_trace["marg_all_rv_model"],percentiles, axis=0)
                self.rvs_to_plot['x']["all"] = {nms[n]:iprcnts[n] for n in range(5)}
                iprcnts = np.percentile(ext_rv_trace["marg_all_rv_model"]+ext_rv_trace["rv_trend"],percentiles, axis=0)
                self.rvs_to_plot['x']["all+trend"] = {nms[n]:iprcnts[n] for n in range(5)}

                iprcnts = np.percentile(np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t']["trend"] = {nms[n]:iprcnts[n] for n in range(5)}
                iprcnts = np.percentile(np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=1),percentiles, axis=0)
                self.rvs_to_plot['t']["all"] = {nms[n]:iprcnts[n] for n in range(5)}
                #print(len(trends_i), len(trends_i[0]), np.vstack(trends_i).shape)
                #print(np.vstack(marg_rv_ts_i[pl]).shape,
                #      np.dstack([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs]).shape,
                #     np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=1).shape)
                iprcnts = np.percentile(np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=0)+np.vstack(trends_i),
                                        percentiles, axis=0)
                self.rvs_to_plot['t']["all+trend"] = {nms[n]:iprcnts[n] for n in range(5)}
            else:
                iprcnts = np.percentile(ext_rv_trace["rv_trend"], percentiles, axis=0)
                self.rvs_to_plot['x']["trend+offset"] = {nms[n]:iprcnts[n] for n in range(5)}
                self.rvs_to_plot['x']["all"] = self.rvs_to_plot['x'][pl]
                self.rvs_to_plot['x']["all+trend"] = self.rvs_to_plot['x'][pl]["marg+trend"]

                iprcnts = np.percentile(np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t']["trend"] = {nms[n]:iprcnts[n] for n in range(5)}
                #print(self.rvs_to_plot['t']["trend"])
                self.rvs_to_plot['t']["all"] = self.rvs_to_plot['t'][pl]
                self.rvs_to_plot['t']["all+trend"] = self.rvs_to_plot['t'][pl]["marg+trend"]
            iprcnts = np.percentile(ext_rv_trace["rv_offsets"], percentiles, axis=0)
            self.rvs_to_plot['x']["offsets"] = {nms[n]:iprcnts[n] for n in range(5)}
        elif hasattr(self,'init_soln'):
            for pl in marg_rv_ts_i:
                self.rvs_to_plot['x'][pl]['marg']={'med':self.init_soln["marg_rv_model_"+pl]}
                self.rvs_to_plot['x'][pl]["marg+trend"]={'med':self.init_soln["marg_rv_model_"+pl]+self.init_soln['rv_trend']}
                self.rvs_to_plot['t'][pl]['marg']={'med':np.array(marg_rv_ts_i[pl]).ravel()}
                self.rvs_to_plot['t'][pl]['marg+trend']={'med':np.array(marg_rv_ts_i[pl]).ravel()+trends_i[0]}
            for pl in marg_rv_ts_i:
                self.rvs_to_plot['x']["all"]={'med':self.init_soln["marg_all_rv_model"]}
                self.rvs_to_plot['x']["all+trend"]={'med':self.init_soln["marg_all_rv_model"]+self.init_soln["rv_trend"]}
                self.rvs_to_plot['t']["all"]=self.rvs_to_plot['t'][pl]["marg"] if len(all_pls_in_rvs)==1 else {'med':np.sum(np.vstack([self.rvs_to_plot['t'][pl]["marg"]['med'] for pl in all_pls_in_rvs]),axis=0)}
                self.rvs_to_plot['t']["all+trend"]=self.rvs_to_plot['t'][pl]["marg+trend"] if len(all_pls_in_rvs)==1 else {'med':np.sum(np.vstack([self.rvs_to_plot['t'][pl]["marg"]['med'] for pl in all_pls_in_rvs]),axis=0)+trends_i[0]}
                if pl in all_rv_ts_i:
                    for i in range(self.n_margs[pl]):
                        if self.n_margs[pl]>1:
                            self.rvs_to_plot['x'][pl][i]={'med':self.init_soln["model_rv_"+pl][:,i]}
                            self.rvs_to_plot['t'][pl][i]={'med':all_rv_ts_i[pl][0][:,i]}
                        else:
                            self.rvs_to_plot['x'][pl][i]={'med':self.init_soln["model_rv_"+pl]}
                            self.rvs_to_plot['t'][pl][i]={'med':all_rv_ts_i[pl]}

            self.rvs_to_plot['t']["trend"] = {'med':trends_i[0]}
            self.rvs_to_plot['x']["trend+offset"] = {'med':self.init_soln["rv_trend"]}
            self.rvs_to_plot['x']["offsets"] = {'med':self.init_soln["rv_offsets"]}

    def init_plot(self, interactive=False, gap_thresh=10, plottype='lc',pointcol='k',palette=None, ncols=None, plot_flat=False,**kwargs):
        """Initialising plotting

        Args:
            interactive (bool, optional): Interactive bokeh plot? Defaults to False.
            gap_thresh (int, optional): Threshold in days above which we cut the plot into smaller figures. Defaults to 10.
            plottype (str, optional): 'lc' or 'rv'. Defaults to 'lc'.
            pointcol (str, optional): colour of smallest raw points. Defaults to 'k' (black)
            palette (str, optional): specify the (i.e. seaborn) colour palette to use. Defaults to None.
            ncols (int, optional): The number of colours to use. Defaults to None.
        """
        import seaborn as sns
        if palette is None:
            ncols = len(self.planets)+4 if ncols is None else ncols
            self.pal = sns.color_palette('viridis', ncols).as_hex()
        else:
            self.pal = sns.color_palette(palette).as_hex()
        if pointcol=="k":
            sns.set_style('whitegrid')
            #Plots bokeh figure
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 0.75

        if not hasattr(self,'savenames'):
            self.get_savename(how='save')
        #Making sure lc is binned to 30mins
        if plottype=='lc':
            if plot_flat:
                if not self.use_GP:
                    self.lc.bin(timeseries=['flux_flat'],binsize=1/48.0)
                    fx_lab='flux_flat'
                    fx_bin_lab='bin_flux_flat'
                else:
                    fx_lab='flux'
                    fx_bin_lab='bin_flux'
            else:
                fx_lab='flux'
                fx_bin_lab='bin_flux'
            time_regions=tools.find_time_regions(self.lc.time[self.lc.mask],split_gap_size=gap_thresh,**kwargs)
            self.lc_regions={}
            for nj in range(len(time_regions)):
                self.lc_regions[nj]={'start':time_regions[nj][0],'end':time_regions[nj][1]}
                self.lc_regions[nj]['total_dur']=self.lc_regions[nj]['end']-self.lc_regions[nj]['start']
                self.lc_regions[nj]['ix']=(self.lc.time>=self.lc_regions[nj]['start'])*(self.lc.time<=self.lc_regions[nj]['end'])*self.lc.mask
                self.lc_regions[nj]['bin_ix']=(self.lc.bin_time>=self.lc_regions[nj]['start'])*(self.lc.bin_time<=self.lc_regions[nj]['end'])*np.isfinite(getattr(self.lc,fx_bin_lab))
                self.lc_regions[nj]['model_ix']=(self.model_time>=self.lc_regions[nj]['start'])&(self.model_time<=self.lc_regions[nj]['end'])
                self.lc_regions[nj]['cadence']=np.nanmedian(np.diff(self.lc.time[self.lc_regions[nj]['ix']]))
                self.lc_regions[nj]['mad']=1.06*np.nanmedian(abs(np.diff(getattr(self.lc,fx_lab)[self.lc_regions[nj]['ix']])))
                self.lc_regions[nj]['minmax']=np.nanpercentile(getattr(self.lc,fx_bin_lab)[self.lc_regions[nj]['bin_ix']],[0.25,99.75])
            transmin=np.min(self.init_soln['marg_all_lc_model']) if not hasattr(self,'trace') else np.min(np.nanmedian(az.extract(self.trace.posterior,var_names=['marg_all_lc_model']).values,axis=0))
            minmax_global = (np.min([np.min([self.lc_regions[nj]['minmax'][0],transmin])-self.lc_regions[nj]['mad'] for nj in self.lc_regions]),
                             np.max([self.lc_regions[nj]['minmax'][1]+self.lc_regions[nj]['mad'] for cad in self.lc_regions]))

            total_time = np.sum([self.lc_regions[i]['total_dur'] for i in self.lc_regions])
            for nj in self.lc_regions:
                #Let's just estimate the number of rows we want per big 
                self.lc_regions[nj]['n_ideal_split'] = np.log10(np.clip(3.333*(self.lc_regions[nj]['total_dur']-80),10,100000))*self.lc_regions[nj]['total_dur']/total_time
                self.lc_regions[nj]['total_time']=total_time
                self.lc_regions[nj]['minmax_global']=minmax_global
        #     lc_limits = tools.find_time_regions(self.lc.time)
        #     if not hasattr(self,'gap_lens'):
        #         #Finding if there's a single enormous gap in the lightcurve, and creating time splits for each region
        #         x_gaps=np.hstack((0, np.where(np.diff(self.lc.time)>gap_thresh)[0]+1, len(self.lc.time)))
        #         self.lc['limits']=[]
        #         self.lc['binlimits']=[]
        #         gap_lens=[]
        #         for ng in range(len(x_gaps)-1):
        #             self.lc['limits']+=[[x_gaps[ng],x_gaps[ng+1]]]
        #             gap_lens+=[self.lc.time[self.lc['limits'][-1][1]-1]-self.lc.time[self.lc['limits'][-1][0]]]
        #             self.lc['binlimits']+=[[np.argmin(abs(self.lc.bin_time-self.lc.time[x_gaps[ng]])),
        #                          np.argmin(abs(self.lc.bin_time-self.lc.time[x_gaps[ng+1]-1]))+1]]
        #         self.lc['gap_lens']=np.array(gap_lens)
        #         all_lens=np.sum(self.lc['gap_lens'])
        #         self.lc['limit_mask']={}
        #         #modlclim_mask={}
        #         for n in range(len(self.lc['gap_lens'])):
        #             #modlclim_mask[n]=np.tile(False,len(self.plot_lc['time']))
        #             #modlclim_mask[n][modlclims[n][0]:modlclims[n][1]][lc['mask'][modlclims[n][0]:modlclims[n][1]]]=True
        #             self.lc['limit_mask'][n]=np.tile(False,len(self.lc.time))
        #             self.lc['limit_mask'][n][self.lc_regions[key]['ix']][self.lc.mask[self.lc_regions[key]['ix']]]=True
        # #elif plottype=='rv':


    def plot_RVs(self, interactive=False, plot_alias='best', nbest=4, n_samp=300, overwrite=False, return_fig=False, plot_resids=False,
                plot_loc=None, palette=None, pointcol='k', plottype='png',raster=False, nmargtoplot=0, save=True,**kwargs):
        """Varied plotting function for RVs of MonoTransit model

        Args:
            interactive (bool, optional): Plot interactive bokeh image? Defaults to False i.e. matplotlib plot
            plot_alias (str, optional): How to plot RV aliaes - 'all' or 'best'. Defaults to 'best'.
            nbest (int, optional): Number of the best aliases to plot. Defaults to 4.
            n_samp (int, optional): Number of samples to use to initialise the RV plotting. Defaults to 300.
            overwrite (bool, optional). Defaults to False.
            return_fig (bool, optional). Return figure variable. Defaults to False.
            plot_resids (bool, optional). Plot residuals under timeseries plot. Defaults to False.
            plot_loc ([type], optional): [description]. Defaults to None.
            palette (str, optional): colour palette to use. Defaults to None.
            pointcol (str, optional): small point colour to use. Defaults to 'k'.
            plottype (str, optional): Type of image to save, if not interactive. Defaults to 'png'.
            raster (bool, optional): Whether to rasterize the image to reduce file size. Defaults to False.
            nmargtoplot (int, optional): Which marginalised planet (Mono/Duo) to plot. Defaults to 0, i.e. first planet
            save (bool, optional). Defaults to True.

        Raises:
            ValueError: [description]
        """
        ################################################################
        #     
        ################################################################
        import seaborn as sns
        sns.set_palette('viridis')

        # plot_alias - can be 'all' or 'best'. All will plot all aliases. Best will assume the highest logprob.
        ncol=3+2*np.max(list(self.n_margs.values())) if plot_alias=='all' else 3+2*nbest
        self.init_plot(plottype='rv', pointcol=pointcol, ncols=ncol,**kwargs)

        if not hasattr(self,'rvs_to_plot') or n_samp!=self.rvs_to_plot['n_samp'] or overwrite:
            self.init_rvs_to_plot(n_samp, plot_alias)

        averr=np.nanmedian(self.rvs['rv_err'])

        all_pls_in_rvs=list(self.planets.keys())+list(self.rvplanets.keys())

        other_pls=self.multis+list(self.rvplanets.keys())
        if len(self.monos+self.duos+self.trios)>0:
            marg_pl=(self.monos+self.duos+self.trios)[nmargtoplot]
        else:
            marg_pl=self.multis[-1]
        #Here we'll choose the best RV curves to plot (in the case of mono/duos)
        nbests = self.n_margs[marg_pl] if plot_alias=='all' else nbest
        if hasattr(self,'trace'):
            ibest = np.nanmedian(az.extract(self.trace.posterior,var_names=['logprob_marg_'+marg_pl]).values,axis=0).argsort()[-1*nbests:]
            heights = np.array([np.clip(np.nanmedian(az.extract(self.trace.posterior,var_names=['K_'+marg_pl]).values[:,i]),0.5*averr,10000) for i in ibest])
        elif hasattr(self,'init_soln'):
            ibest = self.init_soln['logprob_marg_'+marg_pl].argsort()[-1*nbests:]
            heights = np.array([np.clip(self.init_soln['K_'+marg_pl][i],0.5*averr,10000) for i in ibest])
        if len(other_pls)==1:
            heights=np.max(heights)+list(np.array(heights)+np.max(heights))
        heights= np.round(heights[::-1]*24/np.sum(heights[::-1]))
        heights_sort = np.hstack((0,np.cumsum(heights).astype(int)))+6*len(other_pls)

        if interactive:
            from bokeh.plotting import figure, output_file, save, curdoc, show
            from bokeh.models import Band, Whisker, ColumnDataSource, Range1d, arrow_heads
            from bokeh.layouts import gridplot, row, column, layout

            if plot_loc is None:
                savename=self.savenames[0]+'_model_plot.html'
            else:
                savename=plot_loc
            output_file(savename)

            #Initialising figure:
            p = figure(plot_width=800, plot_height=500,title=str(self.ID)+" Transit Fit")
        #Initialising lists of phase-folded plots:
        f_phase={pl:[] for pl in all_pls_in_rvs}

        if not interactive:
            fig=plt.figure(figsize=(7*(0.5 * (1+np.sqrt(5))), 7))
            gs = fig.add_gridspec(heights_sort[-1],3*(3+len(all_pls_in_rvs)),wspace=0.3,hspace=1.25)
            if plot_resids:
                f_alls=fig.add_subplot(gs[:int(np.floor(0.75*heights_sort[-1])),:2*(3+len(all_pls_in_rvs))])
                f_resids=fig.add_subplot(gs[int(np.floor(0.75*heights_sort[-1])):,:2*(3+len(all_pls_in_rvs))])
            else:
                f_alls=fig.add_subplot(gs[:,:2*(3+len(all_pls_in_rvs))])
            #looping through each planet and each alias we want to plot:
            pl=(self.duos+self.monos+self.trios)[nmargtoplot]
            npl=0
            for nplot in range(nbests):
                #print(pl,npl,heights_sort[::-1][nplot+1],"->",heights_sort[::-1][nplot],",",
                #          (2+npl)*(3+len(all_pls_in_rvs)),"->",(3+npl)*(3+len(all_pls_in_rvs)),"/",
                #          heights_sort[-1],3*(3+len(all_pls_in_rvs)) )
                #print(heights_sort[::-1][nplot+1],heights_sort[::-1][nplot])
                if nplot==0:
                    f_phase[pl]+=[fig.add_subplot(gs[heights_sort[::-1][nplot+1]:heights_sort[::-1][nplot],
                                                      (2+npl)*(3+len(all_pls_in_rvs)):(3+npl)*(3+len(all_pls_in_rvs))])]
                else:
                    f_phase[pl]+=[fig.add_subplot(gs[heights_sort[::-1][nplot+1]:heights_sort[::-1][nplot],
                                                      (2+npl)*(3+len(all_pls_in_rvs)):(3+npl)*(3+len(all_pls_in_rvs))],
                                                   sharex=f_phase[pl][0])]
                f_phase[pl][-1].yaxis.tick_right()

            for n_oth,othpl in enumerate(other_pls):
                f_phase[othpl]=fig.add_subplot(gs[(n_oth*6):((n_oth+1)*6),
                                                  (2+npl)*(3+len(all_pls_in_rvs)):(3+npl)*(3+len(all_pls_in_rvs))],
                                               sharex=f_phase[pl][0])

            f_alls.plot(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["trend"]["med"],c='C6')
            if "-2sig" in self.rvs_to_plot['t']["trend"]:
                #print(self.rvs_to_plot['t']['time'].shape,self.rvs_to_plot['t']["trend"]["-2sig"].shape,self.rvs_to_plot['t']["trend"]["+2sig"].shape)
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["trend"]["-2sig"],
                                    self.rvs_to_plot['t']["trend"]["+2sig"],color='C6',alpha=0.1)
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["trend"]["-1sig"],
                                    self.rvs_to_plot['t']["trend"]["+1sig"],color='C6',alpha=0.1,label='trend')
            f_alls.plot(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["med"],c='C1',label='marg. model')
            if "-2sig" in self.rvs_to_plot['t']["all+trend"]:
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["-2sig"],
                                    self.rvs_to_plot['t']["all+trend"]["+2sig"],color='C1',alpha=0.1)
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["-1sig"],
                                    self.rvs_to_plot['t']["all+trend"]["+1sig"],color='C1',alpha=0.1)
            for nc in range(len(self.rvs['scopes'])):
                scope_ix=self.rvs['tele_index']==self.rvs['scopes'][nc]
                f_alls.errorbar(self.rvs['time'][scope_ix],
                                self.rvs['rv'][scope_ix]-self.rvs_to_plot['x']["offsets"]["med"][nc],
                                yerr=self.rvs['rv_err'][scope_ix],fmt='.',markersize=8,ecolor='#bbbbbb',
                                c='C'+str(nc+1),label='scope:'+self.rvs['scopes'][nc])
                if plot_resids:
                    f_resids.errorbar(self.rvs['time'][scope_ix],
                                    self.rvs['rv'][scope_ix]-self.rvs_to_plot['x']["all+trend"]["med"][scope_ix],
                                    yerr=self.rvs['rv_err'][scope_ix],fmt='.',markersize=8,ecolor='#bbbbbb',
                                    c='C'+str(nc+1))

            plt.setp(f_alls.get_xticklabels(), visible=False)
            f_alls.plot(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["med"],c='C4',alpha=0.6,lw=2.5)
            if plot_resids:
                f_resids.plot(self.rvs_to_plot['t']['time'], np.zeros(len(self.rvs_to_plot['t']['time'])),c='C4',alpha=0.4,lw=2.5)


        elif interactive:
            from bokeh.plotting import figure, output_file, save, curdoc, show
            from bokeh.models import Band, Whisker, ColumnDataSource, Range1d, arrow_heads
            from bokeh.layouts import gridplot, row, column, layout

            #For Bokeh plots, we can just use the size in pixels
            f_alls=figure(width=800-200*len(all_pls_in_rvs), plot_height=350, title=None)
            pl=(self.duos+self.monos+self.trios)[nmargtoplot]
            npl=1
            for nplot in range(nbests)[::-1]:
                if nplot==nbests-1:
                    f_phase[pl]+=[figure(width=140,plot_height=500*(heights_sort[nplot+1]-heights_sort[nplot])/heights_sort[-1],
                                          title=None, y_axis_location="right")]
                    f_phase[pl][-1].xaxis.axis_label = 'Phase'
                else:
                    f_phase[pl]+=[figure(width=140,plot_height=500*(heights_sort[nplot+1]-heights_sort[nplot])/heights_sort[-1],
                                          title=None, y_axis_location="right")]+f_phase[pl]
            for n_oth,othpl in enumerate(other_pls):
                f_phase[othpl]=figure(width=140,plot_height=500*6/heights_sort[-1],title=None, y_axis_location="right")

            for nc in range(len(self.rvs['scopes'])):
                scope_ix=self.rvs['tele_index']==self.rvs['scopes'][nc]
                f_alls.circle(self.rvs['time'][scope_ix], self.rvs['rv'][scope_ix]-self.rvs_to_plot['x']["offsets"]["med"][nc],
                             color='black',alpha=0.5,size=0.75)
                errors = ColumnDataSource(data=dict(base=self.rvs['time'][scope_ix]-self.rvs_to_plot['x']["offsets"]["med"][nc],
                    lower=self.rvs['rv'][scope_ix] - self.rvs_to_plot['x']["offsets"]["med"][nc] - self.rvs['rv_err'][scope_ix],
                   upper=self.rvs['rv'][scope_ix] - self.rvs_to_plot['x']["offsets"]["med"][nc] + self.rvs['rv_err'][scope_ix]))
                f_alls.add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                  line_color='#dddddd', line_alpha=0.5,
                                  upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                  lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))

            modelband = ColumnDataSource(data=dict(base=self.rvs['t']['time'],
                                                lower=self.rvs_to_plot['t']['gp_pred'] - self.rvs_to_plot['t']['gp_sd'],
                                                upper=self.rvs_to_plot['t']['gp_pred'] + self.rvs_to_plot['t']['gp_sd']))
            f_alls[n].add_layout(Band(source=modelband,base='base',lower='lower',upper='upper',
                                      fill_alpha=0.4, line_width=0.0, fill_color=self.pal[3]))
            f_alls[n].line(self.rvs_to_plot['t'], self.rvs_to_plot['gp_pred'],
                           line_alpha=0.6, line_width=1.0, color=self.pal[3], legend="RV model")


            residband = ColumnDataSource(data=dict(base=self.rvs['t']['time'],
                                                lower= self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["+2sig"],upper=self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["-2sig"]))
            f_alls[n].add_layout(Band(source=modelband,base='base',lower='lower',upper='upper',
                                      fill_alpha=0.4, line_width=0.0, fill_color=self.pal[3]))

        for n,pl in enumerate(list(self.planets.keys())+list(self.rvplanets.keys())):
            if hasattr(self,'trace'):
                vars=['t0_'+pl,'per_'+pl,'logprob_marg_'+pl,'K_'+pl]
                if pl in self.trios:
                    vars+=['t0_3_'+pl] 
                elif pl in self.duos:
                    vars+=['t0_2_'+pl] 
                    
                ext=az.extract(self.trace.posterior,var_names=vars)
                t0=np.nanmedian(ext['t0_'+pl].values)
                if pl in self.multis or pl in self.rvplanets:
                    per=[np.nanmedian(ext['per_'+pl])]
                    alphas=[1.0]
                else:
                    alphas=np.clip(2*np.exp(np.nanmedian(ext['logprob_marg_'+pl],axis=0)),0.25,1.0)
                    if pl in self.duos:
                        t0=np.nanmedian(ext['t0_2_'+pl])
                        per=np.nanmedian(ext['per_'+pl],axis=0)
                    elif pl in self.monos:
                        per=np.nanmedian(ext['per_'+pl],axis=0)
                        #alphas=[alphas]
            elif hasattr(self,'init_soln'):
                t0=self.init_soln['t0_'+pl]
                if pl in self.multis or pl in self.rvplanets:
                    per=[self.init_soln['per_'+pl]]
                    alphas=[1.0]
                else:
                    per=self.init_soln['per_'+pl]
                    alphas=np.clip(2*np.exp(self.init_soln['logprob_marg_'+pl]),0.25,1.0)
                if pl in self.duos:
                    t0_2=self.init_soln['t0_2_'+pl]
            else:
                raise ValueError()
            if self.n_margs[pl]>1:
                for i in range(self.n_margs[pl]):
                    self.rvs_to_plot['x'][pl][i]['phase']=((self.rvs['time']-t0)/per[i])%1
                    self.rvs_to_plot['t'][pl][i]['phase']=((self.rvs_to_plot['t']['time']-t0)/per[i])%1
            else:
                self.rvs_to_plot['x'][pl]['phase']=((self.rvs['time']-t0)/per[0])%1
                self.rvs_to_plot['t'][pl]['phase']=((self.rvs_to_plot['t']['time']-t0)/per[0])%1

            if len(all_pls_in_rvs)>1:
                other_plsx=np.sum([self.rvs_to_plot['x'][opl]['marg']['med'] for opl in all_pls_in_rvs if opl!=pl],axis=0)
                other_plst=np.sum([self.rvs_to_plot['t'][opl]['marg']['med'] for opl in all_pls_in_rvs if opl!=pl],axis=0)
            else:
                other_plsx=np.zeros(len(self.rvs['time']))
                other_plst=np.zeros(len(self.rvs_to_plot['t']['time']))

            if pl==(self.duos+self.monos)[nmargtoplot]:
                for n,alias in enumerate(ibest):
                    if hasattr(self,'trace'):
                        K=np.clip(np.nanmedian(ext['K_'+pl][:,alias]),averr,100000)
                    else:
                        K=np.clip(self.init_soln['K_'+pl][alias],averr,100000)

                    if interactive:
                        sdbuffer=3
                        errors = ColumnDataSource(data=dict(base=self.rvs_to_plot['x'][pl][alias]['phase'],
                                                lower=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] - self.rvs['rv_err'],
                                                upper=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] + self.rvs['rv_err']))
                        f_phase[pl][n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        f_phase[pl][n].circle(self.rvs_to_plot['x'][pl][i]['phase'],
                                              self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'],
                                          color='C0', alpha=0.6, size=4)


                        f_alls.circle(self.rvs_to_plot['t']['time'],
                                    self.rvs_to_plot['t']['trend']['med']+self.rvs_to_plot['t'][pl][alias]['med'],
                                    c='C'+str(9-int(n)),label=str(np.round(p,1)),
                                    alpha=alphas[alias])


                        if '-2sig' in self.rvs_to_plot['t'][pl][alias]:
                            trband = ColumnDataSource(data=dict(
                                base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
           lower=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['-2sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
           upper=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['+2sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0))
                                                               ))
                            f_phase[pl][n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                   level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[2+n]))
                            trband = ColumnDataSource(data=dict(
                                base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
           lower=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
           upper=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0))
                                                               ))
                            f_phase[pl][n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                                      level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[2+n]))
                        f_phase[pl][n].line(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
                 np.hstack((0,self.rvs_to_plot['t'][pl][alias]['med'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                                        color=self.pal[2+n])
                        f_phase[pl][n].y_range=Range1d(-1.75*K,1.75*K)

                        if n<nbest-1:
                            f_phase[pl][n].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                            f_phase[pl][n].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                            f_phase[pl][n].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

                    else:
                        f_alls.plot(self.rvs_to_plot['t']['time'],
                                    self.rvs_to_plot['t']['trend']['med']+self.rvs_to_plot['t'][pl][alias]['med'],':',
                                    c='C'+str(ncol-1-int(n)),label=pl+'_'+str(np.round(per[alias],1)),linewidth=3.0,
                                    alpha=alphas[alias])

                        f_phase[pl][n].errorbar(self.rvs_to_plot['x'][pl][alias]['phase'],
                                                self.rvs['rv']-other_plsx - self.rvs_to_plot['x']['trend+offset']['med'],
                                            yerr=self.rvs['rv_err'], fmt='.',c='C1',
                                            alpha=0.75, markersize=8,ecolor='#bbbbbb', rasterized=raster)
                        if '+2sig' in self.rvs_to_plot['t'][pl][alias]:
                            f_phase[pl][n].fill_between(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
                  np.hstack((0,self.rvs_to_plot['t'][pl][alias]['-2sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                  np.hstack((0,self.rvs_to_plot['t'][pl][alias]['+2sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                                   alpha=0.25, color='C'+str(ncol-1-int(n)), rasterized=raster)
                            f_phase[pl][n].fill_between(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
                  np.hstack((0,self.rvs_to_plot['t'][pl][alias]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                  np.hstack((0,self.rvs_to_plot['t'][pl][alias]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                                   alpha=0.25, color='C'+str(ncol-1-int(n)), rasterized=raster)
                        f_phase[pl][n].plot(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
                  np.hstack((0,self.rvs_to_plot['t'][pl][alias]['med'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                                            ':', color='C'+str(ncol-1-int(n)), linewidth=3.0, rasterized=raster)
                        #print(K)
                        #print(np.max(self.rvs_to_plot['t'][pl][alias]['med']),
                        #      np.min(self.rvs_to_plot['t'][pl][alias]['med']))
                        f_phase[pl][n].set_ylim(-1.75*K,1.75*K)
                        f_phase[pl][n].yaxis.tick_right()
                        f_phase[pl][n].set_ylabel("RV [m/s]")
                        f_phase[pl][n].yaxis.set_label_position("right")
                        f_phase[pl][n].set_xlim(0.0,1)
                        #f_phase[n].text(0.0,0.0+resid_sigma*1.9,pl,horizontalalignment='center',verticalalignment='top',fontsize=9)
                        #plt.setp(f_phase[n].get_xticklabels(), visible=False)

                    if n==len(all_pls_in_rvs)-1:
                        if interactive:
                            #extra = '[ppt]' if self.lc.flux_unit==0.001 else ''
                            #f_all_resids[n] = 'flux '+extra#<- y-axis label
                            #f_all[n].yaxis.axis_label = 'residuals '+extra#<- y-axis label
                            f_trans[key].xaxis.axis_label = 'Phase' #<- x-axis label
                        else:
                            f_phase[pl][n].set_xlabel("Phase")
                    else:
                        if not interactive:
                            plt.setp(f_phase[pl][n].get_xticklabels(), visible=False)

            elif pl in other_pls:
                if hasattr(self,'trace'):
                    K=np.clip(np.nanmedian(ext['K_'+pl]),averr,100000)
                else:
                    K=np.clip(self.init_soln['K_'+pl],averr,100000)
                    if interactive:
                        sdbuffer=3
                if interactive:
                    errors = ColumnDataSource(data=dict(base=self.rvs_to_plot['x'][pl]['phase'],
                                            lower=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] - self.rvs['rv_err'],
                                            upper=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] + self.rvs['rv_err']))
                    f_phase[pl].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                 line_color='#dddddd', line_alpha=0.5,
                                                 upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                 lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    f_phase[pl].circle(self.rvs_to_plot['x'][pl]['phase'],
                                          self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'],
                                      color='C0', alpha=0.6, size=4)

                    f_alls.circle(self.rvs_to_plot['t']['time'],
                                self.rvs_to_plot['t']['trend']['med']+self.rvs_to_plot['t'][pl]['med'],
                                c='C'+str(9-int(n)),label=str(np.round(p,1)),
                                alpha=1.0)


                    if '-2sig' in self.rvs_to_plot['t'][pl]:
                        trband = ColumnDataSource(data=dict(
                            base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
       lower=np.hstack((0,self.rvs_to_plot['t'][pl]['-2sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
       upper=np.hstack((0,self.rvs_to_plot['t'][pl]['+2sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0))
                                                           ))
                        f_phase[pl].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                               level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[1]))
                        trband = ColumnDataSource(data=dict(
                            base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
       lower=np.hstack((0,self.rvs_to_plot['t'][pl]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
       upper=np.hstack((0,self.rvs_to_plot['t'][pl]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0))
                                                           ))
                        f_phase[pl].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                                  level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[2+n]))
                        f_phase[pl].line(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
             np.hstack((0,self.rvs_to_plot['t'][pl]['marg']['med'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
                                    color=self.pal[1])
                        f_phase[pl].y_range=Range1d(-1.75*K,1.75*K)
                else:
                    f_alls.plot(self.rvs_to_plot['t']['time'],
                                self.rvs_to_plot['t']['trend']['med']+self.rvs_to_plot['t'][pl]['marg']['med'],'-',
                                c='C1',label=pl+'_'+str(np.round(per,1)),linewidth=3.0,
                                alpha=0.4)

                    f_phase[pl].errorbar(self.rvs_to_plot['x'][pl]['phase'],
                                            self.rvs['rv']-other_plsx - self.rvs_to_plot['x']['trend+offset']['med'],
                                        yerr=self.rvs['rv_err'], fmt='.',c='C1',
                                        alpha=0.75, markersize=8,ecolor='#bbbbbb', rasterized=raster)
                    if '+2sig' in self.rvs_to_plot['t'][pl]:
                        f_phase[pl].fill_between(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
              np.hstack((0,self.rvs_to_plot['t'][pl]['-2sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
              np.hstack((0,self.rvs_to_plot['t'][pl]['+2sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
                               alpha=0.25, color='C1', rasterized=raster)
                        f_phase[pl].fill_between(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
              np.hstack((0,self.rvs_to_plot['t'][pl]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
              np.hstack((0,self.rvs_to_plot['t'][pl]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
                               alpha=0.25, color='C1', rasterized=raster)
                    f_phase[pl].plot(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
              np.hstack((0,self.rvs_to_plot['t'][pl]['marg']['med'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
                                        '-', color='C1', linewidth=3.0, rasterized=raster,alpha=0.4)

                    f_phase[pl].set_ylim(-2.5*K,2.5*K)
                    f_phase[pl].yaxis.tick_right()
                    f_phase[pl].set_ylabel("RV [m/s]")
                    f_phase[pl].yaxis.set_label_position("right")
                    f_phase[pl].set_xlim(0.0,1)

            f_alls.set_ylabel("RVs [m/s]")
        f_alls.legend()

        if interactive:
            #Saving
            cols=[]
            for r in range(len(f_alls)):
                cols+=[column(f_alls[r])]
            lastcol=[]
            for r in range(len(f_trans)):
                lastcol+=[f_trans[r]]
            p = gridplot([cols+[column(lastcol)]])
            save(p)
            print("interactive table at:",savename)

            if return_fig:
                return p

        else:
            if save:
                if plot_loc is None and plottype=='png':
                    plt.savefig(self.savenames[0]+'_rv_plot.png',dpi=350,transparent=True)
                    #plt.savefig(self.savenames[0]+'_model_plot.pdf')
                elif plot_loc is None and plottype=='pdf':
                    plt.savefig(self.savenames[0]+'_rv_plot.pdf')
                else:
                    plt.savefig(plot_loc)
            if return_fig:
                return fig

    '''def PlotRVs(nbest=4):

        fig  = plt.figure(figsize=(12,8))

        import seaborn as sns
        sns.set_palette('viridis',10)

        allrvs = fig.add_subplot(gs[:20,:2*(3+len(self.planets))])
        allrvs.plot(mod.rvs['time'],mod.init_soln['rv_trend'],c='C6')
        allrvs.errorbar(mod.rvs['time'],mod.rvs['rv'],yerr=mod.rvs['rv_err'],fmt='.',c='C1')

        rvresids = fig.add_subplot(gs[20:,:8])
        rvresids.errorbar(mod.rvs['time'],mod.rvs['rv']-mod.init_soln['rv_trend']-mod.init_soln['marg_rv_model_00'],
                          yerr=mod.rvs['rv_err'],fmt='.',c='C1')
        for n,i in enumerate(ibest):
            K=mod.init_soln['K_00'][i]
            p=mod.init_soln['per_00'][i]


            allrvs.plot(mod.rvs['time'],mod.init_soln['model_rv_00'][:,i]+mod.init_soln['rv_trend'],'.-',
                        c='C'+str(9-int(n)),alpha=np.clip(np.exp(mod.init_soln['logprob_marg_00'][i]),0.15,1.0),label=str(np.round(p,1)))

            print(i,K,p)
            phase=(mod.rvs['time']-mod.init_soln['t0_2_00']-0.5*p)%p-0.5*p
            rvplot = fig.add_subplot(gs[heights_sort[n]:heights_sort[n+1],8:])
            rvplot.errorbar(phase/p,mod.rvs['rv']-mod.init_soln['rv_trend'],yerr=mod.rvs['rv_err'],fmt='.',c='C1')
            rvplot.plot(np.sort(phase)/p,(mod.init_soln['model_rv_00'][:,i])[np.argsort(phase)],'.-',c='C'+str(9-int(n)))
            rvplot.set_ylim(-2.5*np.clip(K,0.5*averr,10000),2.5*np.clip(K,0.5*averr,10000))
            rvplot.set_xlim(-0.5,0.5)
            rvplot.yaxis.tick_right()

            if n==nbest-1:
                rvplot.set_xlabel("phase")
            else:
                rvplot.set_xticklabels([])
            rvplot.patch.set_linewidth(2)
        allrvs.legend(loc=4)
        '''


    def plot(self, interactive=False, n_samp=None, overwrite=False, interp=True, newgp=False, return_fig=False, max_gp_len=20000, n_intrans_bins=15,
             save=True, plot_loc=None, palette=None, plot_flat=False, pointcol="k", plottype='png',plot_rows=None, ylim=None, xlim=None, **kwargs):
        """Varied photometric plotting function for MonoTransit model

        Args:
            interactive (bool, optional): Plot interactive bokeh image? Defaults to False i.e. matplotlib plot
            n_samp (int, optional): Number of samples to use to initialise the RV plotting. Defaults to 300.
            overwrite (bool, optional). Defaults to False.
            interp (bool, optional). Whether to interpolate the GP fits to create GP timeseries for all t (bin_oot must be True)
            newgp (bool, optional). Whether to use the sampled hyperparametered to inialise a new GP to create GP timeseries for all t
            return_fig (bool, optional). Return figure variable. Defaults to False.
            max_gp_len (int, optional): Maximum length of photometric data to . Defaults to 20000.
            save (bool, optional). Defaults to True.
            bin_gp (bool, optional): Bin GP or use raw x points? Defaults to True.
            plot_loc (str, optional): File location to plot. Defaults to None, which uses `mod.savenames`
            palette ([str, optional): Colour palette to plot with. Defaults to None.
            plot_flat (bool, optional): Plot flatted lightcurve instead of "raw". Defaults to True.
            pointcol (str, optional): Point colour of smallest points. Defaults to "k".
            plottype (str, optional): Type of image to save. Defaults to 'png'.

        Returns:
            [type]: [description]
        """
        ################################################################
        #       Varied plotting function for MonoTransit model
        ################################################################
        self.init_plot(plottype='lc',pointcol=pointcol,plot_flat=plot_flat,**kwargs)
        #Rasterizing matplotlib files if we have a lot of datapoints:
        raster=True if len(self.lc.time>8000) else False

        if not hasattr(self,'trace'):
            n_samp=1
        elif n_samp is None:
            n_samp=99
        
        #Automatically setting the number of rows for the data:
        plot_rows = int(np.clip(np.round(np.sum([self.lc_regions[nj]['n_ideal_split'] for nj in self.lc_regions])),np.clip(len(self.lc_regions),1,2),4)) if plot_rows is None else plot_rows
        subplots_ix = tools.partition_list(np.array([self.lc_regions[j]['total_dur'] for j in self.lc_regions]), plot_rows)

        if interactive:
            from bokeh.plotting import figure, output_file, save, curdoc, show
            from bokeh.models import Band, Whisker, ColumnDataSource, Range1d, arrow_heads
            from bokeh.layouts import gridplot, row, column, layout

            if plot_loc is None:
                savename=self.savenames[0]+'_model_plot.html'
            else:
                savename=plot_loc
            output_file(savename)

            #Initialising figure:
            p = figure(plot_width=1000, plot_height=600,title=str(self.ID)+" Transit Fit")
        else:
            #A4 page: 8.27 x 11.69
            fig=plt.figure(figsize=(11.69,8.27))
            gs = fig.add_gridspec(len(self.planets)*plot_rows*4,32,wspace=0.3,hspace=0.3)

        #####################################
        #       Initialising figures
        #####################################

        f_alls={};f_trans={};f_trans_resids={}
        for irow in range(plot_rows):
            assert np.sum(subplots_ix==irow)>0
            plots_in_this_row = np.array(list(self.lc_regions.keys()))[subplots_ix==irow]
            durs = [np.clip(self.lc_regions[cad2]['total_dur'],4,1000) for cad2 in plots_in_this_row]
            plot_cols = np.hstack((0,np.cumsum(saferound(24*np.array(durs)/np.sum(durs), places=0))))
            for icol,key in enumerate(plots_in_this_row):
                self.lc_regions[key]['n_plot_row']=irow
                self.lc_regions[key]['n_plot_col']=(int(plot_cols[icol]),int(plot_cols[icol+1]))
                #print("row",self.lc_regions[key]['n_plot_row'],"col",self.lc_regions[key]['n_plot_col'])
            #print(plot_rows,irow,plots_in_this_row)

        if not interactive:
            #Creating cumulative list of integers which add up to 24 but round to nearest length ratio:
            #(gs[0, :]) - all top
            #(gs[:, 0]) - all left
            for key in self.lc_regions:      
                f_alls[key]=fig.add_subplot(gs[4*len(self.planets)*self.lc_regions[key]['n_plot_row']:(4*len(self.planets)*(self.lc_regions[key]['n_plot_row']+1)),
                                               self.lc_regions[key]['n_plot_col'][0]:self.lc_regions[key]['n_plot_col'][1]])          
            
            for npl in np.arange(len(self.planets))[::-1]:
                pl=list(self.planets.keys())[npl]
                #print(key,"resids",(npl*4+3)*plot_rows,(npl*4+4)*plot_rows)
                #print(key,"norm",plot_rows*(npl*4),plot_rows*(npl*4+3))
                if npl==len(self.planets)-1:
                    xaxiskey=pl
                    f_trans_resids[pl]=fig.add_subplot(gs[(npl*4+3)*plot_rows:(npl*4+4)*plot_rows,24:])
                    f_trans[pl]=fig.add_subplot(gs[plot_rows*(npl*4):plot_rows*(npl*4+3),24:],sharex=f_trans_resids[xaxiskey])
                else:
                    f_trans[pl]=fig.add_subplot(gs[plot_rows*(npl*4):plot_rows*(npl*4+3),24:],sharex=f_trans_resids[xaxiskey])
                    f_trans_resids[pl]=fig.add_subplot(gs[(npl*4+3)*plot_rows:(npl*4+4)*plot_rows,24:],sharex=f_trans_resids[xaxiskey])

        else:
            #For Bokeh plots, we can just use the size in pixels
            for ng,gaplen in enumerate(self.lc['gap_lens']):
                fwidth=int(np.round(750*gaplen/np.sum(self.lc['gap_lens']))-10)
                if ng==0:
                    f_alls+=[figure(width=fwidth, plot_height=400, title=None)]
                else:
                    f_alls+=[figure(width=fwidth, plot_height=400, title=None,
                                    y_range=f_alls[0].y_range)]
            for npl in np.arange(len(self.planets))[::-1]:
                fheight=int(np.round(0.84*650/len(self.planets)))-3
                if len(f_trans)==0:
                    #Including 30px as space for label in lowermost plot:
                    f_trans=[figure(width=240, plot_height=fheight+30, title=None, y_axis_location="right")]
                else:
                    f_trans=[figure(width=240, plot_height=fheight, title=None,
                                     y_axis_location="right",x_range=f_trans[-1].x_range)]+f_trans

        
        #####################################
        #    Initialising Transit model
        #####################################
        if not hasattr(self, 'trans_to_plot') or 'all' not in self.trans_to_plot or overwrite:
            print("initialising transit")
            self.init_trans_to_plot(n_samp*10)
        #####################################
        #       Initialising GP model
        #####################################
        if self.use_GP and (not hasattr(self, 'gp_to_plot') or 'gp_pred' not in self.gp_to_plot or overwrite):
            self.init_gp_to_plot(n_samp, max_gp_len, interp=interp, newgp=newgp)
        if self.local_spline:
            self.init_spline_to_plot(n_samp)
        '''
        assert hasattr(self,'trace')
        i_gp_pred=[]
        i_gp_var=[]
        print(limits,gap_lens,range(len(gap_lens)),np.arange(len(gap_lens)))
        for n in np.arange(len(gap_lens)):
            for i, sample in enumerate(xo.get_samples_from_trace(self.trace, size=n_samp)):
                with self.model:
                    ii_gp_pred, ii_gp_var = xo.eval_in_model(self.gp['use'].predict(self.lc.time[limits[n][0]:limits[n][1]],
                                                                                return_var=True,return_cov=False),sample)
                i_gp_pred+=[ii_gp_pred]
                i_gp_var+=[ii_gp_var]
            av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
            gp_pred+=[av]
            gp_sd+=[std]

        self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
        self.gp_to_plot['gp_sd']=np.hstack(gp_sd)
        '''
        '''
        with self.model:
            for i, sample in enumerate(xo.get_samples_from_trace(self.trace, size=n_samp)):
                ii_gp_pred, ii_gp_var = xo.eval_in_model(self.gp['use'].predict(self.lc.time.astype(floattype),
                                                                                return_var=True,return_cov=False),sample)
                i_gp_pred+=[ii_gp_pred]
                i_gp_var+=[ii_gp_var]
            av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
        self.gp_to_plot['gp_pred'] = av
        self.gp_to_plot['gp_sd'] = std
        '''

        self.min_trans=abs(np.nanmin(self.trans_to_plot['model']['allpl']['med']))

        #####################################
        #  Plotting full lightcurve regions
        #####################################
        if self.use_GP:
            resid_sigma=np.nanstd(self.lc.flux[self.lc.mask] - self.gp_to_plot['gp_pred'][self.lc.mask] - self.trans_to_plot['all']['allpl']['med'][self.lc.mask])
        elif self.fit_no_flatten:
            resid_sigma=np.nanstd(self.lc.flux[self.lc.mask] - self.trans_to_plot['all']['allpl']['med'][self.lc.mask])
        else:
            resid_sigma=np.nanstd(self.lc.flux_flat[self.lc.mask] - self.trans_to_plot['all']['allpl']['med'][self.lc.mask])

        if not hasattr(self.lc,'bin_time'):
            self.lc.bin()

        for nkey,key in enumerate(self.lc_regions): 
            if self.use_GP:
                if self.lc_regions[key]['cadence']<1/72:
                    bin_detrend=tools.bin_lc_segment(np.column_stack((self.lc.time[self.lc_regions[key]['ix']],
                                   self.lc.flux[self.lc_regions[key]['ix']] - \
                                   self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']],
                                   self.lc.flux_err[self.lc_regions[key]['ix']])),
                                   binsize=29/1440)
            else:
                phot_mean=np.nanmedian(az.extract(self.trace.posterior,var_names=['phot_mean']).values) if hasattr(self,'trace') else self.init_soln['phot_mean']


                #Plotting each part of the lightcurve:
            if interactive:
                if self.use_GP:
                    #Plotting GP region and subtracted flux
                    if np.nanmedian(np.diff(self.lc.time))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        if plot_flat:
                            flux=(self.lc.flux_flat-self.gp_to_plot['gp_pred'])[self.lc_regions[key]['ix']],
                            bin_flux=tools.bin_lc_given_new_x(np.column_stack((self.lc.time[self.lc_regions[key]['ix']],
                                                          (self.lc.flux_flat-self.gp_to_plot['gp_pred'])[self.lc_regions[key]['ix']],
                                                          self.lc.flux_err[self.lc_regions[key]['ix']])),self.lc.bin_time[self.lc_regions[key]['bin_ix']])[:,1]
                        else:
                            flux=self.lc.flux[self.lc_regions[key]['ix']],
                            bin_flux=self.lc.bin_flux[self.lc_regions[key]['bin_ix']],
                        f_alls[key].circle(self.lc.time[self.lc_regions[key]['ix']],
                                           flux, alpha=0.25,size=0.75,color='black')
                        f_alls[key].circle(self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                           bin_flux, alpha=0.65,size=3.5,legend="raw")
                        errors = ColumnDataSource(data=
                                dict(base=self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                    lower=bin_flux - self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']],
                                    upper=bin_flux + self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']]))
                        f_alls[key].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        if plot_flat:
                            flux=(self.lc.flux_flat-self.gp_to_plot['gp_pred'])[self.lc_regions[key]['ix']],
                        else:
                            flux=self.lc.flux[self.lc_regions[key]['ix']],
                        f_alls[key].circle(self.lc.time[self.lc_regions[key]['ix']],
                                           flux, alpha=0.25,size=0.75,color='black')
                    gpband = ColumnDataSource(data=dict(base=self.lc.time[self.lc_regions[key]['ix']],
                              lower=self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']]-self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']],
                              upper=self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']]+self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']]))
                    f_alls[key].add_layout(Band(source=gpband,base='base',lower='lower',upper='upper',
                                              fill_alpha=0.4, line_width=0.0, fill_color=self.pal[3]))
                    f_alls[key].line(self.lc.time[self.lc_regions[key]['ix']], self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']],
                                   line_alpha=0.6, line_width=1.0, color=self.pal[3], legend="GP fit")
                    '''if plot_flat:
                        if np.nanmedian(np.diff(self.lc.time[self.lc_regions[key]['ix']] ))<1/72:
                            #Here we plot the detrended flux:
                            f_alls[n].circle(self.lc.time[self.lc_regions[key]['ix']],
                                             self.lc.flux[self.lc_regions[key]['ix']]-self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']], color='black',
                                             alpha=0.5,size=0.75)
                            f_alls[n].circle(bin_detrend[:,0],bin_detrend[:,1],alpha=0.65,size=3.5,legend='detrended')

                            errors = ColumnDataSource(data=dict(base=bin_detrend[:,0],
                                                         lower=bin_detrend[:,1]+bin_detrend[:,2],
                                                         upper=bin_detrend[:,1]-bin_detrend[:,2]))
                            f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        else:
                            f_alls[n].circle(self.lc.time[self.lc_regions[key]['ix']],
                                             self.lc.flux[self.lc_regions[key]['ix']]-self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']],
                                             legend="detrended",alpha=0.65,
                                             size=3.5)
                    '''

                else:
                    if (self.lc_regions[key]['cadence']*1440)>20 and self.lc_regions[key]['total_time']<500:
                        #Plotting only real points as "binned points" style:
                        f_alls[key].plot(self.lc.time[self.lc_regions[key]['ix']],self.lc.flux[self.lc_regions[key]['ix']],'.',alpha=0.8,markersize=3.0)
                    elif (self.lc_regions[key]['cadence']*1440)>20 and self.lc_regions[key]['total_time']>500:
                        #So much data that we should bin it back down (to 2-hour bins)
                        self.lc_regions[key]['binned']=tools.bin_lc_segment(np.column_stack((self.lc.time[self.lc_regions[key]['ix']],
                                                                                    self.lc.flux[self.lc_regions[key]['ix']],
                                                                                    self.lc.flux_err[self.lc_regions[key]['ix']])),binsize=1/12)
                        f_alls[key].plot(self.lc.time[self.lc_regions[key]['ix']],self.lc.flux[self.lc_regions[key]['ix']],'.k',markersize=0.75,alpha=0.25)
                        f_alls[key].plot(self.lc_regions[key]['binned'][:,0],self.lc_regions[key]['binned'][:,1],'.',alpha=0.8,markersize=3.0)
                    else:
                        #Plotting real points as fine scatters and binned points above:
                        f_alls[key].plot(self.lc.time[self.lc_regions[key]['ix']],self.lc.flux[self.lc_regions[key]['ix']],'.k',markersize=0.75,alpha=0.25)
                        f_alls[key].plot(self.lc.bin_time[self.lc_regions[key]['bin_ix']],self.lc.bin_flux[self.lc_regions[key]['bin_ix']],'.',alpha=0.8,markersize=3.0,color='C'+str(nkey))
                    if self.lc_regions[key]['n_plot_col'][0]!=0.0:
                        f_alls[key].set_yticklabels([])
                    else:
                        f_alls[key].set_ylabel("Relative Flux ["+self.flx_system+"]")
                    if self.lc_regions[key]['n_plot_row']==plot_rows-1:
                        f_alls[key].set_xlabel("Time [BJD-"+str(int(self.jd_base))+"]")
                    f_alls[key].set_xlim(self.lc_regions[key]['start']-self.lc_regions[key]['total_dur']*0.02-0.04,
                                         self.lc_regions[key]['end']+self.lc_regions[key]['total_dur']*0.02+0.04)

                    if np.nanmedian(np.diff(self.lc.time))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        f_alls[key].circle(self.lc.time[self.lc_regions[key]['ix']],
                                         self.lc.flux_flat[self.lc_regions[key]['ix']],#+raw_plot_offset,
                                         color='black',alpha=0.5,size=0.75)
                        if plot_flat:
                            f_alls[key].circle(self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                             self.lc.bin_flux[self.lc_regions[key]['bin_ix']],
                                             legend="detrended",alpha=0.65,size=3.5)

                            errors = ColumnDataSource(data=dict(base=self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                         lower=self.lc.bin_flux[self.lc_regions[key]['bin_ix']] - \
                                         self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']],
                                         upper=self.lc.bin_flux[self.lc_regions[key]['bin_ix']] + \
                                         self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']]))
                            f_alls[key].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))

                        #Here we plot the detrended flux:
                        f_alls[key].circle(self.lc.time[self.lc_regions[key]['ix']],
                                         self.lc.flux_flat[self.lc_regions[key]['ix']],#+raw_plot_offset,
                                         alpha=0.5,size=0.75)
                        if plot_flat:
                            f_alls[nkey].circle(self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                             self.lc.bin_flux_flat[self.lc_regions[key]['bin_ix']],
                                             legend="detrended",alpha=0.65,size=3.5)
                            errors = ColumnDataSource(
                                      data=dict(base=self.lc.bin_time[self.lc_regions[key]['bin_ix']],
                                         lower=self.lc.bin_flux_flat[self.lc_regions[key]['bin_ix']] - \
                                         self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']],
                                         upper=self.lc.bin_flux_flat[self.lc_regions[key]['bin_ix']] + \
                                         self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']]))
                            f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        f_alls[n].circle(self.lc.time[self.lc_regions[key]['ix']],
                                         self.lc.flux[self.lc_regions[key]['ix']]+raw_plot_offset,
                                         legend="raw",alpha=0.65,size=3.5)
                        if plot_flat:
                            f_alls[n].circle(self.lc.time[self.lc_regions[key]['ix']],
                                             self.lc.flux_flat[self.lc_regions[key]['ix']],
                                             legend="detrended",alpha=0.65,size=3.5)
                #Plotting transit
                if len(self.trans_to_plot['all'])>1:
                    trband = ColumnDataSource(data=dict(base=self.lc.time[self.lc_regions[key]['model_ix']],
                                    lower=self.trans_to_plot['all']['-2sig'][self.lc_regions[key]['model_ix']],
                                    upper=self.trans_to_plot['all']['+2sig'][self.lc_regions[key]['model_ix']]))
                    f_alls[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[1]))
                    trband = ColumnDataSource(data=dict(base=self.lc.time[self.lc_regions[key]['model_ix']],
                                    lower=self.trans_to_plot['all']['-1sig'][self.lc_regions[key]['model_ix']],
                                    upper=self.trans_to_plot['all']['+1sig'][self.lc_regions[key]['model_ix']]))
                    f_alls[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[1]))
                f_alls[n].line(self.lc.time[self.lc_regions[key]['model_ix']],
                               self.trans_to_plot["all"]["med"][self.lc_regions[key]['model_ix']],
                               color=self.pal[1], legend="transit fit")

                if n>0:
                    f_alls[key].yaxis.major_tick_line_color = None  # turn off x-axis major ticks
                    f_alls[key].yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                    f_alls[key].yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

                if self.use_GP:
                    #Plotting residuals:
                    if self.lc_regions['cadence']<1/72:
                        #HERE WE BIN
                        errors = ColumnDataSource(data=dict(base=bin_resids[:,0],
                                                     lower=bin_resids[:,1] - bin_resids[:,2],
                                                     upper=bin_resids[:,1] + bin_resids[:,2]))
                        f_alls[key].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        errors = ColumnDataSource(data=dict(base=self.lc.time[self.lc_regions[key]['ix']],
                                          lower=self.lc.flux[self.lc_regions[key]['ix']] - self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] - \
                                           self.trans_to_plot['all']['med'][self.lc_regions[key]['ix']] - self.lc.flux_err[self.lc_regions[key]['ix']],
                                          upper=self.lc.flux[self.lc_regions[key]['ix']] - self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] - \
                                           self.trans_to_plot['all']['med'][self.lc_regions[key]['ix']] + self.lc.flux_err[self.lc_regions[key]['ix']]))
                        f_alls[key].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                else:
                    #Plotting detrended:
                    phot_mean=np.nanmedian(az.extract(self.trace.posterior,var_names=['phot_mean']).values) if hasattr(self,'trace') else self.init_soln['phot_mean']
                    if np.nanmedian(np.diff(self.lc.time[self.lc_regions[key]['ix']]))<1/72:
                        errors = ColumnDataSource(data=dict(base=bin_resids[:,0],
                                                lower=bin_resids[:,1] - bin_resids[:,2],
                                                upper=bin_resids[:,1] + bin_resids[:,2]))
                        f_alls[key].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        f_alls[key].circle(self.lc.time[self.lc_regions[key]['ix']],
                                         self.lc.flux_flat[self.lc_regions[key]['ix']] - phot_mean - \
                                         self.trans_to_plot["all"]["med"][self.lc_regions[key]['ix']],
                                         legend="raw data",alpha=0.65,size=3.5)

                f_alls[key].legend.location = 'bottom_right'
                f_alls[key].legend.background_fill_alpha = 0.1
                f_alls[key].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                f_alls[key].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                f_alls[key].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

            else:
                #Matplotlib plot:
                if plot_flat and not self.use_GP:
                    self.lc.bin(timeseries=np.unique(['flux','flux_flat']))
                    flux=self.lc.flux_flat[self.lc_regions[key]['ix']]
                    bin_flux=self.lc.bin_flux_flat[self.lc_regions[key]['bin_ix']]
                elif plot_flat and self.use_GP:
                    flux=self.lc.flux[self.lc_regions[key]['ix']]-self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']]
                    bin_flux=tools.bin_lc_given_new_x(np.column_stack((self.lc.time[self.lc_regions[key]['ix']],
                                                      (self.lc.flux-self.gp_to_plot['gp_pred'])[self.lc_regions[key]['ix']],
                                                      self.lc.flux_err[self.lc_regions[key]['ix']])),
                                                     self.lc.bin_time[self.lc_regions[key]['bin_ix']])[:,1]
                else:
                    flux=self.lc.flux[self.lc_regions[key]['ix']]
                    bin_flux=self.lc.bin_flux[self.lc_regions[key]['bin_ix']]

                if self.lc_regions[key]['cadence']<1/72:
                    #Plotting flat flux only
                    f_alls[key].plot(self.lc.time[self.lc_regions[key]['ix']], flux,
                                      ".", color=pointcol,alpha=0.15,markersize=0.75, rasterized=raster)
                    f_alls[key].errorbar(self.lc.bin_time[self.lc_regions[key]['bin_ix']],bin_flux,
                                        yerr= self.lc.bin_flux_err[self.lc_regions[key]['bin_ix']], rasterized=raster,
                                        color='C2',fmt=".",label="binned flux", ecolor='#dddddd', alpha=0.5,markersize=3.5)
                    #Plotting residuals:
                else:
                    f_alls[key].errorbar(self.lc.time[self.lc_regions[key]['ix']], flux,
                                    yerr=self.lc.flux_err[self.lc_regions[key]['ix']],color='C2',fmt=".", label="flux",
                                    ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)

                if self.use_GP:
                    if np.nanmedian(np.diff(self.lc.time[self.lc_regions[key]['ix']]))<1/72:
                        if plot_flat:
                            f_alls[key].plot(self.lc.time[self.lc_regions[key]['ix']], self.lc.flux[self.lc_regions[key]['ix']] - \
                                           self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']],
                                           ".", color=pointcol, alpha=0.5,markersize=0.75, rasterized=raster)
                            f_alls[key].errorbar(bin_detrend[:,0], bin_detrend[:,1], yerr=bin_detrend[:,2],color='C2',fmt=".",
                                               label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)
                    else:
                        if plot_flat:
                            f_alls[key].errorbar(self.lc.time[self.lc_regions[key]['ix']],flux,
                                                 yerr=self.lc.flux_err[self.lc_regions[key]['ix']], color='C2', rasterized=raster,
                                                 fmt=".", label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5)

                    if not plot_flat and 'gp_sd' in self.gp_to_plot:
                        #Plotting GP region and subtracted flux
                        f_alls[key].fill_between(self.lc.time[self.lc_regions[key]['ix']],
                                self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] - \
                                2*self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']],
                                self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + \
                                2*self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']], rasterized=raster,
                                color="C3", alpha=0.2,zorder=10)
                        f_alls[key].fill_between(self.lc.time[self.lc_regions[key]['ix']],
                                self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] - \
                                self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']],
                                self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + \
                                self.gp_to_plot['gp_sd'][self.lc_regions[key]['ix']], rasterized=raster,
                                            color="C3", label="GP fit",alpha=0.3,zorder=11)

                #Plotting transit
                ix=self.lc_regions[key]['ix']*self.lc.mask
                if plot_flat:
                    if '-2sig' in self.trans_to_plot['all']['allpl']:
                        f_alls[key].fill_between(self.lc.time[ix],
                                            self.trans_to_plot['all']['allpl']['-2sig'][ix],
                                            self.trans_to_plot['all']['allpl']['+2sig'][ix],
                                            alpha=0.2, color="C0",zorder=10, rasterized=raster)
                        f_alls[key].fill_between(self.lc.time[ix],
                                            self.trans_to_plot['all']['allpl']['-1sig'][ix],
                                            self.trans_to_plot['all']['allpl']['+1sig'][ix],
                                            alpha=0.3, color="C0",zorder=11, rasterized=raster)
                    f_alls[key].plot(self.lc.time[ix],
                                self.trans_to_plot['all']['allpl']['med'][ix],
                                color="C0", label="transit fit", linewidth=2.5,alpha=0.5,zorder=12, rasterized=raster)

                elif not plot_flat and self.use_GP:
                    if '-2sig' in self.trans_to_plot['all']['allpl']:
                        f_alls[key].fill_between(self.lc.time[ix],
                                            self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + self.trans_to_plot['all']['allpl']['-2sig'][ix],
                                            self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + self.trans_to_plot['all']['allpl']['+2sig'][ix],
                                            alpha=0.2, color="C0",zorder=10, rasterized=raster)
                        f_alls[key].fill_between(self.lc.time[ix],
                                            self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + self.trans_to_plot['all']['allpl']['-1sig'][ix],
                                            self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']] + self.trans_to_plot['all']['allpl']['+1sig'][ix],
                                            alpha=0.3, color="C0",zorder=11, rasterized=raster)
                    f_alls[key].plot(self.lc.time[ix],
                                self.gp_to_plot['gp_pred'][self.lc_regions[key]['ix']]+self.trans_to_plot['all']['allpl']['med'][ix],
                                color="C0", label="GP + transit fit", linewidth=2.5,alpha=0.5,zorder=12, rasterized=raster)
                elif not plot_flat and not self.use_GP and self.local_spline:
                    spl=self.lc.flux_spline[ix] if not self.fit_no_flatten else np.zeros(len(self.lc.time[ix]))
                    if '-2sig' in self.trans_to_plot['all']['allpl']:
                        f_alls[key].fill_between(self.lc.time[ix],
                                            spl+self.spline_to_plot['all']['allpl']['-2sig'][ix]+self.trans_to_plot['all']['allpl']['-2sig'][ix],
                                            spl+self.spline_to_plot['all']['allpl']['+2sig'][ix]+self.trans_to_plot['all']['allpl']['+2sig'][ix],
                                            alpha=0.2, color="C0",zorder=10, rasterized=raster)
                        f_alls[key].fill_between(self.lc.time[ix],
                                            spl+self.spline_to_plot['all']['allpl']['-1sig'][ix]+self.trans_to_plot['all']['allpl']['-1sig'][ix],
                                            spl+self.spline_to_plot['all']['allpl']['-1sig'][ix]+self.trans_to_plot['all']['allpl']['+1sig'][ix],
                                            alpha=0.3, color="C0",zorder=11, rasterized=raster)

                    f_alls[key].plot(self.lc.time[ix],
                                spl+self.spline_to_plot['all']['allpl']['med'][ix]+self.trans_to_plot['all']['allpl']['med'][ix],
                                color="C0", label="spline + transit fit", linewidth=2.5,alpha=0.5,zorder=12, rasterized=raster)

                elif not plot_flat and not self.use_GP and not self.local_spline:
                    spl=self.lc.flux_spline[ix] if not self.fit_no_flatten else np.zeros(len(self.lc.time[ix]))
                    if '-2sig' in self.trans_to_plot['all']['allpl']:
                        f_alls[key].fill_between(self.lc.time[ix],
                                            spl+self.trans_to_plot['all']['allpl']['-2sig'][ix],
                                            spl+self.trans_to_plot['all']['allpl']['+2sig'][ix],
                                            alpha=0.2, color="C0",zorder=10, rasterized=raster)
                        f_alls[key].fill_between(self.lc.time[ix],
                                            spl+self.trans_to_plot['all']['allpl']['-1sig'][ix],
                                            spl+self.trans_to_plot['all']['allpl']['+1sig'][ix],
                                            alpha=0.3, color="C0",zorder=11, rasterized=raster)
                    if not self.fit_no_flatten and not self.fit_GP:
                        f_alls[key].plot(self.lc.time[ix],
                                    spl+self.trans_to_plot['all']['allpl']['med'][ix],
                                    color="C0", label="transit fit", linewidth=2.5,alpha=0.5,zorder=12, rasterized=raster)
                    else:
                        f_alls[key].plot(self.lc.time[ix],
                                    spl+self.trans_to_plot['all']['allpl']['med'][ix],
                                    color="C0", label="spline + transit fit", linewidth=2.5,alpha=0.5,zorder=12, rasterized=raster)

                #plt.setp(f_alls[key].get_xticklabels(), visible=False)
                if self.lc_regions[key]['n_plot_col'][0]>0:
                    plt.setp(f_alls[key].get_yticklabels(), visible=False)
                else:
                    extra = '[ppt]' if self.lc.flx_unit==0.001 else ''
                    f_alls[key].set_ylabel('flux '+extra)#<- y-axis label

                if nkey==len(self.lc_regions.keys())-1:
                    f_alls[key].legend()
                f_alls[key].set_xlim(self.lc_regions[key]['start']-self.lc_regions[key]['total_dur']*0.02-0.04,
                                     self.lc_regions[key]['end']+self.lc_regions[key]['total_dur']*0.02+0.04)

                if ylim is None:
                    f_alls[key].set_ylim(self.lc_regions[key]['minmax_global'][0],self.lc_regions[key]['minmax_global'][1])
                else:
                    f_alls[key].set_ylim(ylim)

        max_gp=np.percentile(self.gp_to_plot['gp_pred'],99.5) if self.use_GP else np.nanmax(self.lc.bin_flux)

        if interactive:

            f_alls[0].yaxis.axis_label = 'flux '+extra#<- y-axis label
            '''
            sdbuffer=3
            if self.use_GP:
                f_alls[0].y_range=Range1d(-1*min_trans - sdbuffer*resid_sigma,
                                          raw_plot_offset + np.max(self.gp_to_plot['gp_pred']) + sdbuffer*resid_sigma)
            else:
                f_alls[0].y_range=Range1d(-1*min_trans - sdbuffer*resid_sigma,
                                          raw_plot_offset + np.max(self.lc.bin_flux) + sdbuffer*resid_sigma)

            f_all_resids[0].y_range=Range1d(-1*sdbuffer*resid_sigma, sdbuffer*resid_sigma)
            '''

        #####################################
        #  Plotting individual transits
        #####################################
        maxdur=np.max([self.planets[ipl]['tdur'] for ipl in self.planets])
        setattr(self.lc,'phase',{})
        for n,pl in enumerate(self.planets):
            if hasattr(self,'trace'):
                vars=['t0_'+pl]
                if pl in self.trios:
                    vars+=['t0_3_'+pl]
                if pl in self.trios+self.duos:
                    vars+=['t0_2_'+pl]
                elif pl in self.multis or pl in self.rvplanets:
                    vars+=['per_'+pl]
                if 'tdur_'+pl in self.init_soln:
                    vars+=['tdur_'+pl]
                elif 'tdur_'+pl+'[0]' in self.init_soln:
                    vars+=['tdur_'+pl+'[0]']
                ext=az.extract(self.trace.posterior,var_names=vars)
                t0s=[np.nanmedian(ext['t0_'+pl]), np.nanmedian(ext['t0_2_'+pl]), np.nanmedian(ext['t0_3_'+pl])] if pl in self.trios else [np.nanmedian(ext['t0_'+pl])]
                if pl in self.multis or pl in self.rvplanets:
                    per=np.nanmedian(ext['per_'+pl])
                elif pl in self.duos+self.trios:
                    per=np.max(np.nanmedian(ext['per_'+pl],axis=0))
                elif pl in self.monos:
                    per=3e3
                if 'tdur_'+pl in self.init_soln:
                    binsize=np.nanmedian(ext['tdur_'+pl])/n_intrans_bins
                elif 'tdur_'+pl+'[0]' in self.init_soln:
                    binsize=np.nanmedian(ext['tdur_'+pl+'[0]'])/n_intrans_bins
            elif hasattr(self,'init_soln'):
                t0s=[self.init_soln['t0_'+pl], self.init_soln['t0_2_'+pl], self.init_soln['t0_3_'+pl]] if pl in self.trios else [self.init_soln['t0_'+pl]]
                if pl in self.multis or pl in self.rvplanets:
                    per=self.init_soln['per_'+pl]
                elif pl in self.duos+self.trios:
                    per=np.max(self.init_soln['per_'+pl])
                elif pl in self.monos:
                    per=3e3
                if 'tdur_'+pl in self.init_soln:
                    binsize=self.init_soln['tdur_'+pl]/n_intrans_bins
                elif 'tdur_'+pl+'[0]' in self.init_soln:
                    binsize=self.init_soln['tdur_'+pl+'[0]']/n_intrans_bins
            self.lc.phase[pl]=self.make_phase(self.lc.time,t0s,per)

            for nkey,key in enumerate(self.lc_regions):
                if pl in self.multis or pl in self.rvplanets or pl in self.duos or pl in self.trios:
                    #print(key,pl,per,t0)
                    n_p_sta_end=np.array([np.floor((self.lc_regions[key]['start']-np.min(t0s))/per),np.ceil((self.lc_regions[key]['end']-np.max(t0s))/per)])
                    #Adding ticks for the position of each planet below the data:
                    trans_range=np.arange(n_p_sta_end[0],n_p_sta_end[1],1.0)
                    if interactive:
                            f_alls[key].scatter(np.min(t0s)+trans_range*per,np.tile(self.lc_regions[key]['minmax_global'][0]+0.2*resid_sigma+(0.5*resid_sigma*n/len(self.planets)),
                                                int(len(trans_range))), marker="triangle", s=12.5, fill_color=self.pal[4-n], alpha=0.85)
                    else:
                        f_alls[key].scatter(np.min(t0s)+trans_range*per,
                                        np.tile(self.lc_regions[key]['minmax_global'][0]+0.2*resid_sigma+(resid_sigma*n/len(self.planets)),
                                                int(len(trans_range))),
                                        marker="^", s=12.5, color=self.pal[4-n], alpha=0.85)

                elif pl in self.monos:
                    if (t0s[0]>self.lc_regions[key]['start'])&(t0s[0]<self.lc_regions[key]['end']):
                        if interactive:
                            f_alls[key].scatter(t0s,[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                               marker="triangle", s=12.5, fill_color=self.pal[4-n], alpha=0.85)
                        else:
                            f_alls[key].scatter(t0s,[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                               marker="^", s=12.5, color=self.pal[4-n], alpha=0.85, rasterized=raster)
                '''elif pl in self.duos:
                    if (t0>self.lc_regions[key]['start'])&(t0<self.lc_regions[key]['end']):
                        if interactive:
                            f_alls[key].scatter([t0],[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                           marker="triangle", s=12.5, fill_color=self.pal[4-n], alpha=0.85)
                        else:
                            f_alls[key].scatter([t0],[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                           marker="^", s=12.5, color=self.pal[4-n], alpha=0.85, rasterized=raster)

                    if (t0_2>self.lc_regions[key]['start'])&(t0_2<self.lc_regions[key]['end']):
                        if interactive:
                            f_alls[key].scatter([t0_2],[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                           marker="triangle", s=12.5, fill_color=self.pal[4-n], alpha=0.85)
                        else:
                            f_alls[key].scatter([t0_2],[-1*self.min_trans-0.8*resid_sigma-(resid_sigma*n/len(self.planets))],
                                           marker="^", s=12.5, color=self.pal[4-n], alpha=0.85, rasterized=raster)
                '''
            #Computing
            if len(self.planets)>1:
                other_pls=np.sum([self.trans_to_plot['all'][opl]['med'] for opl in self.planets if opl!=pl],axis=0)
            else:
                other_pls=np.zeros(len(self.lc.time))

            

            dist_dur=2.5
            phasebool=abs(self.lc.phase[pl])<dist_dur*maxdur
            if self.use_GP:
                phaselc=np.column_stack((self.lc.phase[pl][self.lc.mask&phasebool],
                                         self.lc.flux[self.lc.mask&phasebool] - \
                                         self.gp_to_plot['gp_pred'][self.lc.mask&phasebool] - \
                                         other_pls[self.lc.mask&phasebool],
                                         self.lc.flux_err[self.lc.mask&phasebool]))
            elif self.fit_no_flatten:
                phaselc=np.column_stack((self.lc.phase[pl][self.lc.mask&phasebool],
                                         self.lc.flux[self.lc.mask&phasebool] - \
                                         other_pls[self.lc.mask&phasebool],
                                         self.lc.flux_err[self.lc.mask&phasebool]))
            elif self.local_spline:
                phaselc=np.column_stack((self.lc.phase[pl][self.lc.mask&phasebool],
                                         self.lc.flux_flat[self.lc.mask&phasebool] - \
                                         self.spline_to_plot['all']['allpl']['med'][self.lc.mask&phasebool] - \
                                         other_pls[self.lc.mask&phasebool],
                                         self.lc.flux_err[self.lc.mask&phasebool]))
            else:
                phaselc=np.column_stack((self.lc.phase[pl][self.lc.mask&phasebool],
                                         self.lc.flux_flat[self.lc.mask&phasebool] - \
                                         other_pls[self.lc.mask&phasebool],
                                         self.lc.flux_err[self.lc.mask&phasebool]))
            bin_phase=tools.bin_lc_segment(phaselc[np.argsort(phaselc[:,0])],binsize=binsize)

            if interactive:
                sdbuffer=3

                self.min_trans=abs(np.min(self.trans_to_plot[pl]['med']))
                f_trans[pl].circle(phaselc[:,0],phaselc[:,1],
                                  color='black', alpha=0.4, size=0.75)

                f_trans[pl].circle(phaselc[:,0],
                                  phaselc[:,1] - self.trans_to_plot[pl]['med'][self.lc.mask&phasebool] - self.min_trans-sdbuffer*resid_sigma,
                                  color='black', alpha=0.2, size=0.75)
                errors = ColumnDataSource(data=dict(base=bin_phase[:,0],
                                        lower=bin_phase[:,1] - bin_phase[:,2],
                                        upper=bin_phase[:,1] + bin_phase[:,2]))
                f_trans[pl].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                             line_color='#dddddd', line_alpha=0.5,
                                             upper_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5),
                                             lower_head=arrow_heads.TeeHead(line_color='#dddddd',line_alpha=0.5)))
                f_trans[pl].circle(bin_phase[:,0], bin_phase[:,1], alpha=0.65, size=3.5)
                if '-2sig' in self.trans_to_plot[pl]:
                    trband = ColumnDataSource(data=dict(base=np.sort(phaselc[:,0]),
                              lower=self.trans_to_plot[pl]['-2sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                              upper=self.trans_to_plot[pl]['+2sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])]))
                    f_trans[pl].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[2+n]))
                    trband = ColumnDataSource(data=dict(base=np.sort(phaselc[:,0]),
                              lower=self.trans_to_plot[pl]['-1sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                              upper=self.trans_to_plot[pl]['+1sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])]))
                    f_trans[pl].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                              level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=self.pal[2+n]))
                f_trans[pl].line(np.sort(phaselc[:,0]),
                                self.trans_to_plot[pl]["med"][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                                color=self.pal[2+n])
                f_trans[pl].y_range=Range1d(-1*self.min_trans-2*sdbuffer*resid_sigma,sdbuffer*resid_sigma)

                if n<len(self.planets)-1:
                    f_trans[pl].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                    f_trans[pl].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                    f_trans[pl].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

            else:
                f_trans[pl].plot(phaselc[:,0],phaselc[:,1], ".",label="raw data", color=pointcol,
                                 alpha=np.clip(0.15+6*(len(phaselc[:,0])**-0.5),0.1,0.9), markersize=1.25, rasterized=raster)
                f_trans[pl].errorbar(bin_phase[:,0],bin_phase[:,1], yerr=bin_phase[:,2], fmt='.',color='C2',
                                    alpha=0.9, markersize=5, rasterized=raster)
                f_trans_resids[pl].plot(phaselc[:,0],
                                        phaselc[:,1]-self.trans_to_plot['all'][pl]['med'][self.lc.mask&phasebool], ".",
                                        alpha=np.clip(0.15+6*(len(phaselc[:,0])**-0.5),0.1,0.9), color=pointcol, markersize=1.25, rasterized=raster)
                bin_resids=tools.bin_lc_segment(np.column_stack((np.sort(phaselc[:,0]),
                                                                 (phaselc[:,1]-self.trans_to_plot['all'][pl]['med'][self.lc.mask&phasebool])[np.argsort(phaselc[:,0])],
                                                                 phaselc[np.argsort(phaselc[:,0]),2])),binsize)
                nrtrns_resid=np.nanstd(bin_resids[:,1])
                print(nrtrns_resid)
                f_trans_resids[pl].errorbar(bin_resids[:,0],bin_resids[:,1],yerr=bin_resids[:,2],fmt=".",color='C2',
                                         alpha=0.75, markersize=5, rasterized=raster)

                if '+2sig' in self.trans_to_plot['all'][pl]:
                    f_trans[pl].fill_between(np.sort(phaselc[:,0]),
                           self.trans_to_plot['all'][pl]['-2sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                           self.trans_to_plot['all'][pl]['+2sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                           alpha=0.25, color=self.pal[4-n], rasterized=raster)
                    f_trans[pl].fill_between(np.sort(phaselc[:,0]),
                           self.trans_to_plot['all'][pl]['-1sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                           self.trans_to_plot['all'][pl]['+1sig'][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                           alpha=0.25, color=self.pal[4-n], rasterized=raster)
                f_trans[pl].plot(np.sort(phaselc[:,0]),
                                self.trans_to_plot['all'][pl]["med"][self.lc.mask&phasebool][np.argsort(phaselc[:,0])],
                               color=self.pal[4-n], label="transit fit", rasterized=raster,linewidth=2.5)
                f_trans[pl].set_ylim(np.min(self.trans_to_plot['all'][pl]["med"])-4*nrtrns_resid,4*nrtrns_resid)
                f_trans_resids[pl].set_ylim(-5*nrtrns_resid,5*nrtrns_resid)
                f_trans[pl].yaxis.tick_right()
                f_trans_resids[pl].yaxis.tick_right()

                f_trans[pl].text(0.0,0.0+nrtrns_resid*3,pl,horizontalalignment='center',verticalalignment='top',fontsize=9)

                plt.setp(f_trans[pl].get_xticklabels(), visible=False)
                if n<len(self.planets)-1:
                    plt.setp(f_trans_resids[pl].get_xticklabels(), visible=False)

            if pl == xaxiskey:
                if interactive:
                    #extra = '[ppt]' if self.lc.flux_unit==0.001 else ''
                    #f_all[n].yaxis.axis_label = 'residuals '+extra#<- y-axis label
                    f_trans[pl].xaxis.axis_label = 'Time [d] from transit' #<- x-axis label
                else:
                    f_trans_resids[xaxiskey].set_xlabel("Time [d] from transit")

        if not interactive:
            if xlim is None:
                f_trans_resids[xaxiskey].set_xlim(-1*dist_dur*maxdur,dist_dur*maxdur)
            else:
                f_trans_resids[xaxiskey].set_xlim(xlim)

        if interactive:
            #Saving
            cols=[]
            for r in range(len(f_alls)):
                cols+=[column(f_alls[r])]
            lastcol=[]
            for r in range(len(f_trans)):
                lastcol+=[f_trans[r]]
            p = gridplot([cols+[column(lastcol)]])
            save(p)
            print("interactive table at:",savename)

            if return_fig:
                return p

        else:
            if save:
                if plot_loc is None and plottype=='png':
                    plt.savefig(self.savenames[0]+'_model_plot.png',dpi=350)#,transparent=True)
                    #plt.savefig(self.savenames[0]+'_model_plot.pdf')
                elif plot_loc is None and plottype=='pdf':
                    plt.savefig(self.savenames[0]+'_model_plot.pdf')
                else:
                    plt.savefig(plot_loc)
            if return_fig:
                return fig

    def plot_periods(self, plot_loc=None, ylog=True, xlog=True, nbins=25, 
                    pmax=None, pmin=None, ymin=None,ymax=None,extra_factor=1):
        """Plot Marginalised probabilities of the possible periods

        Args:
            plot_loc (str, optional): File location. Defaults to None, which takes location from `savenames`
            ylog (bool, optional): Set y axis as log scale? Defaults to True.
            nbins (int, optional): Number of total bins . Defaults to 25.
            pmax (float, optional): Max period on plot. Defaults to None.
            pmin (float, optional): Min period on plot. Defaults to None.
            ymin (float, optional): Minimum on y axis (logprob). Defaults to None.
            ymax (float, optional): Max on y axis (logprob). Defaults to None.
            xlog (bool, optional): Set x axis (period) to log scale? Defaults to False.
        """
        assert hasattr(self,'trace')
        
        from scipy.special import logsumexp
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib import cm

        # mango=[(0/256,89/256,87/256),(69/256,148/256,52/256),(142/256,183/256,29/256),(208/256,218/256,8/256),
        #     (255/256,235/256,0/256),(248/256,199/256,28/256),(242/256,136/256,35/256),(185/256,68/256,36/256),(116/256,0/256,30/256)]
        # mango_dark=mango[:3]+mango[-3:]
        # sns.set_palette(mango)
        mango=sns.color_palette('viridis',11)[::2]
        sns.set_palette(mango)
        
        # Tableau 20 color palette for demonstration
        c = np.zeros((3, len(mango), 3))
        rgb = ['red', 'green', 'blue']
        for idx, e in enumerate(mango):
            for ii in range(3):
                c[ii, idx, :] = [float(idx) / float(len(mango) - 1), e[ii], e[ii]]

        cdict = dict(zip(rgb, c))
        cmap = LinearSegmentedColormap('tab20', cdict)
        sns.set_palette(mango)
        pal=cmap(np.linspace(0,1,6))#sns.color_palette(mango)
        coldic={-6:"p<1e-5",-5:"p>1e-5",-4:"p>1e-4",-3:"p>0.1%",-2:"p>1%",-1:"p>10%",0:"p>100%"}
        plot_pers=self.duos+self.monos+self.trios
        ymin = 0-15*float(int(ylog)) if ymin is None else ymin
        if len(plot_pers)>0:
            #plt.figure(figsize=(4.5+1.5*np.sqrt(len(self.monos+self.duos+self.trios)),4.2))
            plt.figure(figsize=(3.2+1.5*np.sqrt(len(self.monos+self.duos+self.trios)),3.5))
            for npl, pl in enumerate(plot_pers):
                plt.subplot(1,len(plot_pers),npl+1)
                ext=az.extract(self.trace.posterior,var_names=['logprob_marg_'+pl,'per_'+pl])
                if pl in self.duos+self.trios:
                    #As we're using the nanmedian log10(prob)s for each period, we need to make sure their sums add to 1.0
                    probs=logsumexp(np.log(extra_factor)+ext['logprob_marg_'+pl] - logsumexp(np.log(extra_factor)+ext['logprob_marg_'+pl]),axis=0)/np.log(10)
                    pers = np.nanmedian(ext['per_'+pl],axis=0)
                    pmax = np.nanmax(pers)*1.03 if pmax is None else pmax
                    pmin = np.nanmin(pers)*0.9 if pmin is None else pmin
                    
                    ymax = np.max(probs[pers<pmax])+0.5 if ymax is None else ymax
                    #psum=logsumexp(np.nanmedian(ext['logprob_marg_'+pl],axis=0))/np.log(10)
                    #Plotting lines
                    cols=[]
                    #plt.plot(pers,prob_prcnts[0,:],":",color='C0',alpha=0.65)
                    #plt.plot(pers,prob_prcnts[2,:],":",color='C0',alpha=0.65)
                    ordered_prob_index=np.arange(len(probs))[np.argsort(probs)][::-1]
                    for n in ordered_prob_index:
                        # Density Plot and Histogram of all arrival delays
                        #nprob=probs[n]
                        ncol=int(np.floor(np.clip(probs[n],-6,0)))
                        if ncol not in cols:
                            cols+=[ncol]
                            plt.plot(np.tile(pers[n],2),
                                         [ymin,probs[n]],
                                         linewidth=5.0,color=pal[6+ncol],alpha=0.7,label="$"+coldic[ncol].replace('%','\%')+"$")
                        else:
                            plt.plot(np.tile(pers[n],2),
                                         [ymin,probs[n]],
                                         linewidth=5.0,color=pal[6+ncol],alpha=0.7)

                    plt.title("Duo - "+str(pl))
                    plt.ylim(ymin,ymax)
                    plt.legend()
                    if xlog:
                        plt.xscale('log')
                        plt.xticks([20,30,40,60,80,100,150,200,300,400,600,800,1000,1500,2000,2500,3000,3500],
                                   np.array([20,30,40,60,80,100,150,200,300,400,600,800,1000,1500,2000,2500,3000,3500]).astype(str))
                        #plt.xticks([20,30,40,60,80,100,150,200,250,300,350,400,450,500,600,800,1000,1500,2000,2500,3000,3500],
                        #           np.array([20,30,40,60,80,100,150,200,250,300,350,400,450,500,600,800,1000,1500,2000,2500,3000,3500]).astype(str))
                        #plt.xticklabels([20,40,60,80,100,150,200,250])
                    plt.ylabel("$\log_{10}{p}$")
                    plt.xlabel("Period [d]")
                    plt.xlim(pmin,pmax+1)
                elif pl in self.monos:
                    #if 'logprob_marg_sum_'+pl in self.trace.posterior:
                    #    total_prob=logsumexp((self.trace.posterior['logprob_marg_'+pl]+self.trace.posterior['logprob_marg_sum_'+pl]).ravel())
                    #else:
                    total_prob=logsumexp(ext['logprob_marg_'+pl].ravel())
                    total_av_prob=logsumexp(np.nanmedian(ext['logprob_marg_'+pl],axis=0))
                    pmax = np.nanmax(ext['per_'+pl].ravel()) if pmax is None else pmax
                    pmin = 0.5*np.min(self.planets[pl]['per_gaps']['gap_starts']) if pmin is None else pmin
                    cols=[]
                    for ngap in np.arange(self.planets[pl]['ngaps'])[np.argsort(np.nanmedian(ext['logprob_marg_'+pl],axis=0))]:
                        if self.planets[pl]['per_gaps']['gap_starts'][ngap]<pmax and self.planets[pl]['per_gaps']['gap_ends'][ngap]>pmin:
                            bins=np.arange(np.floor(self.planets[pl]['per_gaps']['gap_starts'][ngap])-0.5,
                                        np.clip(np.ceil(self.planets[pl]['per_gaps']['gap_ends'][ngap])+0.5,0.0,pmax),
                                        1.0)
                            print(bins)
                            ncol=int(np.floor(np.clip(np.nanmedian(ext['logprob_marg_'+pl][:,ngap])-total_av_prob,-6,0)))
                            #print(self.planets[pl]['per_gaps']['gap_starts'][ngap],
                            #      ncol,np.nanmedian(self.trace.posterior['logprob_marg_'+pl][:,ngap])-total_av_prob)
                            #print(ngap,np.exp(self.trace.posterior['logprob_marg_'+pl][:,ngap]-total_prob))
                            if ncol not in cols:
                                cols+=[ncol]
                                plt.hist(ext['per_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                                    weights=np.exp(ext['logprob_marg_'+pl][:,ngap]-total_prob),
                                    color=pal[6+ncol],histtype="stepfilled",label=coldic[ncol])
                            else:
                                plt.hist(ext['per_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                                    weights=np.exp(ext['logprob_marg_'+pl][:,ngap]-total_prob),
                                    color=pal[6+ncol],histtype="stepfilled")

                    plt.title("Mono - "+str(pl))

                    if xlog:
                        plt.xscale('log')
                        plt.xlim(pmin,pmax)
                        ticks=np.array([20,40,60,80,100,150,200,250,300,350,400,450,500,600,800,1000,1500,2000,2500,3000,4000,5000,6000,8000,10000])
                        plt.xticks(ticks[(ticks>pmin)&(ticks<pmax)],ticks[(ticks>pmin)&(ticks<pmax)].astype(str))
                        #plt.xticklabels([20,40,60,80,100,150,200,250])
                    else:
                        pmin=pmin if pmin is not None else 0
                        plt.xlim(pmin,pmax)
                    
                    if ylog:
                        plt.yscale('log')
                        plt.ylabel("$\log_{10}{\\rm prob}$")
                        plt.ylim(ymin, ymax)
                    else:
                        plt.ylim(ymin, ymax)
                        plt.ylabel("prob")
                        #plt.xlim(60,80)
                    plt.xlabel("Period [d]")
                    plt.legend(title="Average prob")
            plt.tight_layout()
            if plot_loc  is None:
                plt.savefig(self.savenames[0]+'_period_dists.pdf')
            else:
                plt.savefig(plot_loc)

    def plot_corner(self,corner_vars=None,use_marg=True,truths=None):
        """Create Corner plot for MCMC samples
        
        Args:
            corner_vars (list, optional): List of parameter names to include. Defaults to None, which uses stellar density & various important orbital params.
            use_marg (bool, optional): Use marginalised parameters (e.g. weighted average across all aliases) or individual alias values. Defaults to True.
            truths (list, optional): "True" parameters for the selected parameters (e.g. for use in testing/comparisons). Defaults to None.
        """
        # Plotting corner for those parameters we're interested in - e.g. orbital parameters
        # If "use_marg" is True - uses the marginalised tdur and period parameters for multis and duos
        # If "use_marg" is False - generates samples for each marginalised distribution and weights by logprob
        import corner

        if corner_vars is None:
            corner_vars=['logrho_S']

            for pl in self.planets:
                for var in self.fit_params:
                    if var+'_'+pl in self.trace.posterior:
                        corner_vars+=[var+'_'+pl]
                if pl in self.duos+self.trios:
                    corner_vars+=['t0_2_'+pl]
                if use_marg:
                    for var in self.marginal_params:
                        if var+'_marg_'+pl in self.trace.posterior:
                            corner_vars+=[var+'_marg_'+pl]

        print("variables for Corner:",corner_vars)
        '''
        for pl in self.duos:
            corner_vars+=['duo_t0_'+pl,'duo_t0_2_'+pl]
            corner_vars+=[var+'_marg_'+pl for var in self.marginal_params]
            if 'tdur' not in self.marginal_params:
                corner_vars+=['duo_tdur_'+pl]
            elif 'b' not in self.marginal_params:
                corner_vars+=['duo_b_'+pl]
            if 'logror' not in self.marginal_params:
                corner_vars+=['duo_logror_'+pl]
            if not self.assume_circ and 'ecc_marg_'+pl not in corner_vars:
                corner_vars+=['duo_ecc_'+pl,'duo_omega_'+pl]
        for pl in self.monos:
            corner_vars+=['mono_t0_'+pl]
            corner_vars+=[var+'_marg_'+pl for var in self.marginal_params]
            if 'tdur' not in self.marginal_params:
                corner_vars+=['mono_tdur_'+pl]
            elif 'b' not in self.marginal_params:
                corner_vars+=['mono_b_'+pl]
            if 'logror' not in self.marginal_params:
                corner_vars+=['mono_logror_'+pl]
            if not self.assume_circ and 'ecc_marg_'+pl not in corner_vars:
                corner_vars+=['mono_ecc_'+pl,'mono_omega_'+pl]
        if len(self.multis)>0:
            corner_vars+=['multi_t0','multi_logror','multi_b','multi_per']
            if not self.assume_circ:
                corner_vars+=['multi_ecc','multi_omega']
        '''
        samples = self.make_table(cols=corner_vars)
        #print(samples.shape,samples.columns)
        assert samples.shape[1]<50

        if use_marg:
            fig = corner.corner(samples,truths=truths)
        else:
            #Not using the marginalised period, and instead using weights:
            logprobs=[]

            all_weighted_periods={}
            all_logprobs={}

            n_mult=np.product([self.planets[mpl]['ngaps'] for mpl in self.monos]) * \
                   np.product([len(self.planets[dpl]['period_aliases']) for dpl in self.duos])
            #print(n_mult,"x samples")
            samples['log_prob']=np.tile(0.0,len(samples))
            samples_len=len(samples)
            samples=pd.concat([samples]*int(n_mult),axis=0)
            #print(samples.shape,samples_len)

            n_pos=0

            vars=[]
            for pl in self.monos+self.duos:
                for v in ['per_'+pl,'b_'+pl,'logprob_marg_'+pl,'tdur_'+pl]:
                    if v in self.trace.posterior:
                        vars+=[v]
            ext=az.extract(self.trace.posterior,varnames=v)
            for mpl in self.monos:
                for n_gap in np.arange(self.planets[mpl]['ngaps']):
                    sampl_loc=np.in1d(np.arange(0,len(samples),1),np.arange(n_pos*samples_len,(n_pos+1)*samples_len,1))
                    samples.loc[sampl_loc,'per_marg_'+mpl]=ext['per_'+mpl][:,n_gap]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+mpl]=ext['tdur_'+mpl][:,n_gap]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+mpl]=ext['b_'+mpl][:,n_gap]
                    samples.loc[sampl_loc,'log_prob']=ext['logprob_marg_'+mpl][:,n_gap]
                    n_pos+=1
            for dpl in self.duos:
                for n_per in np.arange(len(self.planets[dpl]['period_aliases'])):
                    sampl_loc=np.in1d(np.arange(len(samples)),np.arange(n_pos*samples_len,(n_pos+1)*samples_len))
                    samples.loc[sampl_loc,'per_marg_'+dpl]=ext['per_'+dpl][:,n_per]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+dpl]=ext['tdur_'+dpl][:,n_per]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+dpl]=ext['b_'+dpl][:,n_per]
                    samples.loc[sampl_loc,'log_prob'] = ext['logprob_marg_'+dpl][:,n_per]
                    n_pos+=1
            weight_samps = np.exp(samples["log_prob"])
            fig = corner.corner(samples[[col for col in samples.columns if col!='log_prob']],weights=weight_samps);

        fig.savefig(self.savenames[0]+'_corner.pdf')#,dpi=400,rasterized=True)


    def make_table(self,short=True,save=True,cols=['all']):
        """Make table from MCMC Samples

        Args:
            short (bool, optional): Create "short" table (i.e. without hyperparameters). Defaults to True.
            save (bool, optional): Save to csv? Defaults to True.
            cols (list, optional): ['all'] or list of selected column names. Defaults to ['all'].

        Returns:
            pandas DataFrame: Dataframe of parameters and specific parameters for the samples
        """
        assert hasattr(self,'trace')

        if cols[0]=='all':
            #Removing lightcurve, GP and reparameterised hyper-param columns
            cols_to_remove=['gp_', '_gp', 'light_curve','__','model_rv','marg_all_lc','marg_all_rv','rv_model','rv_trend','nonmarg_rvs']
            if short:
                #If we want just the short table, let's remove those params which we derived and which we marginalised
                cols_to_remove+=['mono_uniform_index','logliks','_priors','logprob_marg','logrho_S']
                for col in self.marginal_params:
                    cols_to_remove+=['mono_'+col+'s','duo_'+col+'s']
            medvars=[var for var in self.trace.posterior if not np.any([icol in var for icol in cols_to_remove])]
            #print(cols_to_remove, medvars)
            df = pm.summary(self.trace,var_names=medvars,stat_funcs={"5%": lambda x: np.percentile(x, 5),
                                                                     "-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                                     "median": lambda x: np.percentile(x, 50),
                                                                     "+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                                     "95%": lambda x: np.percentile(x, 95)},round_to=5)
        else:
            df = pm.summary(self.trace,var_names=cols,stat_funcs={"5%": lambda x: np.percentile(x, 5),
                                                                     "-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                                     "median": lambda x: np.percentile(x, 50),
                                                                     "+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                                     "95%": lambda x: np.percentile(x, 95)},round_to=5)

        if save:
            print("Saving sampled model parameters to file with shape: ",df.shape)
            if short:
                df.to_csv(self.savenames[0]+'_mcmc_output_short.csv')
            else:
                df.to_csv(self.savenames[0]+'_mcmc_output.csv')
        return df

    def cheops_planet_properties_table(self,planet=None):
        """Create output compatible with the Cheops "PlanetPropertiesTable". Not yet implemented

        Args:
            planet (str, optional): string name of planet. Defaults to None.
        """
        "target,gaia_id,planet_id,T0,e_T0,P,e_P,ecosw,e_ecosw,esinw,e_esinw,D,e_D,W,e_W,K,e_K"

    def plot_table(self,plot_loc=None,return_table=False):
        """Plot table as PDF (i.e. to assemble PDF report)

        Args:
            plot_loc (str, optional): File location. Defaults to None, which uses `mod.savenames`
            return_table (bool, optional): Return DF figure? Defaults to False.

        Returns:
            pandas DataFrame: Dataframe of parameters and specific parameters (output from `make_table`)
        """

        df = self.make_table(short=True)

        # Making table a plot for PDF:
        fig=plt.figure(figsize=(11.69,8.27))
        ax=fig.add_subplot(111)
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        cols=["name","mean","sd","5%","-$1\sigma$","median","+$1\sigma$","95%"]
        df['name']=df.index.values
        #print(df.loc[:,list(cols.keys())].values)
        tab = ax.table(cellText=df[cols].values,
                  colLabels=cols,
                  loc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)

        tab.auto_set_column_width(col=range(len(cols))) # Provide integer list of columns to adjust

        fig.tight_layout()
        if plot_loc is None:
            fig.savefig(self.savenames[0]+'_table.pdf')
        else:
            plt.savefig(plot_loc)
        if return_table:
            return df



    def load_pickle(self, loadname=None):
        """Load data from saved pickle

        Args:
            loadname (str, optional): Filename to load from. Defaults to None, which loads from `savenames`
        """
        #Pickle file style: folder/TIC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
        if loadname is not None:
            print(self.savenames[0]+'_mcmc.pickle',"exists - loading")
            n_bytes = 2**31
            max_bytes = 2**31 - 1

            ## read
            bytes_in = bytearray(0)
            input_size = os.path.getsize(loadname)
            with open(self.savenames[0]+'_mcmc.pickle', 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            loaded = pickle.loads(bytes_in)
            if type(loaded)==dict:
                for key in loaded:
                    setattr(self,key,loaded[key])
            else:
                self.trace=loaded
        if not hasattr(self, 'savenames') or self.savenames is None:
            self.get_savename(how='load')
        #print(self.savenames, self.savenames is None)
        #[0]+'_mcmc.pickle',os.path.exists(self.savenames[0]+'_mcmc.pickle'))
        if os.path.exists(self.savenames[0]+'_mcmc.pickle'):
            print(self.savenames[0]+'_mcmc.pickle',"exists - loading")
            n_bytes = 2**31
            max_bytes = 2**31 - 1

            ## read
            bytes_in = bytearray(0)
            input_size = os.path.getsize(self.savenames[0]+'_mcmc.pickle')
            with open(self.savenames[0]+'_mcmc.pickle', 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            loaded = pickle.loads(bytes_in)
            if type(loaded)==dict:
                for key in loaded:
                    setattr(self,key,loaded[key])
            else:
                self.trace=loaded

    def predict_future_transits(self, time_start=None, time_end=None, time_dur=180, include_multis=True, 
                              save=True, compute_solsys_dist=True, check_TESS=True):
        """Return a dataframe of potential transits of all Duo candidates between time_start & time_end dates.

        Args:
            time_start (Astropy.Time object or float, optional): Astropy.Time date or julian date (in same base as lc) for the start of the observing period. If not defined, uses today
            time_end (Astropy.Time object or float, optional): Astropy.Time date or julian date (in same base as lc) for the end of the observing period. If not defined, uses today+time_dur
            time_dur (float, optional): Duration (in days) to compute future transits. Default is 100d
            include_multis (bool, optional): Also generate transits for multi-transiting planets? Defaults to True.
            save (bool, optional): Whether to save transit dataframe to file? Defaults to True
            check_TESS (bool, optional): Whether to check which transits will be seen by TESS

        Returns:
            pandas DataFrame: Dataframe of transits observable between time_start and time_end.

        Example:
            # e.g. after running model.RunMcmc():
            df = model.predict_future_transits(Time('2021-06-01T00:00:00.000',format='isot'),Time('2021-10-01T00:00:00.000',format='isot'))
        """
        
        from astropy.time import Time
        from datetime import datetime
        import fractions

        assert hasattr(self,'trace') #We need to have run Mcmc to have samples first.

        if not hasattr(self.lc,'radec'):
            #Getting the coordinate
            self.lc.get_radec()

        #If not defined, we'll take the date today:
        if time_start is None:
            time_start = Time(datetime.now())
        if time_end is None:
            time_end = Time(datetime.now())+time_dur*u.day
        
        if type(time_start)==Time:
            print("time range",time_start.isot,"->",time_end.isot)
            time_start = time_start.jd - self.lc.jd_base
            time_end   = time_end.jd - self.lc.jd_base
        elif type(time_start) in [int, np.int64, float, np.float64] and abs(time_start-np.min([np.nanmedian(self.trace.posterior['t0_'+pl].values) for pl in self.planets]))>5000:
            #This looks like a proper julian date. Let's reformat to match the lightcurve
            time_start -= self.lc.jd_base
            time_end   -= self.lc.jd_base
            print("assuming JD in format 2457...")
            print("time range",Time(time_start,format='jd').isot,"->",Time(time_end,format='jd').isot)
        else:
            assert type(time_start) in [int, np.int64, float, np.float64]
            print("time range",Time(time_start+self.lc.jd_base,format='jd').isot,
                  "->",Time(time_end+self.lc.jd_base,format='jd').isot)
        
        if check_TESS:
            sect_start_ends=self.check_TESS()
        
        all_trans_fin=pd.DataFrame()
        loopplanets = self.duos+self.trios+self.multis if include_multis else self.duos+self.trios
        all_unq_trans=[]
        
        for pl in self.planets:
            for v in ['logprob_marg_'+pl,'t0_'+pl,'t0_2_'+pl,'t0_3_'+pl,'per_'+pl,'tdur_'+pl]:
                if v in self.trace.posterior:
                    vars+=v
        ext=az.extract(self.trace.posterior,var_names=vars)

        for pl in loopplanets:
            all_trans=pd.DataFrame()
            if pl in self.duos+self.trios:
                sum_all_probs=np.logaddexp.reduce(np.nanmedian(ext['logprob_marg_'+pl],axis=0))
                trans_p0=np.floor(np.nanmedian(time_start - ext['t0_2_'+pl].values)/np.nanmedian(ext['per_'+pl].values,axis=0))
                trans_p1=np.ceil(np.nanmedian(time_end -  ext['t0_2_'+pl].values)/np.nanmedian(ext['per_'+pl].values,axis=0))
                n_trans=trans_p1-trans_p0
            elif pl in self.multis:
                trans_p0=[np.floor(np.nanmedian(time_start - ext['t0_'+pl].values)/np.nanmedian(ext['per_'+pl].values))]
                trans_p1=[np.ceil(np.nanmedian(time_end -  ext['t0_'+pl].values)/np.nanmedian(ext['per_'+pl].values))]
                n_trans=[trans_p1[0]-trans_p0[0]]
            #print(pl,trans_p0,trans_p1,n_trans)
            #print(np.nanmedian(self.trace.posterior['t0_2_'+pl])+np.nanmedian(self.trace.posterior['per_'+pl],axis=0)*trans_p0)
            #print(np.nanmedian(self.trace.posterior['t0_2_'+pl])+np.nanmedian(self.trace.posterior['per_'+pl],axis=0)*trans_p1)

            nms=['-2sig','-1sig','med','+1sig','+2sig']
            percentiles=(2.2750132, 15.8655254, 50., 84.1344746, 97.7249868)

            #Getting the important trace info (tcen, dur, etc) for each alias:
            if 'tdur' in self.fit_params or pl in self.multis:
                dur=np.nanpercentile(ext['tdur_'+pl],percentiles)
            naliases=[0] if pl in self.multis else np.arange(self.planets[pl]['npers'])
            idfs=[]
            for nd in naliases:
                if n_trans[nd]>0:
                    if pl in self.duos+self.trios:
                        int_alias=int(self.planets[pl]['period_int_aliases'][nd])
                        transits=np.nanpercentile(np.vstack([ext['t0_2_'+pl].values+ntr*ext['per_'+pl].values[:,:,nd] for ntr in np.arange(trans_p0[nd],trans_p1[nd])]),percentiles,axis=1)
                        if 'tdur' in self.marginal_params:
                            dur=np.nanpercentile(ext['tdur_'+pl][:,nd],percentiles)
                        logprobs=np.nanmedian(ext['logprob_marg_'+pl][:,nd])-sum_all_probs
                    else:
                        transits=np.nanpercentile(np.column_stack([ext['t0_'+pl].values+ntr*ext['per_'+pl].values for ntr in np.arange(trans_p0[nd],trans_p1[nd],1.0)]),percentiles,axis=0)
                        int_alias=1
                        logprobs=np.array([0.0])
                    #Getting the aliases for this:
                    idfs+=[pd.DataFrame({'transit_mid_date':Time(transits[2]+self.lc.jd_base,format='jd').isot,
                                      'transit_mid_med':transits[2],
                                      'transit_dur_med':np.tile(dur[2],len(transits[2])),
                                      'transit_dur_-1sig':np.tile(dur[1],len(transits[2])),
                                      'transit_dur_+1sig':np.tile(dur[3],len(transits[2])),
                                      'transit_start_+2sig':transits[4]-0.5*dur[0],
                                      'transit_start_+1sig':transits[3]-0.5*dur[1],
                                      'transit_start_med':transits[2]-0.5*dur[2],
                                      'transit_start_-1sig':transits[1]-0.5*dur[3],
                                      'transit_start_-2sig':transits[0]-0.5*dur[4],
                                      'transit_end_-2sig':transits[0]+0.5*dur[0],
                                      'transit_end_-1sig':transits[1]+0.5*dur[1],
                                      'transit_end_med':transits[2]+0.5*dur[2],
                                      'transit_end_+1sig':transits[3]+0.5*dur[3],
                                      'transit_end_+2sig':transits[4]+0.5*dur[4],
                                      '1sig_window_dur':transits[3]-transits[1]+dur[3],
                                      '2sig_window_dur':transits[4]-transits[0]+dur[4],
                                      'transit_fractions':np.array([str(fractions.Fraction(i1,int_alias)) for i1 in np.arange(trans_p0[nd],trans_p1[nd]).astype(int)]),
                                      'log_prob':np.tile(logprobs,len(transits[2])),
                                      'prob':np.tile(np.exp(logprobs),len(transits[2])),
                                      'planet_name':np.tile('multi_'+pl,len(transits[2])) if pl in self.multis else np.tile('duo_'+pl,len(transits[2])),
                                      'alias_n':np.tile(nd,len(transits[2])),
                                      'alias_p':np.tile(np.nanmedian(ext['per_'+pl].values[:,nd]),len(transits[2])) if pl in self.duos+self.trios else np.tile(np.nanmedian(ext['per_'+pl].values),len(transits[2]))})]
            all_trans=pd.concat(idfs)
            unq_trans = all_trans.sort_values('log_prob').copy().drop_duplicates('transit_fractions')
            unq_trans = unq_trans.set_index(np.arange(len(unq_trans)))
            unq_trans['aliases_ns']=unq_trans['alias_n'].values.astype(str)
            unq_trans['aliases_ps']=unq_trans['alias_p'].values.astype(str)
            unq_trans['total_prob']=unq_trans['prob']

            for i,row in unq_trans.iterrows():
                oths=all_trans.loc[all_trans['transit_fractions']==row['transit_fractions']]
                #print(row['transit_fractions'],oths['alias_n'].values,oths['alias_p'].values)
                unq_trans.loc[i,'aliases_ns']=','.join(list(oths['alias_n'].values.astype(str)))
                unq_trans.loc[i,'aliases_ps']=','.join(list(np.round(oths['alias_p'].values,4).astype(str)))
                unq_trans.loc[i,'num_aliases']=len(oths)
                unq_trans.loc[i,'total_prob']=np.sum(oths['prob'].values)
                all_unq_trans+=[unq_trans]
        all_trans_fin=pd.concat(all_unq_trans)
        all_trans_fin = all_trans_fin.loc[(all_trans_fin['transit_end_+2sig']>time_start)*(all_trans_fin['transit_start_-2sig']<time_end)].sort_values('transit_mid_med')
        all_trans_fin = all_trans_fin.set_index(np.arange(len(all_trans_fin)))

        if check_TESS and len(sect_start_ends)>0:
            all_trans_fin['in_TESS']=np.any((all_trans_fin['transit_mid_med'].values[:,None]>sect_start_ends[:,0][None,:]-2457000)&(all_trans_fin['transit_mid_med'].values[:,None]<sect_start_ends[:,1][None,:]-2457000),axis=1)
        elif check_TESS and len(sect_start_ends)==0:
            all_trans_fin['in_TESS']=np.tile(False,len(all_trans_fin))

        if compute_solsys_dist:
            from astropy.time import Time
            from astropy.coordinates import SkyCoord, get_body
            sun_coo = get_body('sun', Time(all_trans_fin['transit_mid_med'].values+self.lc.jd_base,format='jd',scale='tdb'))
            sun_sep = sun_coo.separation(self.lc.radec)
            all_trans_fin['sun_separation'] = sun_sep.deg
            moon_coo = get_body('moon', Time(all_trans_fin['transit_mid_med'].values+self.lc.jd_base,format='jd',scale='tdb'))
            moon_sep = moon_coo.separation(self.lc.radec)
            all_trans_fin['moon_separation'] = moon_sep.deg
        if save:
            all_trans_fin.to_csv(self.savenames[0]+"_list_all_trans.csv")
        return all_trans_fin

    def check_TESS(self,**kwargs):
        """Returns time frames in the future when TESS is observing
        """
        import importlib
        tess_stars2px = importlib.import_module("tess-point.tess_stars2px")
        from astropy.time import Time
        result = tess_stars2px.tess_stars2px_function_entry(self.lc.all_ids['tess']['id'], self.lc.radec.ra.deg, self.lc.radec.dec.deg)
        sectdiffs=np.diff(result[-1].midtimes)
        sectdiffs=np.hstack((sectdiffs[0],0.5*(sectdiffs[:-1]+sectdiffs[1:]),sectdiffs[-1]))
        future_sect_ix=np.isin(result[-1].sectors,result[3])&(result[-1].midtimes>Time.now().jd)
        midtimes = result[-1].midtimes[future_sect_ix]
        return np.column_stack((midtimes-0.5*sectdiffs[future_sect_ix]+0.2,midtimes+0.5*sectdiffs[future_sect_ix]-0.2))
        #Now we have sector start & end times, let's check which future transit will be TESS observed:

    def cheops_RMS(self, Gmag, tdur):
        #RMS polynomial fits for 3 hour durations:
        rms_brightfit = np.array([ 2.49847572, -6.41232409])
        rms_faintfit = np.array([  30.2599025 , -256.41381477])
        rms = np.max([np.polyval(rms_faintfit,Gmag),np.polyval(rms_brightfit,Gmag)])
        return rms/np.sqrt(tdur/0.125)

    def make_cheops_OR(self, DR2ID=None, pl=None, min_eff=45, oot_min_orbits=1.0, timing_sigma=3, t_start=None, t_end=None, Texp=None,
                     max_orbits=14, min_pretrans_orbits=0.5, min_intrans_orbits=None, orbits_flex=1.4, observe_sigma=2, 
                     observe_threshold=None, max_ORs=None,prio_1_threshold=0.25, prio_3_threshold=0.0, targetnamestring=None,
                     min_orbits=4.0, outfilesuffix='_output_ORs.csv',avoid_TESS=True, pre_post_TESS="pre", prog_id="0072"):
        """Given a list of observable transits (which are outputted from `trace_to_cheops_transits`), 
            create a csv which can be run by pycheops make_xml_files to produce input observing requests (both to FC and observing tool).

        Args:
            DR2ID (int, optional):  Gaia DR2 ID. If not present, will take DR2ID from lc object (under self.lc.all_ids['tess']['data'])
            pl (str, optional): name of planet in self.planets dict to process. Defaults to None, i.e. assumes all planets)
            min_eff (int, optional): minimum efficiency in percent. Defaults to 45
            oot_min_orbits (int, optional): minimum number of out-of-transit orbits to include both before & after (this may be higher if timing uncertainty is worse). Defaults to 2.
            timing_sigma (float,optional): Number of uncertainty
            t_start (float, optional): time of start of Cheops observations, in same jd as model (e.g. TESS HJD BJD-2457000). Defaults to None.
            t_end (float, optional): time of end of Cheops observations, in same jd as model (e.g. TESS HJD BJD-2457000). Defaults to None.
            Texp (float, optional): Exposure Time of CHEOPS observations. Defaults to None, i.e. generated by PyCheops
            max_orbits(float, optional): Maximum number of orbits per visit. Defaults to 14 (i.e. 1d)
            min_pretrans_orbits (float, optional): Number of orbits to insist happen before (and after) transit, rather than either before/after (as with oot_min_orbits). Defaults to 0.5
            min_intrans_orbits (float, optional): In the case that the visit duration does not cover the full transit, this tells us how much of the transit we have to observe
            orbits_flex (float, optional): Flexibility in orbits. Defaults to None
            observe_sigma (float, optional): threshold in sigma to make sure to observe. Defaults to 2.
            observe_threshold (float, optional): threshold above which to create ORs. If None, uses `observe_sigma` to decide which to observe. Defaults to None
            max_ORs (int, optional): Maximum number of ORs to create. Default is 14
            prio_1_threshold (float, optional): Rough percentage of ORs we want to make P1 on Cheops. Defaults to 0.25.
            prio_3_threshold (float, optional): Rough percentage of ORs we want to make P3 on Cheops. Defaults to 0.0 - i.e. no P3 observations
            targetnamestring (str, optional): String for target name. Defaults to None (and using the ID)
            min_orbits (float, optional): Minimum number of total orbits to observe. Defaults to 4.0
            outfilesuffix (str, optional): suffix place to save CSV. Defaults to '_output_ORs.csv'
            avoid_TESS (bool, optional): Whether to use time constraints to specifically avoid TESS photometry. Defaults to True
            pre_post_TESS (str, optional): If we are avoiding TESS, should we create the pre-TESS ORs, or the post-TESS ORs? Defaults to "pre"

        Returns:
            df = model.predict_future_transits: panda DF to save as csv in location where one can run make_xml_files. e.g. `make_xml_files output.csv --auto-expose -f`
        """
        #radec, SpTy, Vmag, e_Vmag,

        #Deriving spectral type:
        from astropy.io import ascii
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord, Distance
        from astropy import units as u
        from astropy.time import Time
        from scipy import stats

        if DR2ID is None:
            assert 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'GAIA' in self.lc.all_ids['tess']['data'] and not pd.isnull(self.lc.all_ids['tess']['data']['GAIA']), "Must provide Gaia DR2 ID"
            DR2ID=int(self.lc.all_ids['tess']['data']['GAIA'])

        tab=ascii.read("https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt",header_start=23,data_start=24,data_end=118).to_pandas().loc[:,['SpT','Teff']]
        SpTy = tab['SpT'].values[np.argmin(abs(self.Teff[0]-tab['Teff']))][:2]

        gaiainfo=Gaia.launch_job_async("SELECT * \
                                        FROM gaiadr2.gaia_source \
                                        WHERE gaiadr2.gaia_source.source_id="+str(DR2ID)).results.to_pandas().iloc[0]
        gaia_colour=(gaiainfo['phot_bp_mean_mag']-gaiainfo['phot_rp_mean_mag'])
        V=gaiainfo['phot_g_mean_mag']+0.0176+0.00686*gaia_colour+0.1732*gaia_colour**2
        Verr=1.09/gaiainfo['phot_g_mean_flux_over_error']+0.045858

        c = SkyCoord(ra=gaiainfo['ra']* u.deg,dec=gaiainfo['dec']* u.deg,
                     distance=Distance(parallax=gaiainfo['parallax'] * u.mas),
                     pm_ra_cosdec=gaiainfo['pmra'] * u.mas/u.yr, pm_dec=gaiainfo['pmdec'] * u.mas/u.yr,
                     obstime=Time(gaiainfo['ref_epoch'], format='jyear'))
        old_radec=c.apply_space_motion(Time(2000, format='jyear'))

        #if Texp is None:
        #    print("* WARNING - MUST SET EXPOSURE TIME (Texp) FOR REAL OBSERVATIONS. USING 1SEC HERE *")
        #    Texp=1

        next_vernal = 2459659.14792+np.ceil((Time.now().jd-2459659.14792)/365.25)*365.25
        #print(next_vernal)
        if t_start is None:
            import datetime
            from astropy.time import Time
            today=Time.now().jd
            #RA is defined as 0 for the Sun at the vernal equinox.
            #Therefore their RA, converted to fractional days and shifted by half a year, gives the time of opposition.
            
            end_next_obs   = next_vernal-365.25+(old_radec.ra.deg/360-0.5)*365.25+90
            start_next_obs = next_vernal+(old_radec.ra.deg/360-0.5)*365.25-90

            if today < end_next_obs:
                #observable now?
                t_start= today
                t_end  = end_next_obs
            else:
                t_start = start_next_obs
                t_end  = end_next_obs+365.25
        if t_end is None:
            #Using date 60d after it's at opposition in 2022:
            t_end = next_vernal-(old_radec.ra.deg/360)*365.25+60

        #We always need an array of all possible transits/aliases also saved to file to check:
        all_trans = self.predict_future_transits(t_start-self.lc.jd_base,t_end-self.lc.jd_base, check_TESS=avoid_TESS)
        all_trans.to_csv(self.savenames[0]+outfilesuffix.replace("_ORs","").replace(".csv","_list_all_trans.csv"))

        if not hasattr(self,'savenames'):
            self.get_savename(how='save')

        if avoid_TESS and np.any(all_trans['in_TESS']) and pre_post_TESS=="pre":
            t_end=2457000+np.min(all_trans.loc[all_trans['in_TESS'],"transit_mid_med"].values)-0.5
            outfilesuffix=outfilesuffix.replace('.csv',"_preTESS.csv")
        elif avoid_TESS and np.any(all_trans['in_TESS']) and pre_post_TESS=="post":
            t_start=2457000+np.max(all_trans.loc[all_trans['in_TESS'],"transit_mid_med"].values)+0.5
            outfilesuffix=outfilesuffix.replace('.csv',"_postTESS.csv")
        #print(Time(t_end,format='jd').isot,t_end,Time(t_start,format='jd').isot,t_start)
        out_tab=pd.DataFrame()
        if pl is None:
            searchpls=list(self.planets.keys())
        else:
            searchpls=[pl]

        for ipl in searchpls:
            vars+=['logprob_marg_'+ipl,'per_'+ipl,'ror_'+ipl,'tdur_'+ipl,'t0_'+ipl]
        ext=az.extract(self.trace.posterior,var_names=vars)
        
        for ipl in searchpls:
            if self.n_margs[ipl]>1:
                allprobs=np.exp(np.nanmedian(ext['logprob_marg_'+ipl],axis=0))
                allprobs/=np.sum(allprobs) #normalising
                allpers=np.arange(ext['per_'+ipl].shape[1])
            else:
                allprobs=np.array([1.0])
                allpers=np.array([0])
            assert observe_sigma is not None or observe_threshold is not None or max_ORs is not None, "Must use either observe_sigma, observe_threshold or max_ORs to set the number of aliases to observe"
            # 1) Observing some fraction of aliases which covers more probability that the observe_sigma fraction. i.e. 2-sigma = cover 95%
            sorted_probs = np.sort(allprobs)[::-1]
            if ipl in self.multis:
                observe_threshold=0
            else:
                if observe_sigma is not None and observe_threshold is None:
                    frac = stats.norm.cdf(observe_sigma)
                    #print(frac,np.cumsum(sorted_probs),np.searchsorted(np.cumsum(sorted_probs),frac)-1)
                    observe_threshold=sorted_probs[np.searchsorted(np.cumsum(sorted_probs),frac)-1]+1e-9
                # 2) Observing up to some maximum number of ORs (e.g. 14)
                if max_ORs is not None:
                    if observe_threshold is None or ipl in self.multis:
                        observe_threshold=0
                    if max_ORs<len(allprobs) and np.sum(sorted_probs>observe_threshold)>max_ORs and ipl not in self.multis:
                        observe_threshold=np.sort(allprobs)[::-1][max_ORs]
            
            depth=1e6*np.nanmedian(ext['ror_'+ipl])**2
            print("SNR for whole transit is: ",depth/self.cheops_RMS(gaiainfo['phot_g_mean_mag'], np.nanmedian(ext['tdur_'+ipl])))
            print("SNR for single orbit in/egress is: ",depth/self.cheops_RMS(gaiainfo['phot_g_mean_mag'], 0.5*98/1440))

            prio_1_prob_threshold = np.ceil(np.sum(allprobs>observe_threshold)*prio_1_threshold)
            prio_3_prob_threshold = np.ceil(np.sum(allprobs>observe_threshold)*(1-prio_3_threshold))
            print(allprobs,observe_threshold,allpers[allprobs>observe_threshold])
            for nper in allpers:
                #print(allpers,nper,allprobs[nper],observe_threshold)
                if allprobs[nper]>observe_threshold:
                    ser={}
                    iper=np.nanmedian(ext['per_'+ipl][:,nper]) if len(ext['per_'+ipl].shape)>1 else np.nanmedian(ext['per_'+ipl])
                    ser['ObsReqName']=self.id_dic[self.mission]+str(self.ID)+'_'+ipl+'_period'+str(np.round(iper,2)).replace('.',';')+'_prob'+str(allprobs[nper])[:4]
                    ser['Target']=self.id_dic[self.mission]+str(self.ID) if targetnamestring is None else targetnamestring
                    ser['_RAJ2000']=old_radec.ra.to_string(unit=u.hourangle, sep=':')
                    ser['_DEJ2000']=old_radec.dec.to_string(sep=':')
                    ser['pmra']=gaiainfo['pmra']
                    ser['pmdec']=gaiainfo['pmdec']
                    
                    ser['parallax']=gaiainfo['plx'] if 'plx' in gaiainfo else gaiainfo['parallax']
                    ser['SpTy']=SpTy
                    ser['Gmag']=gaiainfo['phot_g_mean_mag']
                    ser['dr2_g_mag']=gaiainfo['phot_g_mean_mag']
                    ser['e_Gmag']=1.09/gaiainfo['phot_g_mean_flux_over_error']
                    ser['e_dr2_g_mag']=1.09/gaiainfo['phot_g_mean_flux_over_error']

                    ser['Vmag']=V
                    ser['e_Vmag']=Verr

                    ser['Programme_ID']=prog_id
                    ser['BJD_early']=t_start
                    ser['BJD_late']=t_end
                    #Total observing time must cover duration, and either the full timing bound (i.e. assuming 3 sigma), or the oot_min_orbits (if the timing precision is better than the oot_min_orbits)

                    dur=np.nanpercentile(ext['tdur_'+ipl],[16,50,84])
                    n_trans_av = np.round(((0.5*(t_end+t_start)-self.lc.jd_base)-np.nanmedian(ext['t0_'+ipl]))/iper)
                    if len(ext['per_'+ipl].shape)>1:
                        i_timing_bounds = np.percentile(ext['t0_'+ipl]+n_trans_av*ext['per_'+ipl][:,nper],[100*(1-stats.norm.cdf(timing_sigma)), 50, 100*stats.norm.cdf(timing_sigma)])
                    else:
                        i_timing_bounds = np.percentile(ext['t0_'+ipl]+n_trans_av*ext['per_'+ipl],[100*(1-stats.norm.cdf(timing_sigma)), 50, 100*stats.norm.cdf(timing_sigma)])
                    timing_bounds = (i_timing_bounds[-1] - i_timing_bounds[0])*1440/98.7
                    dur_bounds = (dur[-1]-dur[0])*1440/98.7
                    if min_intrans_orbits is None:
                        #For normal (non-duotransits), we want to cover ALL of the intrans orbits
                        protected_T = dur[1]*1440/98.77 + np.clip(timing_bounds+dur_bounds,2*oot_min_orbits,100)
                        flexi_T = orbits_flex
                        ideal_T_visit = protected_T + flexi_T
                    else:
                        #We're not covering the whole transit, so we only need to cover half the timing/duration bounds
                        ideal_T_visit = min_intrans_orbits + np.clip(0.5*timing_bounds+0.5*dur_bounds,oot_min_orbits,100) + orbits_flex
                    #print(ideal_T_visit,min_orbits,max_orbits)
                    ser['T_visit']=np.clip(ideal_T_visit,min_orbits,max_orbits)# in orbits

                    #np.clip(*86400,(min_orbits*99.77*60), 2.5e5)
                    ser['N_Visits']=1

                    rank=np.argsort(allprobs)[::-1][nper]#list(np.sort(allprobs)).index(sprob)/len(allprobs)

                    if rank<prio_1_prob_threshold:
                        ser['Priority']=1
                    elif rank>prio_3_prob_threshold:
                        ser['Priority']=3
                    else:
                        ser['Priority']=2
                    #print(rank,allprobs[nper],ser['Priority'],prio_1_prob_threshold,prio_3_prob_threshold,np.argsort(allprobs)[::-1][nper])

                    if Texp is not None:
                        ser['Texp']=Texp
                    ser['MinEffDur']=min_eff
                    ser['Gaia_DR2']=str(DR2ID)
                    ser['BJD_0']=self.lc.jd_base+np.nanmedian(ext['t0_'+ipl])
                    ser['Period']=iper
                    #ser['T_visit']*0.5
                    
                    if min_intrans_orbits is None:
                        # Here we cannot cover the full transit or timing uncertainty - a DEEP DUO
                        # So, we need to position the visit duration such that it definitely catches ingress/egress even with timing/duration variation
                        # We also need at least an orbit in-transit
                        latest_before_t0=-0.5*protected_T
                        earliest_before_t0=-0.5*protected_T-orbits_flex
                        #print("DEEP",(1-ser['Ph_early'])*ser['Period']," to ",(1-ser['Ph_late'])*ser['Period'])
                    else:
                        #The latest we can possibly start the observation is half the duration minus min_pretrans_orbits plus the timing uncertainty before t0
                        earliest_before_t0 = -0.5*dur[1]*1440/98.77 + 0.5*timing_bounds + min_intrans_orbits - ser['T_visit']#-0.5*orbits_flex min_intrans_orbits
                        latest_before_t0 = 0.5*dur[1]*1440/98.77 - 0.5*timing_bounds - min_intrans_orbits
                    ser['Ph_late']=(((latest_before_t0*98.77/1440)/ser['Period'])+1)%1
                    ser['Ph_early']=(((earliest_before_t0*98.77/1440)/ser['Period'])+1)%1
                        #(((-0.5*(dur[1]+(timing_bounds+dur_bounds)*98.77/1440)-oot_min_orbits*98.77/1440)/ser['Period'])+1)%1
                        #The earliest we can possibly start is therefore this minus the actual visit length
                        #ser['Ph_late']
                        #ser['Ph_early']=((0.5*(dur[1]+(timing_bounds+dur_bounds)*98.77/1440)+oot_min_orbits*98.77/1440-ser['T_visit']*98.77/1440)/ser['Period'])+1
                        #print("NORMAL",earliest_before_t0,(1-ser['Ph_early'])*ser['Period']," to ",latest_before_t0,(1-ser['Ph_late'])*ser['Period'])

                    ser['Old_T_eff']=-99.
                    #ser["BegPh1"]=1-(row['mid']-row['start_latest'])/100
                    #ser["EndPh1"]=((row['end_earliest']-row['mid'])/100)
                    #ser["Effic1"]=50
                    ser['N_Ranges']=0
                    out_tab.loc[nper,list(ser.keys())]=pd.Series(ser,name=nper)
        out_tab['MinEffDur']=out_tab['MinEffDur'].values.astype(int)
        #print(98.77*60*out_tab['T_visit'].values)
        out_tab['T_visit']=(98.77*60*out_tab['T_visit'].values).astype(int) #in seconds
        #print(out_tab['T_visit'].values)
        out_tab['N_Ranges']=out_tab['N_Ranges'].values.astype(int)
        out_tab['N_Visits']=out_tab['N_Visits'].values.astype(int)
        out_tab['Priority']=out_tab['Priority'].values.astype(int)
       
        out_tab = out_tab.set_index(np.arange(len(out_tab)))
        out_tab.to_csv(self.savenames[0]+outfilesuffix)

        command="make_xml_files "+self.savenames[0]+outfilesuffix
        if Texp is None:
            command+=" --auto-expose"
        command+=" -f"
        print("Run the following command in a terminal to generate ORs:\n\""+command+"\"")
        return out_tab

    def get_lds(self,n_samples,mission='tess',how='2'):
        """Gets theoretical quadratic Limb Darkening parameters for any specified mission.
            This is done by first interpolating the theoretical samples (e.g. Claret) onto Teff and logg axes. FeH is typically fixed to the closest value.
            Then, using stellar samples from normally-distributed Teff and logg, a distribution of values for each LD parameter are retrieved.
            This can be performed for TESS, Kepler, CoRoT and CHEOPS bandpasses.

        Args:
            n_samples (int): Number of samples to generate
            mission (str, optional): [description]. Defaults to 'tess'.
            how (str, optional): [description]. Defaults to '2'.

        Returns:
            np.ndarray: Two-column array of quadratic limb darkening parameters (u1 and u2) as generated from the interpolated Normally-distributed Teff & logg
        """
        #
        Teff_samples = np.random.normal(self.Teff[0],np.average(abs(self.Teff[1:])),n_samples)
        logg_samples = np.random.normal(self.logg[0],np.average(abs(self.logg[1:])),n_samples)

        from scipy.interpolate import CloughTocher2DInterpolator as ct2d
        import pandas as pd

        if mission[0].lower()=="t":
            import pandas as pd
            from astropy.io import ascii
            TessLDs=ascii.read(os.path.join(MonoData_tablepath,'tessLDs.txt')).to_pandas()
            TessLDs=TessLDs.rename(columns={'col1':'logg','col2':'Teff','col3':'FeH','col4':'L/HP','col5':'a',
                                               'col6':'b','col7':'mu','col8':'chi2','col9':'Mod','col10':'scope'})
            if self.FeH!=0.0:
                #Finding nearest by FeH:
                unq_FeHs=pd.unique(TessLDs['FeH'])
                TessLDs=TessLDs.loc[TessLDs['FeH']==unq_FeHs[np.argmin(abs(self.FeH-unq_FeHs.astype(float)))]]

            a_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),
                                           TessLDs.logg.values.astype(float))),
                          TessLDs.a.values.astype(float))
            b_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),
                                           TessLDs.logg.values.astype(float))),
                          TessLDs.b.values.astype(float))

            outarr=np.column_stack((a_interp(np.clip(Teff_samples,2300,12000),np.clip(logg_samples,0,5)),
                                    b_interp(np.clip(Teff_samples,2300,12000),np.clip(logg_samples,0,5))))
            return outarr
        elif mission[0].lower()=="k":
            #Get Kepler Limb darkening coefficients.
            types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
            if how in types:
                checkint = types[how]
                #print(checkint)
            else:
                print("no key...")

            arr = np.genfromtxt(os.path.join(MonoData_tablepath,"KeplerLDlaws.txt"),skip_header=2)
            #Selecting FeH manually:
            feh_ix=arr[:,2]==np.unique(arr[:, 2])[np.argmin(self.FeH-np.unique(arr[:, 2]))]
            a_interp=ct2d(np.column_stack((arr[feh_ix,0],arr[feh_ix,1])),arr[feh_ix,4])
            b_interp=ct2d(np.column_stack((arr[feh_ix,0],arr[feh_ix,1])),arr[feh_ix,5])
            outarr=np.column_stack((a_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5)),
                                    b_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5))))
            return outarr
        elif mission.lower()=='corot':
            from astroquery.vizier import Vizier
            Vizier.ROW_LIMIT = -1
            arr = Vizier.get_catalogs('J/A+A/618/A20/COROTq')[0].to_pandas()
            a_interp=ct2d(np.column_stack((arr['Teff'],arr['logg'])),arr['a'])
            b_interp=ct2d(np.column_stack((arr['Teff'],arr['logg'])),arr['b'])
            outarr=np.column_stack((a_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5)),
                                    b_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5))))
            return outarr
        elif mission.lower()=='cheops':
            tab=pd.read_fwf(os.path.join(MonoData_tablepath,"Cheops_Quad_LDs_AllFeHs.txt"),header=None,widths=[5,7,5,5,9])
            tab=pd.DataFrame({'logg':tab.iloc[3::3,0].values.astype(float),'Teff':tab.iloc[3::3,1].values.astype(float),
                              'logZ':tab.iloc[3::3,2].values.astype(float),'vuturb':tab.iloc[3::3,3].values.astype(float),
                              'a':tab.iloc[3::3,4].values.astype(float),'b':tab.iloc[4::3,4].values.astype(float),
                              'CHI2':tab.iloc[5::3,4].values.astype(float)})
            #Sorting by metallicity:
            tab=tab[abs(self.FeH-tab['logZ'].values)<0.2]
            '''
            from astroquery.vizier import Vizier
            Vizier.ROW_LIMIT = -1
            arr = Vizier.get_catalogs('J/A+A/618/A20/COROTq')[0].to_pandas()
            '''
            a_interp=ct2d(np.column_stack((tab['Teff'],tab['logg'])),tab['a'])
            b_interp=ct2d(np.column_stack((tab['Teff'],tab['logg'])),tab['b'])
            outarr=np.column_stack((a_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5)),
                                    b_interp(np.clip(Teff_samples,3500,50000),np.clip(logg_samples,0,5))))
            return outarr

    def vals_to_latex(self, vals):
        """Function to turn percentiles (i.e. -1, 0, and +1 sigma values) into rounded latex strings for a table.
        This function identifies when errorbars are unequal (i.e. val ^ +sig1 _ -sig2), and when they are equal (val +/- sig)

        Args:
            vals (list): Values in the form [-1-sigma, median, +1-sigma]

        Returns:
            str: Latex string for a single parameter's value and errors
        """
        #
        try:
            roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))
            errs=[vals[2]-vals[1],vals[1]-vals[0]]
            if np.round(errs[0],roundval-1)==np.round(errs[1],roundval-1):
                #Errors effectively the same...
                if roundval<0:
                    return " $ "+str(int(np.round(vals[1],roundval)))+" \pm "+str(int(np.round(np.average(errs),roundval)))+" $ "
                else:
                    return " $ "+str(np.round(vals[1],roundval))+" \pm "+str(np.round(np.average(errs),roundval))+" $ "
            else:
                if roundval<0:
                    return " $ "+str(int(np.round(vals[1],roundval)))+"^{+"+str(int(np.round(errs[0],roundval)))+"}_{-"+str(int(np.round(errs[1],roundval)))+"} $ "
                else:
                    return " $ "+str(np.round(vals[1],roundval))+"^{+"+str(np.round(errs[0],roundval))+"}_{-"+str(np.round(errs[1],roundval))+"} $ "
        except:
            return " - "

    def to_latex_table(self,varnames='all',order='columns'):
        """Creating a Latex table for specfic parameters

        Args:
            varnames (str, optional): . Defaults to 'all'.
            order (str, optional): Whether to stack by 'columns' or 'rows'. Defaults to 'columns'.

        Returns:
            str: Multi-line latex table string.
        """
        #Plotting corner of the parameters to see correlations
        print("Making Latex Table")
        if not hasattr(self,'savenames'):
            self.get_savename(how='save')
        if not hasattr(self,'tracemask') or self.tracemask is None:
            self.tracemask=np.tile(True,len(mod.trace.posterior.chain)*len(mod.trace.posterior.draw))
        if varnames is None or varnames == 'all':
            varnames=[var for var in self.trace.posterior if var[-2:]!='__' and var not in ['gp_pred','light_curves']]

        self.samples = self.make_table(cols=varnames)
        self.samples = self.samples.loc[self.tracemask]
        facts={'r_pl':109.07637,'Ms':1.0,'rho':1.0,"t0":1.0,"period":1.0,"vrel":1.0,"tdur":24}
        units={'r_pl':"$ R_\\oplus $",'Ms':"$ M_\\odot $",'rho':"$ \\rho_\\odot $",
               "t0":"BJD-2458433","period":'d',"vrel":"$R_s/d$","tdur":"hours"}
        if order=="rows":
            #Table has header as a single row and data as a single row
            rowstring=str("ID")
            valstring=str(ID)
            for row in self.samples.columns:
                fact=[fact for fact in list(facts.keys()) if fact in row]
                if fact is not []:
                    rowstring+=' & '+str(row)+' ['+units[fact[0]]+']'
                    valstring+=' & '+vals_to_latex(np.percentile(facts[fact[0]]*self.samples[row],[16,50,84]))
                else:
                    rowstring+=' & '+str(row)
                    valstring+=' & '+vals_to_latex(np.percentile(self.samples[row],[16,50,84]))
            outstring=rowstring+"\n"+valstring
        else:
            #Table has header as a single column and data as a single column
            outstring="ID & "+str(ID)
            for row in self.samples.columns:
                fact=[fact for fact in list(facts.keys()) if fact in row]
                if len(fact)>0:
                    outstring+="\n"+row+' ['+units[fact[0]]+']'+" & "+vals_to_latex(np.percentile(facts[fact[0]]*self.samples[row],[16,50,84]))
                else:
                    outstring+="\n"+row+" & "+vals_to_latex(np.percentile(self.samples[row],[16,50,84]))
        with open(self.savenames[0]+'_table.txt','w') as file_to_write:
            file_to_write.write(outstring)
        #print("appending to file,",savename,"not yet supported")
        return outstring
