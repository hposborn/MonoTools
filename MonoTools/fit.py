import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.plotting import figure, output_file, save, curdoc, show
from bokeh.models import Band, Whisker, ColumnDataSource, Range1d
from bokeh.models.arrow_heads import TeeHead
from bokeh.layouts import gridplot, row, column, layout
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
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs

import eleanor

import pickle
import os.path
from datetime import datetime
import requests
import httplib2
from lxml import html

import glob

import warnings
warnings.filterwarnings("ignore")


MonoData_tablepath = os.path.join(os.path.dirname( __file__ ),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname( __file__ ),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')
if not os.path.isdir(MonoData_savepath):
    os.mkdir(MonoData_savepath)


from . import tools
from . import starpars
from . import search
#from . import tools
#from .stellar import starpars
#from . import MonoSearch


#creating new hidden directory for theano compilations:
theano_dir=MonoData_savepath+'/.theano_dir_'+str(np.random.randint(8))
if not os.path.isdir(theano_dir):
    os.mkdir(theano_dir)

theano_pars={'device':'cpu',
             'floatX':'float64',
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
        os.environ["THEANO_FLAGS"] = os.environ["THEANO_FLAGS"]+","+key+"="+theano_pars[key]

#setting float type:
floattype=np.float64

import theano.tensor as tt
import pymc3 as pm
import theano
theano.config.print_test_value = True
theano.config.exception_verbosity='high'

#print("theano config:",config)#['device'],config['floatX'],config['cxx'],config['compiledir'],config['base_compiledir'])


class monoModel():
    #The default monoModel class. This is what we will use to build a Pymc3 model
    
    def __init__(self, ID, mission, lc=None, rvs=None, planets=None, overwrite=False, savefileloc=None, **kwargs):
        #Initialising default modelling parameters. These can also be updated in init_model()
        self.overwrite=overwrite
        self.defaults={'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'use_GP':True,           # use_GP - bool - Fit for "second light" (
                       'train_GP':True,         # train_GP - bool - Fit for "second light" (
                       'constrain_LD':True,     # constrain_LD - bool - Use constrained LDs from model or unconstrained?
                       'ld_mult':3.,            # ld_mult - float - How much to multiply theoretical LD param uncertainties
                       'useL2':False,           # useL2 - bool - Fit for "second light" (i.e. a binary or planet+blend)
                       'FeH':0.0,               # FeH - float - Stellar FeH
                       'LoadFromFile':False,    # LoadFromFile - bool - Load previous model? 
                       'cutDistance':3.75,      # cutDistance - float - cut out points further than cutDistance*Tdur. 0.0 means no cutting
                       'maskdist': 0.666,        #Distance, in transit durations, from set transits, to "mask" as in-transit data when e.g. flattening.
                       'force_match_input':None,# force_match_input - Float/None add potential with this the sigma between the input and the output logror and logdur to force MCMC to match the input duration & maximise logror [e.g. 0.1 = match to 1-sigma=10%]
                       'debug':False,           # debug - bool - print debug statements?
                       'pred_all_time':False,   # pred_all_time - bool - use the GP to predict all times, or only near transits?
                       'fit_params':['logror','b','tdur', 't0'], # fit_params - list of strings - fit these parameters. Options: ['logror', 'b' or 'tdur', 'ecc', 'omega']

                       'marginal_params':['per','ecc','omega'], # marginal_params - list of strings - marginalise over these parameters. Options: ['per', 'b' ´or 'tdur', 'ecc', 'omega','logror']
                       'interpolate_v_prior':True, # Whether to use interpolation to produce transit velocity prior
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'per_index':-8/3,        # per_index - float - period prior index e.g. P^{index}. -8/3 in to Kipping 2018
                       'derive_K':True,         # If we have RVs, do we derive K for each alias or fit for a single K param
                       'use_multinest':False,   # use_multinest - bool - currently not supported
                       'use_pymc3':True,        # use_pymc3 - bool
                       'bin_oot':True}          # bin_oot - bool - Bin points outside the cutDistance to 30mins
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
        if self.LoadFromFile and not self.overwrite:
            #Catching the case where the file doesnt exist:
            success = self.LoadModelFromFile(loadfile=savefileloc)
            LoadFromFile = success
            
        #If we don;t have a past model to load, we load the lightcurve and, if a "planets" dict was passes, initialise those:
        if not self.LoadFromFile or self.overwrite:
            assert ID is not None and mission is not None and lc is not None
            if 'mask' not in lc:
                lc['mask']=np.tile(True,len(lc['time']))

            if (np.sort(lc['time'])!=lc['time']).all():
                if self.debug: print("#SORTING")
                for key in [key for key in lc if type(key)==np.ndarray and key!='time']:
                    if len(lc[key])==len(lc['time']):
                        lc[key]=lc[key][np.argsort(lc['time'])][:]
                lc['time']=np.sort(lc['time'])
            
            if rvs is not None:
                self.add_rvs(rvs)
            
            #Making sure all lightcurve arrays match theano.floatX:
            lc.update({key:lc[key].astype(floattype) for key in lc if type(lc[key])==np.ndarray and type(lc[key][0]) in [floattype,float,floattype]})
            
            lc['near_trans'] = np.tile(False, len(lc['time'])) if 'near_trans' not in lc else lc['near_trans']
            lc['in_trans']   = np.tile(False, len(lc['time'])) if 'in_trans' not in lc else lc['in_trans']
            
            self.lc=lc
            self.planets={};self.rvplanets={}
            self.multis=[];self.monos=[];self.duos=[]

            if planets is not None:
                for pl in planets:
                    add_planet(self, planets[pl]['orbit_flag'], planets[pl], pl)

            self.savefileloc=savefileloc
    
    def LoadModelFromFile(self, loadfile=None):
        if loadfile is None:
            self.GetSavename(how='load')
            loadfile=self.savenames[0]+'_model.pickle'
            if self.debug: print(self.savenames)
        if os.path.exists(loadfile):
            #Loading from pickled dictionary
            pick=pickle.load(open(loadfile,'rb'))
            assert not isinstance(pick, monoModel)
            #print(In this case, unpickle your object separately)
            for key in pick:
                setattr(self,key,pick[key])
            return True
        else:
            return False
    
    def SaveModelToFile(self, savefile=None, limit_size=False):
        if savefile is None:
            if not hasattr(self,'savenames'):
                self.GetSavename(how='save')
            savefile=self.savenames[0]+'_model.pickle'
            
        #Loading from pickled dictionary
        saving={}
        if limit_size and hasattr(self,'trace'):
            #We cannot afford to store full arrays of GP predictions and transit models
            # But first we need to turn the predicted arrays into percentiles now for plotting:
            if self.use_GP:
                self.init_gp_to_plot()
            self.init_trans_to_plot()
            
            #And let's clip gp and lightcurves and pseudo-variables from the trace:
            medvars=[var for var in self.trace.varnames if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var and '__' not in var]
            for key in medvars:
                #Permanently deleting these values from the trace.
                self.trace.remove_values(key)
            #medvars=[var for var in self.trace.varnames if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
        n_bytes = 2**31
        max_bytes = 2**31 - 1
        
        bytes_out = pickle.dumps(self.__dict__)
        #bytes_out = pickle.dumps(self)
        with open(savefile, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
        del saving
        #pick=pickle.dump(self.__dict__,open(loadfile,'wb'))
            
    def drop_planet(self, name):
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

    def add_planet(self, pltype, pl_dic, name):
        # A flexible input function to the model, which is able to fill in gaps.
        # pl_dic needs: depth, tdur, tcen, period (if multi), tcen_2 (if duo)
        #Making sure we have the necessary info:
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
            pl_dic['r_pl']=pl_dic['ror']*self.Rstar[0]*11.2
        
        #Adding dict as planet:
        if pltype=='multi':
            self.add_multi(pl_dic, name)
        elif pltype=='rvplanet':
            self.add_rvplanet(pl_dic, name)
        else:
            if 'period_err' not in pl_dic:
                pl_dic['period_err']=999
            if pltype=='duo':
                if 'period' not in pl_dic:
                    pl_dic['period']=abs(pl_dic['tcen_2']-pl_dic['tcen'])
                self.add_duo(pl_dic, name)
            elif pltype=='mono':
                if 'period' not in pl_dic:
                    pl_dic['period']=999
                self.add_mono(pl_dic, name)
    
    def add_rvplanet(self, pl_dic, name):
        assert name not in list(self.planets.keys())+list(self.rvplanets.keys())
        # Adds non-transiting planet seen only in RVs. 
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
        
    def add_multi(self, pl_dic, name):
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
        phase=(self.lc['time']-pl_dic['tcen']-0.5*pl_dic['period'])%pl_dic['period']-0.5*pl_dic['period']
        self.lc['near_trans'] += abs(phase)<self.cutDistance*pl_dic['tdur']
        self.lc['in_trans']   += abs(phase)<self.maskdist*pl_dic['tdur']
        
        self.planets[name]=pl_dic
        self.multis+=[name]
        
    def add_mono(self, pl_dic, name):
        #Adds planet with single eclipses
        
        #Adding the planet to the lightcurve mask arrays first (as compute_period_gaps performs flattening).
        self.lc['near_trans'] += abs(self.lc['time']-pl_dic['tcen'])<self.cutDistance*pl_dic['tdur']
        self.lc['in_trans']   += abs(self.lc['time']-pl_dic['tcen'])<self.maskdist*pl_dic['tdur']
        
        #Calculating whether there are period gaps:
        assert name not in self.planets
        p_gaps,rms_series=self.compute_period_gaps(pl_dic['tcen'],tdur=pl_dic['tdur'],depth=pl_dic['depth'])
        pl_dic['per_gaps']={'gap_starts':p_gaps[:,0],'gap_ends':p_gaps[:,1],
                           'gap_widths':p_gaps[:,1]-p_gaps[:,0],'gap_probs':-5/3*(p_gaps[:,1]**(-5/3)-p_gaps[:,0]**(-5/3))}
        pl_dic['per_gaps']['gap_probs']/=np.sum(pl_dic['per_gaps']['gap_probs'])
        pl_dic['P_min']=p_gaps[0,0]
        pl_dic['rms_series']=rms_series
        if 'log_ror' not in pl_dic:
            if 'ror' in pl_dic:
                pl_dic['log_ror']=np.log(pl_dic['ror'])
            elif 'depth' in pl_dic:
                assert pl_dic['depth']<0.25 #Depth must be a ratio (not in mmags)
                pl_dic['ror']=pl_dic['depth']**0.5
                pl_dic['log_ror']=np.log(pl_dic['ror'])
        pl_dic['ngaps']=len(p_gaps)
        
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
        
    def compute_rms_series(self,tdur,split_gap_size=2.0,n_steps_per_dur=7):
        # Computing an RMS time series for the lightcurve by binning
        # split_gap_size = Duration at which to cut the lightcurve and compute in loops
        # n_steps_per_dur = number of steps with which to cut up each duration. Odd numbers work most uniformly
        
        if not hasattr(self.lc,'flux_flat') or len(self.lc['flux_flat'])!=len(self.lc['flux_err']):
            if len(self.planets)>0:
                self.lc = tools.lcFlatten(self.lc,transit_mask=~self.lc['in_trans'],
                                          stepsize=0.133*np.min([self.planets[pl]['tdur'] for pl in self.planets]),
                                          winsize=6.5*np.max([self.planets[pl]['tdur'] for pl in self.planets]))
            else:
                self.lc = tools.lcFlatten(self.lc,transit_mask=~self.lc['in_trans'],
                                          stepsize=0.04,
                                          winsize=1.5)

        rms_series=np.zeros((len(self.lc['time'])))
        binsize=(1/n_steps_per_dur)*tdur
        if np.nanmax(np.diff(self.lc['time']))>split_gap_size:
            loop_blocks=np.array_split(np.arange(len(self.lc['time'])),np.where(np.diff(self.lc['time'])>split_gap_size)[0])
        else:
            loop_blocks=[np.arange(len(self.lc['time']))]
        rms_series_sh=[]
        bins=[]
        for sh_time in loop_blocks:
            thesebins=np.arange(np.nanmin(self.lc['time'][sh_time])-tdur,
                                np.nanmax(self.lc['time'][sh_time])+tdur+binsize, binsize)
            theserms=np.zeros_like(thesebins)
            for n,b in enumerate(thesebins):
                ix=(abs(b-self.lc['time'][sh_time])<(0.5*tdur))*self.lc['mask'][sh_time]
                if np.sum(ix)>1:
                    theserms[n]=tools.weighted_avg_and_std(self.lc['flux_flat'][sh_time][ix],
                                                           self.lc['flux_err'][sh_time][ix])[1]
                else:
                    theserms[n]=np.nan
            bins+=[thesebins]
            rms_series_sh+=[np.array(theserms)]

            '''
            lc_segment=np.column_stack((self.lc['time'][sh_time],self.lc['flux_flat'][sh_time],
                                        self.lc['flux_err'][sh_time],self.lc['mask'][sh_time].astype(int)))
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

    def compute_period_gaps(self,tcen,tdur,depth,max_per=8000,SNR_thresh=4):
        # Given the time array, the t0 of transit, and the fact that another transit is not observed, 
        #   we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
        # In this case, a list of periods is returned, with all points within 0.5dur to be cut
        
        rmsseries = self.compute_rms_series(tdur)
        
        dist_from_t0=abs(tcen-rmsseries[:,0])
        
        #Here we only want the parts of the timeseries where we could have detected a transit (if there was one):
        dist_from_t0=dist_from_t0[(((depth/self.lc['flux_unit'])/rmsseries[:,1])>SNR_thresh)*(~np.isnan(rmsseries[:,1]))]
        dist_from_t0=np.sort(dist_from_t0)
        gaps=np.where(np.diff(dist_from_t0)>(0.9*tdur))[0]
        if len(gaps)>0:
            #Looping from minimum distance from transit to gap, to maximum distance from transit to end-of-lc
            checkpers=np.arange(dist_from_t0[gaps[0]]-tdur,np.max(dist_from_t0)+tdur,tdur*0.166)
            checkpers_ix=self.CheckPeriodsHaveGaps(checkpers,tdur,tcen).astype(int) #Seeing if each period has data coverage
            
            #Creating an array of tuples which form start->end of specific gaps:
            starts=checkpers[:-1][np.diff(checkpers_ix)==1.0]
            #Because the above array ends beyond the max lc extent, we need to add the max period to this array:
            ends=np.hstack((checkpers[1:][np.diff(checkpers_ix)==-1.0],max_per))
            #print(starts,ends)
            gap_start_ends=np.array([(starts[n],ends[n]) for n in range(len(starts))])
        else:
            gap_start_ends=np.array([(np.max(dist_from_t0),max_per)])
        return gap_start_ends,rmsseries
    
    def CheckPeriodsHaveGaps(self,pers,tdur,tcen,tcen_2=None,coverage_thresh=0.15):
        #Looping through potential periods and counting points in-transit
        trans=abs(self.lc['time'][self.lc['mask']]-tcen)<0.45*tdur
        if np.sum(trans)==0:
            trans=abs(self.lc['time'][self.lc['mask']]-tcen)<0.5*tdur
        if self.debug: print(np.sum(trans),"points in transit")
        #Adding up in-transit cadences to give days in transit:
        days_in_known_transits = np.sum(np.array([cad[1:] for cad in self.lc['cadence'][self.lc['mask']][trans]]).astype(float))/1440
        if tcen_2 is not None:
            trans2=abs(self.lc['time'][self.lc['mask']]-tcen_2)<0.45*tdur
            days_in_known_transits += np.sum(np.array([cad[1:] for cad in self.lc['cadence'][self.lc['mask']][trans2]]).astype(float))/1440
            coverage_thresh*=0.5 #Two transits already in number count, so to compensate we must decrease the thresh
            
        check_pers_ix=[]
        #Looping through periods
        for per in pers:
            phase=(self.lc['time'][self.lc['mask']]-tcen-per*0.5)%per-per*0.5
            intr=abs(phase)<0.45*tdur
            #Here we need to add up the cadences in transit (and not simply count the points) to check coverage:
            days_in_tr=np.sum([float(self.lc['cadence'][ncad][1:]) for ncad in np.arange(len(self.lc['cadence']))[self.lc['mask']][intr]])/1440.
            check_pers_ix+=[days_in_tr<(1.0+coverage_thresh)*days_in_known_transits]
            #Less than 15% of another eclipse is covered
        return np.array(check_pers_ix)

    def compute_duo_period_aliases(self,duo,dur=0.5):
        # Given the time array, the t0 of transit, and the fact that two transits are observed, 
        #   we want to calculate a distribution of impossible periods to remove from the period alias list
        #finding the longest unbroken observation for P_min guess
        #P_min = np.max(np.hstack((self.compute_period_gaps(duo['tcen'],dur=duo['tdur']),
        #                          self.compute_period_gaps(duo['tcen_2'],dur=duo['tdur']))))
        #print(P_min,np.ceil(duo['period']/P_min),np.ceil(duo['period']/P_min))
        check_pers_ints = np.arange(1,np.ceil(duo['period']/10),1.0)
        check_pers_ix = self.CheckPeriodsHaveGaps(duo['period']/check_pers_ints,duo['tdur'],duo['tcen'],tcen_2=duo['tcen_2'])
        
        duo['period_int_aliases']=check_pers_ints[check_pers_ix]
        if duo['period_int_aliases']==[]:
            print("problem in computing Duotransit aliases")
        else:
            duo['period_aliases']=duo['period']/duo['period_int_aliases']
            duo['P_min']=np.min(duo['period_aliases'])
        return duo

    def calc_gap_edge_likelihoods(self,mono,n_check=100):
        # In the case that we are not creating transit models for each period gap, we need to calculate how the "edges" of those gaps affect the log probability.
        # Effectively we'll calculate the likelihood of the edges of the gaps w.r.t the initial-fit transit model
        # This will then become a 1D (e.g. linear) polynomial which sets the logprior at the edges of each monotransit gap.
        from scipy import interpolate as interp
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
                (abs((self.lc['time']-self.planets[mono]['tcen']-0.5*pmin)%pmin-0.5*pmin)<5*self.planets[mono]['tdur']) + \
                (abs((self.lc['time']-self.planets[mono]['tcen']-0.5*pmax)%pmax-0.5*pmax)<5*self.planets[mono]['tdur'])) * \
                       (abs(self.lc['time']-self.planets[mono]['tcen'])>4*self.planets[mono]['tdur'])
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

                bfmodel=interp.interp1d(np.hstack((-1000,tzoom-self.planets[mono]['tcen'],1000)),
                                        np.hstack((0.0,light_curve.T[0],0.0)))

            sigma2 = lc['flux_err'][round_gaps] ** 2
            pers=np.hstack((start_gaps,end_gaps,mid_gap))[None,:]
            phases=(self.lc['time'][round_gaps,None]-self.planets[mono]['t0']-0.5*pers)%pers-0.5*pers
            #Calculating delta logliks (where the final "mid_gap" should be the furthest point from data, i.e. max loglik
            logliks=-0.5 * np.sum((self.lc['flux_flat'][round_gaps,None]*self.lc['flux_unit'] - bfmodel(phases)) ** 2 / sigma2[:,None] + np.log(sigma2[:,None]),axis=0)
            logliks-=logliks[-1]
            #Adding the polynomial fits to the 'per_gaps' dict:
            starts+=[np.polyfit(start_gaps[logliks[:int(n_check*0.5)]<0]-pmin,logliks[:int(n_check*0.5)][logliks[:int(n_check*0.5)]<0],1)]
            if ngap<(self.planets[mono]['ngaps']-1):
                ends+=[np.polyfit(end_gaps[logliks[int(n_check*0.5):-1]<0]-pmax,logliks[int(n_check*0.5):-1][logliks[int(n_check*0.5):-1]<0],1)]
            else:
                ends+=[np.array([0.0,0.0])]
        self.planets[mono]['per_gaps']['start_loglik_polyvals']=np.vstack(starts)
        self.planets[mono]['per_gaps']['end_loglik_polyvals']=np.vstack(ends)

    def add_duo(self, pl_dic,name):
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
        pl_dic=self.compute_duo_period_aliases(pl_dic)
        pl_dic['npers']=len(pl_dic['period_int_aliases'])
        
        pl_dic['ror']=np.sqrt(pl_dic['depth']) if not hasattr(pl_dic,'ror') else 0.01
        
        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            
            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+pl_dic['ror'])**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*np.median(pl_dic['period_aliases'])*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        
        for per in pl_dic['period_aliases']:
            phase=(self.lc['time']-pl_dic['tcen']-0.5*per)%per-0.5*per
            self.lc['near_trans']+=abs(phase)<self.cutDistance*pl_dic['tdur']
        self.lc['in_trans']+=abs(self.lc['time']-pl_dic['tcen'])<self.maskdist*pl_dic['tdur']
        self.lc['in_trans']+=abs(self.lc['time']-pl_dic['tcen_2'])<self.maskdist*pl_dic['tdur']

        self.planets[name]=pl_dic
        self.duos+=[name]
    
    def init_starpars(self,Rstar=np.array([1.0,0.08,0.08]),
                      Teff=np.array([5227,100,100]),
                      logg=np.array([4.3,1.0,1.0]),
                      FeH=0.0,rhostar=None,Mstar=None):
        #Adds stellar parameters to model. Arrays are [value, neg_err, pos_err]
        #All parameters (except Teff) are *relative to the Sun*
        self.Rstar=np.array(Rstar)
        self.Teff=np.array(Teff)
        self.logg=np.array(logg)
        self.FeH=FeH
        
        if Mstar is not None:
            self.Mstar = Mstar if type(Mstar)==float else Mstar[0]
        #Here we only have a mass, radius, logg- Calculating rho two ways (M/R^3 & logg/R), and doing weighted average
        if rhostar is None:
            rho_logg=[np.power(10,self.logg[0]-4.43)/self.Rstar[0]]
            rho_logg+=[np.power(10,self.logg[0]+self.logg[1]-4.43)/(self.Rstar[0]-self.Rstar[1])/rho_logg[0]-1.0,
                       1.0-np.power(10,self.logg[0]-self.logg[2]-4.43)/(self.Rstar[0]+self.Rstar[2])/rho_logg[0]]
            if Mstar is not None:
                rho_MR=[Mstar[0]/self.Rstar[0]**3]
                rho_MR+=[(Mstar[0]+Mstar[1])/(self.Rstar[0]-abs(self.Rstar[1]))**3/rho_MR[0]-1.0,
                         1.0-(Mstar[0]-abs(Mstar[2]))/(self.Rstar[0]+self.Rstar[2])**3/rho_MR[0]]
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
                
            self.rhostar=np.array(rhostar)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3
        else:
            self.rhostar=np.array(rhostar)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3

    
    def GetSavename(self, how='load',overwrite=None):
        '''
        # Get unique savename (defaults to MCMC suffic) with format:
        # [savefileloc]/[T/K]IC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
        #
        # INPUTS:
        # - ID
        # - mission - (TESS/K2/Kepler)
        # - how : 'load' or 'save'
        # - overwrite : if 'save', whether to overwrite past save or not.
        # - savefileloc : file location of files to save (default: 'MonoTools/[T/K]ID[11-number ID]/
        #
        # OUTPUTS:
        # - filepath
        '''
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
        # Add a dictionary of rvs with arrays of: 
        # necessary: "time", "rv", "rv_err"
        # optional: e.g.: "rv_unit" (assumes m/s), "tele_index" (unique telescope id for each RV), "jd_base" (assumes same as lc)
        #
        # PLEASE NOTE - Due to the fact that, unlike transits, the RVs of each planet affect *all* observed RV data
        # It is not yet possible to isolate individual planet contributions (as it is with transits) and treat seperately
        # Therefore marginalising over multiple planets with ambiguous periods is not yet possible.
        # However, this should work for multi-transiting planets (known periods) with single outer companions.
        #
        # TBD - activity detrending, multiple RV sources, trends, etc
        
        if 'jd_base' in rv_dic and rv_dic['jd_base']!=self.lc['jd_base']:
            rv_dic['time']+=(rv_dic['jd_base']-self.lc['jd_base'])
            rv_dic['jd_base']=self.lc['jd_base']
        if rv_dic['rv_unit']=='kms' or rv_dic['rv_unit']==1000:
            #We want 
            rv_dic['rv']*=1000
            rv_dic['rv_err']*=1000
            rv_dic['rv_unit']='ms'
        elif 'rv_unit' not in rv_dic or rv_dic['rv_unit']!='ms':
            print("Assuming RV unit is in m/s")
        if 'tele_index' not in rv_dic or (rv_dic['tele_index'])!=len(rv_dic['time']):
            print("Assuming all one telescope (HARPS).")
            rv_dic['tele_index']=np.tile('h',len(rv_dic['time']))
        rv_dic['scopes'] = np.unique(rv_dic['tele_index'])
        #Building an array of ones and zeros to use later
        rv_dic['tele_index_arr']=np.zeros((len(rv_dic['time']),len(rv_dic['scopes'])))
        for ns in range(len(rv_dic['scopes'])):
            rv_dic['tele_index_arr'][:,ns]+=(rv_dic['tele_index']==rv_dic['scopes'][ns]).astype(int)
        
        self.rvs={}
        for key in rv_dic:
            if type(rv_dic)==np.ndarray and type(rv_dic)[0] in [float,np.float32,np.float64]:
                self.rvs[key]=rv_dic[key][:].astype(floattype)
            else:
                self.rvs[key]=rv_dic[key]
        self.rv_tref = np.round(np.nanmedian(rv_dic['time']),-1)
        #Setting polynomial trend
        self.rv_npoly=n_poly_trend
        
        assert len(self.duos+self.monos)<2 #Cannot fit more than one planet with uncertain orbits with RVs (currently)
    
    def init_lc(self, **kwargs):
        #Initialising LC. This can be done either after or before model initialisation.
        
        step=0.133*np.min([self.init_soln['tdur_'+pl] for pl in self.planets]) if hasattr(self,'init_soln') else 0.133*np.min([self.planets[pl]['tdur'] for pl in self.planets])
        win=6.5*np.max([self.init_soln['tdur_'+pl] for pl in self.planets]) if hasattr(self,'init_soln') else 6.5*np.max([self.planets[pl]['tdur'] for pl in self.planets])

        self.lc['near_trans'] = np.tile(False, len(self.lc['time']))
        self.lc['in_trans']   = np.tile(False, len(self.lc['time']))
        for pl in self.planets:
            if pl in self.multis:
                t0 = self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                p = self.init_soln['per_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['period']
                phase=(self.lc['time']-t0-0.5*p)%p-0.5*p
            elif pl in self.duos:
                t0= self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                p=abs(self.init_soln['t0_2_'+pl]-self.init_soln['t0_'+pl]) if hasattr(self,'init_soln') else abs(self.planets[pl]['tcen_2']-self.planets[pl]['tcen'])
                phase=(self.lc['time']-t0-0.5*p)%p-0.5*p
            elif pl in self.monos:
                t0= self.init_soln['t0_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tcen']
                phase=abs(self.lc['time']-t0)
            dur = self.init_soln['tdur_'+pl] if hasattr(self,'init_soln') else self.planets[pl]['tdur']
            self.lc['near_trans'] += abs(phase)<self.cutDistance*dur
            self.lc['in_trans']   += abs(phase)<self.maskdist*dur
        
        self.lc=tools.lcFlatten(self.lc,transit_mask=~self.lc['in_trans'], stepsize=step, winsize=win)

        if self.cutDistance>0 or not self.use_GP:
            if self.bin_oot:
                #Creating a pseudo-binned dataset where out-of-transit LC is binned to 30mins but near-transit is not.
                oot_binsize=1/12 if self.mission.lower()=='kepler' else 1/48
                oot_binlc=tools.lcBin(self.lc,use_flat=~self.use_GP,extramask=~self.lc['near_trans'],
                                      modify_lc=False,binsize=oot_binsize)
                oot_binlc['mask']=np.tile(True,len(oot_binlc['time']))
                oot_binlc['in_trans']=np.tile(False,len(oot_binlc['time']))
                oot_binlc['near_trans']=np.tile(False,len(oot_binlc['time']))
                self.pseudo_binlc={}
                for key in oot_binlc:
                    lckey = key.replace("bin_","") if 'bin_' in key else key
                    lckey = 'flux_err' if lckey=='flux_flat_err' else lckey
                    #print(key,lckey,oot_binlc[key],self.lc[lckey])
                    self.pseudo_binlc[lckey]=np.hstack((oot_binlc[key],self.lc[lckey][self.lc['near_trans']]))
                for key in [key for key in self.pseudo_binlc.keys() if key!='time']:
                    if 'bin_' in key:
                        _=self.pseudo_binlc.pop(key)
                    else:
                        if self.debug: print(key,len(self.pseudo_binlc[key]),len(np.argsort(self.pseudo_binlc['time'])))
                        self.pseudo_binlc[key]=self.pseudo_binlc[key][np.argsort(self.pseudo_binlc['time'])]
                self.pseudo_binlc['time']=np.sort(self.pseudo_binlc['time'])
                self.pseudo_binlc['flux_unit']=self.lc['flux_unit']
            elif not self.bin_oot:
                self.lc_near_trans={}
                lc_len=len(self.lc['time'])
                for key in self.lc:
                    if type(self.lc[key])==np.ndarray and len(self.lc[key])==lc_len:
                        self.lc_near_trans[key]=self.lc[key][self.lc['near_trans']]
                    else:
                        self.lc_near_trans[key]=self.lc[key]
                
            if self.debug: print(np.sum(self.lc['near_trans']&self.lc['mask']),"points in new lightcurve, compared to ",np.sum(self.lc['mask'])," in original mask, leaving ",np.sum(self.lc['near_trans']),"points in the lc")        

        
    def init_model(self, overwrite=False, **kwargs):
        # lc - dictionary with arrays:
        #   -  'time' - array of times, (x)
        #   -  'flux' - array of flux measurements (y)
        #   -  'flux_err'  - flux measurement errors (yerr)        

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
        assert not (self.assume_circ and self.interpolate_v_prior) #Cannot interpolate_v_prior and assume circular.
        assert not ((len(self.duos+self.monos)>1)*hasattr(self,'rvs')) #Cannot fit more than one planet with uncertain orbits with RVs (currently)

        n_pl=len(self.planets)
        assert n_pl>0

        if self.debug: print(len(self.planets),'planets |','monos:',self.monos,'multis:',self.multis,'duos:',self.duos, "use GP=",self.use_GP)

        ######################################
        #   Masking out-of-transit flux:
        ######################################
        # To speed up computation, here we loop through each planet and add the region around each transit to the data to keep
        
        self.init_lc()
        ######################################
        #   Creating flux & telescope index func:
        ######################################
        self.cads=np.unique(self.lc['cadence'])
        #In the case of different cadence/missions, we need to separate their respective errors to fit two logs2
        self.lc['flux_err_index']=np.column_stack([np.where(self.lc['cadence']==cad,1.0,0.0) for cad in self.cads])
        if self.bin_oot:
            self.pseudo_binlc['flux_err_index']=np.column_stack([np.where(self.pseudo_binlc['cadence']==cad,1.0,0.0) for cad in self.cads])
        else:
            self.lc_near_trans['flux_err_index']=self.lc['flux_err_index'][self.lc['near_trans']]

        if not hasattr(self.lc,'tele_index'):
            #Here we're making an index for which telescope (kepler vs tess) did the observations,
            # then we multiply the output n_time array by the n_time x 2 index and sum along the 2nd axis
            
            self.lc['tele_index']=np.zeros((len(self.lc['time']),4))
            if self.bin_oot:
                self.pseudo_binlc['tele_index']=np.zeros((len(self.pseudo_binlc['time']),4))
            else:
                self.lc_near_trans['tele_index']=np.zeros((len(self.lc_near_trans['time']),4))
            for ncad in range(len(self.cads)):
                if self.cads[ncad][0].lower()=='t':
                    self.lc['tele_index'][:,0]+=self.lc['flux_err_index'][:,ncad]
                    if self.debug: print(self.bin_oot,'making tele_index')
                    if self.bin_oot:
                        self.pseudo_binlc['tele_index'][:,0]+=self.pseudo_binlc['flux_err_index'][:,ncad]
                    else:
                        self.lc_near_trans['tele_index'][:,0]+=self.lc_near_trans['flux_err_index'][:,ncad]
                elif self.cads[ncad][0].lower()=='k':
                    self.lc['tele_index'][:,1]+=self.lc['flux_err_index'][:,ncad]
                    if self.bin_oot:
                        self.pseudo_binlc['tele_index'][:,1]+=self.pseudo_binlc['flux_err_index'][:,ncad]
                    else:
                        self.lc_near_trans['tele_index'][:,1]+=self.lc_near_trans['flux_err_index'][:,ncad]
                elif self.cads[ncad][0].lower()=='c':
                    self.lc['tele_index'][:,2]+=self.lc['flux_err_index'][:,ncad]
                    if self.bin_oot:
                        self.pseudo_binlc['tele_index'][:,2]+=self.pseudo_binlc['flux_err_index'][:,ncad]
                    else:
                        self.lc_near_trans['tele_index'][:,2]+=self.lc_near_trans['flux_err_index'][:,ncad]
                elif self.cads[ncad][0].lower()=='x':
                    self.lc['tele_index'][:,3]+=self.lc['flux_err_index'][:,ncad]
                    if self.bin_oot:
                        self.pseudo_binlc['tele_index'][:,3]+=self.pseudo_binlc['flux_err_index'][:,ncad]
                    else:
                        self.lc_near_trans['tele_index'][:,3]+=self.lc_near_trans['flux_err_index'][:,ncad]

        if self.use_GP:
            self.gp={}
            if self.train_GP and not hasattr(self,'gp_init_trace'):
                self.GP_training()
        
        ######################################
        #   Initialising sampling models:
        ######################################

        if self.use_pymc3:
            self.init_pymc3()
        elif self.use_multinest:
            self.run_multinest(**kwargs)
        
    def GP_training(self,n_draws=900,max_len_lc=25000,uselc=None):
        print("initialising and training the GP")
        if uselc is None:
            uselc=self.lc
            
        with pm.Model() as gp_train_model:
            #####################################################
            #     Training GP kernel on out-of-transit data
            #####################################################
            mean=pm.Normal("mean",mu=np.median(uselc['flux'][uselc['mask']]),
                                  sd=np.std(uselc['flux'][uselc['mask']]))

            self.log_flux_std=np.array([np.log(np.nanstd(uselc['flux'][~uselc['in_trans']][uselc['cadence'][~uselc['in_trans']]==c])) for c in self.cads]).ravel().astype(floattype)
            if self.debug: print(self.log_flux_std)
            print(np.sum(~uselc['in_trans']),len(~uselc['in_trans']))
            if self.debug: print(np.unique(uselc['cadence'][~uselc['in_trans']]))
            if self.debug: print(self.cads,np.unique(uselc['cadence']),uselc['time'][uselc['cadence']=='t'],
                  self.log_flux_std,np.sum(~uselc['in_trans']))

            logs2 = pm.Normal("logs2", mu = self.log_flux_std+1, 
                              sd = np.tile(1.0,len(self.log_flux_std)), shape=len(self.log_flux_std))

            # Transit jitter & GP parameters
            #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
            lcrange=uselc['time'][-1]-uselc['time'][0]
            max_cad = np.nanmax([np.nanmedian(np.diff(uselc['time'][uselc['near_trans']&(uselc['cadence']==c)])) for c in self.cads])
            av_dur = np.nanmean([self.planets[pl]['tdur'] for pl in self.planets])
            #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
            success=False;target=0.05
            while not success and target<0.21:
                try:
                    low=(2*np.pi)/(lcrange/(target/0.01))
                    up=(2*np.pi)/(av_dur*(target/0.01))
                    w0 = pm.InverseGamma("w0",testval=(2*np.pi)/10,**xo.estimate_inverse_gamma_parameters(lower=low,upper=up))
                    success=True
                except:
                    low=(2*np.pi)/(10)
                    up=(2*np.pi)/(6*max_cad)
                    target*=1.15
                    success=False
            if self.debug: print(success, target,lcrange,low,up)
            maxpower=1.0*np.nanstd(uselc['flux'][(uselc['in_trans'])&uselc['mask']])
            minpower=0.02*np.nanmedian(abs(np.diff(uselc['flux'][(uselc['in_trans'])&uselc['mask']])))
            if self.debug: print(np.nanmedian(abs(np.diff(uselc['flux'][uselc['near_trans']]))),np.nanstd(uselc['flux'][uselc['near_trans']]),minpower,maxpower)
            success=False;target=0.01
            while not success and target<0.2:
                try:
                    power = pm.InverseGamma("power",testval=minpower*5,
                                            **xo.estimate_inverse_gamma_parameters(lower=minpower,
                                                                                   upper=maxpower/(target/0.01),
                                                                                   target=0.1))
                    success=True
                except:
                    target*=1.15
                    success=False
            if self.debug: print(success, target)
            S0 = pm.Deterministic("S0", power/(w0**4))

            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))

            if len(uselc['time']>max_len_lc):
                mask=(uselc['in_trans'])*(np.arange(0,len(uselc['time']),1)<max_len_lc)
            else:
                mask=uselc['in_trans']

            self.gp['train'] = xo.gp.GP(kernel, uselc['time'][uselc['in_trans']].astype(floattype),
                                   uselc['flux_err'][uselc['in_trans']].astype(floattype)**2 + \
                                   tt.dot(uselc['flux_err_index'][uselc['in_trans']].astype(floattype),tt.exp(logs2)),
                                   J=2)

            self.gp['train'].log_likelihood(uselc['flux'][uselc['in_trans']].astype(floattype)-mean)

            self.gp_init_soln = xo.optimize(start=None, vars=[logs2, power, w0, mean],verbose=True)

            if self.debug: print("sampling init GP",int(n_draws*0.66),"times with",len(uselc['flux'][uselc['in_trans']]),"-point lightcurve") 

            self.gp_init_trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=self.gp_init_soln, chains=2,
                                           step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

    def init_interpolated_Mp_prior(self):
        MRarray=np.genfromtxt(os.path.join(MonoData_tablepath,"LogMePriorFromRe.txt"))
        self.interpolated_mu = xo.interp.RegularGridInterpolator([np.hstack((0.01,MRarray[:,0])).astype(np.float64)],
                                                            np.hstack((np.log(0.1),MRarray[:,1])).astype(np.float64)[:, None])
        self.interpolated_sigma = xo.interp.RegularGridInterpolator([np.hstack((0.01,MRarray[:,0])).astype(np.float64)],
                                                               np.hstack((1.25,MRarray[:,2])).astype(np.float64)[:, None])
    
    def init_interpolated_v_prior(self):
        #Initialising the interpolated functions for log prob vs log velocity and marginalised eccentricity vs log velocity
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
                                                                                       emarg_arr[1:,1:]))[:,:,None],
                                                                      nout=1)
        f_logprob=gzip.open(os.path.join(MonoData_tablepath,
                                         "logprob_array_"+interp_locs[self.ecc_prior.lower()]+".txt.gz"),"rb")
        logprob_arr=np.genfromtxt(io.BytesIO(f_logprob.read()))
        
        np.genfromtxt(os.path.join(MonoData_tablepath,"logprob_array_"+interp_locs[self.ecc_prior.lower()]+".txt"))
        #logprob_arr = pd.read_csv(os.path.join(MonoData_tablepath,"logprob_array_"+interp_locs[self.ecc_prior.lower()]+".csv"),
        #                          index_col=0)
        #Rewriting infinite or nan values as <-300:
        self.interpolator_logprob = xo.interp.RegularGridInterpolator([logprob_arr[1:,0], logprob_arr[0,1:]],
                                                                      logprob_arr[1:,1:,None], nout=1)
    
    def init_pymc3(self,ld_mult=1.5,):
        ######################################
        #       Selecting lightcurve:
        ######################################
        if self.bin_oot:
            lc=self.pseudo_binlc
        elif self.cutDistance>0:
            lc=self.lc_near_trans
        else:
            lc=self.lc
        
        if self.interpolate_v_prior:
            #Setting up interpolation functions for v_priors here:
            self.init_interpolated_v_prior()
            
        if hasattr(self,'rvs'):
            self.init_interpolated_Mp_prior()
        start=None
        with pm.Model() as model:            
            
            if self.debug: print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)
            
            ######################################
            #   Intialising Stellar Params:
            ######################################
            #Using log rho because otherwise the distribution is not normal:
            logrho_S = pm.Bound(pm.Normal,upper=2.5,lower=-6)("logrho_S", mu=np.log(self.rhostar[0]), 
                                 sd=np.average(abs(self.rhostar[1:]/self.rhostar[0])),
                                 testval=np.log(self.rhostar[0]))
            rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S)) #Converting from rho_sun into g/cm3
            Rs = pm.Normal("Rs", mu=self.Rstar[0], sd=np.average(abs(self.Rstar[1:])),testval=self.Rstar[0],shape=1)
            Ms = pm.Deterministic("Ms",(rho_S)*Rs**3)

            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if self.useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0
            
            BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
            
            ######################################
            #     Initialising dictionaries
            ######################################
            pers={};t0s={};logrors={};rors={};rpls={};bs={};dist_in_transits={};a_Rs={};tdurs={};vels={};logvels={};
            self.n_margs={}
            if not self.assume_circ:
                eccs={};omegas={}
            if len(self.monos+self.duos)>0:
                max_eccs={};min_eccs={}
                if 'b' not in self.fit_params:
                    b_priors={}
            if len(self.duos)>0:
                t0_2s={}
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
                t0s[pl] = pm.Bound(pm.Normal, 
                                    upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.33,
                                    lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.33
                                   )("t0_"+pl,mu=self.planets[pl]['tcen'],
                                          sd=self.planets[pl]['tdur']*0.1,
                                          testval=self.planets[pl]['tcen'])
                
                #############################################
                #     Initialising specific duo/mono terms
                #############################################

                if pl in self.monos:
                    self.n_margs[pl]=self.planets[pl]['ngaps']
                    ind_min=np.power(self.planets[pl]['per_gaps']['gap_ends']/self.planets[pl]['per_gaps']['gap_starts'],self.per_index)
                    per_meds[pl]=np.power(((1-ind_min)*0.5+ind_min),self.per_index)*self.planets[pl]['per_gaps']['gap_starts']
                    if 'log_per' in self.planets[pl]:
                        testindex=[]
                        for ngap in np.arange(len(self.planets[pl]['per_gaps']['gap_ends'])):
                            if np.exp(self.planets[pl]['log_per'])>self.planets[pl]['per_gaps'][ngap,0] and np.exp(self.planets[pl]['log_per'])<self.planets[pl]['per_gaps']['gap_ends'][ngap]:
                                testindex+=[((np.exp(self.planets[pl]['log_per'])/self.planets[pl]['per_gaps']['gap_starts'][ngap])**self.per_index - ind_min[ngap])/(1-ind_min[ngap])]
                            else:
                                testindex+=[np.clip(np.random.normal(0.5,0.25),0.00001,0.99999)]
                        mono_uniform_index_period[pl]=xo.distributions.UnitUniform("mono_uniform_index_"+str(pl),
                                                        shape=len(self.planets[pl]['per_gaps']['gap_starts']),
                                                        testval=testindex)
                    else:
                        mono_uniform_index_period[pl]=xo.distributions.UnitUniform("mono_uniform_index_"+str(pl),
                                                        shape=len(self.planets[pl]['per_gaps']['gap_starts']))
                    pers[pl]=pm.Deterministic("per_"+str(pl), tt.power(((1-ind_min)*mono_uniform_index_period[pl]+ind_min),1/self.per_index)*self.planets[pl]['per_gaps']['gap_starts'])
                elif pl in self.duos:
                    self.n_margs[pl]=self.planets[pl]['npers']
                    t0_2s[pl] = pm.Bound(pm.Normal, 
                                                   upper=self.planets[pl]['tcen_2']+self.planets[pl]['tdur']*0.5, 
                                                   lower=self.planets[pl]['tcen_2']-self.planets[pl]['tdur']*0.5
                                                  )("t0_2_"+pl,mu=self.planets[pl]['tcen_2'],
                                                                      sd=self.planets[pl]['tdur']*0.2,
                                                                      testval=self.planets[pl]['tcen_2'])
                    pers[pl]=pm.Deterministic("per_"+pl, tt.tile(tt.abs_(t0s[pl] - t0_2s[pl]),self.n_margs[pl])/self.planets[pl]['period_int_aliases'])
                elif pl in self.multis:
                    self.n_margs[pl]=1
                    pers[pl] = pm.Normal("per_"+pl, 
                                         mu=self.planets[pl]['period'],
                                         sd=np.clip(self.planets[pl]['period_err']*0.25,0.005,0.02*self.planets[pl]['period']),
                                         testval=self.planets[pl]['period'])

                #############################################
                #     Initialising shared planet params
                #############################################

                if 'logror' in self.marginal_params:
                    logrors[pl]=pm.Uniform("logror_"+pl,lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                                testval=np.log(self.planets[pl]['ror']),
                                                shape=self.n_margs[pl])
                else:
                    logrors[pl]=pm.Uniform("logror_"+pl,lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                            testval=np.log(self.planets[pl]['ror']))
                rors[pl]=pm.Deterministic("ror_"+pl,tt.exp(logrors[pl]))
                rpls[pl]=pm.Deterministic("rpl_"+pl,109.2*rors[pl]*Rs)
                
                if not self.assume_circ:
                    #Marginalising over, so one value for each period:
                    if pl not in self.multis and not self.interpolate_v_prior and ('ecc' in self.marginal_params or 'omega' in self.marginal_params):
                        if self.ecc_prior.lower()=='kipping' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = BoundedBeta("ecc_"+pl, alpha=0.867,beta=3.03,
                                                         testval=0.05,shape=self.n_margs[pl])
                        elif self.ecc_prior.lower()=='vaneylen' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = pm.Bound(pm.Weibull, lower=1e-5, 
                                                     upper=1-1e-5)("ecc_"+pl,alpha=0.049,beta=2,testval=0.05,
                                                                   shape=self.n_margs[pl])
                        elif self.ecc_prior.lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5,
                                                       shape=self.n_margs[pl])

                        omegas[pl] = xo.distributions.Angle("omega_"+pl,
                                                                 shape=self.n_margs[pl])

                    elif not self.interpolate_v_prior or pl in self.multis:
                        #Fitting for a single ecc and omega (not marginalising or doing the v interpolation)
                        if self.ecc_prior.lower()=='kipping' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = BoundedBeta("ecc_"+pl, alpha=0.867,beta=3.03,testval=0.05)
                        elif self.ecc_prior.lower()=='vaneylen' or (self.ecc_prior.lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("ecc_"+pl,alpha= 0.049,beta=2,
                                                                                           testval=0.05)
                        elif self.ecc_prior.lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5)

                        omegas[pl] = xo.distributions.Angle("omega_"+pl)
                if 'b' in self.fit_params or pl in self.multis:
                    if 'logror' in self.marginal_params and pl not in self.multis:
                        # The Espinoza (2018) parameterization for the joint radius ratio and
                        bs[pl] = xo.distributions.ImpactParameter("b_"+pl,ror=rors[pl],shape=self.n_margs[pl])
                    else:
                        bs[pl] = xo.distributions.ImpactParameter("b_"+pl, ror=rors[pl], testval=self.planets[pl]['b'])
                if 'tdur' in self.fit_params and pl not in self.multis:
                    tdurs[pl] = pm.Uniform("tdur_"+pl,
                                           lower=0.33*self.planets[pl]['tdur'],
                                           upper=3*self.planets[pl]['tdur'],
                                           testval=self.planets[pl]['tdur'])
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
                                                     (1+eccs[pl]*tt.cos(omegas[pl]-np.pi/2))/(1-eccs[pl]**2)
                                                    )
                    bs[pl] = pm.Deterministic("b_"+pl,tt.clip(bsqs[pl], 1e-5, 100)**0.5)
                    # Combining together prior from db/dtdur (which is needed to renormalise to constant b) ~ P**(-2/3)/b
                    # And custom prior which reduces the logprob of all models with bsq<0 (i.e. unphysical) by 5-25 
                if 'b' in self.fit_params and 'tdur' in self.fit_params and pl not in self.multis:
                    #We fit for both duration and b, so we can derive velocity (v/v_circ) directly:
                    vels[pl]=pm.Deterministic("vel_"+pl, tt.sqrt((1+rors[pl])**2 - bs[pl]**2)/(tdurs[pl]*86400) * ((3*pers[pl]*86400)/(np.pi**2*6.67e-11*rho_S*1409.78))**(1/3))
                    logvels[pl]= pm.Deterministic("logvel_"+pl, tt.log(vels[pl]))
                    
                    #Minimum eccentricity (and the associated omega) are then derived from vel, but are one of a range of values
                    min_eccs[pl] = pm.Deterministic("min_ecc_"+pl,tt.clip(tt.abs_(2/(1 + vels[pl]**2) - 1), 1e-4, 1.0-1e-4))
                    omegas[pl] = pm.Deterministic("omega_"+pl,np.pi-0.5*np.pi*(logvels[pl]/tt.abs_(logvels[pl])) )
                
                ######################################
                #         Initialising RVs
                ######################################

                #Circular a_Rs
                a_Rs[pl]=pm.Deterministic("a_Rs_"+pl,((6.67e-11*(rho_S*1409.78)*(86400*pers[pl])**2)/(3*np.pi))**(1/3))
                if hasattr(self,'rvs'):
                    # Using density directly to connect radius (well-constrained) to mass (likely poorly constrained).
                    # Testval uses mass assuming std of RVs is from planet mass
                    #logrhos[pl] = pm.Bound(pm.Normal, lower=np.log(0.01), upper=np.log(3))("logrho_"+pl,mu=0.0,sd=1,
                    #                         testval=np.clip(np.log(0.22*np.std(self.rvs['rv']) * \
                    #                                                (self.planets[pl]['ror'] * self.Rstar[0]*109.1)**(-3)),
                    #                                         np.log(0.01),np.log(3)))
                    #print("K est.",0.22*np.std(self.rvs['rv']),"Rp est:",self.planets[pl]['ror'] * self.Rstar[0]*109.1,"Rp^3 est:",(self.planets[pl]['ror'] * self.Rstar[0]*109.1)**(-3))
                    #tt.printing.Print("logrhos")(logrhos[pl])
                    #logMp_wrt_normals[pl]=pm.Normal("logmass_wrt_normal_"+pl,mu=0.0,sd=1.0)
                    #Calculating the mass using a prior derived from the MR distribution:
                    #logMps[pl] = pm.Deterministic("logMp_"+pl, (logMp_wrt_normals[pl] * \
                    #                              self.interpolated_sigma.evaluate(rpls[pl].dimshuffle(0,'x')) + \
                    #                              self.interpolated_mu.evaluate(rpls[pl].dimshuffle(0,'x'))).T[0])
                    #logMps[pl] = pm.Deterministic("logMp_"+pl, logrhos[pl] + 3*tt.log(rpls[pl]))
                    #tt.printing.Print("logMps")(logMps[pl])
                    #Mps[pl]=pm.Deterministic("Mp_"+pl, tt.exp(logMps[pl]))
                    #tt.printing.Print("Mps")(Mps[pl])
                    
                    #sin(incl) = tt.sqrt(1-(bs[pl]/a_Rs[pl])**2)
                    
                    #Data-driven prior:
                    if not self.derive_K or (pl in self.multis and 'K' in self.planets[pl]):
                        if 'K' in self.planets[pl]:
                            logKs[pl] = pm.Normal("logK_"+pl, mu=np.log(self.planets[pl]['K']), sd=0.5)
                        else:
                            logKs[pl] = pm.Normal("logK_"+pl, mu=np.log(np.std(self.rvs['rv'])/np.sqrt(len(self.planets))), sd=0.5)
                        Ks[pl] = pm.Deterministic("K_"+pl,tt.exp(logKs[pl]))
                        if pl in self.duos+self.monos and self.interpolate_v_prior:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,tt.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-min_eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        elif not self.assume_circ:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,tt.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        else:
                            Mps[pl]=pm.Deterministic("Mp_"+pl,tt.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1.9884e30*Ms)**(2/3)/5.972e24)
                        rhos[pl]=pm.Deterministic("rho_"+pl,Mps[pl]/rpls[pl]**3)
                    
                    #if pl in self.duos+self.monos and self.interpolate_v_prior:
                    #    #Using minimum eccentricity as this is most likely.
                    #    #37768.355 = Me*Msun^-2/3
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3) * \
                    #                            tt.tile(5.972e24*Mps[pl]*(1.9884e30*Ms)**(-2/3),self.n_margs[pl]) * \
                    #                            sin_incls[pl]*(1-min_eccs[pl]**2)**-0.5)
                    #    tt.printing.Print("Ks")(Ks[pl])
                    #elif not self.assume_circ:
                    #    #Using global ecc parameter
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3)* \
                    #                            5.972e24*Mps[pl]*sin_incls[pl]*(1.9884e30*Ms)**(-2/3)*(1-eccs[pl]**2)**-0.5)
                    #else:
                    #    #Without eccentricity
                    #    Ks[pl]=pm.Deterministic("K_"+pl,((2*np.pi*6.67e-11)/(86400*pers[pl]))**(1/3)* \
                    #                            5.972e24*Mps[pl]*sin_incls[pl]*(1.9884e30*Ms)**(-2/3))

            #############################################
            #     Initialising RV_only planet params
            #############################################

            if len(self.rvplanets)>0:
                for pl in self.rvplanets:
                    self.n_margs[pl]=1
                    #Making sure tcen can't drift far onto other repeating tcens
                    t0s[pl] = pm.Bound(pm.Normal,upper=self.rvplanets[pl]['tcen']+0.55*self.rvplanets[pl]['period'], 
                                                   lower=self.rvplanets[pl]['tcen']-0.55*self.rvplanets[pl]['period']
                                          )("t0_"+pl, mu=self.rvplanets[pl]['period'], sd=self.rvplanets[pl]['period_err'])
                    pers[pl] = pm.Normal("per_"+pl, mu=self.rvplanets[pl]['period'], sd=self.rvplanets[pl]['period_err'])
                    logKs[pl] = pm.Normal("logK_"+pl, mu=self.rvplanets[pl]['logK'], sd=self.rvplanets[pl]['logK_err'])
                    Ks[pl] = pm.Deterministic("K_"+pl, tt.exp(logKs[pl]))

                    if not self.rvplanets[pl]['assume_circ']:
                        if self.rvplanets[pl]['ecc_prior'].lower()=='kipping' or (self.rvplanets[pl]['ecc_prior'].lower()=='auto' and (len(self.planets)+len(self.rvplanets))==1):
                            eccs[pl] = BoundedBeta("ecc_"+pl, alpha=0.867,beta=3.03,
                                                         testval=0.05)
                        elif self.rvplanets[pl]['ecc_prior'].lower()=='vaneylen' or (self.rvplanets[pl]['ecc_prior'].lower()=='auto' and (len(self.planets)+len(self.rvplanets))>1):
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            eccs[pl] = pm.Bound(pm.Weibull, lower=1e-5, 
                                                     upper=1-1e-5)("ecc_"+pl,alpha=0.049,beta=2,testval=0.05)
                        elif self.rvplanets[pl]['ecc_prior'].lower()=='uniform':
                            eccs[pl] = pm.Uniform("ecc_"+pl,lower=1e-5, upper=1-1e-5)

                        omegas[pl] = xo.distributions.Angle("omega_"+pl)
                    else:
                        eccs[pl] = pm.Deterministic("ecc_"+pl,tt.constant(0.0))
                        omegas[pl] = pm.Deterministic("omega_"+pl,tt.constant(0.0))
                    Mps[pl]=pm.Deterministic("Mp_"+pl,tt.exp(logKs[pl]) * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                      (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24) #This is Mpsini

            ######################################
            #     Initialising Limb Darkening
            ######################################
            # Here we either constrain the LD params given the stellar info, OR we let exoplanet fit them
            # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.

            #Single mission
            if np.any([c[0].lower()=='t' for c in self.cads]) and self.constrain_LD:
                ld_dists=self.getLDs(n_samples=1200,mission='tess')
                u_star_tess = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_tess", 
                                                mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                                sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[0].lower()=='t' for c in self.cads]) and not self.constrain_LD:
                u_star_tess = xo.distributions.QuadLimbDark("u_star_tess", testval=np.array([0.3, 0.2]))
            if np.any([c[0].lower()=='k' for c in self.cads]) and self.constrain_LD:
                ld_dists=self.getLDs(n_samples=3000,mission='kepler')
                if self.debug: print("LDs",ld_dists)
                u_star_kep = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_kep", 
                                            mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                            sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[0].lower()=='k' for c in self.cads]) and self.constrain_LD:
                u_star_kep = xo.distributions.QuadLimbDark("u_star_kep", testval=np.array([0.3, 0.2]))
            if np.any([c[0].lower()=='c' for c in self.cads]) and self.constrain_LD:
                ld_dists=self.getLDs(n_samples=1200,mission='corot')
                u_star_corot = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_corot", 
                                                mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                                sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[0].lower()=='c' for c in self.cads]) and not self.constrain_LD:
                u_star_corot = xo.distributions.QuadLimbDark("u_star_corot", testval=np.array([0.3, 0.2]))
            if np.any([c[0].lower()=='x' for c in self.cads]) and self.constrain_LD:
                ld_dists=self.getLDs(n_samples=1200,mission='cheops')
                u_star_cheops = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_cheops", 
                                                mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                                sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[0].lower()=='x' for c in self.cads]) and not self.constrain_LD:
                u_star_cheops = xo.distributions.QuadLimbDark("u_star_cheops", testval=np.array([0.3, 0.2]))

            mean=pm.Normal("mean",mu=np.median(lc['flux'][lc['mask']]),
                                  sd=np.std(lc['flux'][lc['mask']]))
            if not hasattr(self,'log_flux_std'):
                self.log_flux_std=np.array([np.log(np.nanstd(lc['flux'][~lc['in_trans']][lc['cadence'][~lc['in_trans']]==c])) for c in self.cads]).ravel().astype(floattype)
            if self.debug: print(self.log_flux_std,np.sum(~lc['in_trans']),"/",len(~lc['in_trans']))
            
            logs2 = pm.Normal("logs2", mu = self.log_flux_std+1, 
                              sd = np.tile(2.0,len(self.log_flux_std)), shape=len(self.log_flux_std))
            ######################################
            #     Initialising RV background
            ######################################

            if hasattr(self,'rvs'):
                #One offset for each telescope:
                rv_offsets = pm.Normal("rv_offsets", 
                             mu=np.array([np.nanmedian(self.rvs['rv'][self.rvs['tele_index']==s]) for s in self.rvs['scopes']]),
                              sd=np.array([np.nanstd(self.rvs['rv'][self.rvs['tele_index']==s]) for s in self.rvs['scopes']]),
                                       shape=len(self.rvs['scopes']))
                #Now doing the polynomials with a vander
                if self.rv_npoly>2:
                    #We have encapsulated the offset into rv_offsets, so here we form a poly with rv_npoly-1 terms
                    rv_polys = pm.Normal("rv_polys",mu=0,
                                         sd=np.nanstd(self.rvs['rv'])*(10.0**-np.arange(self.rv_npoly)[::-1])[:-1],
                                         shape=self.rv_npoly-1,testval=np.zeros(self.rv_npoly-1))
                    rv_trend = pm.Deterministic("rv_trend", tt.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1) + \
                                         tt.dot(np.vander(self.rvs['time']-self.rv_tref,self.rv_npoly)[:,:-1],rv_polys))
                elif self.rv_npoly==2:
                    #We have encapsulated the offset into rv_offsets, so here we just want a single-param trend term
                    rv_polys = pm.Normal("rv_polys", mu=0, sd=0.1*np.nanstd(self.rvs['rv']), testval=0.0)
                    #tt.printing.Print("trend")(rv_polys*(self.rvs['time']-self.rv_tref))
                    #tt.printing.Print("offset")(rv_offsets* self.rvs['tele_index_arr'])
                    rv_trend = pm.Deterministic("rv_trend", tt.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1) + \
                                                            rv_polys*(self.rvs['time']-self.rv_tref))
                else:
                    #No trend - simply an offset
                    rv_trend = pm.Deterministic("rv_trend", tt.sum(rv_offsets*self.rvs['tele_index_arr'],axis=1))
                #tt.sum(rv_offsets*tele_index_arr,axis=1)+tt.dot(np.vander(time,3),np.hstack((trend,0)))
                #rv_mean = pm.Normal("rv_mean", mu=np.nanmedian(self.rvs['rv']),sd=2.5*np.nanstd(self.rvs['rv']))
                rv_logs2 = pm.Uniform("rv_logs2", lower = -10, upper = 1.5)
            
            if self.use_GP:
                ######################################
                #     Initialising GP kernel
                ######################################
                if self.debug: print(np.isnan(lc['time']),np.isnan(lc['flux']),np.isnan(lc['flux_err']))
                if self.train_GP:
                    #Taking trained values from out-of-transit to use as inputs to GP:
                    minmax=np.percentile(self.gp_init_trace["w0"],[0.5,99.5]).astype(floattype)
                    w0=pm.Interpolated("w0", x_points=np.linspace(minmax[0],minmax[1],201)[1::2],
                                          pdf_points=np.histogram(self.gp_init_trace["w0"],
                                                                  np.linspace(minmax[0],minmax[1],101))[0]
                                         )
                    minmax=np.percentile(self.gp_init_trace["power"],[0.5,99.5]).astype(floattype)
                    power=pm.Interpolated("power", x_points=np.linspace(minmax[0],minmax[1],201)[1::2],
                                          pdf_points=np.histogram(self.gp_init_trace["power"],
                                                                  np.linspace(minmax[0],minmax[1],101))[0]
                                         )
                else:
                    # Transit jitter & GP parameters
                    #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
                    lcrange=lc['time'][slc['near_trans']][-1]-lc['time'][lc['near_trans']][0]
                    max_cad = np.nanmax([np.nanmedian(np.diff(lc['time'][lc['near_trans']&(lc['cadence']==c)])) for c in self.cads])
                    #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
                    success=False;target=0.05
                    while not success and target<0.4:
                        try:

                            w0 = pm.InverseGamma("w0",testval=(2*np.pi)/10,
                                         **xo.estimate_inverse_gamma_parameters(lower=(2*np.pi)/(lcrange),
                                                                                upper=(2*np.pi)/(6*max_cad)))
                            success=True
                        except:
                            target+=0.05
                            success=False

                    maxpower=12.5*np.nanstd(lc['flux'][lc['near_trans']])
                    minpower=0.2*np.nanmedian(abs(np.diff(lc['flux'][lc['near_trans']])))
                    if self.debug: print(np.nanmedian(abs(np.diff(lc['flux'][lc['near_trans']]))),np.nanstd(lc['flux'][lc['near_trans']]),minpower,maxpower)
                    success=False;target=0.05
                    while not success and target<0.4:
                        try:
                            power = pm.InverseGamma("power",testval=minpower*5,
                                                    **xo.estimate_inverse_gamma_parameters(lower=minpower, upper=maxpower,target=0.1))
                            success=True
                        except:
                            target+=0.05
                            success=False
                    if self.debug: print(target," after ",int(target/0.05),"attempts")

                    if self.debug: print("input to GP power:",maxpower-1)
                S0 = pm.Deterministic("S0", power/(w0**4))

                # GP model for the light curve
                kernel = xo.gp.terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))
                
                if hasattr(self.lc,'near_trans') and np.sum(lc['near_trans'])!=len(lc['time']):
                    self.gp['use'] = xo.gp.GP(kernel, lc['time'][lc['near_trans']].astype(floattype),
                                       lc['flux_err'][lc['near_trans']].astype(floattype)**2 + \
                                       tt.dot(lc['flux_err_index'][lc['near_trans']],tt.exp(logs2)),
                                       J=2)

                    self.gp['all'] = xo.gp.GP(kernel, lc['time'].astype(floattype),
                                           lc['flux_err'].astype(floattype)**2 + \
                                           tt.dot(lc['flux_err_index'],tt.exp(logs2)),
                                           J=2)
                else:
                    self.gp['use'] = xo.gp.GP(kernel, lc['time'][lc['mask']].astype(floattype),
                                              lc['flux_err'][lc['mask']].astype(floattype)**2 + \
                                              tt.dot(lc['flux_err_index'][lc['mask']],tt.exp(logs2)),
                                           J=2)

            ################################################
            #  Creating function to generate transit models
            ################################################
            def gen_lc(i_orbit, i_r, n_pl, mask=None,prefix='',make_deterministic=False):
                # Short method to create stacked lightcurves, given some input time array and some input cadences:
                # This function is needed because we may have 
                #   -  1) multiple cadences and 
                #   -  2) multiple telescopes (and therefore limb darkening coefficients)
                trans_pred=[]
                mask = ~np.isnan(lc['time']) if mask is None else mask
                cad_index=[]
                
                if n_pl>1:
                    r=tt.tile(i_r,n_pl)
                else:
                    r=i_r
                
                for cad in self.cads:
                    cadmask=mask&(lc['cadence']==cad)
                    
                    #print(self.lc['tele_index'][mask,0].astype(bool),len(self.lc['tele_index'][mask,0]),cadmask[mask],len(cadmask[mask]))
                    
                    if cad[0].lower()=='t':
                        #Taking the "telescope" index, and adding those points with the matching cadences to the cadmask
                        cad_index+=[(lc['tele_index'][mask,0].astype(bool))&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0].lower()=='k':
                        cad_index+=[(lc['tele_index'][mask,1]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0].lower()=='c':
                        cad_index+=[(lc['tele_index'][mask,2]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_corot).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0].lower()=='x':
                        cad_index+=[(lc['tele_index'][mask,3]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_cheops).get_light_curve(
                                                                 orbit=i_orbit, r=r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]

                # transit arrays (ntime x n_pls x 2) * telescope index (ntime x n_pls x 2), summed over dimension 2
                if n_pl>1 and make_deterministic:
                    
                    return pm.Deterministic(prefix+"light_curves", 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               tt.stack(cad_index).dimshuffle(1,'x',0),axis=2))
                elif n_pl==1 and make_deterministic:
                    return pm.Deterministic(prefix+"light_curves", 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2)))
                elif n_pl>1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                  tt.stack(cad_index).dimshuffle(1,'x',0),axis=2)

                elif n_pl==1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2))

            def create_orbit(pl, Rs, rho_S, pers, t0s, bs, n_marg=1, eccs=None, omegas=None):
                #Setting up Exoplanet orbit
                if pl in self.multis or self.interpolate_v_prior:
                    #Single orbit expected:
                    i_t0s=t0s;i_pers=pers;i_bs=bs
                    if not self.assume_circ:
                        i_eccs=eccs;i_omegas=omegas
                else:
                    #Multiple orbits expected
                    i_t0s=tt.tile(t0s,n_marg)
                    i_pers=tt.tile(pers,n_marg)
                    i_bs=tt.tile(bs,n_marg) if 'b' not in self.marginal_params else bs
                    if not self.assume_circ:
                        i_eccs=tt.tile(eccs,n_marg) if 'ecc' not in self.marginal_params else eccs
                        i_omegas=tt.tile(omegas,n_marg) if 'omega' not in self.marginal_params else omegas
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
            logpriors={};rvlogliks={};lclogliks={}
            if len(self.duos+self.monos)>0:
                #Initialising priors:
                per_priors={};b_priors={};geom_ecc_priors={};ecc_lim_priors={}
                edge_priors={};v_priors={};gap_width_priors={};
            if self.force_match_input is not None:
                match_input_potentials={}
            
            for pl in self.rvplanets:
                rvorbits[pl] = xo.orbits.KeplerianOrbit(period=pers[pl], t0=t0s[pl], ecc=eccs[pl], omega=omegas[pl])
                model_rvs[pl] = pm.Deterministic("model_rv_"+pl,
                                                 rvorbits[pl].get_radial_velocity(self.rvs['time'], K=tt.exp(logKs[pl])))
            
            for pl in self.multis+self.duos+self.monos:
                #Making orbit and lightcurve(s)
                if self.assume_circ:
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl], t0s[pl], bs[pl], n_marg=self.n_margs[pl])
                    light_curves[pl] = gen_lc(orbits[pl], rors[pl], self.n_margs[pl], mask=None,
                                              prefix=pl+'_', make_deterministic=True)
                elif self.interpolate_v_prior and pl in self.duos+self.monos:
                    #  We only need to create one orbit if we're not marginalising over N periods 
                    #      (i.e. when we only have the lightcurve and we're interpolating a velocity prior)
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl][tt.argmin(min_eccs[pl])], t0s[pl], bs[pl], n_marg=1, 
                                              omegas=omegas[pl][tt.argmin(min_eccs[pl])], eccs=tt.min(min_eccs[pl]))
                    light_curves[pl] = gen_lc(orbits[pl], rors[pl], 1, mask=None,
                                              prefix=pl+'_', make_deterministic=True)
                else:
                    orbits[pl] = create_orbit(pl, Rs, rho_S, pers[pl], t0s[pl], bs[pl], n_marg=self.n_margs[pl],  
                                              eccs=eccs[pl], omegas=omegas[pl])
                    light_curves[pl] = gen_lc(orbits[pl], rors[pl], 1, mask=None,
                                              prefix=pl+'_', make_deterministic=pl in self.multis)
                if hasattr(self,'rvs'):
                    if pl in self.duos+self.monos and self.interpolate_v_prior:
                        #In this case, we need to create N orbits but only one lightcurve (from the min eccentricity)
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
                    vels[pl] = pm.Deterministic("vel_"+pl,tt.sqrt(vx[pl]**2 + vy[pl]**2))
                if pl not in logvels:
                    logvels[pl]= pm.Deterministic("logvel_"+pl,tt.log(vels[pl]))
                if pl not in a_Rs:
                    a_Rs[pl] = pm.Deterministic("a_Rs_"+pl, orbits[pl].a)
                if pl not in tdurs:
                    #if 'tdur' in self.marginal_params:
                    tdurs[pl]=pm.Deterministic("tdur_"+pl,
                            (2*Rs*tt.sqrt( (1+rors[pl])**2 - bs[pl]**2)) / vels[pl] )
                #else:
                #    vx, vy, vz=orbits[pl].get_relative_velocity(t0s[pl])
                
                

                ################################################
                #                   Priors:
                ################################################
                #Force model to match expected/input depth_duration with sigmoid (not used in default)
                if self.force_match_input is not None:
                    match_input_potentials[pl]=tt.sum(tt.exp( -(tdurs[pl]**2 + self.planets[pl]['tdur']**2) / (2*(self.force_match_input*self.planets[multi]['tdur'])**2) )) + \
                                     tt.sum(tt.exp( -(logrors[pl]**2 + self.planets[pl]['log_ror']**2) / (2*(self.force_match_input*self.planets[pl]['log_ror'])**2) ))
                    pm.Potential("all_match_input_potentials",
                                 tt.sum([match_input_potentials[i] for i in match_input_potentials]))
                
                ################################################
                #       Generating RVs for each submodels:
                ################################################
                if hasattr(self,'rvs'):
                    new_rverr = ((1+tt.exp(rv_logs2))*self.rvs['rv_err'].astype(floattype))
                    sum_log_rverr = tt.sum(-len(self.rvs['rv'])/2 * tt.log(2*np.pi*(new_rverr**2)))
                    #model_rvs[pl] = pm.Deterministic('model_rv_'+pl, tt.tile(Ks[pl].dimshuffle('x',0),(len(self.rvs['time']),1)))
                    if self.derive_K:
                        #Deriving the best-fit K from the data:
                        sinf, cosf = rvorbits[pl]._get_true_anomaly(self.rvs['time'])
                        #tt.printing.Print("cosf")(cosf)
                        #tt.printing.Print("sinf")(sinf)
                        normalised_rv_models[pl]=(rvorbits[pl].cos_omega * cosf - rvorbits[pl].sin_omega * sinf + \
                                                  rvorbits[pl].ecc * rvorbits[pl].cos_omega)
                        #tt.printing.Print("normalised_rv_models")(normalised_rv_models[pl])
                        
                        if pl in self.duos+self.monos and not hasattr(model,'nonmarg_rvs'):
                            if (len(self.multis)+len(self.rvplanets))>1:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs", (rv_trend + tt.sum([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())],axis=1)))
                            elif (len(self.multis)+len(self.rvplanets))==1:
                                onlypl=self.multis+list(self.rvplanets.keys())
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",(rv_trend+model_rvs[onlypl[0]]))
                            else:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",rv_trend)                        
                        if pl in self.duos+self.monos:
                            #Mono or duo. Removing multi orbit if we have one:
                            Ks[pl] = pm.Deterministic("K_"+pl, tt.clip(tt.batched_tensordot(tt.tile(self.rvs['rv'] - nonmarg_rvs,(self.n_margs[pl],1)), normalised_rv_models[pl].T, axes=1) / tt.sum(normalised_rv_models[pl]**2,axis=0),0.05,1e5))
                        elif pl in self.mulits and pl not in Ks and self.derive_K:
                            #Multi:
                            Ks[pl] = pm.Deterministic("K_"+pl, tt.clip(tt.dot(self.rvs['rv'] - rv_trend, normalised_rv_models[pl].T) / tt.sum(normalised_rv_models[pl]**2,axis=0),np.log(0.05),1e5))
                        model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'], K=Ks[pl]))
                        if pl in self.duos+self.monos and self.interpolate_v_prior:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-min_eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        elif not self.assume_circ:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2) * (1-eccs[pl]**2)**0.5*(1.9884e30*Ms)**(2/3)/5.972e24)
                        else:
                            Mps[pl]=pm.Deterministic("Mp_"+pl, Ks[pl] * ((2*np.pi*6.67e-11)/(86400*pers[pl]))**(-1/3)/\
                                                 tt.sqrt(1-(bs[pl]/a_Rs[pl])**2)*(1.9884e30*Ms)**(2/3)/5.972e24)
                        rhos[pl]=pm.Deterministic("rho_"+pl,Mps[pl]/rpls[pl]**3)
                                                   
                    else:
                        if pl in self.duos+self.monos:
                            model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'],
                                                                                                      K=tt.exp(logKs[pl])))
                        else:
                            model_rvs[pl] = pm.Deterministic('model_rv_'+pl, rvorbits[pl].get_radial_velocity(self.rvs['time'],tt.tile(Ks[pl].dimshuffle('x',0),(len(self.rvs['time']),1))))
                            
                
                if pl in self.duos+self.monos:
                    #Need the minimum period to normalise
                    per_priors[pl] = pm.Deterministic("per_prior_"+pl, 
                                                      self.per_index * tt.log(pers[pl]/self.planets[pl]['P_min']))
                    if len(self.multis)>0:
                        max_eccs[pl] = pm.Deterministic("max_ecc_"+pl,
                                                        1 - tt.max([a_Rs[i] for i in self.multis])/a_Rs[pl])
                    else:
                        max_eccs[pl] = pm.Deterministic("max_ecc_"+pl,1 - 2/a_Rs[pl])
                    if not self.assume_circ and not self.interpolate_v_prior:
                        #A correction to the period prior from the increased geometric probability of high-ecc planets:
                        geom_ecc_priors[pl]=pm.Deterministic("geom_ecc_prior_"+pl,
                                                             -1*tt.log(dist_in_transits[pl]/a_Rs[pl]))
                        
                        #A sigmoid prior rejecting orbits that either star-graze or cross the orbits of inner planets
                        ecc_lim_priors[pl]=pm.Deterministic("star_ecc_lim_prior_"+pl,
                                            (500 / (1 + tt.exp(-30*(max_eccs[pl] - eccs[pl])))-500))
                    else:
                        #These are incorporated into the interpolated velocity prior:
                        geom_ecc_priors[pl]=tt.zeros(self.n_margs[pl])
                        ecc_lim_priors[pl]=tt.zeros(self.n_margs[pl])
                    if 'b' not in self.fit_params:
                        #Require prior on b to account for the fact that high impact parameters less likely
                        b_priors[pl]=pm.Deterministic("b_prior_"+pl, tt.log(tt.max(bs[pl])/bs[pl]) - \
                                                                      5/2*tt.log(pers[pl]/tt.max(pers[pl])) + \
                                                                      tt.switch(tt.lt(bsqs[pl],0),bsqs[pl]*40-15,0))
                        #tt.log( (tt.max(bs[pl])/bs[pl]) * (pers[pl]/tt.max(pers[pl]))**(-5/2) ) + tt.switch(tt.lt(bsqs[pl],0),bsqs[pl]*40-15,0)
                    else:
                        b_priors[pl]=tt.zeros(self.n_margs[pl])
                        
                    if self.interpolate_v_prior:
                        #Prior on derived velocity implied by period (coming from geometry and eccentricity)
                        v_priors[pl]=pm.Deterministic("v_prior_"+pl,self.interpolator_logprob.evaluate(
                                                                    tt.stack([logvels[pl],max_eccs[pl]],axis=-1))[:,0])
                        edge_priors[pl]=tt.zeros(self.n_margs[pl])
                        
                        '''#Prior here mimics presence of data edges to likelihood, but applied as a prior
                        #Assumes data is at 0 and planet transit is at rp/rs**2
                        if pl in self.monos:
                            edge_ix=self.lc['mask'][:,None] * \
                                (abs((self.lc['time'][:,None]-t0s[pl]-0.5*pers[pl])%pers[pl]-0.5*pers[pl])<tdurs[pl]*0.5) * \
                                (abs(self.lc['time'][:,None]-t0s[pl])>tdurs[pl]*0.5)
                        elif pl in self.duos:
                            edge_ix=self.lc['mask'][:,None] * \
                            (abs((self.lc['time'][:,None]-t0s[pl]-0.5*pers[pl])%pers[pl]-0.5*pers[pl])<tdurs[pl]*0.5) * \
                            (abs(self.lc['time'][:,None]-t0_2s[pl])>tdurs[pl]*0.55) * \
                            (abs(self.lc['time'][:,None]-t0s[pl])>tdurs[pl]*0.55) * \
                            self.lc['mask'][:,None]

                        #depth^2/errs^2 = ror^4/errs^2
                        edge_priors[pl]=pm.Deterministic("edge_prior_"+pl,
                                       -tt.sum(rors[pl]**4/(self.lc['flux_unit']*self.lc['flux_err'][:,None])**2*edge_ix,axis=0)
                                                        )
                        '''
                    else:
                        v_priors[pl]=tt.zeros(self.n_margs[pl])
                        edge_priors[pl]=tt.zeros(self.n_margs[pl])
                    
                    '''
                    #We want models to prefer high-K solutions over flat lines, so we include a weak prior on log(K)
                    if hasattr(self,'rvs') and self.derive_K:
                        Krv_priors[pl] = pm.Deterministic("Krv_prior_"+pl, 0.25*tt.log(Ks[pl]))
                    elif hasattr(self,'rvs') and not self.derive_K:
                        Krv_priors[pl] = pm.Deterministic("Krv_prior_"+pl, 0.25*tt.tile(tt.log(Ks[pl]),self.n_margs[pl]) )
                    else:
                        Krv_priors[pl] = tt.zeros(self.n_margs[pl])
                    '''
                    
                    if pl in self.monos:
                        #For monotransits, there is a specific term required from the width of the gap (not necessary for duos)
                        gap_width_priors[pl] = pm.Deterministic("gap_width_prior_"+pl,
                                                                tt._shared(self.planets[pl]['per_gaps']['gap_probs']))
                    else:
                        gap_width_priors[pl] = tt.zeros(self.n_margs[pl])
                    
                    #tt.printing.Print("per_priors")(per_priors[pl])
                    #tt.printing.Print("geom_ecc_priors")(geom_ecc_priors[pl])
                    #tt.printing.Print("ecc_lim_priors")(ecc_lim_priors[pl])
                    #tt.printing.Print("b_priors")(b_priors[pl])
                    #tt.printing.Print("v_priors")(v_priors[pl])
                    #tt.printing.Print("edge_priors")(edge_priors[pl])
                    #tt.printing.Print("gap_width_priors")(gap_width_priors[pl])
                    #tt.printing.Print("Krv_priors")(Krv_priors[pl])
                    #Summing up for total log prior for each alias/gap:
                    logpriors[pl]=pm.Deterministic("logprior_"+pl, per_priors[pl] + geom_ecc_priors[pl] + ecc_lim_priors[pl] + \
                                           b_priors[pl] + v_priors[pl] + edge_priors[pl] + gap_width_priors[pl])#+Krv_priors[pl]
                
                    if pl in self.duos+self.monos and hasattr(self,'rvs'):
                        if not hasattr(model,'nonmarg_rvs'):
                            if (len(self.multis)+len(self.rvplanets))>1:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs", (rv_trend + tt.sum([model_rvs[ipl] for ipl in self.multis+list(self.rvplanets.keys())],axis=1)))
                            elif (len(self.multis)+len(self.rvplanets))==1:
                                onlypl=self.multis+list(self.rvplanets.keys())
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",(rv_trend+model_rvs[onlypl[0]]))
                            else:
                                nonmarg_rvs = pm.Deterministic("nonmarg_rvs",rv_trend)                        
                        rvlogliks[pl]=pm.Deterministic("rv_loglik_"+pl, tt.tile(sum_log_rverr,self.n_margs[pl]) - tt.sum((self.rvs['rv'][:,None] - (nonmarg_rvs.dimshuffle(0,'x') + model_rvs[pl]))**2/(new_rverr.dimshuffle(0,'x')**2),axis=0))
                    elif hasattr(self,'rvs'):
                        rvlogliks[pl] = pm.Deterministic("rv_loglik_"+pl, sum_log_rverr - tt.sum((self.rvs['rv'] - (model_rvs[pl] + rv_trend))**2/new_rverr**2))

                        #tt.printing.Print("rvlogprobs")(rvlogliks[pl])
                        '''model_rvs_i=[]
                        irvlogliks=[]
                        for i in range(self.n_margs[pl]):
                            model_rvs_i+=[rvorbits[pl].get_radial_velocity(self.rvs['time'],Ks[pl])]
                            imodel = rv_mean + model_rvs_i[-1] + tt.sum([model_rvs[ipl] for ipl in self.multis],axis=1)
                            irvlogliks+=[sum_log_rverr - tt.sum(-(self.rvs['rv']-imodel)**2/(new_rverr2))]
                        rvlogliks[pl] = pm.Deterministic('rvloglik_'+pl, tt.stack(irvlogliks,axis=-1))
                        model_rvs[pl] = pm.Deterministic('model_rv_'+pl, tt.stack(model_rvs_i,axis=-1))'''
                    else:
                        rvlogliks[pl]=0.0

            if not self.use_GP:
                #Calculating some extra info to speed up the loglik calculation
                new_yerr = lc['flux_err'][lc['mask']].astype(floattype)**2 + \
                           tt.sum(lc['flux_err_index'][lc['mask']]*tt.exp(logs2).dimshuffle('x',0),axis=1)
                new_yerr_sq = new_yerr**2
                sum_log_new_yerr = tt.sum(-np.sum(lc['mask'])/2 * tt.log(2*np.pi*(new_yerr_sq)))

            stacked_marg_lc={};resids={};logprobs={};logprob_sums={};logprob_margs={}
            
            for pl in self.multis+self.duos+self.monos+list(self.rvplanets.keys()):
                if pl in self.multis:
                    #No marginalisation needed for multi-transit candidates, or in the case where we interpolate v_priors
                    stacked_marg_lc[pl]=light_curves[pl]
                    if hasattr(self,'rvs'):
                        marg_rv_models[pl] = model_rvs[pl][:,0]
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
                            resids[pl][n] = lc['flux'][lc['mask']].astype(floattype) - \
                                     (light_curves[pl][lc['mask'],n] + tt.sum([ilc for ilc in stacked_marg_lc],axis=1)[lc['mask']] + \
                                      mean.dimshuffle('x'))
                            if self.debug: 
                                tt.printing.Print("rawflux_"+str(n))(tt._shared(lc['flux'][lc['mask']]))
                                tt.printing.Print("models_"+str(n))(iter_models[pl]['lcs'][lc['mask'],n])
                                tt.printing.Print("resids_"+str(n))(resids[pl][n])
                                tt.printing.Print("resids_max_"+str(n))(tt.max(resids[pl][n]))
                                tt.printing.Print("resids_min_"+str(n))(tt.min(resids[pl][n]))
                            if self.use_GP:
                                ilogliks+=[self.gp['use'].log_likelihood(y=resids[pl][n])]
                            else:
                                ilogliks+=[sum_log_new_yerr - tt.sum((resids[pl][n])**2/(new_yerr_sq))]
                                #Saving models:
                                
                        lclogliks[pl] = pm.Deterministic('lcloglik_'+pl, tt.stack(ilogliks))
                    elif self.interpolate_v_prior:
                        #Assume there is no loglikelihood difference (all difference comes from prior)
                        lclogliks[pl]=tt.zeros(self.n_margs[pl])
                        
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
                        pm.Deterministic('ecc_marg_'+pl,tt.sum(eccs[pl]*tt.exp(logprob_margs[pl])))
                        pm.Deterministic('omega_marg_'+pl,tt.sum(omegas[pl]*tt.exp(logprob_margs[pl])))
                    elif self.interpolate_v_prior:
                        #Getting double-marginalised eccentricity (across omega space given v and then period space)
                        #tt.printing.Print("input_coords")(tt.stack([logvels[pl],max_eccs[pl]],axis=-1))
                        eccs[pl] = pm.Deterministic('ecc_'+pl, 
                                         self.interpolator_eccmarg.evaluate(tt.stack([logvels[pl],max_eccs[pl]],axis=-1)).T[0])
                        #tt.printing.Print("eccs[pl]")(eccs[pl])
                        pm.Deterministic('ecc_marg_'+pl,tt.sum(eccs[pl]*tt.exp(logprob_margs[pl])))
                    if 'tdur' not in self.fit_params:
                        pm.Deterministic('tdur_marg_'+pl,tt.sum(tdurs[pl]*tt.exp(logprob_margs[pl])))
                    if 'b' not in self.fit_params:
                        pm.Deterministic('b_marg_'+pl,tt.sum(bs[pl]*tt.exp(logprob_margs[pl])))
                    if 'logror' not in self.fit_params:
                        pm.Deterministic('logror_marg_'+pl,tt.sum(logrors[pl]*tt.exp(logprob_margs[pl])))
                    pm.Deterministic('vel_marg_'+pl,tt.sum(vels[pl]*tt.exp(logprob_margs[pl])))
                    pm.Deterministic('per_marg_'+pl,tt.sum(pers[pl]*tt.exp(logprob_margs[pl])))
                    if not self.interpolate_v_prior:
                        stacked_marg_lc[pl] = pm.Deterministic('marg_light_curve_'+pl,
                                              tt.sum(light_curves[pl] * tt.exp(logprob_margs[pl]).dimshuffle('x',0),axis=1))
                    else:
                        stacked_marg_lc[pl] = light_curves[pl]
                    if hasattr(self,'rvs'):
                        marg_rv_models[pl] = pm.Deterministic('marg_rv_model_'+pl,
                                              tt.sum(model_rvs[pl] * tt.exp(logprob_margs[pl]).dimshuffle('x',0),axis=1))
                        #tt.printing.Print("lclogliks")(lclogliks[pl])
                        #tt.printing.Print("logpriors")(logpriors[pl])
                        #tt.printing.Print("rvlogliks")(rvlogliks[pl])
                        #tt.printing.Print("logprobs[pl]")(logprobs[pl])
                        #tt.printing.Print("logprobmargs")(logprob_margs[pl])
                        #tt.printing.Print("exp(logprobmargs)")(tt.exp(logprob_margs[pl]).dimshuffle('x',0))
                        #tt.printing.Print("marg_rv_models")(marg_rv_models[pl])
                        pm.Deterministic('K_marg_'+pl,tt.sum(Ks[pl]*tt.exp(logprob_margs[pl])))
                        pm.Deterministic('Mp_marg_'+pl,tt.sum(Mps[pl]*tt.exp(logprob_margs[pl])))
                        pm.Deterministic('rho_marg_'+pl,tt.sum(rhos[pl]*tt.exp(logprob_margs[pl])))
            
            ################################################
            #     Compute combined model & log likelihood
            ################################################
            marg_all_lc_model = pm.Deterministic("marg_all_lc_model",
                                                    tt.sum([stacked_marg_lc[pl] for pl in self.planets],axis=0))
            
            if hasattr(self,'rvs'):
                if (len(self.planets)+len(self.rvplanets))>1:
                    marg_all_rv_model = pm.Deterministic("marg_all_rv_model",
                            tt.sum([marg_rv_models[pl] for pl in list(self.planets.keys())+list(self.rvplanets.keys())],axis=0))
                else:
                    rvkey=list(self.planets.keys())+list(self.rvplanets.keys())
                    marg_all_rv_model = pm.Deterministic("marg_all_rv_model", marg_rv_models[rvkey[0]])
                margrvloglik = pm.Normal("margrvloglik", mu=marg_all_rv_model+rv_trend, sd=new_rverr, observed=self.rvs['rv'])
            if self.use_GP:
                total_llk = pm.Deterministic("total_llk",self.gp['use'].log_likelihood(lc['flux'][lc['mask']] - \
                                                                                (marg_all_lc_model[lc['mask']] + mean)))
                gp_pred = pm.Deterministic("gp_pred", self.gp['use'].predict(lc['time'].astype(floattype),
                                                                             return_var=False))
                pm.Potential("llk_gp", total_llk)
                #pm.Normal("all_obs",mu=(marg_all_lc_model + gp_pred + mean),sd=new_yerr,
                #          observed=self.lc['flux'][self.lc['near_trans']].astype(floattype))
            else:
                
                marglcloglik=pm.Normal("marglcloglik",mu=(marg_all_lc_model[lc['mask']] + mean), sd=new_yerr,
                                    observed=lc['flux_flat'][lc['mask']].astype(floattype))
                #tt.printing.Print("marglcloglik")(marglcloglik)
                
            #all_loglik = pm.Normal("all_loglik", mu=tt.concatenate([(marg_all_rv_model+rv_trend).flatten(),
            #                                                  (marg_all_lc_model[lc['mask']] + mean).flatten()]),
            #                                                  sd=tt.concatenate([new_rverr.flatten(),new_yerr.flatten()]),
            #                                                  observed=np.hstack([self.rvs['rv'],lc['flux'][lc['mask']]]))

                
            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            if self.debug: print("optimizing model",model.test_point)
            
            ################################################
            #   Creating initial model optimisation menu:
            ################################################
            #Setting up optimization depending on what planet models we have:
            initvars0=[]#r,b
            initvars1=[]#P
            initvars2=[logrho_S]#r,b,t0
            initvars3=[logs2]
            initvars4=[]#r,b,P
            for pl in self.planets:
                initvars2+=[t0s[pl]]
                initvars0+=[logrors[pl]];initvars1+=[logrors[pl]];initvars2+=[logrors[pl]];initvars4+=[logrors[pl]]
                if 'b' in self.fit_params or pl in self.multis:
                    initvars0+=[bs[pl]];initvars2+=[bs[pl]];initvars4+=[bs[pl]]
                if 'tdur' in self.fit_params and pl not in self.multis:
                    initvars0+=[tdurs[pl]];initvars2+=[tdurs[pl]];initvars4+=[tdurs[pl]]
                if pl in self.multis:
                    initvars1+=[pers[pl]];initvars4+=[pers[pl]]
                if pl in self.monos:
                    initvars1+=[mono_uniform_index_period[pl]];initvars4+=[mono_uniform_index_period[pl]]
                if pl in self.duos:
                    initvars1+=[t0_2s[pl]]
                if not self.assume_circ and (not self.interpolate_v_prior or pl in self.multis):
                    initvars2+=[eccs[pl], omegas[pl]]
                if hasattr(self,'rvs') and not self.derive_K:
                    initvars0+=[logKs[pl]]
                    initvars2+=[logKs[pl]]
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
                initvars3+=[logs2, power, w0, mean]
            else:
                initvars3+=[mean]
            initvars5=initvars2+initvars3+[logs2,Rs,Ms]
            if np.any([c[0].lower()=='t' for c in self.cads]):
                initvars5+=[u_star_tess]
            if np.any([c[0].lower()=='k' for c in self.cads]):
                initvars5+=[u_star_kep]
            if np.any([c[0].lower()=='c' for c in self.cads]):
                initvars5+=[u_star_corot]
            if np.any([c[0].lower()=='x' for c in self.cads]):
                initvars5+=[u_star_cheops]

            ################################################
            #                  Optimising:
            ################################################

            if self.debug: print("before",model.check_test_point())
            map_soln = xo.optimize(start=start, vars=initvars0,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars1,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars2,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars3,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars4,verbose=True)
            #Doing everything except the marginalised periods:
            map_soln = xo.optimize(start=map_soln, vars=initvars5)
            map_soln = xo.optimize(start=map_soln)

            if self.debug: print("after",model.check_test_point())

            self.model = model
            self.init_soln = map_soln
    
    def RunMcmc(self, n_draws=500, plot=True, n_burn_in=None, overwrite=False, continuesampling=False, chains=4, **kwargs):
        #if not hasattr(self,'trace') and self.use_GP:
        #    #Adding a step to re-do the lightcurve flattening using the new transt durations in the non-GP case
        #    self.init_lc()
        
        if not overwrite:
            self.LoadPickle()
            if hasattr(self,'trace') and self.debug:
                print("LOADED MCMC")
        
        if not (hasattr(self,'trace') or hasattr(self,'trace_df')) or overwrite or continuesampling:
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
                if hasattr(self,'trace') and continuesampling:
                    print("Using already-generated MCMC trace as start point for new trace")
                    self.trace = pm.sample(tune=n_burn_in, draws=n_draws, chains=chains, trace=self.trace,
                                           step=xo.get_dense_nuts_step(target_accept=0.9), compute_convergence_checks=False)
                else:
                    self.trace = pm.sample(tune=n_burn_in, draws=n_draws, start=self.init_soln, chains=chains,
                                           step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)
            #Saving both the class and a pandas dataframe of output data.
            self.SaveModelToFile()
            _=self.MakeTable(save=True)
        elif not (hasattr(self,'trace') or hasattr(self,'trace_df')):
            print("Trace or trace df exists...")
            
        
    def Table(self):
        if LoadFromFile and not self.overwrite and os.path.exists(self.savenames[0]+'_results.txt'):
            with open(self.savenames[0]+'_results.txt', 'r', encoding='UTF-8') as file:
                restable = file.read()
        else:
            restable=self.ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
                                       savename=self.savenames[0]+'_results.txt', overwrite=False,
                                       savefileloc=None, tracemask=tracemask)
    
    
    def init_gp_to_plot(self,n_samp=7,max_gp_len=12000):
        n_samp = 7 if n_samp is None else n_samp
        print("Initalising GP models for plotting with n_samp=",n_samp)
        import celerite
        from celerite import terms
        if not hasattr(self,'gap_lens'):
            self.init_plot()
        gp_pred=[]
        gp_sd=[]
        self.gp_to_plot={'n_samp':n_samp}
        if hasattr(self,'trace'):
            #Using the output of the model trace
            medvars=[var for var in self.trace.varnames if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
            self.meds={}
            for mv in medvars:
                if len(self.trace[mv].shape)>1:
                    self.meds[mv]=np.median(self.trace[mv],axis=0)
                elif len(self.trace[mv].shape)==1:
                    self.meds[mv]=np.median(self.trace[mv])  
        else:
            self.meds=self.init_soln

        limit_mask_bool={}

        # TBD HERE

        if n_samp==1:
            #Creating the median model:

            for n in np.arange(len(self.lc['limits'])):

                #Only creating out-of-transit GP for the binned (e.g. 30min) data
                cutBools = tools.cutLc(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],max_gp_len,
                                       transit_mask=~self.lc['in_trans'][self.lc['limits'][n][0]:self.lc['limits'][n][1]])

                limit_mask_bool[n]={}
                for nc,c in enumerate(cutBools):
                    limit_mask_bool[n][nc]=np.tile(False,len(self.lc['time']))
                    limit_mask_bool[n][nc][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]=self.lc['limit_mask'][n][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]
                    i_kernel = terms.SHOTerm(log_S0=np.log(self.meds['S0']), log_omega0=np.log(self.meds['w0']), log_Q=np.log(1/np.sqrt(2)))
                    i_gp = celerite.GP(i_kernel,mean=self.meds['mean'],fit_mean=False)

                    i_gp.compute(self.lc['time'][limit_mask_bool[n][nc]].astype(floattype),
                                 np.sqrt(self.lc['flux_err'][limit_mask_bool[n][nc]]**2 + \
                                 np.dot(self.lc['flux_err_index'][limit_mask_bool[n][nc]],np.exp(self.meds['logs2']))))
                    #llk=i_gp.log_likelihood(mod.lc['flux'][mod.lc['mask']][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]-mod.trans_to_plot['all']['med'][mod.lc['mask']][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]-mod.meds['mean'])
                    #print(llk.eval())
                    i_gp_pred, i_gp_var= i_gp.predict(self.lc['flux'][limit_mask_bool[n][nc]] - \
                                        self.trans_to_plot['all']['med'][limit_mask_bool[n][nc]],
                                        t=self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c].astype(floattype),
                                        return_var=True,return_cov=False)
                    gp_pred+=[i_gp_pred]
                    gp_sd+=[np.sqrt(i_gp_var)]
            ''''
            gp_pred=[];gp_sd=[]
            for n in np.arange(len(self.lc['limits'])):
                with self.model:
                    pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],
                                                            return_var=True,return_cov=False),self.meds)                    
                gp_pred+=[pred]
                gp_sd+=[np.sqrt(var)]
                print(n,len(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]]),'->',len(gp_sd[-1]),len(gp_pred[-1]))
            '''
            self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
            self.gp_to_plot['gp_sd']=np.hstack(gp_sd)
            '''
            with self.model:
                pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc['time'].astype(floattype),
                                                                 return_var=True,return_cov=False),self.meds)                    
            self.gp_to_plot['gp_pred']=pred
            self.gp_to_plot['gp_sd']=np.sqrt(var)
            '''
        elif n_samp>1:
            assert hasattr(self,'trace')
            #Doing multiple samples and making percentiles:
            for n in np.arange(len(self.lc['limits'])):
                #Need to break up the lightcurve even further to avoid GP burning memory:
                cutBools = tools.cutLc(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],max_gp_len,
                                       transit_mask=~self.lc['in_trans'][self.lc['limits'][n][0]:self.lc['limits'][n][1]])
                i_kernel = terms.SHOTerm(log_S0=np.log(self.meds['S0']), log_omega0=np.log(self.meds['w0']), log_Q=np.log(1/np.sqrt(2)))
                i_gp = celerite.GP(i_kernel,mean=self.meds['mean'],fit_mean=False)
                limit_mask_bool[n]={}
                for nc,c in enumerate(cutBools):
                    limit_mask_bool[n][nc]=np.tile(False,len(self.lc['time']))
                    limit_mask_bool[n][nc][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]=self.lc['limit_mask'][n][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c]
                    i_gp_pred=[]
                    i_gp_var=[]
                    for i, sample in enumerate(xo.get_samples_from_trace(self.trace, size=n_samp)):
                        #print(np.exp(sample['logs2']))
                        i_gp.set_parameter('kernel:log_S0',np.log(sample['S0']))
                        i_gp.set_parameter('kernel:log_omega0',np.log(sample['w0']))
                        i_gp.set_parameter('mean:value',sample['mean'])
                        i_gp.compute(self.lc['time'][limit_mask_bool[n][nc]],
                                     np.sqrt(self.lc['flux_err'][limit_mask_bool[n][nc]]**2 + \
                                      np.dot(self.lc['flux_err_index'][limit_mask_bool[n][nc]], np.exp(sample['logs2']))))
                        marg_lc=np.tile(0.0,len(self.lc['time']))
                        if hasattr(self,'pseudo_binlc') and len(self.trans_to_plot_i['all']['med'])==len(self.pseudo_binlc['time']):
                            marg_lc[self.lc['near_trans']]=sample['marg_all_lc_model'][self.pseudo_binlc['near_trans']]
                        elif hasattr(self,'lc_near_trans') and len(self.trans_to_plot_i['all']['med'])==len(self.lc_near_trans['time']):
                            marg_lc[self.lc['near_trans']]=sample['marg_all_lc_model'][key1][key2]
                        elif len(self.trans_to_plot_i['all']['med'])==len(self.lc['time']):
                            marg_lc[self.lc['near_trans']]=sample['marg_all_lc_model'][key1][key2][self.lc['near_trans']]

                        
                        #marg_lc[self.lc['near_trans']]=sample['marg_all_lc_model'][self.lc['near_trans']]
                        ii_gp_pred, ii_gp_var= i_gp.predict(self.lc['flux'][limit_mask_bool[n][nc]] - marg_lc[limit_mask_bool[n][nc]],
                                        t=self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]][c].astype(floattype),
                                        return_var=True,return_cov=False)

                        i_gp_pred+=[ii_gp_pred]
                        i_gp_var+=[ii_gp_var]
                    av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
                    gp_pred+=[av]
                    gp_sd+=[std]
            self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
            self.gp_to_plot['gp_sd']=np.hstack(gp_sd)

    def init_trans_to_plot(self,n_samp=None):
        n_samp=len(self.trace['mean']) if n_samp is None else n_samp
        print("Initalising Transit models for plotting with n_samp=",n_samp)
        if not hasattr(self,'gap_lens'):
            self.init_plot()
        self.trans_to_plot_i={}
        self.trans_to_plot_i['all']={}
        percentiles=(2.2750132, 15.8655254, 50., 84.1344746, 97.7249868)
        if hasattr(self,'trace') and 'marg_all_lc_model' in self.trace.varnames:
            prcnt=np.percentile(self.trace['marg_all_lc_model'],percentiles,axis=0)
            nms=['-2sig','-1sig','med','+1sig','+2sig']
            self.trans_to_plot_i['all']={nms[n]:prcnt[n] for n in range(5)}
        elif 'marg_all_lc_model' in self.init_soln:
            self.trans_to_plot_i['all']['med']=self.init_soln['marg_all_lc_model']
        else:
            print("marg_all_lc_model not in any optimised models")
        for pl in self.planets:
            if hasattr(self,'trace') and pl+"_light_curves" in self.trace.varnames:
                prcnt = np.percentile(self.trace[pl+"_light_curves"], percentiles, axis=0)
                nms=['-2sig','-1sig','med','+1sig','+2sig']
                self.trans_to_plot_i[pl] = {nms[n]:prcnt[n] for n in range(5)}
            elif hasattr(self,'init_soln') and pl+"_light_curves" in self.init_soln:
                self.trans_to_plot_i[pl] = {'med':self.init_soln[pl+"_light_curves"]}
        '''
            self.trans_to_plot_i[pl]={}
            if pl in self.multis or self.interpolate_v_prior:
                if hasattr(self,'trace') and 'light_curve_'+pl in self.trace.varnames:
                    if len(self.trace['mask_light_curves'].shape)>2:
                        prcnt = np.percentile(self.trace['multi_mask_light_curves'][:,:,self.multis.index(pl)],
                                                  (5,16,50,84,95),axis=0)
                    else:
                        prcnt = np.percentile(self.trace['multi_mask_light_curves'], (5,16,50,84,95), axis=0)

                elif 'multi_mask_light_curves' in self.init_soln:
                    if len(self.init_soln['multi_mask_light_curves'].shape)==1:
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves']
                    else:    
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves'][:,self.multis.index(pl)]
                else:
                    print('multi_mask_light_curves not in any optimised models')
            elif pl in self.duos or self.monos and not self.interpolate_v_prior:
                if hasattr(self,'trace') and 'marg_light_curve_'+pl in self.trace.varnames:
                    prcnt=np.percentile(self.trace['marg_light_curve_'+pl],(5,16,50,84,95),axis=0)
                    nms=['-2sig','-1sig','med','+1sig','+2sig']
                    self.trans_to_plot_i[pl] = {nms[n]:prcnt[n] for n in range(5)}
                elif 'marg_light_curve_'+pl in self.init_soln:
                    self.trans_to_plot_i[pl]['med'] = self.init_soln['marg_light_curve_'+pl]
                else:
                    print('marg_light_curve_'+pl+' not in any optimised models')
        '''
        self.trans_to_plot={'n_samp':n_samp}

        #Adding zeros to other regions where we dont have transits (not in the out of transit mask):
        for key1 in self.trans_to_plot_i:
            self.trans_to_plot[key1]={}
            for key2 in self.trans_to_plot_i[key1]:
                self.trans_to_plot[key1][key2]=np.zeros(len(self.lc['time']))
                if hasattr(self,'pseudo_binlc') and len(self.trans_to_plot_i[key1]['med'])==len(self.pseudo_binlc['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2][self.pseudo_binlc['near_trans']]
                elif hasattr(self,'lc_near_trans') and len(self.trans_to_plot_i[key1]['med'])==len(self.lc_near_trans['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2]
                elif len(self.trans_to_plot_i[key1]['med'])==len(self.lc['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2][self.lc['near_trans']]
        if len(self.planets)==1 and list(self.planets.keys())[0] not in self.trans_to_plot:
            self.trans_to_plot[list(self.planets.keys())[0]] = self.trans_to_plot['all']
    
    def init_rvs_to_plot(self, n_samp=None, plot_alias='all'):
        #Going from the outputted samples back through exoplanet model to create times on a fine grid
        
        n_samp = 300 if n_samp is None else n_samp
        
        all_pls_in_rvs=list(self.planets.keys())+list(self.rvplanets.keys())
        
        self.rvs_to_plot={'t':{pl:{} for pl in all_pls_in_rvs},
                          'x':{pl:{} for pl in all_pls_in_rvs}}
        self.rvs_to_plot['n_samp']=n_samp

        self.rvs_to_plot['t']['time']=np.arange(np.min(self.rvs['time'])-5,np.max(self.rvs['time'])+5,0.5)
        if hasattr(self,'trace'):
            samples=xo.get_samples_from_trace(self.trace, size=n_samp)
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
                            rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                              rho_star=sample['rho_S']*1.40978, 
                                                              period=sample['per_'+pl],t0=sample['t0_'+pl],b=sample['b_'+pl]
                                                  ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]),sample)
                        else:
                            rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                              rho_star=sample['rho_S']*1.40978, 
                                                              period=sample['per_'+pl],t0=sample['t0_'+pl],b=sample['b_'+pl],
                                                              ecc=sample['ecc_'+pl],omega=sample['omega_'+pl]
                                                   ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]),sample)
                        marg_rv_ts_i[pl]+=[rvs]
                    elif pl in self.rvplanets:
                        
                        rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(period=sample['per_'+pl], t0=sample['t0_'+pl],
                                                                        ecc=sample['ecc_'+pl], omega=sample['omega_'+pl]
                                               ).get_radial_velocity(self.rvs_to_plot['t']['time'], K=sample['K_'+pl]),sample)
                        marg_rv_ts_i[pl]+=[rvs]

                    else:
                        if self.interpolate_v_prior:
                            rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978, 
                                                    period=sample['per_'+pl],
                                                    t0=tt.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tt.tile(sample['b_'+pl],self.n_margs[pl]),
                                                    ecc=sample['min_ecc_'+pl],omega=sample['omega_'+pl]
                                              ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]),sample)
                        elif not self.assume_circ:
                            rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978, 
                                                    period=sample['per_'+pl],
                                                    t0=tt.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tt.tile(sample['b_'+pl],self.n_margs[pl]),
                                                    ecc=tt.tile(sample['ecc_'+pl],self.n_margs[pl]),
                                                    omega=tt.tile(sample['omega_'+pl],self.n_margs[pl])
                                             ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]),sample)
                        elif self.assume_circ:
                            rvs = xo.eval_in_model(xo.orbits.KeplerianOrbit(r_star=sample['Rs'],
                                                    rho_star=sample['rho_S']*1.40978, 
                                                    period=sample['per_'+pl],
                                                    t0=tt.tile(sample['t0_'+pl],self.n_margs[pl]),
                                                    b=tt.tile(sample['b_'+pl],self.n_margs[pl])
                                              ).get_radial_velocity(self.rvs_to_plot['t']['time'],sample['K_'+pl]),sample)
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
                xprcnts      = np.percentile(self.trace["marg_rv_model_"+pl], percentiles, axis=0)
                xtrendprcnts = np.percentile(self.trace["marg_rv_model_"+pl]+self.trace["rv_trend"], percentiles, axis=0)
                
                self.rvs_to_plot['x'][pl]['marg'] = {nms[n]:xprcnts[n] for n in range(5)}
                self.rvs_to_plot['x'][pl]['marg+trend'] = {nms[n]:xtrendprcnts[n] for n in range(5)}
                tprcnts = np.percentile(np.vstack(marg_rv_ts_i[pl]), percentiles, axis=0)
                ttrendprcnts = np.percentile(np.vstack(marg_rv_ts_i[pl])+np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t'][pl]['marg'] = {nms[n]:tprcnts[n] for n in range(5)}
                self.rvs_to_plot['t'][pl]['marg+trend'] = {nms[n]:ttrendprcnts[n] for n in range(5)}
                if pl in self.duos or pl in self.monos:
                    alltrvs = np.dstack(all_rv_ts_i[pl])
                    for i in range(self.n_margs[pl]):
                        xiprcnts=np.percentile(self.trace["model_rv_"+pl][:,:,i], percentiles, axis=0)
                        self.rvs_to_plot['x'][pl][i]={nms[n]:xiprcnts[n] for n in range(5)}
                        tiprcnts=np.percentile(alltrvs[:,i,:], percentiles, axis=1)
                        self.rvs_to_plot['t'][pl][i]={nms[n]:tiprcnts[n] for n in range(5)}                        
            #print(self.rvs_to_plot)    
            if len(all_pls_in_rvs)>1:
                iprcnts = np.percentile(self.trace["rv_trend"],percentiles, axis=0)
                self.rvs_to_plot['x']["trend+offset"] = {nms[n]:iprcnts[n] for n in range(5)}
                iprcnts = np.percentile(self.trace["marg_all_rv_model"],percentiles, axis=0)
                self.rvs_to_plot['x']["all"] = {nms[n]:iprcnts[n] for n in range(5)}
                iprcnts = np.percentile(self.trace["marg_all_rv_model"]+self.trace["rv_trend"],percentiles, axis=0)
                self.rvs_to_plot['x']["all+trend"] = {nms[n]:iprcnts[n] for n in range(5)}

                iprcnts = np.percentile(np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t']["trend"] = {nms[n]:iprcnts[n] for n in range(5)}
                print(self.rvs_to_plot['t']["trend"])
                iprcnts = np.percentile(np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=1),percentiles, axis=0)
                self.rvs_to_plot['t']["all"] = {nms[n]:iprcnts[n] for n in range(5)}
                print(len(trends_i), len(trends_i[0]), np.vstack(trends_i).shape)
                print(np.vstack(marg_rv_ts_i[pl]).shape,
                      np.dstack([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs]).shape,
                     np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=1).shape)
                iprcnts = np.percentile(np.sum([np.vstack(marg_rv_ts_i[pl]) for pl in all_pls_in_rvs],axis=0)+np.vstack(trends_i),
                                        percentiles, axis=0)
                self.rvs_to_plot['t']["all+trend"] = {nms[n]:iprcnts[n] for n in range(5)}
            else:
                iprcnts = np.percentile(self.trace["rv_trend"], percentiles, axis=0)
                self.rvs_to_plot['x']["trend+offset"] = {nms[n]:iprcnts[n] for n in range(5)}
                self.rvs_to_plot['x']["all"] = self.rvs_to_plot['x'][pl]
                self.rvs_to_plot['x']["all+trend"] = self.rvs_to_plot['x'][pl]["marg+trend"]

                iprcnts = np.percentile(np.vstack(trends_i), percentiles, axis=0)
                self.rvs_to_plot['t']["trend"] = {nms[n]:iprcnts[n] for n in range(5)}
                print(self.rvs_to_plot['t']["trend"])
                self.rvs_to_plot['t']["all"] = self.rvs_to_plot['t'][pl]
                self.rvs_to_plot['t']["all+trend"] = self.rvs_to_plot['t'][pl]["marg+trend"]
            iprcnts = np.percentile(self.trace["rv_offsets"], percentiles, axis=0)
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
        
    def init_plot(self, interactive=False, gap_thresh=10,plottype='lc',pointcol='k',palette=None, ncols=None):
        import seaborn as sns
        if palette is None:
            ncols = len(self.planets)+3 if ncols is None else ncols
            self.pal = sns.color_palette('viridis', ncols).as_hex()
        else:
            self.pal = sns.color_palette(palette).as_hex()
        if pointcol=="k":
            sns.set_style('whitegrid')
            #Plots bokeh figure
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 0.75
        
        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        #Making sure lc is binned to 30mins
        if plottype=='lc':
            self.lc=tools.lcBin(self.lc,binsize=1/48.0)
        
            if not hasattr(self,'gap_lens'):
                #Finding if there's a single enormous gap in the lightcurve, and creating time splits for each region
                x_gaps=np.hstack((0, np.where(np.diff(self.lc['time'])>gap_thresh)[0]+1, len(self.lc['time'])))
                self.lc['limits']=[]
                self.lc['binlimits']=[]
                gap_lens=[]
                for ng in range(len(x_gaps)-1):
                    self.lc['limits']+=[[x_gaps[ng],x_gaps[ng+1]]]
                    gap_lens+=[self.lc['time'][self.lc['limits'][-1][1]-1]-self.lc['time'][self.lc['limits'][-1][0]]]
                    self.lc['binlimits']+=[[np.argmin(abs(self.lc['bin_time']-self.lc['time'][x_gaps[ng]])),
                                 np.argmin(abs(self.lc['bin_time']-self.lc['time'][x_gaps[ng+1]-1]))+1]]
                self.lc['gap_lens']=np.array(gap_lens)
                all_lens=np.sum(self.lc['gap_lens'])
                self.lc['limit_mask']={}
                #modlclim_mask={}
                for n in range(len(self.lc['gap_lens'])):
                    #modlclim_mask[n]=np.tile(False,len(self.plot_lc['time']))
                    #modlclim_mask[n][modlclims[n][0]:modlclims[n][1]][lc['mask'][modlclims[n][0]:modlclims[n][1]]]=True
                    self.lc['limit_mask'][n]=np.tile(False,len(self.lc['time']))
                    self.lc['limit_mask'][n][self.lc['limits'][n][0]:self.lc['limits'][n][1]][self.lc['mask'][self.lc['limits'][n][0]:self.lc['limits'][n][1]]]=True
        #elif plottype=='rv':
    

    def PlotRVs(self, interactive=False, plot_alias='best', nbest=4, n_samp=None, overwrite=False, return_fig=False, 
                plot_loc=None, palette=None, pointcol='k', plottype='png',raster=False, nmargtoplot=0, save=True):
        ################################################################
        #     Varied plotting function for RVs of MonoTransit model
        ################################################################
        import seaborn as sns
        sns.set_palette('viridis')
        
        # plot_alias - can be 'all' or 'best'. All will plot all aliases. Best will assume the highest logprob.
        ncol=3+2*np.max(list(self.n_margs.values())) if plot_alias=='all' else 3+2*nbest
        self.init_plot(plottype='rv', pointcol=pointcol, ncols=ncol)
        
        if not hasattr(self,'rvs_to_plot') or n_samp!=self.rvs_to_plot['n_samp'] or overwrite:
            self.init_rvs_to_plot(n_samp, plot_alias)

        averr=np.nanmedian(self.rvs['rv_err'])
        
        all_pls_in_rvs=list(self.planets.keys())+list(self.rvplanets.keys())
        
        other_pls=self.multis+list(self.rvplanets.keys())
        
        marg_pl=(self.monos+self.duos)[nmargtoplot]
        #Here we'll choose the best RV curves to plot (in the case of mono/duos)
        nbests = self.n_margs[marg_pl] if plot_alias=='all' else nbest
        if hasattr(self,'trace'):
            ibest = np.nanmedian(self.trace['logprob_marg_'+marg_pl],axis=0).argsort()[-1*nbests:]
            heights = np.array([np.clip(np.nanmedian(self.trace['K_'+marg_pl][:,i]),0.5*averr,10000) for i in ibest])
        elif hasattr(self,'init_soln'):
            ibest = self.init_soln['logprob_marg_'+marg_pl].argsort()[-1*nbests:]
            heights = np.array([np.clip(self.init_soln['K_'+marg_pl][i],0.5*averr,10000) for i in ibest])
        if len(other_pls)==1:
            heights=np.max(heights)+list(np.array(heights)+np.max(heights))
        heights= np.round(heights[::-1]*24/np.sum(heights[::-1]))
        heights_sort = np.hstack((0,np.cumsum(heights).astype(int)))+6*len(other_pls)
                    
        if interactive:
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
            gs = fig.add_gridspec(heights_sort[-1],3*(3+len(all_pls_in_rvs)),wspace=0.3,hspace=0.001)
            f_all_resids = fig.add_subplot(gs[int(np.floor(0.75*heights_sort[-1])):,:2*(3+len(all_pls_in_rvs))])
            #f_all_resids=fig.add_subplot(gs[4,:(2*len(all_pls_in_rvs))])
            f_alls=fig.add_subplot(gs[:int(np.floor(0.75*heights_sort[-1])),:2*(3+len(all_pls_in_rvs))],sharex=f_all_resids)
            #looping through each planet and each alias we want to plot:
            pl=(self.duos+self.monos)[nmargtoplot]
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
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["trend"]["-2sig"],
                                    self.rvs_to_plot['t']["trend"]["+2sig"],'--',color='C6',alpha=0.1)
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["trend"]["-1sig"],
                                    self.rvs_to_plot['t']["trend"]["+1sig"],'--',color='C6',alpha=0.1,label='trend')
            f_alls.plot(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["med"],c='C1',label='marg. model')
            if "-2sig" in self.rvs_to_plot['t']["all+trend"]:
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["-2sig"],
                                    self.rvs_to_plot['t']["all+trend"]["+2sig"],'--',color='C1',alpha=0.1)
                f_alls.fill_between(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["-1sig"],
                                    self.rvs_to_plot['t']["all+trend"]["+1sig"],'--',color='C1',alpha=0.1)
            for nc in range(len(self.rvs['scopes'])):
                scope_ix=self.rvs['tele_index']==self.rvs['scopes'][nc]
                f_alls.errorbar(self.rvs['time'][scope_ix],
                                self.rvs['rv'][scope_ix]-self.rvs_to_plot['x']["offsets"]["med"][nc],
                                yerr=self.rvs['rv_err'][scope_ix],fmt='.',markersize=8,ecolor='#bbbbbb',
                                c='C'+str(nc+1),label='scope:'+self.rvs['scopes'][nc])
                f_all_resids.errorbar(self.rvs['time'][scope_ix], 
                                      self.rvs['rv'][scope_ix] - self.rvs_to_plot["x"]["all+trend"]["med"][scope_ix],
                              yerr=self.rvs['rv_err'][scope_ix] ,fmt='.',markersize=8,ecolor='#bbbbbb',c='C'+str(nc+1))

            plt.setp(f_alls.get_xticklabels(), visible=False)
            f_alls.plot(self.rvs_to_plot['t']['time'], self.rvs_to_plot['t']["all+trend"]["med"],c='C4',alpha=0.6,lw=2.5)
            if "-2sig" in self.rvs_to_plot["x"]["all+trend"]:
                f_all_resids.fill_between(self.rvs['time'], 
                                      self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["+2sig"],
                                      self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["-2sig"],
                                      color='C6',alpha=0.1)
                f_all_resids.fill_between(self.rvs['time'], 
                                      self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["+1sig"],
                                      self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["-1sig"],
                                      color='C6',alpha=0.1)
            f_all_resids.plot(self.rvs['time'], np.tile(0.0,len(self.rvs['time'])),':',color='C6',alpha=0.4,linewidth=3.0)

        elif interactive:
            #For Bokeh plots, we can just use the size in pixels
            f_all_resids=figure(width=800-200*len(all_pls_in_rvs), plot_height=150, title=None)
            f_alls=figure(width=800-200*len(all_pls_in_rvs), plot_height=350, title=None, x_range=f_all_resids.x_range)
            pl=(self.duos+self.monos)[nmargtoplot]
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
                                  upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                  lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                f_all_resids[n].circle(self.rvs['time'][scope_ix] ,
                                       self.rvs['rv'][scope_ix] - self.rvs_to_plot['x']["offsets"]["med"][nc] - \
                                       self.rvs_to_plot["all"]["med"][scope_ix], color='black',
                                       alpha=0.5,size=0.75)

            modelband = ColumnDataSource(data=dict(base=self.rvs['t']['time'],
                                                lower=self.rvs_to_plot['t']['gp_pred'] - self.rvs_to_plot['t']['gp_sd'], 
                                                upper=self.rvs_to_plot['t']['gp_pred'] + self.rvs_to_plot['t']['gp_sd']))
            f_alls[n].add_layout(Band(source=modelband,base='base',lower='lower',upper='upper',
                                      fill_alpha=0.4, line_width=0.0, fill_color=pal[3]))
            f_alls[n].line(self.rvs_to_plot['t'], self.rvs_to_plot['gp_pred'], 
                           line_alpha=0.6, line_width=1.0, color=pal[3], legend="RV model")
            
            
            residband = ColumnDataSource(data=dict(base=self.rvs['t']['time'],
                                                lower= self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["+2sig"],upper=self.rvs_to_plot["x"]["all+trend"]["med"]-self.rvs_to_plot["x"]["all+trend"]["-2sig"]))
            f_alls[n].add_layout(Band(source=modelband,base='base',lower='lower',upper='upper',
                                      fill_alpha=0.4, line_width=0.0, fill_color=pal[3]))

        for n,pl in enumerate(list(self.planets.keys())+list(self.rvplanets.keys())):
            if hasattr(self,'trace'):
                t0=np.nanmedian(self.trace['t0_'+pl])
                if pl in self.multis or pl in self.rvplanets:
                    per=[np.nanmedian(self.trace['per_'+pl])]
                    alphas=1.0
                else:
                    alphas=np.clip(2*np.exp(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0)),0.25,1.0)
                    if pl in self.duos:
                        t0=np.nanmedian(self.trace['t0_2_'+pl])
                        per=np.nanmedian(self.trace['per_'+pl],axis=0)
                    elif pl in self.monos:
                        per=np.nanmedian(self.trace['per_'+pl],axis=0)
            elif hasattr(self,'init_soln'):
                t0=self.init_soln['t0_'+pl]
                if pl in self.multis or pl in self.rvplanets:
                    per=[self.init_soln['per_'+pl]]
                else:
                    per=self.init_soln['per_'+pl]
                if pl in self.duos:
                    t0_2=self.init_soln['t0_2_'+pl]
                    alphas=np.clip(2*np.exp(self.init_soln['logprob_marg_'+pl]),0.25,1.0)
                else:
                    alphas=1.0
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
                    print(pl,n,alias)
                    if hasattr(self,'trace'):
                        K=np.clip(np.nanmedian(self.trace['K_'+pl][:,alias]),averr,100000)
                    else:
                        K=np.clip(self.init_soln['K_'+pl][alias],averr,100000)

                    if interactive:
                        sdbuffer=3
                        errors = ColumnDataSource(data=dict(base=self.rvs_to_plot['x'][pl][alias]['phase'],
                                                lower=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] - self.rvs['rv_err'],
                                                upper=self.rvs['rv']-other_plsx-self.rvs_to_plot['x']['trend+offset']['med'] + self.rvs['rv_err']))
                        f_phase[pl][n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
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
                                   level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[2+n]))
                            trband = ColumnDataSource(data=dict(
                                base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
           lower=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
           upper=np.hstack((0,self.rvs_to_plot['t'][pl][alias]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0))
                                                               ))
                            f_phase[pl][n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                                      level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[2+n]))
                        f_phase[pl][n].line(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl][alias]['phase']),1)),
                 np.hstack((0,self.rvs_to_plot['t'][pl][alias]['med'][np.argsort(self.rvs_to_plot['t'][pl][alias]['phase'])],0)),
                                        color=pal[2+n])
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
                        #f_phase[n].text(0.0,0.0+resid_sd*1.9,pl,horizontalalignment='center',verticalalignment='top',fontsize=9)
                        #plt.setp(f_phase[n].get_xticklabels(), visible=False) 

                    if n==len(all_pls_in_rvs)-1:
                        if interactive:
                            #extra = '[ppt]' if self.lc.flux_unit==0.001 else ''
                            #f_all_resids[n] = 'flux '+extra#<- y-axis label
                            #f_all[n].yaxis.axis_label = 'residuals '+extra#<- y-axis label
                            f_trans[n].xaxis.axis_label = 'Phase' #<- x-axis label
                        else:
                            f_phase[pl][n].set_xlabel("Phase")
                    else:
                        if not interactive:
                            plt.setp(f_phase[pl][n].get_xticklabels(), visible=False)
                
            elif pl in other_pls:
                if hasattr(self,'trace'):
                    K=np.clip(np.nanmedian(self.trace['K_'+pl]),averr,100000)
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
                                                 upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                 lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
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
                               level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[1]))
                        trband = ColumnDataSource(data=dict(
                            base=np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
       lower=np.hstack((0,self.rvs_to_plot['t'][pl]['-1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
       upper=np.hstack((0,self.rvs_to_plot['t'][pl]['+1sig'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0))
                                                           ))
                        f_phase[pl].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                                  level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[2+n]))
                        f_phase[pl].line(np.hstack((0,np.sort(self.rvs_to_plot['t'][pl]['phase']),1)),
             np.hstack((0,self.rvs_to_plot['t'][pl]['marg']['med'][np.argsort(self.rvs_to_plot['t'][pl]['phase'])],0)),
                                    color=pal[1])
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
                
            f_all_resids.set_xlabel("Time [HJD-"+str(self.rvs['jd_base'])+"]")
            f_alls.set_ylabel("RVs [m/s]")
            f_all_resids.set_ylabel("RV residuals [m/s]")
        f_alls.legend()
            
        if interactive:
            #Saving
            cols=[]
            for r in range(len(f_alls)):
                cols+=[column(f_alls[r],f_all_resids[r])]
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
            
            
    def Plot(self, interactive=False, n_samp=None, overwrite=False,return_fig=False, max_gp_len=20000, save=True,
             bin_gp=True, plot_loc=None, palette=None, plot_flat=True, pointcol="k", plottype='png'):
        ################################################################
        #       Varied plotting function for MonoTransit model
        ################################################################
        self.init_plot(plottype='lc',pointcol=pointcol)
        #Rasterizing matplotlib files if we have a lot of datapoints:
        raster=True if len(self.lc['time']>8000) else False
        
        if not hasattr(self,'trace'):
            n_samp=1
        
        if interactive:
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
            gs = fig.add_gridspec(len(self.planets)*4,32,wspace=0.3,hspace=0.001)
        
        #####################################
        #       Initialising figures
        #####################################
        
        f_alls=[];f_all_resids=[];f_trans=[];f_trans_resids=[]
        if not interactive:
            #Creating cumulative list of integers which add up to 24 but round to nearest length ratio:
            n_pl_widths=np.hstack((0,np.cumsum(1+np.array(saferound(list((24-len(self.lc['gap_lens']))*self.lc['gap_lens']/np.sum(self.lc['gap_lens'])), places=0))))).astype(int)
            #(gs[0, :]) - all top
            #(gs[:, 0]) - all left
            print(self.lc['gap_lens']/np.sum(self.lc['gap_lens']),n_pl_widths,32,range(len(n_pl_widths)-1))
            for ng in range(len(n_pl_widths)-1):
                if ng==0:
                    f_all_resids+=[fig.add_subplot(gs[len(self.planets)*3:,n_pl_widths[ng]:n_pl_widths[ng+1]])]
                    f_alls+=[fig.add_subplot(gs[:len(self.planets)*3,n_pl_widths[ng]:n_pl_widths[ng+1]],
                                                   sharex=f_all_resids[0])]
                else:
                    f_all_resids+=[fig.add_subplot(gs[len(self.planets)*3:, n_pl_widths[ng]:n_pl_widths[ng+1]],
                                                   sharey=f_all_resids[0])]
                    f_alls+=[fig.add_subplot(gs[:len(self.planets)*3,n_pl_widths[ng]:n_pl_widths[ng+1]],
                                             sharey=f_alls[0],sharex=f_all_resids[ng])]
            for npl in np.arange(len(self.planets))[::-1]:
                if npl==len(self.planets)-1:
                    f_trans_resids+=[fig.add_subplot(gs[(npl*4+3),-7:])]
                    f_trans+=[fig.add_subplot(gs[(npl*4):(npl*4+3),-7:],sharex=f_trans_resids[0])]
                else:
                    f_trans+=[fig.add_subplot(gs[(npl*4):(npl*4+3),-7:],sharex=f_trans_resids[0])]
                    f_trans_resids+=[fig.add_subplot(gs[(npl*4+3),-7:],sharex=f_trans_resids[0])]

        else:
            #For Bokeh plots, we can just use the size in pixels
            for ng,gaplen in enumerate(self.lc['gap_lens']):
                fwidth=int(np.round(750*gaplen/np.sum(self.lc['gap_lens']))-10)
                if ng==0:
                    f_all_resids+=[figure(width=fwidth, plot_height=150, title=None)]
                    f_alls+=[figure(width=fwidth, plot_height=400, title=None,x_range=f_all_resids[0].x_range)]
                else:
                    f_all_resids+=[figure(width=fwidth, plot_height=150, title=None,
                                          y_range=f_all_resids[0].y_range)]
                    f_alls+=[figure(width=fwidth, plot_height=400, title=None,
                                    x_range=f_all_resids[ng].x_range, y_range=f_alls[0].y_range)]
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
            self.init_gp_to_plot(n_samp,max_gp_len)
        
        '''
        assert hasattr(self,'trace')
        i_gp_pred=[]
        i_gp_var=[]
        print(limits,gap_lens,range(len(gap_lens)),np.arange(len(gap_lens)))
        for n in np.arange(len(gap_lens)):
            for i, sample in enumerate(xo.get_samples_from_trace(self.trace, size=n_samp)):
                with self.model:
                    ii_gp_pred, ii_gp_var = xo.eval_in_model(self.gp['use'].predict(self.lc['time'][limits[n][0]:limits[n][1]],
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
                ii_gp_pred, ii_gp_var = xo.eval_in_model(self.gp['use'].predict(self.lc['time'].astype(floattype),
                                                                                return_var=True,return_cov=False),sample)
                i_gp_pred+=[ii_gp_pred]
                i_gp_var+=[ii_gp_var]
            av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
        self.gp_to_plot['gp_pred'] = av
        self.gp_to_plot['gp_sd'] = std
        '''

        self.min_trans=abs(np.nanmin(self.trans_to_plot['all']['med']))

        #####################################
        #  Plotting full lightcurve regions
        #####################################
        if self.use_GP:
            resid_sd=np.nanstd(self.lc['flux'][self.lc['mask']] - self.gp_to_plot['gp_pred'][self.lc['mask']] - self.trans_to_plot['all']['med'][self.lc['mask']])
        else:
            resid_sd=np.nanstd(self.lc['flux_flat'][self.lc['mask']] - self.trans_to_plot['all']['med'][self.lc['mask']])
        if plot_flat:
            raw_plot_offset = 2.5*abs(self.min_trans) if not self.use_GP else 1.25*abs(self.min_trans) + resid_sd +\
                                                                     abs(np.min(self.gp_to_plot['gp_pred'][self.lc['mask']]))
        else:
            raw_plot_offset=0.0
            
        for n in np.arange(len(self.lc['gap_lens'])):
            low_lim = self.lc['time'][self.lc['limits'][n][0]]
            upp_lim = self.lc['time'][self.lc['limits'][n][1]-1]
            unmasked_lim_bool = (self.lc['time']>=(low_lim-0.5))&(self.lc['time']<(upp_lim+0.5))
            
            if self.use_GP:
                if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                    bin_detrend=tools.bin_lc_segment(np.column_stack((self.lc['time'][self.lc['limit_mask'][n]],
                                   self.lc['flux'][self.lc['limit_mask'][n]] - \
                                   self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]],
                                   self.lc['flux_err'][self.lc['limit_mask'][n]])),
                                   binsize=29/1440)
                    bin_resids=tools.bin_lc_segment(np.column_stack((self.lc['time'][self.lc['limit_mask'][n]],
                                   self.lc['flux'][self.lc['limit_mask'][n]] - \
                                   self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                   self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                   self.lc['flux_err'][self.lc['limit_mask'][n]])),
                                   binsize=29/1440)
            else:
                if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                    bin_resids=tools.bin_lc_segment(np.column_stack((self.lc['time'][self.lc['limit_mask'][n]],
                                   self.lc['flux_flat'][self.lc['limit_mask'][n]] - \
                                   self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                   self.lc['flux_err'][self.lc['limit_mask'][n]])),
                                   binsize=29/1440)
                mean=np.nanmedian(self.trace['mean']) if hasattr(self,'trace') else self.init_soln['mean']


                #Plotting each part of the lightcurve:
            if interactive:
                if self.use_GP:
                    #Plotting GP region and subtracted flux
                    if np.nanmedian(np.diff(self.lc['time']))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                         self.lc['flux'][self.lc['limit_mask'][n]]+raw_plot_offset,
                                         alpha=0.5,size=0.75,color='black')
                        f_alls[n].circle(self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                         self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + raw_plot_offset,
                                         alpha=0.65,size=3.5,legend="raw")
                        errors = ColumnDataSource(data=
                         dict(base=self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                              lower=self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + \
                              raw_plot_offset - self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                              upper=self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + \
                              raw_plot_offset + self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]]))

                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                         self.lc['flux'][self.lc['limit_mask'][n]]+raw_plot_offset,
                                         legend="raw",alpha=0.65,size=3.5)


                    gpband = ColumnDataSource(data=dict(base=self.lc['time'][self.lc['limit_mask'][n]],
                              lower=raw_plot_offset+self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]]-self.gp_to_plot['gp_sd'][self.lc['limit_mask'][n]], 
                              upper=raw_plot_offset+self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]]+self.gp_to_plot['gp_sd'][self.lc['limit_mask'][n]]))
                    f_alls[n].add_layout(Band(source=gpband,base='base',lower='lower',upper='upper',
                                              fill_alpha=0.4, line_width=0.0, fill_color=pal[3]))
                    f_alls[n].line(self.lc['time'][self.lc['limit_mask'][n]], self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]]+raw_plot_offset, 
                                   line_alpha=0.6, line_width=1.0, color=pal[3], legend="GP fit")
                    if plot_flat:
                        if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]] ))<1/72:
                            #Here we plot the detrended flux:
                            f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                             self.lc['flux'][self.lc['limit_mask'][n]]-self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]], color='black',
                                             alpha=0.5,size=0.75)
                            f_alls[n].circle(bin_detrend[:,0],bin_detrend[:,1],alpha=0.65,size=3.5,legend='detrended')

                            errors = ColumnDataSource(data=dict(base=bin_detrend[:,0],
                                                         lower=bin_detrend[:,1]+bin_detrend[:,2],
                                                         upper=bin_detrend[:,1]-bin_detrend[:,2]))
                            f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        else:
                            f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                             self.lc['flux'][self.lc['limit_mask'][n]]-self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]],
                                             legend="detrended",alpha=0.65,
                                             size=3.5)

                else:
                    if np.nanmedian(np.diff(self.lc['time']))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                         self.lc['flux_flat'][self.lc['limit_mask'][n]]+raw_plot_offset, 
                                         color='black',alpha=0.5,size=0.75)
                        if plot_flat:
                            f_alls[n].circle(self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                             self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                             legend="detrended",alpha=0.65,size=3.5)

                            errors = ColumnDataSource(data=dict(base=self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                         lower=self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] - \
                                         self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                         upper=self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + \
                                         self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]]))
                            f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))

                        #Here we plot the detrended flux:
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                         self.lc['flux_flat'][self.lc['limit_mask'][n]]+raw_plot_offset, 
                                         alpha=0.5,size=0.75)
                        if plot_flat:
                            f_alls[n].circle(self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                             self.lc['bin_flux_flat'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                             legend="detrended",alpha=0.65,size=3.5)
                            errors = ColumnDataSource(
                                      data=dict(base=self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                         lower=self.lc['bin_flux_flat'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] - \
                                         self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                         upper=self.lc['bin_flux_flat'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + \
                                         self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]]))
                            f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                         line_color='#dddddd', line_alpha=0.5,
                                                         upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                         lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                         self.lc['flux'][self.lc['limit_mask'][n]]+raw_plot_offset,
                                         legend="raw",alpha=0.65,size=3.5)
                        if plot_flat:
                            f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                             self.lc['flux_flat'][self.lc['limit_mask'][n]],
                                             legend="detrended",alpha=0.65,size=3.5)
                #Plotting transit
                if len(self.trans_to_plot['all'])>1:
                    trband = ColumnDataSource(data=dict(base=self.lc['time'][unmasked_lim_bool],
                                    lower=self.trans_to_plot['all']['-2sig'][unmasked_lim_bool],
                                    upper=self.trans_to_plot['all']['+2sig'][unmasked_lim_bool]))
                    f_alls[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[1]))
                    trband = ColumnDataSource(data=dict(base=self.lc['time'][unmasked_lim_bool],
                                    lower=self.trans_to_plot['all']['-1sig'][unmasked_lim_bool],
                                    upper=self.trans_to_plot['all']['+1sig'][unmasked_lim_bool]))
                    f_alls[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[1]))
                f_alls[n].line(self.lc['time'][unmasked_lim_bool],
                               self.trans_to_plot["all"]["med"][unmasked_lim_bool],
                               color=pal[1], legend="transit fit")

                if n>0:
                    f_alls[n].yaxis.major_tick_line_color = None  # turn off x-axis major ticks
                    f_alls[n].yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                    f_alls[n].yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
                    f_all_resids[n].yaxis.major_tick_line_color = None  # turn off x-axis major ticks
                    f_all_resids[n].yaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                    f_all_resids[n].yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

                if self.use_GP:
                    #Plotting residuals:
                    if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                        #HERE WE BIN
                        f_all_resids[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                               self.lc['flux'][self.lc['limit_mask'][n]] - self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                               self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]], color='black',
                                               alpha=0.5,size=0.75)

                        errors = ColumnDataSource(data=dict(base=bin_resids[:,0],
                                                     lower=bin_resids[:,1] - bin_resids[:,2],
                                                     upper=bin_resids[:,1] + bin_resids[:,2]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        f_all_resids[n].circle(bin_resids[:,0],bin_resids[:,1],
                                               legend="residuals",alpha=0.65,size=3.5)
                    else:
                        errors = ColumnDataSource(data=dict(base=self.lc['time'][self.lc['limit_mask'][n]],
                                          lower=self.lc['flux'][self.lc['limit_mask'][n]] - self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                           self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]] - self.lc['flux_err'][self.lc['limit_mask'][n]],
                                          upper=self.lc['flux'][self.lc['limit_mask'][n]] - self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                           self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]] + self.lc['flux_err'][self.lc['limit_mask'][n]]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        f_all_resids[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                               self.lc['flux'][self.lc['limit_mask'][n]] - self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                               self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                               alpha=0.65,
                                               size=3.5)

                else:
                    #Plotting detrended:
                    mean=np.nanmedian(self.trace['mean']) if hasattr(self,'trace') else self.init_soln['mean']
                    if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                        f_all_resids[n].circle(self.lc['time'][self.lc['limit_mask'][n]],
                                               self.lc['flux_flat'][self.lc['limit_mask'][n]] - mean - \
                                               self.trans_to_plot["all"]["med"][self.lc['limit_mask'][n]], color='black',
                                               alpha=0.5,size=0.75)
                        errors = ColumnDataSource(data=dict(base=bin_resids[:,0],
                                                lower=bin_resids[:,1] - bin_resids[:,2],
                                                upper=bin_resids[:,1] + bin_resids[:,2]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        f_all_resids[n].circle(bin_resids[:,0],bin_resids[:,1],
                                               alpha=0.65,size=3.5)

                    else:
                        f_alls[n].circle(self.lc['time'][self.lc['limit_mask'][n]], 
                                         self.lc['flux_flat'][self.lc['limit_mask'][n]] - mean - \
                                         self.trans_to_plot["all"]["med"][self.lc['limit_mask'][n]],
                                         legend="raw data",alpha=0.65,size=3.5)

                f_all_resids[n].xaxis.axis_label = 'Time [BJD-'+str(int(self.lc['jd_base']))+']' #<- x-axis label
                f_alls[n].legend.location = 'bottom_right'
                f_alls[n].legend.background_fill_alpha = 0.1
                f_alls[n].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                f_alls[n].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                f_alls[n].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

            else:
                #Matplotlib plot:
                if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                    f_alls[n].plot(self.lc['time'][self.lc['limit_mask'][n]], self.lc['flux'][self.lc['limit_mask'][n]] + raw_plot_offset,
                                   ".", color=pointcol, alpha=0.5,markersize=0.75, rasterized=raster)
                    f_alls[n].errorbar(self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                       self.lc['bin_flux'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]] + raw_plot_offset, 
                                       yerr=self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],color='C1',
                                       fmt=".", label="raw", ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)
                else:
                    f_alls[n].errorbar(self.lc['time'][self.lc['limit_mask'][n]], self.lc['flux'][self.lc['limit_mask'][n]] + raw_plot_offset, 
                                       yerr=self.lc['flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],color='C1',
                                       fmt=".", label="raw", ecolor='#dddddd', 
                                       alpha=0.5,markersize=3.5, rasterized=raster)

                if self.use_GP:
                    
                    if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                        if plot_flat:
                            f_alls[n].plot(self.lc['time'][self.lc['limit_mask'][n]], self.lc['flux'][self.lc['limit_mask'][n]] - \
                                           self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]],
                                           ".", color=pointcol, alpha=0.5,markersize=0.75, rasterized=raster)
                            f_alls[n].errorbar(bin_detrend[:,0], bin_detrend[:,1], yerr=bin_detrend[:,2],color='C2',fmt=".",
                                               label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)

                        #Plotting residuals:
                        f_all_resids[n].plot(self.lc['time'][self.lc['limit_mask'][n]],
                                             self.lc['flux'][self.lc['limit_mask'][n]] - self.gp_to_plot['gp_pred'][self.lc['limit_mask'][n]] - \
                                             self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                             ".", color=pointcol, alpha=0.5,markersize=0.75, rasterized=raster)

                        f_all_resids[n].errorbar(bin_resids[:,0],bin_resids[:,1], yerr=bin_resids[:,2], fmt=".", 
                                                 ecolor='#dddddd',alpha=0.5,markersize=0.75, rasterized=raster)

                    else:
                        if plot_flat:
                            f_alls[n].errorbar(self.lc['time'][self.lc['limit_mask'][n]], 
                                               self.lc['flux_flat'][self.lc['limit_mask'][n]],
                                               yerr=self.lc['flux_err'][self.lc['limit_mask'][n]],color='C2', rasterized=raster,
                                               fmt=".", label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5)

                        f_all_resids[n].errorbar(self.lc['time'][self.lc['limit_mask'][n]], 
                                                 self.lc['flux_flat'][self.lc['limit_mask'][n]] - \
                                                 self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                                 yerr=self.lc['flux_err'][self.lc['limit_mask'][n]], fmt=".", rasterized=raster,
                                                 ecolor='#dddddd',label="residuals", alpha=0.5,markersize=0.75)
                    #Plotting GP region and subtracted flux
                    f_alls[n].fill_between(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],
                               self.gp_to_plot['gp_pred'][self.lc['limits'][n][0]:self.lc['limits'][n][1]] + raw_plot_offset - \
                               2*self.gp_to_plot['gp_sd'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],
                               self.gp_to_plot['gp_pred'][self.lc['limits'][n][0]:self.lc['limits'][n][1]] + raw_plot_offset + \
                               2*self.gp_to_plot['gp_sd'][self.lc['limits'][n][0]:self.lc['limits'][n][1]], rasterized=raster,
                               color="C3", alpha=0.2,zorder=10)
                    f_alls[n].fill_between(self.lc['time'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],
                               self.gp_to_plot['gp_pred'][self.lc['limits'][n][0]:self.lc['limits'][n][1]] + raw_plot_offset - \
                               self.gp_to_plot['gp_sd'][self.lc['limits'][n][0]:self.lc['limits'][n][1]],
                               self.gp_to_plot['gp_pred'][self.lc['limits'][n][0]:self.lc['limits'][n][1]] + raw_plot_offset + \
                               self.gp_to_plot['gp_sd'][self.lc['limits'][n][0]:self.lc['limits'][n][1]], rasterized=raster,
                                           color="C3", label="GP fit",alpha=0.3,zorder=11)


                else:
                    #GP not used.
                    if np.nanmedian(np.diff(self.lc['time'][self.lc['limit_mask'][n]]))<1/72:
                        #Plotting flat flux only
                        if plot_flat:
                            f_alls[n].plot(self.lc['time'][self.lc['limit_mask'][n]], self.lc['flux_flat'][self.lc['limit_mask'][n]],
                                       ".", color=pointcol,alpha=0.5,markersize=0.75, rasterized=raster)
                            f_alls[n].errorbar(self.lc['bin_time'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                           self.lc['bin_flux_flat'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]],
                                           yerr= self.lc['bin_flux_err'][self.lc['binlimits'][n][0]:self.lc['binlimits'][n][1]], rasterized=raster, 
                                           color='C2',fmt=".",label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5)
                        #Plotting residuals:
                        f_all_resids[n].plot(self.lc['time'][self.lc['limit_mask'][n]],
                                             self.lc['flux_flat'][self.lc['limit_mask'][n]] - \
                                             self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                             ".", color=pointcol,label="raw data", alpha=0.5,markersize=0.75, rasterized=raster)

                        f_all_resids[n].errorbar(bin_resids[:,0],bin_resids[:,1],yerr=bin_resids[:,2], fmt=".",
                                                 ecolor='#dddddd',label="residuals", alpha=0.5,
                                                 markersize=0.75, rasterized=raster)

                    else:
                        if plot_flat:
                            f_alls[n].errorbar(self.lc['time'][self.lc['limit_mask'][n]], self.lc['flux_flat'][self.lc['limit_mask'][n]],
                                           yerr=self.lc['flux_err'][self.lc['limit_mask'][n]],fmt=".", label="detrended", 
                                           ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)

                        f_all_resids[n].errorbar(self.lc['time'][self.lc['limit_mask'][n]], 
                                                 self.lc['flux_flat'][self.lc['limit_mask'][n]] - \
                                                 self.trans_to_plot['all']['med'][self.lc['limit_mask'][n]],
                                                 yerr=self.lc['flux_err'][self.lc['limit_mask'][n]], fmt=".",
                                                 ecolor='#dddddd', alpha=0.5,markersize=0.75, rasterized=raster)

                #Plotting transit
                if '-2sig' in self.trans_to_plot['all']:
                    f_alls[n].fill_between(self.lc['time'][unmasked_lim_bool],
                                           self.trans_to_plot['all']['-2sig'][unmasked_lim_bool],
                                           self.trans_to_plot['all']['+2sig'][unmasked_lim_bool],
                                           alpha=0.2, color="C4",zorder=10, rasterized=raster)
                    f_alls[n].fill_between(self.lc['time'][unmasked_lim_bool],
                                           self.trans_to_plot['all']['-1sig'][unmasked_lim_bool],
                                           self.trans_to_plot['all']['+1sig'][unmasked_lim_bool],
                                           alpha=0.3, color="C4",zorder=11, rasterized=raster)
                f_alls[n].plot(self.lc['time'][unmasked_lim_bool],
                               self.trans_to_plot['all']['med'][unmasked_lim_bool],
                               color="C4", label="transit fit", linewidth=1.5,alpha=0.5,zorder=12, rasterized=raster)
                
                plt.setp(f_alls[n].get_xticklabels(), visible=False) 
                if n>0:
                    plt.setp(f_alls[n].get_yticklabels(), visible=False) 
                    plt.setp(f_all_resids[n].get_yticklabels(), visible=False) 
                
                f_all_resids[n].set_xlabel = 'Time [BJD-'+str(int(self.lc['jd_base']))+']' #<- x-axis label
                f_all_resids[n].set_xlim(self.lc['time'][self.lc['limits'][n][0]]-0.1,self.lc['time'][self.lc['limits'][n][1]-1]+0.1)
                if self.lc['gap_lens'][n]==np.max(self.lc['gap_lens']):
                    f_alls[n].legend()
                    
        extra = '[ppt]' if self.lc['flux_unit']==0.001 else ''
        
        max_gp=np.percentile(self.gp_to_plot['gp_pred'],99.5) if self.use_GP else np.nanmax(self.lc['bin_flux'])
        
        if interactive:
            
            f_all_resids[0].yaxis.axis_label = 'residuals '+extra#<- y-axis label
            f_alls[0].yaxis.axis_label = 'flux '+extra#<- y-axis label
            '''
            sdbuffer=3
            if self.use_GP:
                f_alls[0].y_range=Range1d(-1*min_trans - sdbuffer*resid_sd, 
                                          raw_plot_offset + np.max(self.gp_to_plot['gp_pred']) + sdbuffer*resid_sd)
            else:
                f_alls[0].y_range=Range1d(-1*min_trans - sdbuffer*resid_sd,
                                          raw_plot_offset + np.max(self.lc['bin_flux']) + sdbuffer*resid_sd)

            f_all_resids[0].y_range=Range1d(-1*sdbuffer*resid_sd, sdbuffer*resid_sd)
            '''
            
        else:
            f_alls[0].set_ylabel('flux '+extra)#<- y-axis label
            f_all_resids[0].set_ylabel('residuals '+extra)#<- y-axis label

            f_alls[0].set_ylim(-1*abs(self.min_trans)-1.5*resid_sd,raw_plot_offset+max_gp+1.5*resid_sd)
            f_all_resids[0].set_ylim(-2.25*resid_sd,2.25*resid_sd)

        #####################################
        #  Plotting individual transits
        #####################################
        maxdur=1.25*np.max([self.planets[ipl]['tdur'] for ipl in self.planets])
        self.lc['phase']={}
        for n,pl in enumerate(self.planets):
            if hasattr(self,'trace'):
                t0=np.nanmedian(self.trace['t0_'+pl])
                if pl in self.multis or pl in self.rvplanets:
                    per=np.nanmedian(self.trace['per_'+pl])
                elif pl in self.duos:
                    t0_2=np.nanmedian(self.trace['t0_2_'+pl])
                    per=np.nanmedian(abs(self.trace['t0_'+pl]-self.trace['t0_2_'+pl]))
                    
                elif pl in self.monos:
                    per=2e3
            elif hasattr(self,'init_soln'):
                t0=self.init_soln['t0_'+pl]
                if pl in self.multis or pl in self.rvplanets:
                    per=self.init_soln['per_'+pl]
                elif pl in self.duos:
                    t0_2=self.init_soln['t0_2_'+pl]
                    per=abs(self.init_soln['t0_'+pl]-self.init_soln['t0_2_'+pl])
                elif pl in self.monos:
                    per=2e3
            self.lc['phase'][pl]=(self.lc['time']-t0-0.5*per)%per-0.5*per

            for ns in range(len(self.lc['limits'])):
                if pl in self.multis or pl in self.rvplanets:
                    n_p_sta=np.ceil((t0-self.lc['time'][self.lc['limits'][ns][0]])/per)
                    n_p_end=(t0-self.lc['time'][self.lc['limits'][ns][1]-1])/per
                    
                    if (t0>self.lc['limits'][ns][0])&(t0<self.lc['limits'][ns][0]):
                        #Adding ticks for the position of each planet below the data:
                        f_alls[ns].scatter(t0+np.arange(n_p_sta,n_p_end,1.0)*per,
                                           np.tile(-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets)),
                                                   int(np.ceil(n_p_end-n_p_sta))),
                                           marker="triangle", size=12.5, line_color=pal[2+n], fill_color=col, alpha=0.85)

                elif pl in self.monos:
                    if (t0>self.lc['limits'][ns][0])&(t0<self.lc['limits'][ns][0]):
                        if interactive:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                               marker="triangle", size=12.5, fill_color=pal[n+2], alpha=0.85)
                        else:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                               "^", markersize=12.5, color=pal[n+2], alpha=0.85, rasterized=raster)
                elif pl in self.duos:
                    if (t0>self.lc['limits'][ns][0])&(t0<self.lc['limits'][ns][0]):
                        if interactive:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           marker="triangle", size=12.5, fill_color=pal[2+n], alpha=0.85)
                        else:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           "^", markersize=12.5, color=pal[2+n], alpha=0.85, rasterized=raster)

                    if (t0_2>self.lc['limits'][ns][0])&(t0_2<self.lc['limits'][ns][0]):
                        if interactive:
                            f_alls[ns].scatter([t0_2],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           marker="triangle", size=12.5, fill_color=pal[2+n], alpha=0.85)
                        else:
                            f_alls[ns].scatter([t0_2],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           "^", markersize=12.5, color=pal[2+n], alpha=0.85, rasterized=raster)
                
            #Computing
            if len(self.planets)>1:
                other_pls=np.sum([self.trans_to_plot[opl]['med'] for opl in self.planets if opl!=pl],axis=0)
            else:
                other_pls=np.zeros(len(self.lc['time']))
            
            phasebool=abs(self.lc['phase'][pl])<1.25*maxdur
            if self.use_GP:
                phaselc=np.column_stack((self.lc['phase'][pl][self.lc['mask']&phasebool],
                                         self.lc['flux'][self.lc['mask']&phasebool] - \
                                         self.gp_to_plot['gp_pred'][self.lc['mask']&phasebool] - \
                                         other_pls[self.lc['mask']&phasebool],
                                         self.lc['flux_err'][self.lc['mask']&phasebool]))
            else:
                phaselc=np.column_stack((self.lc['phase'][pl][self.lc['mask']&phasebool],
                                         self.lc['flux_flat'][self.lc['mask']&phasebool] - \
                                         other_pls[self.lc['mask']&phasebool],
                                         self.lc['flux_err'][self.lc['mask']&phasebool]))
            bin_phase=tools.bin_lc_segment(phaselc[np.argsort(phaselc[:,0])],binsize=maxdur/15.0)
            
            if interactive:
                sdbuffer=3
                
                self.min_trans=abs(np.min(self.trans_to_plot[pl]['med']))
                f_trans[n].circle(phaselc[:,0],phaselc[:,1], 
                                  color='black', alpha=0.4, size=0.75)

                f_trans[n].circle(phaselc[:,0],
                                  phaselc[:,1] - self.trans_to_plot[pl]['med'][self.lc['mask']&phasebool] - self.min_trans-sdbuffer*resid_sd, 
                                  color='black', alpha=0.2, size=0.75)
                errors = ColumnDataSource(data=dict(base=bin_phase[:,0],
                                        lower=bin_phase[:,1] - bin_phase[:,2],
                                        upper=bin_phase[:,1] + bin_phase[:,2]))
                f_trans[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                             line_color='#dddddd', line_alpha=0.5,
                                             upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                             lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                f_trans[n].circle(bin_phase[:,0], bin_phase[:,1], alpha=0.65, size=3.5)
                if '-2sig' in self.trans_to_plot[pl]:
                    trband = ColumnDataSource(data=dict(base=np.sort(phaselc[:,0]),
                              lower=self.trans_to_plot[pl]['-2sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                              upper=self.trans_to_plot[pl]['+2sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])]))
                    f_trans[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                           level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[2+n]))
                    trband = ColumnDataSource(data=dict(base=np.sort(phaselc[:,0]),
                              lower=self.trans_to_plot[pl]['-1sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                              upper=self.trans_to_plot[pl]['+1sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])]))
                    f_trans[n].add_layout(Band(source=trband,base='base',lower='lower',upper='upper',
                                              level='underlay',fill_alpha=0.25, line_width=0.0, fill_color=pal[2+n]))
                f_trans[n].line(np.sort(phaselc[:,0]),
                                self.trans_to_plot[pl]["med"][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                                color=pal[2+n])
                f_trans[n].y_range=Range1d(-1*self.min_trans-2*sdbuffer*resid_sd,sdbuffer*resid_sd)
                
                if n<len(self.planets)-1:
                    f_trans[n].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                    f_trans[n].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                    f_trans[n].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

            else:
                f_trans[n].plot(phaselc[:,0],phaselc[:,1], ".",label="raw data", color=pointcol,
                                alpha=0.5,markersize=0.75, rasterized=raster)
                f_trans[n].errorbar(bin_phase[:,0],bin_phase[:,1], yerr=bin_phase[:,2], fmt='.', 
                                    alpha=0.75, markersize=5, rasterized=raster)
                f_trans_resids[n].plot(phaselc[:,0],
                                       phaselc[:,1]-self.trans_to_plot[pl]['med'][self.lc['mask']&phasebool], ".",
                                         alpha=0.5, color=pointcol,markersize=0.75, rasterized=raster)
                if '+2sig' in self.trans_to_plot[pl]:
                    f_trans[n].fill_between(np.sort(phaselc[:,0]),
                           self.trans_to_plot[pl]['-2sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                           self.trans_to_plot[pl]['+2sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                           alpha=0.25, color="C3", rasterized=raster)
                    f_trans[n].fill_between(np.sort(phaselc[:,0]),
                           self.trans_to_plot[pl]['-1sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                           self.trans_to_plot[pl]['+1sig'][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                           alpha=0.25, color="C3", rasterized=raster)
                f_trans[n].plot(np.sort(phaselc[:,0]),
                                self.trans_to_plot[pl]["med"][self.lc['mask']&phasebool][np.argsort(phaselc[:,0])],
                               color="C3", label="transit fit", rasterized=raster)
                f_trans[n].set_ylim(np.min(self.trans_to_plot[pl]["med"])-2*resid_sd,2*resid_sd)
                f_trans_resids[n].set_ylim(-2.25*resid_sd,2.25*resid_sd)
                f_trans[n].yaxis.tick_right()
                f_trans_resids[n].yaxis.tick_right()
                
                f_trans[n].text(0.0,0.0+resid_sd*1.9,pl,horizontalalignment='center',verticalalignment='top',fontsize=9)
                
                plt.setp(f_trans[n].get_xticklabels(), visible=False) 
                if n>0:
                    plt.setp(f_trans_resids[n].get_xticklabels(), visible=False) 

            if n==len(self.planets)-1:
                if interactive:
                    #extra = '[ppt]' if self.lc.flux_unit==0.001 else ''
                    #f_all_resids[n] = 'flux '+extra#<- y-axis label
                    #f_all[n].yaxis.axis_label = 'residuals '+extra#<- y-axis label
                    f_trans[n].xaxis.axis_label = 'Time [d] from transit' #<- x-axis label
                    
        if not interactive:
            f_trans_resids[0].set_xlim(-1.3*maxdur,1.3*maxdur)
            f_trans_resids[0].set_xlabel("Time [d] from transit")
            
        if interactive:
            #Saving
            cols=[]
            for r in range(len(f_alls)):
                cols+=[column(f_alls[r],f_all_resids[r])]
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
                    plt.savefig(self.savenames[0]+'_model_plot.png',dpi=350,transparent=True)
                    #plt.savefig(self.savenames[0]+'_model_plot.pdf')
                elif plot_loc is None and plottype=='pdf':
                    plt.savefig(self.savenames[0]+'_model_plot.pdf')
                else:
                    plt.savefig(plot_loc)
            if return_fig:
                return fig

    def PlotPeriods(self, plot_loc=None, ylog=True, nbins=25, pmax=None, ymin=None,xlog=False):
        assert hasattr(self,'trace')
        import seaborn as sns
        from scipy.special import logsumexp
        pal=sns.color_palette('viridis_r',7)
        coldic={-6:"p<1e-5",-5:"p>1e-5",-4:"p>1e-4",-3:"p>0.1%",-2:"p>1%",-1:"p>10%",0:"p>100%"}
        plot_pers=self.duos+self.monos
        if ymin is None and len(self.duos)>0:
            ymin=np.min(np.hstack([np.nanmedian(self.trace['logprob_marg_'+pl],axis=0) for pl in self.duos]))/np.log(10)-2.0
        elif ymin is None:
            ymin=1e-12
        
        if len(plot_pers)>0:
            plt.figure(figsize=(8.5,4.2))
            for npl, pl in enumerate(plot_pers):
                plt.subplot(1,len(plot_pers),npl+1)
                if pl in self.duos:
                    #As we're using the nanmedian log10(prob)s for each period, we need to make sure their sums add to 1.0
                    psum=logsumexp(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0))/np.log(10)
                    #Plotting lines
                    cols=[]
                    
                    probs=np.nanmedian(self.trace['logprob_marg_'+pl],axis=0)/np.log(10)
                    for n in np.arange(len(probs))[np.argsort(probs)][::-1]:
                        # Density Plot and Histogram of all arrival delays        
                        nprob=probs[n]
                        ncol=int(np.floor(np.clip(nprob-psum,-6,0)))
                        
                        if ncol not in cols:
                            cols+=[ncol]
                            plt.plot(np.tile(np.nanmedian(self.trace['per_'+pl][:,n]),2),
                                     [ymin-psum,nprob-psum],
                                     linewidth=5.0,color=pal[6+ncol],alpha=0.6,label=coldic[ncol])
                        else:
                            plt.plot(np.tile(np.nanmedian(self.trace['per_'+pl][:,n]),2),
                                     [ymin-psum,nprob-psum],
                                     linewidth=5.0,color=pal[6+ncol],alpha=0.6)

                    plt.title("Duo - "+str(pl))
                    plt.ylim(ymin-psum,1.0)
                    plt.legend()
                    if xlog:
                        plt.xscale('log')
                        plt.xticks([20,40,60,80,100,150,200,250,300,350,400,450,500,600,700],
                                   np.array([20,40,60,80,100,150,200,250,300,350,400,450,500,600,700]).astype(str))
                        #plt.xticklabels([20,40,60,80,100,150,200,250])
                    plt.ylabel("$\log_{10}{p}$")
                    plt.xlabel("Period [d]")
                elif pl in self.monos:
                    
                    #if 'logprob_marg_sum_'+pl in self.trace.varnames:
                    #    total_prob=logsumexp((self.trace['logprob_marg_'+pl]+self.trace['logprob_marg_sum_'+pl]).ravel())
                    #else:
                    total_prob=logsumexp(self.trace['logprob_marg_'+pl].ravel())
                    total_av_prob=logsumexp(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0))
                    pmax = np.nanmax(self.trace['per_'+pl].ravel()) if pmax is None else pmax
                    cols=[]
                    for ngap in np.arange(self.planets[pl]['ngaps']):
                        bins=np.arange(np.floor(self.planets[pl]['per_gaps']['gap_starts'][ngap]),
                                       np.clip(np.ceil(self.planets[pl]['per_gaps']['gap_ends'][ngap])+1.0,0.0,pmax),
                                       1.0)
                        
                        ncol=int(np.floor(np.clip(np.nanmedian(self.trace['logprob_marg_'+pl][:,ngap])-total_av_prob,-6,0)))
                        print(self.planets[pl]['per_gaps']['gap_starts'][ngap],
                              ncol,np.nanmedian(self.trace['logprob_marg_'+pl][:,ngap])-total_av_prob)
                        #print(ngap,np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob))
                        if ncol not in cols:
                            cols+=[ncol]
                            plt.hist(self.trace['per_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                                 weights=np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob),
                                 color=pal[6+ncol],histtype="stepfilled",label=coldic[ncol])
                        else:
                            plt.hist(self.trace['per_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                                 weights=np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob),
                                 color=pal[6+ncol],histtype="stepfilled")

                    plt.title("Mono - "+str(pl))
                    if ylog:
                        plt.yscale('log')
                        plt.ylabel("$\log_{10}{\\rm prob}$")
                        plt.ylim(ymin,1.0)
                    else:
                        plt.ylabel("prob")
                    
                    if xlog:
                        plt.xscale('log')
                        plt.xlim(0.5*np.min(self.planets[pl]['per_gaps']['gap_starts']),pmax)
                        plt.xticks([20,40,60,80,100,150,200,250,300,350,400,450,500,600,800,1000,1500,2000,2500,3000],
                                   np.array([20,40,60,80,100,150,200,250,300,350,400,450,500,600,800,1000,1500,2000,2500,3000]).astype(str))
                        #plt.xticklabels([20,40,60,80,100,150,200,250])
                    else:
                        plt.xlim(0,pmax)
                    #plt.xlim(60,80)
                    plt.ylim(1e-12,1.0)
                    plt.xlabel("Period [d]")
                    plt.legend(title="Average prob")
            
            if plot_loc is None:
                plt.savefig(self.savenames[0]+'_period_dists.pdf')
            else:
                plt.savefig(plot_loc)
    
    def PlotCorner(self,corner_vars=None,use_marg=True,truths=None):
        # Plotting corner for those parameters we're interested in - e.g. orbital parameters
        # If "use_marg" is True - uses the marginalised tdur and period parameters for multis and duos
        # If "use_marg" is False - generates samples for each marginalised distribution and weights by logprob
        import corner
        
        if corner_vars is None:
            corner_vars=['logrho_S']

            for pl in self.planets:
                for var in self.fit_params:
                    if var+'_'+pl in self.trace.varnames:
                        corner_vars+=[var+'_'+pl]
                if pl in self.duos:
                    corner_vars+=['t0_2_'+pl]
                if use_marg:
                    for var in self.marginal_params:
                        if var+'_marg_'+pl in self.trace.varnames:
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
        samples = pm.trace_to_dataframe(self.trace, varnames=corner_vars)
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
            print(n_mult,"x samples")
            samples['log_prob']=np.tile(0.0,len(samples))
            samples_len=len(samples)
            samples=pd.concat([samples]*int(n_mult),axis=0)
            print(samples.shape,samples_len)
            
            n_pos=0
            
            for mpl in self.monos:
                for n_gap in np.arange(self.planets[mpl]['ngaps']):
                    sampl_loc=np.in1d(np.arange(0,len(samples),1),np.arange(n_pos*samples_len,(n_pos+1)*samples_len,1))
                    samples.loc[sampl_loc,'per_marg_'+mpl]=self.trace['per_'+mpl][:,n_gap]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+mpl]=self.trace['tdur_'+mpl][:,n_gap]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+mpl]=self.trace['b_'+mpl][:,n_gap]
                    samples.loc[sampl_loc,'log_prob']=self.trace['logprob_marg_'+mpl][:,n_gap]
                    n_pos+=1
            for dpl in self.duos:
                for n_per in np.arange(len(self.planets[dpl]['period_aliases'])):
                    sampl_loc=np.in1d(np.arange(len(samples)),np.arange(n_pos*samples_len,(n_pos+1)*samples_len))
                    samples.loc[sampl_loc,'per_marg_'+dpl]=self.trace['per_'+dpl][:,n_per]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+dpl]=self.trace['tdur_'+dpl][:,n_per]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+dpl]=self.trace['b_'+dpl][:,n_per]
                    samples.loc[sampl_loc,'log_prob'] = self.trace['logprob_marg_'+dpl][:,n_per]
                    n_pos+=1
            weight_samps = np.exp(samples["log_prob"])
            fig = corner.corner(samples[[col for col in samples.columns if col!='log_prob']],weights=weight_samps);
        
        fig.savefig(self.savenames[0]+'_corner.pdf',dpi=400,rasterized=True)
        
        
    def MakeTable(self,short=True,save=True,cols='all'):
        assert hasattr(self,'trace')
        
        if cols=='all':
            #Removing lightcurve, GP and reparameterised hyper-param columns
            cols_to_remove=['gp_', '_gp', 'light_curve','__']
            if short:
                #If we want just the short table, let's remove those params which we derived and which we marginalised
                cols_to_remove+=['mono_uniform_index','logliks','_priors','logprob_marg','mean', 'logrho_S']
                for col in self.marginal_params:
                    cols_to_remove+=['mono_'+col+'s','duo_'+col+'s']
            medvars=[var for var in self.trace.varnames if not np.any([icol in var for icol in cols_to_remove])]
            print(cols_to_remove, medvars)
            df = pm.summary(self.trace,var_names=medvars,stat_funcs={"5%": lambda x: np.percentile(x, 5),
                                                                     "-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                                     "median": lambda x: np.percentile(x, 50),
                                                                     "+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                                     "95%": lambda x: np.percentile(x, 95)})
        else:
            df = pm.summary(self.trace,var_names=cols,stat_funcs={"5%": lambda x: np.percentile(x, 5),
                                                                     "-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                                     "median": lambda x: np.percentile(x, 50),
                                                                     "+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                                     "95%": lambda x: np.percentile(x, 95)})

        if save:
            print("Saving sampled model parameters to file with shape: ",df.shape)
            if short:
                df.to_csv(self.savenames[0]+'_mcmc_output_short.csv')
            else:
                df.to_csv(self.savenames[0]+'_mcmc_output.csv')
        return df

    def PlotTable(self,plot_loc=None,return_table=False):
        
        df = self.MakeTable(short=True)
        
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
    

        
    def LoadPickle(self, loadname=None):
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
            self.GetSavename(how='load')
        print(self.savenames, self.savenames is None)
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
    
    def PredictFutureTransits(self, time_start, time_end, include_multis=True):
        '''
        # Return a dataframe of potential transits of all Duo candidates between time_start & time_end dates.
        # time_start - Astropy.Time date or julian date (in same base as lc) for the start of the observing period
        # time_end - Astropy.Time date or julian date for the start of the period we might want to observe aliases
        # include_multis - boolean - default:True - whether to also generate transits for multi-transiting planets.
        #
        #
        e.g. after running model.RunMcmc():
        df = model.PredictFutureTransits(Time('2021-06-01T00:00:00.000',format='isot'),Time('2021-10-01T00:00:00.000',format='isot'))
        '''
        from astropy.time import Time
        import fractions

        assert hasattr(self,'trace') #We need to have run Mcmc to have samples first.
        
        if type(time_start)==Time:
            time_start = time_start.jd - self.lc['jd_base']
            time_end   = time_start.jd - self.lc['jd_base']
            print("time range",time_start.isot,"->",time_end.isot)
        elif type(time_start) in [int, np.int64, float, np.float64] and abs(time_start-np.nanmedian(self.trace['t0_'+list(self.planets.keys())[0]]))>5000:
            #This looks like a proper julian date. Let's reformat to match the lightcurve
            time_start -= self.lc['jd_base']
            time_end   -= self.lc['jd_base']
            print("assuming JD in format 2457...")
            print("time range",Time(time_start,format='jd').isot,"->",Time(time_end,format='jd').isot)
        else:
            assert type(time_start) in [int, np.int64, float, np.float64]
            print("time range",Time(time_start+self.lc['jd_base'],format='jd').isot,
                  "->",Time(time_end+self.lc['jd_base'],format='jd').isot)
        
        all_trans=pd.DataFrame()
        loopplanets = self.duos+self.multis if include_multis else self.duos
        for pl in loopplanets:
            if pl in self.duos:
                sum_all_probs=np.logaddexp.reduce(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0))
                trans_p0=np.floor(np.nanmedian(time_start - self.trace['t0_2_'+pl])/np.nanmedian(self.trace['per_'+pl],axis=0))
                trans_p1=np.ceil(np.nanmedian(time_end -  self.trace['t0_2_'+pl])/np.nanmedian(self.trace['per_'+pl],axis=0))
            elif pl in self.multis:
                trans_p0=np.floor(np.nanmedian(time_start - self.trace['t0_'+pl])/np.nanmedian(self.trace['per_'+pl],axis=0))
                trans_p1=np.ceil(np.nanmedian(time_end -  self.trace['t0_'+pl])/np.nanmedian(self.trace['per_'+pl],axis=0))
            #print(np.nanmedian(self.trace['t0_2_'+pl])+np.nanmedian(self.trace['per_'+pl],axis=0)*trans_p0)
            #print(np.nanmedian(self.trace['t0_2_'+pl])+np.nanmedian(self.trace['per_'+pl],axis=0)*trans_p1)
            n_trans=trans_p1-trans_p0
            all_trans=pd.DataFrame()
            
            nms=['-2sig','-1sig','med','+1sig','+2sig']
            percentiles=(2.2750132, 15.8655254, 50., 84.1344746, 97.7249868)
            
            #Getting the important trace info (tcen, dur, etc) for each alias:
            if 'tdur' in self.fit_params or pl in self.multis:
                dur=np.nanpercentile(self.trace['tdur_'+pl],percentiles)
            naliases=[1] if pl in self.multis else np.arange(self.planets[pl]['npers'])
            for nd in naliases:
                if n_trans[nd]>0:
                    if pl in self.duos:
                        transits=np.nanpercentile(np.vstack([self.trace['t0_2_'+pl]+ntr*self.trace['per_'+pl][:,nd] for ntr in np.arange(trans_p0[nd],trans_p1[nd])]),percentiles,axis=1)

                        if 'tdur' in self.marginal_params:
                            dur=np.nanpercentile(self.trace['tdur_'+pl][:,nd],percentiles)
                        logprobs=np.nanmedian(self.trace['logprob_marg_'+pl][:,nd])-sum_all_probs
                    else:
                        transits=np.nanpercentile(np.vstack([self.trace['t0_'+pl][nd]+ntr*self.trace['per_'+pl] for ntr in np.arange(trans_p0[nd],trans_p1[nd])]),percentiles,axis=1)

                        logprobs=np.array([0.0])
                    idf=pd.DataFrame({'transit_mid_date':Time(transits[2]+self.lc['jd_base'],format='jd').isot,
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
                                      'transit_fractions':np.array([str(fractions.Fraction(i1,int(nd+1))) for i1 in np.arange(trans_p0[nd],trans_p1[nd]).astype(int)]),
                                      'log_prob':np.tile(logprobs,len(transits[2])),
                                      'prob':np.exp(logprobs),
                                      'planet_name':np.tile('multi_'+pl,len(transits[2])) if pl in self.multis else np.tile('duo_'+pl,len(transits[2])),
                                      'alias_n':np.tile(nd,len(transits[2])),
                                      'alias_p':np.tile(np.nanmedian(self.trace['per_'+pl][:,nd]),len(transits[2])) if pl in self.duos else np.nanmedian(self.trace['per_'+pl])})
                    all_trans=all_trans.append(idf)
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
        unq_trans = unq_trans.loc[(unq_trans['transit_end_+2sig']>time_start)*(unq_trans['transit_start_-2sig']<time_end)].sort_values('transit_mid_med')
        return unq_trans.set_index(np.arange(len(unq_trans)))
    
    def getLDs(self,n_samples,mission='tess',how='2'):
        #Gets theoretical Limb Darkening parameters
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
            tab=pd.read_fwf("/Users/hosborn/Postdoc/MonoTools/data/tables/Cheops_Quad_LDs_AllFeHs.txt",header=None,widths=[5,7,5,5,9])
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
        #Function to turn -1,0, and +1 sigma values into round latex strings for a table
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

    def ToLatexTable(self,varnames='all',order='columns'):
        #Plotting corner of the parameters to see correlations
        print("MakingLatexTable")
        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        if self.tracemask is None:
            self.tracemask=np.tile(True,len(self.trace['Rs']))
        if varnames is None or varnames == 'all':
            varnames=[var for var in trace.varnames if var[-2:]!='__' and var not in ['gp_pred','light_curves']]

        self.samples = pm.trace_to_dataframe(self.trace, varnames=varnames)
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
