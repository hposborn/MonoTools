import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


MonoData_tablepath = os.path.join('/'.join(os.path.dirname( __file__ ).split('/')[:-1]),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname(os.path.dirname( __file__ )),'data')
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
    
    def __init__(self, ID, mission, lc=None, planets=None, overwrite=False, savefileloc=None, **kwargs):
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
                       'fit_params':['logror','tdur'], # fit_params - list of strings - fit these parameters. Options: ['logror', 'b' or 'tdur', 'ecc', 'omega']

                       'marginal_params':['per','b'], # marginal_params - list of strings - marginalise over these parameters. Options: ['per', 'b' or 'tdur', 'ecc', 'omega','logror']
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'N_ecc_samples':15,      # N_ecc_samples - int - in the case that we're fitting for ingress and duration therefore deriving velocity therefore marginalising over period and eccentricity, we want to know how many eccentricity samples to produce for each period
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
                lc['mask']=np.tile(True,len(x))

            if (np.sort(lc['time'])!=lc['time']).all():
                if self.debug: print("#SORTING")
                for key in [k for key in lc if type(k)==np.ndarray and k!='time']:
                    if len(lc[key])==len(lc['time']):
                        lc[key]=lc[key][np.argsort(lc['time'])][:]
                lc['time']=np.sort(lc['time'])
            
            #Making sure all lightcurve arrays match theano.floatX:
            lc.update({key:lc[key].astype(floattype) for key in lc if type(lc[key])==np.ndarray and type(lc[key][0]) in [floattype,float,floattype]})
            
            lc['near_trans'] = np.tile(False, len(lc['time'])) if 'near_trans' not in lc else lc['near_trans']
            lc['in_trans']   = np.tile(False, len(lc['time'])) if 'in_trans' not in lc else lc['in_trans']
            
            self.lc=lc
            self.planets={}
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
                pl_dic['ror']=pl_dic['depth']**0.5
                pl_dic['log_ror']=np.log(pl_dic['ror'])
        if 'ror' not in pl_dic:
            pl_dic['ror']=np.exp(pl_dic['log_ror'])
        
        if 'r_pl' not in pl_dic and hasattr(self,'Rstar'):
            pl_dic['r_pl']=pl_dic['ror']*self.Rstar[0]*11.2
        
        #Adding dict as planet:
        if pltype=='multi':
            self.add_multi(pl_dic, name)
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

    def add_multi(self, pl_dic, name):
        assert name not in self.planets
        #Adds planet with multiple eclipses
        if not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.5*pl_dic['tdur']/pl_dic['period']
        
        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            #Estimating b from simple geometry:
            ror=np.sqrt(pl_dic['depth']) if hasattr(self,'depth') else 0.0
            pl_dic['b']=np.clip((1+ror)**2 - (pl_dic['tdur']*86400)**2 * \
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
        p_gaps=self.compute_period_gaps(pl_dic['tcen'],tdur=pl_dic['tdur'],depth=pl_dic['depth'])
        pl_dic['per_gaps']=np.column_stack((p_gaps,
                                            p_gaps[:,1]-p_gaps[:,0],
                                            -5/3*(p_gaps[:,1]**(-5/3)-p_gaps[:,0]**(-5/3))))
        pl_dic['per_gaps'][:,-1]/=np.sum(pl_dic['per_gaps'][:,-1])
        pl_dic['P_min']=p_gaps[0,0]
        
        if 'log_ror' not in pl_dic:
            if 'ror' in pl_dic:
                pl_dic['log_ror']=np.log(pl_dic['ror'])
            elif 'depth' in pl_dic:
                pl_dic['ror']=pl_dic['depth']**0.5
                pl_dic['log_ror']=np.log(pl_dic['ror'])
        pl_dic['ngaps']=len(pl_dic['per_gaps'])
        
        if 'b' not in pl_dic and 'depth' in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            ror=np.sqrt(pl_dic['depth']) if hasattr(self,'depth') else 0.0
            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+ror)**2 - (pl_dic['tdur']*86400)**2 * \
                                ((3*np.min(pl_dic['per_gaps'][:,0])*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        
        self.planets[name]=pl_dic
        self.monos+=[name]
        
    def compute_rms_series(self,tdur,split_gap_size=2.0,n_steps_per_dur=7):
        # Computing an RMS time series for the lightcurve by binning
        # split_gap_size = Duration at which to cut the lightcurve and compute in loops
        # n_steps_per_dur = number of steps with which to cut up each duration. Odd numbers work most uniformly
        
        if not hasattr(self.lc,'flux_flat') or len(self.lc['flux_flat'])!=len(self.lc['flux_err']):
            self.lc = tools.lcFlatten(self.lc,transit_mask=~self.lc['in_trans'],
                                      stepsize=0.133*np.min([self.planets[pl]['tdur'] for pl in self.planets]),
                                      winsize=6.5*np.max([self.planets[pl]['tdur'] for pl in self.planets]))

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
        return gap_start_ends
    
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
    
    def add_duo(self, pl_dic,name):
        assert name not in self.planets
        #Adds planet with two eclipses and unknown period between these
        assert pl_dic['period']==abs(pl_dic['tcen']-pl_dic['tcen_2'])
        if not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.5*pl_dic['tdur']/pl_dic['period']
        tcens=np.array([pl_dic['tcen'],pl_dic['tcen_2']])
        pl_dic['tcen']=np.min(tcens)
        pl_dic['tcen_2']=np.max(tcens)
        #Calculating P_min and the integer steps
        pl_dic=self.compute_duo_period_aliases(pl_dic)
        pl_dic['npers']=len(pl_dic['period_int_aliases'])
        
        if 'b' not in pl_dic:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            ror=np.sqrt(pl_dic['depth']) if hasattr(self,'depth') else 0.0
            #Estimating b from simple geometry:
            pl_dic['b']=np.clip((1+ror)**2 - (pl_dic['tdur']*86400)**2 * \
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
        #Adds stellar parameters to model
        self.Rstar=np.array(Rstar)
        self.Teff=np.array(Teff)
        self.logg=np.array(logg)
        self.FeH=FeH
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
            else:
                rhostar=[rho_logg[0],rho_logg[0]*rho_logg[1],rho_logg[0]*rho_logg[2]]
            self.rhostar=np.array(rhostar)
    
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
        self.fit_params=self.fit_params+['ecc'] if self.assume_circ and 'ecc' not in self.fit_params and 'ingress' not in self.fit_params else self.fit_params
        self.fit_params=self.fit_params+['omega'] if self.assume_circ and 'omega' not in self.fit_params and 'ingress' not in self.fit_params else self.fit_params
        assert self.use_multinest^self.use_pymc3 #Must have either use_multinest or use_pymc3, though use_multinest doesn't work
        
        n_pl=len(self.planets)
        assert n_pl>0

        if self.debug: print(len(self.planets),'planets |','monos:',self.monos,'multis:',self.multis,'duos:',self.duos, "use GP=",self.use_GP)

        ######################################
        #   Masking out-of-transit flux:
        ######################################
        # To speed up computation, here we loop through each planet and add the region around each transit to the data to keep
        self.lc=tools.lcFlatten(self.lc,transit_mask=~self.lc['in_trans'],
                                     stepsize=0.133*np.min([self.planets[pl]['tdur'] for pl in self.planets]),
                                     winsize=6.5*np.max([self.planets[pl]['tdur'] for pl in self.planets]))

        if self.cutDistance>0 or not self.use_GP:
            if self.bin_oot:
                #Creating a pseudo-binned dataset where out-of-transit LC is binned to 30mins but near-transit is not.
                oot_binsize=1/8 if self.mission.lower()=='kepler' else 1/48
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
            
            self.lc['tele_index']=np.zeros((len(self.lc['time']),3))
            if self.bin_oot:
                self.pseudo_binlc['tele_index']=np.zeros((len(self.pseudo_binlc['time']),3))
            else:
                self.lc_near_trans['tele_index']=np.zeros((len(self.lc_near_trans['time']),3))
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
            #max_cad = np.nanmax([np.nanmedian(np.diff(uselc['time'][uselc['near_trans']&(uselc['cadence']==c)])) for c in self.cads])
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
        
        start=None
        with pm.Model() as model:            
            
            if self.debug: print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)
            
            ######################################
            #   Intialising Stellar Params:
            ######################################
            #Using log rho because otherwise the distribution is not normal:
            logrho_S = pm.Normal("logrho_S", mu=np.log(self.rhostar[0]), 
                                 sd=np.average(abs(self.rhostar[1:]/self.rhostar[0])),
                                 testval=np.log(self.rhostar[0]))
            rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S)*1.4097798) #Converting from rho_sun into g/cm3
            Rs = pm.Normal("Rs", mu=self.Rstar[0], sd=np.average(abs(self.Rstar[1:])),testval=self.Rstar[0],shape=1)
            Ms = pm.Deterministic("Ms",(rho_S)*Rs**3)

            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if self.useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0
            
            ######################################
            #     Initialising Mono params
            ######################################
            if len(self.monos)>0:
                # The period distributions of monotransits are tricky as we often have gaps to contend with
                # We cannot sample the full period distribution while some regions have p=0.
                # Therefore, we need to find each possible period region and marginalise over each
                
                min_Ps=np.array([self.planets[pls]['P_min'] for pls in self.monos])
                if self.debug: print(min_Ps)
                #From Dan Foreman-Mackey's thing:
                test_ps=np.array([self.planets[pls]['period'] if self.planets[pls]['period']>self.planets[pls]['P_min'] else 1.25*self.planets[pls]['P_min'] for pls in self.monos])
                mono_uniform_index_period={}
                mono_tdurs={};mono_bsqs={};mono_b_priors={}
                mono_pers={};mono_t0s={};mono_logrors={};mono_bs={};
                mono_v_vcircs={}; mono_reparam_omegas={};mono_reparam_omega_base={}
                per_meds={} #median period from each bin
                per_index=-8/3
                
                for pl in self.monos:
                    #P_index = xo.distributions.UnitUniform("P_index", shape=n_pl, testval=pertestval)#("P_index", mu=0.5, sd=0.3)
                    #P_index = pm.Bound("P_index", upper=1.0, lower=0.0)("P_index", mu=0.5, sd=0.33, shape=n_pl)
                    #period = pm.Deterministic("period", tt.power(P_index,1/per_index)*P_min)
                    ind_min=np.power(self.planets[pl]['per_gaps'][:,1]/self.planets[pl]['per_gaps'][:,0],per_index)
                    per_meds[pl]=np.power(((1-ind_min)*0.5+ind_min),per_index)*self.planets[pl]['per_gaps'][:,0]
                    if 'log_per' in self.planets[pl]:
                        testindex=[]
                        for ngap in np.arange(len(self.planets[pl]['per_gaps'][:,1])):
                            if np.exp(self.planets[pl]['log_per'])>self.planets[pl]['per_gaps'][ngap,0] and np.exp(self.planets[pl]['log_per'])<self.planets[pl]['per_gaps'][ngap,1]:
                                testindex+=[((np.exp(self.planets[pl]['log_per'])/self.planets[pl]['per_gaps'][ngap,0])**per_index - ind_min[ngap])/(1-ind_min[ngap])]
                            else:
                                testindex+=[np.clip(np.random.normal(0.5,0.25),0.00001,0.99999)]
                        #p = ( ( (1-(max/min^i)) * index+(max/min^i) )^(1/i))*min
                        #index = ((p/min)^(i) - ((max/min)^i))/(1-(max/min^i))
                        #print(np.exp(self.planets[pl]['log_per']),"->",testindex,"compared to",
                        #      np.power(((1-ind_min)*0.5+ind_min),1/per_index)*self.planets[pl]['per_gaps'][0,0])
                        mono_uniform_index_period[pl]=xo.distributions.UnitUniform("mono_uniform_index_"+str(pl),
                                                        shape=len(self.planets[pl]['per_gaps'][:,0]),
                                                        testval=testindex)
                    else:
                        mono_uniform_index_period[pl]=xo.distributions.UnitUniform("mono_uniform_index_"+str(pl),
                                                        shape=len(self.planets[pl]['per_gaps'][:,0]))

                    mono_pers[pl]=pm.Deterministic("mono_pers_"+str(pl), tt.power(((1-ind_min)*mono_uniform_index_period[pl]+ind_min),1/per_index)*self.planets[pl]['per_gaps'][:,0])
                    mono_t0s[pl] = pm.Bound(pm.Normal, 
                                            upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.33,
                                            lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.33
                                           )("mono_t0s_"+pl,mu=self.planets[pl]['tcen'],
                                                  sd=self.planets[pl]['tdur']*0.1,
                                                  testval=self.planets[pl]['tcen'])

                   
                    if self.debug: print(np.log(0.001),"->",np.log(0.25+int(self.useL2)),
                                   np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))
                    mono_logrors[pl]=pm.Uniform("mono_logrors_"+pl,lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                                testval=np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))

                    mono_tdurs[pl] = pm.Uniform("mono_tdurs_"+pl,
                                           lower=0.33*self.planets[pl]['tdur'],
                                           upper=3*self.planets[pl]['tdur'],
                                           testval=self.planets[pl]['tdur'])
                    
                    mono_bs[pl] = xo.distributions.ImpactParameter("mono_bs_"+pl, ror=tt.exp(mono_logrors[pl]))
                    
                    #mono_ingress_repar[pl]= pm.Uniform("__mono_ingress_repar"+pl,lower=0.0,upper=1.0,
                    #                        testval=np.clip(1/(1/(4*self.planets[pl]['r_pl']/(109.1*self.Rstar[0]))-1/2),0,1.0))
                    #As ingress can never be <2Rp/Rs, we reparameterise to this space:
                    #mono_ingresses[pl] = pm.Deterministic("mono_ingresses_"+pl,
                    #                                     mono_ingress_repar[pl]*(1.0-2*tt.exp(mono_logrors[pl])))

                    #Deriving b from ingress
                    #mono_bsqs[pl] = pm.Deterministic("mono_bsqs_"+pl, (tt.exp(mono_logrors[pl])**2*mono_ingresses[pl] - 2*tt.exp(mono_logrors[pl]) + mono_ingresses[pl])/mono_ingresses[pl])
                    
                    #if self.debug: tt.printing.Print("ingress:")(mono_ingresses[pl])

                    #tt.printing.Print("derived b^2 values:")(mono_bsqs[pl])
                    #Deriving the difference in velocity compared to each circular velocity given the periods:
                    mono_v_vcircs[pl] = pm.Deterministic("mono_v_vcircs_"+pl, 
                                                         ((1+tt.exp(mono_logrors[pl]))**2 - mono_bs[pl]**2) / \
                                                         ((mono_tdurs[pl]*86400)**2 * \
                                                         ((3*mono_pers[pl]*86400)/(np.pi*6.67e-11*rho_S*1000))**(-2/3))
                                                        )
                    mono_reparam_omega_base[pl] = pm.Uniform("__mono_reparam_omega_base_"+pl,lower=0,upper=1,
                                                         testval=np.random.random(self.planets[pl]['ngaps']),
                                                         shape=(self.planets[pl]['ngaps']))
                    mono_reparam_omegas[pl] = pm.Deterministic("mono_reparam_omegas_"+pl, 
                                                               (mono_reparam_omega_base[pl].dimshuffle('x',0) + \
                                                                tt.arange(int(self.N_ecc_samples)).dimshuffle(0,'x')) / \
                                                               int(self.N_ecc_samples))

                    #mono_bs[pl] = pm.Deterministic("mono_bs_"+pl,tt.clip(mono_bsqs[pl], 1e-5, 100)**0.5)
                    # Combining together prior from db/dtdur (which is needed to renormalise to constant b) ~ P**(-2/3)/b
                    # And custom prior which reduces the logprob of all models with bsq<0 (i.e. unphysical) by 5-25 
                    #mono_b_priors[pl]=pm.Deterministic("mono_b_priors_"+pl,tt.log( (tt.max(mono_bs[pl])/mono_bs[pl]) * \
                    #                                                (mono_pers[pl]/tt.max(mono_pers[pl]))**(-5/2) ) + 
                    #                                                tt.switch(tt.lt(mono_bsqs[pl],0),mono_bsqs[pl]*40-15,0))
                    
            ######################################
            #     Initialising Duo params
            ######################################
            if len(self.duos)>0:
                #Again, in the case of a duotransit, we have a series of possible periods between two know transits.
                # TO model these we need to compute each and marginalise over them
                duo_pers={};duo_t0s={};duo_t0_2s={};duo_logrors={};duo_bs={};duo_tdurs={}
                duo_v_vcircs={}; duo_reparam_omega_base={};duo_reparam_omegas={};#duo_ingress_repar={};duo_ingresses={}
                duo_tdurs={};
                
                for npl,pl in enumerate(self.duos):
                    duo_t0s[pl] = pm.Bound(pm.Normal, 
                                           upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.5, 
                                           lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.5
                                          )("duo_t0s_"+pl,mu=self.planets[pl]['tcen'], 
                                                 sd=self.planets[pl]['tdur']*0.2,
                                                 testval=self.planets[pl]['tcen'])
                    duo_t0_2s[pl] = pm.Bound(pm.Normal, 
                                                   upper=self.planets[pl]['tcen_2']+self.planets[pl]['tdur']*0.5, 
                                                   lower=self.planets[pl]['tcen_2']-self.planets[pl]['tdur']*0.5
                                                  )("duo_t0_2s_"+pl,mu=self.planets[pl]['tcen_2'],
                                                                      sd=self.planets[pl]['tdur']*0.2,
                                                                      testval=self.planets[pl]['tcen_2'])
                    duo_pers[pl]=pm.Deterministic("duo_pers_"+pl,
                                                     abs(duo_t0_2s[pl]-duo_t0s[pl])/self.planets[pl]['period_int_aliases'])
                    duo_logrors[pl]=pm.Uniform("duo_logrors_"+pl, lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                               testval=np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))

                    #Fitting for tdur, deriving b from other quantities:
                    duo_tdurs[pl] = pm.Uniform("duo_tdurs_"+pl,
                                           lower=0.33*self.planets[pl]['tdur'],
                                           upper=3*self.planets[pl]['tdur'],
                                           testval=self.planets[pl]['tdur'])
                    #initror=3*self.planets[pl]['r_pl']/(109.1*self.Rstar[0])
                    #duo_ingress_repar[pl] = pm.Uniform("__duo_ingress_repar"+pl,lower=0.0,upper=1.0,
                    #                                   testval=np.clip(1/(1/(4*initror)-1/2),0,1.0))
                    #As ingress can never be <2Rp/Rs, we reparameterise to this space:
                    #duo_ingresses[pl] = pm.Deterministic("duo_ingresses_"+pl,
                    #                                     duo_ingress_repar[pl]*(1.0-2*tt.exp(duo_logrors[pl])))
                    #
                    #if self.debug: tt.printing.Print("ingress:")(duo_ingresses[pl])
                    #Deriving b from ingress:
                    #duo_bsqs[pl] = pm.Deterministic("__duo_bsqs_"+pl, (tt.exp(duo_logrors[pl])**2*duo_ingresses[pl] - 2*tt.exp(duo_logrors[pl]) + duo_ingresses[pl])/duo_ingresses[pl])
                    #duo_ingresses[pl] = 

                    #if self.debug: tt.printing.Print("derived b^2:")(duo_bsqs[pl])
                    #Deriving the difference in velocity compared to each circular velocity given the periods:
                    duo_bs[pl] = xo.distributions.ImpactParameter("duo_bs_"+pl,
                                                                  ror=tt.exp(duo_logrors[pl]),
                                                                  testval = 0.4)
                    
                    duo_v_vcircs[pl] = pm.Deterministic("duo_v_vcircs_"+pl, 
                                                         ((1+tt.exp(duo_logrors[pl]))**2 - duo_bs[pl]**2) / \
                                                         ((duo_tdurs[pl]*86400)**2 * \
                                                         ((3*duo_pers[pl]*86400)/(np.pi*6.67e-11*rho_S*1000))**(-2/3))
                                                        )
                    if self.debug: tt.printing.Print("derived v/vcircs:")(duo_v_vcircs[pl])
                    duo_reparam_omega_base[pl] = pm.Uniform("__duo_reparam_omega_base_"+pl,lower=0,upper=1,
                                                         testval=np.random.random(self.planets[pl]['npers']),
                                                         shape=(self.planets[pl]['npers']))
                    if self.debug: tt.printing.Print("random 0 to 1 base for omega selection:")(duo_reparam_omega_base[pl])
                    duo_reparam_omegas[pl]=pm.Deterministic("duo_reparam_omegas_"+pl,
                                                           (duo_reparam_omega_base[pl].dimshuffle('x',0) + tt.arange(int(self.N_ecc_samples)).dimshuffle(0,'x')) / \
                                                           int(self.N_ecc_samples))
                    if self.debug: tt.printing.Print("array from 0 to 1 for omega selection:")(duo_reparam_omegas[pl])
                    
                    # Combining together prior from db/dtdur (which is needed to renormalise to constant b) ~ P**(-2/3)/b
                    # And custom prior which reduces the logprob of all models with bsq<0 (i.e. unphysical) by 5-25 
                    #duo_b_priors[pl]=pm.Deterministic("duo_b_priors_"+pl, tt.log((tt.max(duo_bs[pl])/duo_bs[pl]) * \
                    #                                                 (duo_pers[pl]/tt.max(duo_pers[pl]))**(-5/2) ) + 
                    #                                                 tt.switch(tt.lt(duo_bsqs[pl],0),duo_bsqs[pl]*40-15,0) )
                    if self.debug:
                        tt.printing.Print("bs")(duo_bs[pl])
                        #tt.printing.Print("unphysical_priors")(tt.switch(tt.lt(duo_bsqs[pl],0),duo_bsqs[pl]*40-15,0))
                        #tt.printing.Print("combined_priors")(duo_b_priors[pl])
                            
            ######################################
            #     Initialising Multi params
            ######################################
            
            if len(self.multis)>0:
                #In the case of multitransiting plaets, we know the periods already, so we set a tight normal distribution
                inipers=np.array([self.planets[pls]['period'] for pls in self.multis])
                inipererrs=np.array([self.planets[pls]['period_err'] for pls in self.multis])
                if self.debug: print("init periods:", inipers,inipererrs)
                multi_pers = pm.Normal("multi_pers", 
                                          mu=inipers,
                                          sd=np.clip(inipererrs*0.25,np.tile(0.005,len(inipers)),0.02*inipers),
                                          shape=len(self.multis),
                                          testval=inipers)
                tcens=np.array([self.planets[pls]['tcen'] for pls in self.multis]).ravel()
                tdurs=np.array([self.planets[pls]['tdur'] for pls in self.multis]).ravel()
                multi_t0s = pm.Bound(pm.Normal, upper=tcens+tdurs*0.5, lower=tcens-tdurs*0.5,
                                    )("multi_t0s",mu=tcens, sd=tdurs*0.05,
                                            shape=len(self.multis),testval=tcens)
                if self.debug: print(np.log(0.001),"->",np.log(0.25+float(int(self.useL2))),'rors:',
                               np.log(np.array([self.planets[pls]['r_pl']/(109.1*self.Rstar[0]) for pls in self.multis])))
                multi_logrors=pm.Uniform("multi_logrors",lower=np.log(0.001), upper=np.log(0.25+float(int(self.useL2))),
                             testval=np.log(np.array([self.planets[pls]['r_pl']/(109.1*self.Rstar[0]) for pls in self.multis])),
                                        shape=len(self.multis))
                
                if not self.assume_circ:
                    if self.ecc_prior.lower()=='kipping' or (self.ecc_prior.lower()=='auto' and len(self.planets)==1):
                        multi_eccs = BoundedBeta("multi_eccs", alpha=0.867, beta=3.03, testval=0.05,shape=len(self.multis))
                    elif self.ecc_prior.lower()=='vaneylen' or (self.ecc_prior.lower()=='auto' and len(self.planets)>1):
                        # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                        multi_eccs = pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("multi_eccs",alpha= 0.049,beta=2,
                                                                                       testval=0.05,shape=len(self.multis))
                    elif self.ecc_prior.lower()=='uniform':
                        multi_eccs = pm.Uniform('multi_eccs',lower=1e-5, upper=1-1e-5,shape=len(self.multis))
                    multi_omegas = xo.distributions.Angle("multi_omegas",shape=len(self.multis))
                
                #Fixing the multiplanets to explore impact parameter (as tdur reparameterisation requires long-period planets)
                multi_bs = xo.distributions.ImpactParameter("multi_bs",ror=tt.exp(multi_logrors),shape=len(self.multis),
                                                            testval=np.array([self.planets[pls]['b'] for pls in self.multis]))

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

            mean=pm.Normal("mean",mu=np.median(lc['flux'][lc['mask']]),
                                  sd=np.std(lc['flux'][lc['mask']]))
            if not hasattr(self,'log_flux_std'):
                self.log_flux_std=np.array([np.log(np.nanstd(lc['flux'][~lc['in_trans']][lc['cadence'][~lc['in_trans']]==c])) for c in self.cads]).ravel().astype(floattype)
            if self.debug: print(self.log_flux_std,np.sum(~lc['in_trans']),"/",len(~lc['in_trans']))
            
            logs2 = pm.Normal("logs2", mu = self.log_flux_std+1, 
                              sd = np.tile(2.0,len(self.log_flux_std)), shape=len(self.log_flux_std))

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
            def gen_lc(i_orbit, i_r, n_pl, mask=None, prefix='',name=None,make_deterministic=False):
                # Short method to create stacked lightcurves, given some input time array and some input cadences:
                # This function is needed because we may have 
                #   -  1) multiple cadences and 
                #   -  2) multiple telescopes (and therefore limb darkening coefficients)
                trans_pred=[]
                mask = ~np.isnan(lc['time']) if mask is None else mask
                cad_index=[]
                for cad in self.cads:
                    cadmask=mask&(lc['cadence']==cad)
                    
                    #print(self.lc['tele_index'][mask,0].astype(bool),len(self.lc['tele_index'][mask,0]),cadmask[mask],len(cadmask[mask]))
                    
                    if cad[0].lower()=='t':
                        #Taking the "telescope" index, and adding those points with the matching cadences to the cadmask
                        cad_index+=[(lc['tele_index'][mask,0].astype(bool))&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0].lower()=='k':
                        cad_index+=[(lc['tele_index'][mask,1]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0].lower()=='c':
                        cad_index+=[(lc['tele_index'][mask,2]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_corot).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=lc['time'][mask].astype(floattype),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                # transit arrays (ntime x n_pls x 2) * telescope index (ntime x n_pls x 2), summed over dimension 2
                name = prefix+"light_curves" if name is None and make_deterministic else name
                if n_pl>1 and make_deterministic:
                    
                    return pm.Deterministic(name, 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               tt.stack(cad_index).dimshuffle(1,'x',0),axis=2))
                elif n_pl==1 and make_deterministic:
                    return pm.Deterministic(name, 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                               tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2)))
                elif n_pl>1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * \
                                  tt.stack(cad_index).dimshuffle(1,'x',0),axis=2)

                elif n_pl==1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2))

            
            #####################################################
            #  Multiplanet lightcurve model & derived parameters 
            #####################################################
            if len(self.multis)>0:
                if self.assume_circ:
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S, period=multi_pers, t0=multi_t0s, b=multi_bs)
                else:
                    # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S, ecc=multi_eccs, omega=multi_omegas,
                        period=multi_pers, t0=multi_t0s, b=multi_bs)
                if self.debug: print("generating multi lcs:")
                
                multi_mask_light_curves = gen_lc(multi_orbit, tt.exp(multi_logrors),
                                                 len(self.multis),mask=None,
                                                 prefix='multi_mask_',make_deterministic=True)
                
                multi_a_Rs = pm.Deterministic("multi_a_Rs", multi_orbit.a)
                #if 'tdur' in self.marginal_params:
                multi_vx, multi_vy, multi_vz = multi_orbit.get_relative_velocity(multi_t0s)
                multi_tdur=pm.Deterministic("multi_tdurs",
                                            (2 * tt.tile(Rs,len(self.multis)) * tt.sqrt( (1+tt.exp(multi_logrors))**2 - multi_bs**2)) / tt.sqrt(multi_vx**2 + multi_vy**2) )
                if self.force_match_input is not None:
                    #pm.Bound("bounded_tdur_multi",upper=0.5,lower=-0.5,
                    #         tt.log(multi_tdur)-tt.log([self.planets[multi]['tdur'] for multi in self.multis]))
                    #pm.Potential("match_input_potential_multi",tt.sum(multi_logrors - tt.exp((tt.log(multi_tdur)-tt.log([self.planets[multi]['tdur'] for multi in self.multis]))**2))
                    pm.Potential("match_input_potential_multi",tt.sum(
                                     tt.exp( -(multi_tdur**2 + [self.planets[multi]['tdur']**2 for multi in self.multis]) / ([2*(self.force_match_input*self.planets[multi]['tdur'])**2 for multi in self.multis]) ) + \
                                     tt.exp( -(multi_logrors**2 + [self.planets[multi]['log_ror']**2 for multi in self.multis]) / ([2*(self.force_match_input*self.planets[multi]['log_ror'])**2 for multi in self.multis]) )
                                    ))
                if self.debug: print("summing multi lcs:")
                if len(self.multis)>1:
                    multi_mask_light_curve = pm.math.sum(multi_mask_light_curves, axis=1) #Summing lightcurve over n planets
                else:
                    multi_mask_light_curve=multi_mask_light_curves
                #Multitransiting planet potentials:
                '''if self.use_GP:
                    pm.Potential("multi_obs",
                                 self.gp['oot'].log_likelihood(self.lc['flux'][self.lc['near_trans']]-(multi_mask_light_curve+ mean)))
                else:
                    new_yerr = self.lc['flux_err'][self.lc['near_trans']].astype(floattype)**2 + \
                               tt.dot(self.lc['flux_err_index'][self.lc['near_trans']],tt.exp(logs2))
                    pm.Normal("multiplanet_obs",mu=(multi_mask_light_curve + mean),sd=new_yerr,
                              observed=self.lc['flux_flat'][self.lc['near_trans']].astype(floattype))
                '''
            else:
                multi_mask_light_curve = tt.as_tensor_variable(np.zeros(len(lc['time'])))
                #print(multi_mask_light_curve.shape.eval())
                #np.zeros_like(self.lc['flux'][self.lc['near_trans']])
                
            ################################################
            #    Marginalising over Duotransit periods
            ################################################
            if len(self.duos)>0:
                peri_dist={}
                duo_per_info={}
                duo_dist_in_trans={}
                duo_omegas={};duo_omegas_all={}
                duo_eccs={};duo_eccs_all={}
                duo_ecc_priors={}
                duo_ecc_logprobmarg={}
                omega_prior_all={}
                ecc_prior_to_per_marg={}
                for nduo,duo in enumerate(self.duos):
                    if self.debug: print("#Marginalising over ",len(self.planets[duo]['period_int_aliases'])," period aliases for ",duo)
                    #Marginalising over each possible period
                    #Single planet with two transits and a gap
                    duo_per_info[duo]={'lc':{}}
                    #Here we are going to marginalise over ecc and omega as well as P.

                    #Step 1 - calculate limits in omega and eccentricity (e.g. max given stellar radius)
                    a_circ = ((rho_S*1000*6.67e-11*(86400*duo_pers[duo])**2)/(3*np.pi))**(1/3)
                    max_eccs = 1 - 1/a_circ
                    #Step 2 - convert reparam_omegas (a range of random evenly-spaced values between 0 and 1) to omega:
                    duo_omegas_all[duo]=pm.Deterministic("duo_omegas_all_"+duo,
                                                     tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                         (np.arccos(-1*np.sqrt(1 - duo_v_vcircs[duo]**2))+np.pi/2)+duo_reparam_omegas[duo]*2*(np.pi-np.arccos(-1*np.sqrt(1 - duo_v_vcircs[duo]**2))),
                         (5*np.pi/2 + (2*duo_reparam_omegas[duo] - 1)*np.arccos((-1 + np.sqrt(1 - max_eccs**2)* duo_v_vcircs[duo])/max_eccs))%(2*np.pi)))
                    if self.debug: tt.printing.Print("all_omegas:")(duo_omegas_all[duo])

                    #Step 3 - compute eccentricity for each of the omegas:
                    duo_eccs_all[duo]=pm.Deterministic("duo_eccs_all_"+duo,
                                                   tt.clip(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                               (duo_v_vcircs[duo]**2*(-1*np.sqrt((duo_v_vcircs[duo]**2*(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2-1))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2)**2)) - np.cos(duo_omegas_all[duo]-np.pi/2)**2*np.sqrt((duo_v_vcircs[duo]**2*(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2-1))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2)**2)-np.cos(duo_omegas_all[duo]-np.pi/2))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2 + duo_v_vcircs[duo]**2),
                                (duo_v_vcircs[duo]**2*(np.sqrt((duo_v_vcircs[duo]**2*(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2-1))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2)**2)) + np.cos(duo_omegas_all[duo]-np.pi/2)**2*np.sqrt((duo_v_vcircs[duo]**2*(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2-1))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2+duo_v_vcircs[duo]**2)**2)-np.cos(duo_omegas_all[duo]-np.pi/2))/(np.cos(duo_omegas_all[duo]-np.pi/2)**2 + duo_v_vcircs[duo]**2)),
                                                           1e-4,1-1e-4)
                                                  )
                    if self.debug: tt.printing.Print("all_eccentricities:")(duo_eccs_all[duo])

                    #Step 4 - compute prior prob for each of the omega+eccentricity combinations
                    min_eccs = tt.clip(abs(2/(1 + duo_v_vcircs[duo]**2) - 1), 1e-4, 1.0-1e-4) #Minimum eccentricity given velocity and 2Rs
                    #Perihelion distance from eccentricities and circular velocities:
                    peri_dist[duo]=pm.Deterministic("peri_dist_"+duo,
                                                    ((rho_S*1000*6.67e-11*(duo_pers[duo]*86400)**2)/(3*np.pi))**(1/3)* \
                                                    (1-duo_eccs_all[duo]))
                    if self.ecc_prior.lower() =='kipping' or (len(self.planets)==1 and self.ecc_prior.lower()=='auto'):
                        alpha=0.867;beta=3.03
                        if len(self.multis)>0:
                            #1) PDF of each eccentricities (normalised to beta function given alpha/beta: 0.427186)
                            #2) d omega - i.e. width of each omega sample given N_ecc_samples spread across a certain width
                            #3) Normalised to the PDF of the median eccentricity, 0.173379575 -> 2.95669
                            #4) Geometric prior from the inverse distance during transit (peri/rs) as a ratio of s.m.a.
                            #5) Strong prior against orbits which intersect the star
                            #6) Strong prior against orbits which intersect the largest possible orbits of known planets
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(((duo_eccs_all[duo]-1e-4)**(alpha-1)*(1-1e-4-duo_eccs_all[duo])**(-1e-4))/(0.427186*(1-2e-4)**(alpha+beta-1))) - \
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(2*np.pi))).dimshuffle('x',0) + \
                                  tt.log(2.95669) + \
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100) + \
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo]/tt.max(multi_a_Rs) - 1)))-100)
                                                                  )
                        else:
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(((duo_eccs_all[duo]-1e-4)**(alpha-1)*(1-1e-4-duo_eccs_all[duo])**(-1e-4))/(0.427186*(1-2e-4)**(alpha+beta-1))) + \
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(self.N_ecc_samples*2*np.pi))).dimshuffle('x',0)+\
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100)
                                                                  )
                        #The probabilities of eccentricities available / total probabilities
                    elif self.ecc_prior.lower() =='vaneylen' or (len(self.planets)>1 and self.ecc_prior.lower()=='auto'):
                        sigma=0.049
                        #1) PDF of each eccentricities
                        #2) Normalised to the PDF of the median eccentricity, 0.05769309110325826 -> 12.014387984851783
                        #3) Omega prior from the fraction of regions of omega possible to create transit
                        #4) Geometric prior from the inverse distance during transit (peri/rs) as a ratio of s.m.a.
                        #5) Strong prior against orbits which intersect the star
                        #6) Strong prior against orbits which intersect the largest possible orbits of known planets

                        # probability of each from the PDF of the Rayleigh distribution
                        # and from the geometric probability, i.e. (planet-star transit separation)/s.m.a.
                        if len(self.multis)>0:
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(duo_eccs_all[duo] * tt.exp((-1*duo_eccs_all[duo]**2)/(2*sigma**2)) / sigma**2) - \
                                  tt.log(12.014387984851783) + \
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(2*np.pi))).dimshuffle('x',0)+\
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100) + \
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo]/tt.max(multi_a_Rs) - 1)))-100)
                                                                  )
                        else:
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(duo_eccs_all[duo] * np.exp((-1*duo_eccs_all[duo]**2)/(2*sigma**2)) / sigma**2) - \
                                  tt.log(12.014387984851783) + \
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(2*np.pi))).dimshuffle('x',0)+\
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100)
                                                                  )
                    else:
                        if len(self.multis)>0:
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(2*np.pi))).dimshuffle('x',0)+\
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100) + \
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo]/tt.max(multi_a_Rs) - 1)))-100)
                                                                  )
                        else:
                            duo_ecc_priors[duo] = pm.Deterministic("duo_ecc_priors_"+duo,
                                  tt.log(tt.switch(tt.lt(duo_v_vcircs[duo],1.0),
                                            (np.pi-tt.arccos(-1*tt.sqrt(1 - duo_v_vcircs[duo]**2)))/(2*np.pi),
                                            (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*duo_v_vcircs[duo]-1)/max_eccs))/(2*np.pi)))+\
                                  tt.log((1+duo_eccs_all[duo]*tt.cos(duo_omegas_all[duo]+0.5*np.pi))/(1+duo_eccs_all[duo]**2))+\
                                  (100 / (1 + tt.exp(-20*(peri_dist[duo] - 2)))-100)                                           )
                    
                    #Step 6 - compute marginal prob for all omega/ecc samples
                    '''
                    #Lets do this with the period priors:
                    duo_ecc_logprobmarg[duo] = pm.Deterministic("duo_ecc_logprobmarg_"+duo,
                                               (duo_ecc_priors[duo]+omega_prior_all[duo].dimshuffle('x',0)) - \
                                               pm.math.logsumexp(duo_ecc_priors[duo])+omega_prior_all[duo].dimshuffle('x',0))
                    '''
                    #Step 7 - Taking the minimum eccentricity to compute the lightcurve model:
                    duo_omegas[duo]=tt.switch(tt.lt(duo_v_vcircs[duo],1.0),1.5*np.pi,0.5*np.pi)
                    duo_eccs[duo] = min_eccs
                    #duo_omegas[duo]=duo_omegas_all[duo][tt.eq(duo_ecc_logprobmarg[duo],tt.max(duo_ecc_logprobmarg[duo],axis=0))]
                    #duo_eccs[duo]=duo_eccs_all[duo][tt.eq(duo_ecc_logprobmarg[duo],tt.max(duo_ecc_logprobmarg[duo],axis=0))]
                    #Step 7 - Compute prior for omega (e.g. omega range, which depends on vcirc>1.0, divided by 2pi)
                    '''
                    #Step 8 - Compute prior for whole eccentricity region (i.e. sum of beta distribution PDF within eccentricity range)
                    if self.ecc_prior.lower() =='kipping' or (len(self.planets)==1 and self.ecc_prior.lower()=='auto'):
                        ecc_prior_to_per_marg[duo]=pm.Deterministic("ecc_prior_to_per_marg_"+duo,
                                                                    tt.sum(duo_ecc_priors[duo]) + \
                                         tt.log((1.1534*max_eccs**0.8672*(1 - 0.942694*max_eccs + 0.316151*max_eccs**2 - \
                                                                          0.00234395*max_eccs**3 - 0.00045162*max_eccs**4 - \
                                                                          0.000147609*max_eccs**5) - \
                                                 1.1534*min_eccs**0.8672*(1 - 0.942694*min_eccs + 0.316151*min_eccs**2 - \
                                                                          0.00234395*min_eccs**3 - 0.00045162*min_eccs**4 - \
                                                                          0.000147609*min_eccs**5)) / \
                                                            0.4276486107761747))
                    elif self.ecc_prior.lower() =='vaneylen' or (len(self.planets)>1 and self.ecc_prior.lower()=='auto'):
                        ecc_prior_to_per_marg[duo]=pm.Deterministic("ecc_prior_to_per_marg_"+duo,
                                                            tt.sum(duo_ecc_priors[duo]) + \
                                                            tt.log((1 - tt.exp((-1*max_eccs**2)/(2*sigma**2))) - \
                                                                   (1 - tt.exp((-1*min_eccs**2)/(2*sigma**2)))
                                                                   ))

                    else:
                        ecc_prior_to_per_marg[duo]=pm.Deterministic("ecc_prior_to_per_marg_"+duo,
                                                                    tt.sum(duo_ecc_priors[duo]) + \
                                                                    tt.tile(0.0,self.planets[duo]['npers']))
                    #This step is not actually necessary as the prior from the sum of PDF points is propagated
                    
                    ecc_prior_to_per_marg[duo]=pm.Deterministic("ecc_prior_to_per_marg_"+duo,
                                                                pm.math.logsumexp(duo_ecc_priors[duo],axis=0))
                    '''
                    # We are only creating a _single_ transit lightcurve and orbit. 
                    duoorbit = xo.orbits.KeplerianOrbit(r_star=Rs,
                                                        rho_star=rho_S, 
                                                        period=duo_pers[duo][tt.argmin(duo_eccs[duo])],
                                                        t0=duo_t0s[duo],
                                                        b=duo_bs[duo],
                                                        ecc=tt.min(duo_eccs[duo]),
                                                        omega=duo_omegas[duo][tt.argmin(duo_eccs[duo])])
                    
                    duo_per_info[duo]['logpriors'] = pm.Deterministic('logpriors_'+duo,
                                                                   -2 * tt.log(duo_pers[duo].dimshuffle('x',0)) + \
                                                                   duo_ecc_priors[duo])
                    
                    #Including this in the potential:
                    pm.Potential("period_prior_into_model_"+duo,-2.666666 * tt.log(duo_pers[duo] / \
                                                                              np.min(self.planets[duo]['period_aliases'])))
                    #pm.Potential('sum_logprior_'+duo, pm.math.logsumexp(duo_per_info[duo]['logpriors']))
                    
                    duo_per_info[duo]['lcs'] = gen_lc(duoorbit,tt.exp(duo_logrors[duo]),1,
                                                      mask=None, name='marg_light_curve_'+duo,make_deterministic=True)

            ################################################
            #    Marginalising over Monotransit gaps
            ################################################
            if len(self.monos)>0:
                mono_a_Rs={}
                mono_per_info={}
                mono_dist_in_trans={}
                mono_gap_info={}
                if 'tdur' in self.marginal_params:
                    mono_vels={}
                    mono_tdurs={}
                for nmono,mono in enumerate(self.monos):
                    if self.debug: print("#Marginalising over ",len(self.planets[mono]['per_gaps'])," period gaps for ",mono)
                    
                mono_omegas={};mono_omegas_all={}
                mono_eccs={};mono_eccs_all={}
                mono_ecc_priors={}
                mono_ecc_logprobmarg={}
                omega_prior_all={}
                ecc_prior_to_per_marg={}
                #Here we are going to marginalise over ecc and omega as well as P.

                #Step 1 - calculate limits in omega and eccentricity (e.g. max given stellar radius)
                a_circ = ((rho_S*1000*6.67e-11*(86400*mono_pers[mono])**2)/(3*np.pi))**(1/3)
                max_eccs = 1 - 1/a_circ
                #Step 2 - convert reparam_omegas to omega:
                mono_omegas_all[mono]=pm.Deterministic("mono_omegas_all_"+mono,
                                                 tt.switch(tt.lt(mono_v_vcircs[mono],1.0),
                     (np.arccos(-1*np.sqrt(1 - mono_v_vcircs[mono]**2))+np.pi/2)+mono_reparam_omegas[mono]*2*(np.pi-np.arccos(-1*np.sqrt(1 - mono_v_vcircs[mono]**2))),
                     (5*np.pi/2 + (2*mono_reparam_omegas[mono] - 1)*np.arccos((-1 + np.sqrt(1 - max_eccs**2)* mono_v_vcircs[mono])/max_eccs))%(2*np.pi)))

                #Step 3 - compute eccentricity for each of the omegas:
                mono_eccs_all[mono]=pm.Deterministic("mono_eccs_all_"+mono,
                                               tt.switch(tt.lt(mono_v_vcircs[mono],1.0),
                           (mono_v_vcircs[mono]**2*(-1*np.sqrt((mono_v_vcircs[mono]**2*(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2-1))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2)**2)) - np.cos(mono_omegas_all[mono]-np.pi/2)**2*np.sqrt((mono_v_vcircs[mono]**2*(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2-1))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2)**2)-np.cos(mono_omegas_all[mono]-np.pi/2))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2 + mono_v_vcircs[mono]**2),
                            (mono_v_vcircs[mono]**2*(np.sqrt((mono_v_vcircs[mono]**2*(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2-1))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2)**2)) + np.cos(mono_omegas_all[mono]-np.pi/2)**2*np.sqrt((mono_v_vcircs[mono]**2*(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2-1))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2+mono_v_vcircs[mono]**2)**2)-np.cos(mono_omegas_all[mono]-np.pi/2))/(np.cos(mono_omegas_all[mono]-np.pi/2)**2 + mono_v_vcircs[mono]**2))
                                              )
                #Step 4 - compute prior prob for each of the omega+eccentricity combinations
                if self.ecc_prior.lower() =='kipping' or (len(self.planets)==1 and self.ecc_prior.lower()=='auto'):
                    alpha=0.867;beta=3.03
                    min_eccs = tt.clip(abs(2/(1 + mono_v_vcircs[mono]**2) - 1), 1e-4, 1.0) #Minimum eccentricity given velocity
                    mono_ecc_priors[mono] = pm.Deterministic("mono_ecc_priors_"+mono,
                                                           ((mono_eccs_all[mono]-min_eccs)**(alpha-1)*(max_eccs-mono_eccs_all[mono])**(max_eccs-1.0))/((max_eccs-min_eccs)**(alpha+beta-1)))
                    #The probabilities of eccentricities available / total probabilities.
                elif self.ecc_prior.lower() =='vaneylen' or (len(self.planets)>1 and self.ecc_prior.lower()=='auto'):
                    sigma=0.049
                    mono_ecc_priors[mono] = pm.Deterministic("mono_ecc_priors_"+mono,
                                                      mono_eccs_all[mono]*np.exp((-1*mono_eccs_all[mono]**2)/(2*sigma**2))/sigma**2)
                else:
                    mono_ecc_priors[mono] = pm.Deterministic("mono_ecc_priors_"+mono,0)

                #Step 5 - Compute prior for omega (e.g. omega range, which depends on vcirc>1.0, divided by 2pi)
               
                omega_prior_all[mono]=pm.Deterministic("omega_prior_to_per_marg_"+mono,
                                               tt.switch(tt.lt(mono_v_vcircs[mono],1.0),
                                     (np.pi-tt.arccos(-1*tt.sqrt(1 - mono_v_vcircs[mono]**2)))/(2*np.pi),
                                     (2*tt.arccos((tt.sqrt(1 - max_eccs**2)*mono_v_vcircs[mono] - 1)/max_eccs))/(2*np.pi)))
                
                #Step 6 - compute marginal prob for all omega/ecc samples
                '''mono_ecc_logprobmarg[mono] = (mono_ecc_priors[mono]+omega_prior_all[mono].dimshuffle('x',0)) - pm.math.logsumexp(mono_ecc_priors[mono]+omega_prior_all[mono].dimshuffle('x',0))
                '''
                #Step 6 - using marginal prob, take weighted median sample
                ecc_prob_argmax = tt.argmax(mono_ecc_logprobmarg, axis=axis, keepdims=True)
                mono_omegas[mono]=mono_omegas_all[mono][ecc_prob_argmax]
                mono_eccs[mono]=mono_eccs_all[mono][ecc_prob_argmax]
                #
                '''
                #Step 8 - Compute prior for whole eccentricity region (e.g. sum of beta distribution PDF within eccentricity range)
                if self.ecc_prior.lower() =='kipping' or (len(self.planets)==1 and self.ecc_prior.lower()=='auto'):
                    ecc_prior_to_per_marg[mono]=pm.Deterministic("ecc_prior_to_per_marg_"+mono,
                                            (1.1534*max_eccs**0.8672*(1 - 0.942694*max_eccs + 0.316151*max_eccs**2 - \
                                                                      0.00234395*max_eccs**3 - 0.00045162*max_eccs**4 - \
                                                                      0.000147609*max_eccs**5) - \
                                             1.1534*min_eccs**0.8672*(1 - 0.942694*min_eccs + 0.316151*min_eccs**2 - \
                                                                      0.00234395*min_eccs**3 - 0.00045162*min_eccs**4 - \
                                                                      0.000147609*min_eccs**5)) / \
                                                        0.4276486107761747)
                elif self.ecc_prior.lower() =='vaneylen' or (len(self.planets)>1 and self.ecc_prior.lower()=='auto'):
                    ecc_prior_to_per_marg[mono]=pm.Deterministic("ecc_prior_to_per_marg_"+mono,
                                                               (1 - tt.exp((-1*max_eccs**2)/(2*sigma**2))) - \
                                                               (1 - tt.exp((-1*min_eccs**2)/(2*sigma**2)))
                                                               )

                else:
                    ecc_prior_to_per_marg[mono]=pm.Deterministic("ecc_prior_to_per_marg_"+mono,
                                                                tt.tile(0.0,self.planets[mono]['n_gaps']))

                '''
                #Single planet with one transits and multiple period gaps
                mono_gap_info[mono]={'lc':{}}
                # Set up a Keplerian orbit for the planets
                #print(r[mono_ind].ndim,tt.tile(r[mono_ind],len(self.planets[mono]['per_gaps'][:,0])).ndim)
                # We are only creating a _single_ transit lightcurve and orbit. 
                monoorbit = xo.orbits.KeplerianOrbit(r_star=Rs,rho_star=rho_S, 
                                                    period=mono_pers[duo][tt.argmin(mono_eccs[mono])],
                                                    t0=mono_t0s[mono],b=mono_bs[mono],ecc=tt.min(mono_eccs[mono]),
                                                    omega=mono_omegas[mono][tt.argmin(mono_eccs[mono])])

                if hasattr(self,'multis') and len(self.multis)>0:
                    mono_gap_info[mono]['logpriors'] = -1 * tt.log(mono_dist_in_trans[mono]) + \
                                                       -2 * tt.log(mono_pers[mono]) + \
                                                       tt.log(self.planets[mono]['per_gaps'][:,3]) + \
                                     (500 / (1 + tt.exp(-20*(mono_a_Rs[mono]*(1 - mono_eccs[mono]) - 2)))-500) + \
                                     (500 / (1 + tt.exp(-20*((mono_a_Rs[mono]/tt.max(multi_a_Rs))*(1 - mono_eccs[mono]) - 1)))-500) +\
                                     mono_b_priors[mono]
                else:
                    mono_per_info[mono]['logpriors'] = -1 * tt.log(mono_dist_in_trans[mono]) + \
                                                       -2 * tt.log(mono_pers[mono]) + \
                                                       tt.log(self.planets[mono]['per_gaps'][:,3]) + \
                                      (500 / (1 + tt.exp(-20*(mono_a_Rs[mono]*(1 - mono_eccs[mono]) - 2)))-500) + \
                                      mono_b_priors[mono]
                mono_gap_info[mono]['lcs'] = gen_lc(monoorbit, tt.exp(mono_logrors[mono]),1, mask=None,
                                                    prefix='mono_mask_'+mono+'_',make_deterministic=True)
                
                pm.Deterministic('mono_priors_'+mono, mono_gap_info[mono]['logpriors'])
            #Priors - we have an occurrence rate prior (~1/P), a geometric prior (1/distance in-transit = dcosidb)
            # a window function log(1/P) -> -1*logP and  a factor for the width of the period bin - i.e. log(binsize)
            #mono_gap_info[mono]['logpriors'] = 0.0
            #This is also complicated by the fact that each gap has its own internal gradient
            # but between regions these are not normalised, so we include a factor w.r.t. the median period in the bin
            #I have no idea if we also need to incorporate the *width* of the bin here - I need to test this.
                    
            ################################################
            #        Compute marginalised LCs:
            ################################################
            
            # Marginalising together - say we have 3 models to marginalise and [4,6 and 8] regions in each:
            # This means we need to create a sum of each combination (e.g. a loglike)
            if len(self.duos+self.monos)>0:
                iter_models = {}
                n_mod=0
                for duo in self.duos:
                    iter_models[n_mod]={'name':duo,
                                        'n_points':np.sum((abs(lc['time']-self.planets[duo]['tcen'])<0.5*self.planets[duo]['tdur'])|(abs(lc['time']-self.planets[duo]['tcen_2'])<0.5*self.planets[duo]['tdur'])),
                                       'tdurs':duo_tdurs[duo],
                                       'eccs':duo_eccs_all[duo],
                                       'omegas':duo_omegas_all[duo],
                                       'v_vcircs':duo_v_vcircs[duo],
                                       'bs':duo_bs[duo],
                                       'logrors':duo_logrors[duo],
                                       'pers':duo_pers[duo],
                                       'len':len(self.planets[duo]['period_int_aliases']),
                                       'range':np.arange(len(self.planets[duo]['period_int_aliases'])),
                                       'lcs':duo_per_info[duo]['lcs'],
                                       'logpriors':duo_per_info[duo]['logpriors'],
                                       'type':'duo'}
                    if 'ecc' in self.marginal_params:
                        iter_models[n_mod]['eccs']=duo_eccs[duo]
                        iter_models[n_mod]['omegas']=duo_omegas[duo]

                    n_mod+=1
                for mono in self.monos:
                    iter_models[n_mod]={'name':mono,
                                        'n_points':np.sum(abs(lc['time']-self.planets[mono]['tcen'])<0.5*self.planets[mono]['tdur']),
                                        'tdurs':mono_tdurs[mono],
                                        'eccs':mono_eccs_all[duo],
                                        'omegas':mono_omegas_all[duo],
                                        'bs':mono_bs[mono],
                                        'v_vcircs':mono_v_vcircs[duo],
                                        'logrors':mono_logrors[mono],
                                        'pers':mono_pers[mono],
                                        'len':len(self.planets[mono]['per_gaps']),
                                        'range':np.arange(len(self.planets[mono]['per_gaps'])),
                                        'lcs':mono_gap_info[mono]['lcs'],
                                        'logpriors':mono_gap_info[mono]['logpriors'],
                                        'type':'mono'}
                    if 'ecc' in self.marginal_params:
                        iter_models[n_mod]['eccs']=mono_eccs[mono]
                        iter_models[n_mod]['omegas']=mono_omegas[mono]
                    n_mod+=1

                #For each combination we will create a combined model and compute the loglik
                new_yerr = lc['flux_err'][lc['mask']].astype(floattype)**2 + \
                               tt.sum(lc['flux_err_index'][lc['mask']]*tt.exp(logs2).dimshuffle('x',0),axis=1)

                if not self.use_GP:
                    #Calculating some extra info to speed up the loglik calculation
                    new_yerr_sq = new_yerr**2
                    sum_log_new_yerr = tt.sum(-np.sum(lc['mask'])/2 * tt.log(2*np.pi*(new_yerr_sq)))

                #Marginalising over all models individually:
                resids         = {}
                log_prob_margs = {}
                for pl in iter_models:
                    resids[pl]={}
                    iter_models[pl]['logliks']={}
                    if iter_models[pl]['len']>1:
                        iter_models[pl]['logprob_marg_sum'] = pm.Deterministic('logprob_marg_sum_'+str(iter_models[pl]['name']),
                                                                               pm.math.logsumexp(iter_models[pl]['logpriors']))
                        log_prob_margs[pl] = pm.Deterministic('logprob_marg_'+str(iter_models[pl]['name']), 
                                                              iter_models[pl]['logpriors'] - iter_models[pl]['logprob_marg_sum'])
                        #iter_models[pl]['marg_lc'] = pm.Deterministic('marg_light_curve_'+str(iter_models[pl]['name']),
                        #        tt.sum(iter_models[pl]['lcs'] * tt.exp(log_prob_margs[pl]).dimshuffle('x',0),axis=1))
                    for par in self.marginal_params:
                        if par in ['ecc','omega']:
                            #Weighted sum of ecc and omega 2D arrays using marginalised probabilities:
                            pm.Deterministic(par+'_marg_'+str(iter_models[pl]['name']),
                                             tt.sum(iter_models[pl][par+'s']*tt.exp(log_prob_margs[pl])))
                        else:
                            #Doing 1D weighted average:
                            pm.Deterministic(par+'_marg_'+str(iter_models[pl]['name']),
                                             tt.sum(iter_models[pl][par+'s']*tt.sum(tt.exp(log_prob_margs[pl]),axis=0)))

                #Now summing over all lcs:
                marg_all_light_curves = tt.stack([iter_models[pl]['lcs'] for pl in iter_models], axis=1)
                marg_all_light_curve = pm.Deterministic("marg_all_light_curve",
                                                        tt.sum(marg_all_light_curves,axis=1) + multi_mask_light_curve)
                    
                if self.use_GP:
                    total_llk = pm.Deterministic("total_llk",self.gp['use'].log_likelihood(lc['flux'][lc['mask']] - \
                                                                                    (marg_all_light_curve[lc['mask']] + mean)))
                    gp_pred = pm.Deterministic("gp_pred", self.gp['use'].predict(lc['time'].astype(floattype),
                                                                                 return_var=False))
                    pm.Potential("llk_gp", total_llk)
                    #pm.Normal("all_obs",mu=(marg_all_light_curve + gp_pred + mean),sd=new_yerr,
                    #          observed=self.lc['flux'][self.lc['near_trans']].astype(floattype))
                else:
                    pm.Normal("all_obs",mu=(marg_all_light_curve[lc['mask']] + mean),sd=new_yerr,
                              observed=lc['flux'][lc['mask']].astype(floattype))

            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            if self.debug: print("optimizing model",model.test_point)
            map_soln = xo.optimize(start=start)
            ################################################
            #   Creating initial model optimisation menu:
            ################################################

            #Setting up optimization depending on what planet models we have:
            initvars0=[]#r,b
            initvars1=[]#P
            initvars2=[rho_S]#r,b,t0
            initvars3=[logs2]
            initvars4=[]#r,b,P
            if len(self.multis)>0:
                #if 'tdur' in self.marginal_params:
                initvars0+=[multi_bs]
                initvars2+=[multi_bs]
                initvars4+=[multi_bs]
                #elif 'b' in self.marginal_params:
                #    initvars0+=[multi_tdurs]
                #    initvars2+=[multi_tdurs]
                #    initvars4+=[multi_tdurs]
                initvars0+=[multi_logrors]
                initvars1+=[multi_pers]
                initvars2+=[multi_logrors,multi_t0s]
                initvars4+=[multi_logrors,multi_bs,multi_pers]
                if not self.assume_circ:
                    initvars2+=[multi_eccs, multi_omegas]

            if len(self.monos)>0:
                for pl in self.monos:
                    if 'b' in self.fit_params:
                        initvars0+=[mono_bs[pl]]
                        initvars2+=[mono_bs[pl]]
                        initvars4+=[mono_bs[pl]]
                    if 'tdur' in self.fit_params:
                        initvars0+=[mono_tdurs[pl]]
                        initvars2+=[mono_tdurs[pl]]
                        initvars4+=[mono_tdurs[pl]]
                    if 'ingress' in self.fit_params:
                        initvars0+=[mono_ingress_repar[pl]]
                        initvars2+=[mono_ingress_repar[pl]]
                        initvars4+=[mono_ingress_repar[pl]]
                    initvars1 += [mono_logrors[pl]]
                    initvars2 += [mono_logrors[pl],mono_t0s[pl]]
                    initvars4 += [mono_logrors[pl]]
                    for n in range(len(self.planets[pl]['per_gaps'][:,0])):
                        initvars0 += [mono_pers[pl][n]]
                        initvars1 += [mono_pers[pl][n]]
                        initvars4 += [mono_pers[pl][n]]

                        #exec("initvars1 += [mono_pers_"+pl+"_"+str(int(n))+"]")
                        #exec("initvars4 += [mono_pers_"+pl+"_"+str(int(n))+"]")
            if len(self.duos)>0:
                #for pl in self.duos:
                #    eval("initvars1+=[duo_period_"+pl+"]")
                for pl in self.duos:
                    if 'b' in self.fit_params:
                        initvars0+=[duo_bs[pl]]
                        initvars2+=[duo_bs[pl]]
                        initvars4+=[duo_bs[pl]]
                    if 'tdur' in self.fit_params:
                        initvars0+=[duo_tdurs[pl]]
                        initvars2+=[duo_tdurs[pl]]
                        initvars4+=[duo_tdurs[pl]]
                    if 'ingress' in self.fit_params:
                        initvars0+=[duo_ingress_repar[pl]]
                        initvars2+=[duo_ingress_repar[pl]]
                        initvars4+=[duo_ingress_repar[pl]]

                    initvars0 += [duo_logrors[pl]]
                    initvars2 += [duo_logrors[pl],duo_t0s[pl],duo_t0_2s[pl]]
                    initvars4 += [duo_logrors[pl],duo_pers[pl]]
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

            if self.debug: print("after",model.check_test_point())

            self.model = model
            self.init_soln = map_soln
    
    def RunMcmc(self, n_draws=500, plot=True, n_burn_in=None, overwrite=False, continuesampling=False, chains=4, **kwargs):
        if not overwrite:
            self.LoadPickle()
            if hasattr(self,'trace') and self.debug:
                print("LOADED MCMC")
        
        if not (hasattr(self,'trace') or hasattr(self,'trace_df')) or overwrite or continuesampling:
            if not hasattr(self,'init_soln'):
                self.init_model()
            #Running sampler:
            np.random.seed(int(self.ID))
            with self.model:
                n_burn_in=np.clip(int(n_draws*0.66),125,15000) if n_burn_in is None else n_burn_in
                if self.debug: print(type(self.init_soln))
                if self.debug: print(self.init_soln.keys())
                if hasattr(self,'trace') and continuesampling:
                    print("Using already-generated MCMC trace as start point for new trace")
                    self.trace = pm.sample(tune=n_burn_in, draws=n_draws, chains=chains, trace=self.trace,
                                           step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)
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
    def run_multinest(self, ld_mult=2.5, verbose=False,max_iter=1500,**kwargs):
        import pymultinest
        
        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        
        if os.path.exists(self.savenames[0]+'_mnest_out'):
            if not self.overwrite:
                os.system('rm '+self.savenames[0]+'_mnest_out'+'/*')
        else:
            os.mkdir(self.savenames[0]+'_mnest_out')
            out_mnest_folder = self.savenames[0]+'_mnest_out'


        #Setting up necessary sampling functions:
        from scipy.stats import norm,beta,truncnorm,cosine

        def transform_uniform(x,a,b):
            return a + (b-a)*x

        def transform_loguniform(x,a,b):
            la=np.log(a)
            lb=np.log(b)
            return np.exp(la + x*(lb-la))

        def transform_normal(x,mu,sigma):
            return norm.ppf(x,loc=mu,scale=sigma)

        def transform_beta(x,a,b):
            return beta.ppf(x,a,b)

        def transform_truncated_normal(x,mu,sigma,a=0.,b=1.):
            ar, br = (a - mu) / sigma, (b - mu) / sigma
            return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)

        def transform_omega(e,omega):
            #2-pi region
            if np.random.random()<(1-e):
                return omega*np.pi*2
            else:
                #Sinusoidal region using a shifted cosine:
                return (cosine.ppf(omega)+np.pi*0.5)%(np.pi*2)
        
        log_flux_std={c:np.log(np.std(self.lc['flux'][self.lc['near_trans']&(self.lc['cadence']==c)])) for c in self.cads}

        if self.use_GP:
            import celerite
            from celerite import terms
            kernel = terms.SHOTerm(log_S0=np.log(np.nanstd(self.lc['flux'])) - 4*np.log(np.pi/7), log_Q=np.log(1/np.sqrt(2)), log_omega0=np.log(np.pi/7))
            kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term
            self.mnest_gps={}
            for ncad,cad in enumerate(self.cads):
                cadmask=self.lc['cadence']==cad
                self.mnest_gps[cad]=celerite.GP(kernel + terms(JitterTerm,log_sigma=log_flux_std[cad]),mean=0.0, fit_mean=False)
                self.mnest_gps[cad].compute(self.lc['time'][cadmask&self.lc['near_trans']], self.lc['flux_err'][cadmask&self.lc['near_trans']])
            #Initial compue here ^
            
        per_index=-8/3
        
        tess_ld_dists=self.getLDs(n_samples=3000,mission='tess');
        tess_lds=[np.clip(np.nanmedian(tess_ld_dists,axis=0),0,1),np.clip(ld_mult*np.nanstd(tess_ld_dists,axis=0),0.05,1.0)]
        kep_ld_dists=self.getLDs(n_samples=3000,mission='tess')
        kep_lds=[np.clip(np.nanmedian(kep_ld_dists,axis=0),0,1),np.clip(ld_mult*np.nanstd(kep_ld_dists,axis=0),0.05,1.0)]

        # Creating the two necessary functions:
        nparams=0
        self.cube_indeces={}
        self.cube_indeces['logrho_S']=nparams;nparams+=1
        self.cube_indeces['Rs']=nparams;nparams+=1
        self.cube_indeces['mean']=nparams;nparams+=1
        if self.useL2:
            self.cube_indeces['deltamag_contam']=nparams;nparams+=1
        for pl in self.multis+self.monos+self.duos:
            self.cube_indeces['t0_'+pl]=nparams;nparams+=1
            if pl in self.monos:
                self.cube_indeces['per_'+pl]=nparams;nparams+=1
            elif pl in self.duos:
                self.cube_indeces['t0_2_'+pl]=nparams;nparams+=1
                self.cube_indeces['duo_per_int_'+pl]=nparams;nparams+=1
            elif pl in self.multis:
                self.cube_indeces['per_'+pl]=nparams;nparams+=1
            self.cube_indeces['r_pl_'+pl]=nparams;nparams+=1
            self.cube_indeces['b_'+pl]=nparams;nparams+=1
            if not self.assume_circ:
                self.cube_indeces['ecc_'+pl]=nparams;nparams+=1
                self.cube_indeces['omega_'+pl]=nparams;nparams+=1
        if self.constrain_LD:
            if 't' in np.unique([c[0].lower() for c in self.cads]):
                self.cube_indeces['u_star_tess_0']=nparams;nparams+=1
                self.cube_indeces['u_star_tess_1']=nparams;nparams+=1
            if 'k' in np.unique([c[0].lower() for c in self.cads]):
                self.cube_indeces['u_star_kep_0']=nparams;nparams+=1
                self.cube_indeces['u_star_kep_1']=nparams;nparams+=1
        else:
            if 't' in np.unique([c[0].lower() for c in self.cads]):
                self.cube_indeces['q_star_tess_0']=nparams;nparams+=1
                self.cube_indeces['q_star_tess_1']=nparams;nparams+=1
            if 'k' in np.unique([c[0].lower() for c in self.cads]):
                self.cube_indeces['q_star_kep_0']=nparams;nparams+=1
                self.cube_indeces['q_star_kep_1']=nparams;nparams+=1
        for ncad,cad in enumerate(self.cads):
            self.cube_indeces['logs2_'+cad]=nparams;nparams+=1
        if self.use_GP:
            self.cube_indeces['logw0']=nparams;nparams+=1
            self.cube_indeces['logpower']=nparams;nparams+=1
        if verbose: print(cube_indeces)
        
        print("Forming PyMultiNest model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)

        def prior(cube, ndim, total_nparams):
            # do prior transformation
            ######################################
            #   Intialising Stellar Params:
            ######################################
            #Using log rho because otherwise the distribution is not normal:
            if verbose: print('logrho_S',cube[self.cube_indeces['logrho_S']])
            cube[self.cube_indeces['logrho_S']] = transform_normal(cube[self.cube_indeces['logrho_S']],
                                                                   mu=np.log(self.rhostar[0]), 
                                        sigma=np.average(abs(self.rhostar[1:]/self.rhostar[0])))

            if verbose: print('->',cube[self.cube_indeces['logrho_S']],'Rs',cube[self.cube_indeces['Rs']])
            cube[self.cube_indeces['Rs']] = transform_normal(cube[self.cube_indeces['Rs']],mu=self.Rstar[0],
                                                        sigma=np.average(abs(self.Rstar[1:])))
            # The baseline flux

            if verbose: print('->',cube[self.cube_indeces['Rs']],'mean',cube[self.cube_indeces['mean']])
            cube[self.cube_indeces['mean']]=transform_normal(cube[self.cube_indeces['mean']],
                                                             mu=np.median(self.lc['flux'][self.lc['mask']]),
                                                  sigma=np.std(self.lc['flux'][self.lc['mask']]))
            if verbose: print('->',cube[self.cube_indeces['mean']])
            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if self.useL2:

                if verbose: print('deltamag_contam',cube[self.cube_indeces['deltamag_contam']])
                cube[self.cube_indeces['deltamag_contam']] = transform_uniform(cube[self.cube_indeces['deltamag_contam']], -20.0, 20.0)
                if verbose: print('->',cube[self.cube_indeces['deltamag_contam']])
                mult = (1+np.power(2.511,-1*cube[self.cube_indeces['deltamag_contam']])) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0


            ######################################
            #     Initialising Periods & tcens
            ######################################
            for pl in self.multis+self.monos+self.duos:
                tcen=self.planets[pl]['tcen']
                tdur=self.planets[pl]['tdur']

                if verbose: print('t0_'+pl,cube[self.cube_indeces['t0_'+pl]])
                cube[self.cube_indeces['t0_'+pl]]=transform_truncated_normal(cube[self.cube_indeces['t0_'+pl]],
                                                                        mu=tcen,sigma=tdur*0.1,
                                                                        a=tcen-tdur*0.5,b=tcen+tdur*0.5)
                if verbose: print('->',cube[self.cube_indeces['t0_'+pl]])
                if pl in self.monos:

                    if verbose: print('per_'+pl+'.  (mono)',cube[self.cube_indeces['per_'+pl]])
                    per_index = -8/3
                    #Need to loop through possible period gaps, according to their prior probability *density* (hence the final term diiding by gap width)
                    rel_gap_prob = (self.planets[pl]['per_gaps'][:,0]**(per_index)-self.planets[pl]['per_gaps'][:,1]**(per_index))/self.planets[pl]['per_gaps'][:,2]
                    rel_gap_prob /= np.sum(rel_gap_prob)
                    rel_gap_prob = np.hstack((0.0,np.cumsum(rel_gap_prob)))
                    #incube=cube[self.cube_indeces['per_'+pl]]
                    #outcube=incube[:]
                    #looping through each gpa, cutting the "cube" up according to their prior probability and making a p~P^-8/3 distribution for each gap:
                    for i in range(len(rel_gap_prob)-1):
                        ind_min=np.power(self.planets[pl]['per_gaps'][i,1]/self.planets[pl]['per_gaps'][i,0],per_index)
                        if (cube[self.cube_indeces['per_'+pl]]>rel_gap_prob[i])&(cube[self.cube_indeces['per_'+pl]]<=rel_gap_prob[i+1]):
                            cube[self.cube_indeces['per_'+pl]]=np.power(((1-ind_min)*(cube[self.cube_indeces['per_'+pl]]-rel_gap_prob[i])/(rel_gap_prob[i+1]-rel_gap_prob[i])+ind_min),1/per_index)*self.planets[pl]['per_gaps'][i,0]
                    if verbose: print('->',cube[self.cube_indeces['per_'+pl]])
                # The period distributions of monotransits are tricky as we often have gaps to contend with
                # We cannot sample the full period distribution while some regions have p=0.
                # Therefore, we need to find each possible period region and marginalise over each

                if pl in self.duos:
                    #In the case of a duotransit, we have a discrete series of possible periods between two know transits.
                    #If we want to model this duo transit exactly like the first (as a bounded normal)
                    # we can work out the necessary periods to cause these dips

                    if verbose: print('t0_2_'+pl+'.  (duo)',cube[self.cube_indeces['t0_2_'+pl]])

                    tcen2=self.planets[pl]['tcen_2']
                    cube[self.cube_indeces['t0_2_'+pl]]=transform_truncated_normal(cube[self.cube_indeces['t0_2_'+pl]],
                                                                            mu=tcen2,sigma=tdur*0.1,
                                                                            a=tcen2-tdur*0.5,b=tcen2+tdur*0.5)
                    if verbose: print('->',cube[self.cube_indeces['t0_2_'+pl]],'duo_per_int_'+pl+'  (duo)',
                          cube[self.cube_indeces['duo_per_int_'+pl]])
                    rel_per_prob = self.planets[pl]['period_aliases']**(per_index)
                    rel_per_prob /= np.sum(rel_per_prob)
                    rel_per_prob = np.cumsum(rel_per_prob)
                    rel_per_prob[-1]=1.0000000001
                    #incube=cube[self.cube_indeces['duo_per_int_'+pl]]
                    #outcube=incube[:]
                    #looping through each gp,p cutting the "cube" up according to their prior probability and making a p~P^-8/3 distribution for each gap:
                    if verbose: print(rel_per_prob,cube[self.cube_indeces['duo_per_int_'+pl]],
                                      cube[:self.cube_indeces['duo_per_int_'+pl]])
                    cube[self.cube_indeces['duo_per_int_'+pl]]=self.planets[pl]['period_int_aliases'][np.searchsorted(rel_per_prob,cube[self.cube_indeces['duo_per_int_'+pl]])]
                    if verbose: print('->',cube[self.cube_indeces['duo_per_int_'+pl]])

                if pl in self.multis:

                    p=self.planets[pl]['period']
                    perr=self.planets[pl]['period_err']
                    cube[self.cube_indeces['per_'+pl]]=transform_normal(cube[self.cube_indeces['per_'+pl]],
                                                                            mu=p,
                                                                            sigma=np.clip(perr*0.25,0.005,0.02*p))
                    #In the case of multitransiting plaets, we know the periods already, so we set a tight normal distribution

                ######################################
                #     Initialising R_p & b
                ######################################
                # The Espinoza (2018) parameterization for the joint radius ratio and
                # impact parameter distribution
                rpl=self.planets[pl]['r_pl']/(109.1*self.Rstar[0])
                maxr=1.0 if self.useL2 else 0.2

                if verbose: print('r_pl_'+pl,cube[self.cube_indeces['r_pl_'+pl]])
                cube[self.cube_indeces['r_pl_'+pl]] = transform_uniform(cube[self.cube_indeces['r_pl_'+pl]],0.0,maxr)

                if verbose: print('->',cube[self.cube_indeces['r_pl_'+pl]],'b_'+pl,cube[self.cube_indeces['b_'+pl]])
                cube[self.cube_indeces['b_'+pl]] = transform_uniform(cube[self.cube_indeces['b_'+pl]],0.0,1.0)
                #We can do the adjustment for b later when sampling in the model

                if not self.assume_circ:
                    if verbose: print('->',cube[self.cube_indeces['b_'+pl]],'ecc_'+pl,cube[self.cube_indeces['ecc_'+pl]])

                    cube[self.cube_indeces['ecc_'+pl]] = transform_beta(cube[self.cube_indeces['ecc_'+pl]],a=0.867, b=3.03)

                    #Here we have a joint disribution of omega and eccentricity
                    # This isn't perfect but it better includes the fact that high-ecc planets are more likely to transit close to periasteron
                    if verbose: print('->',cube[self.cube_indeces['ecc_'+pl]],'omega_'+pl,cube[self.cube_indeces['omega_'+pl]])
                    cube[self.cube_indeces['omega_'+pl]] = transform_omega(cube[self.cube_indeces['ecc_'+pl]],
                                                                           cube[self.cube_indeces['omega_'+pl]])
                    if verbose: print('->',cube[self.cube_indeces['omega_'+pl]])


            ######################################
            #     Initialising Limb Darkening
            ######################################
            # Here we either constrain the LD params given the stellar info, OR we let exoplanet fit them
            if self.constrain_LD:
                # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.

                #Single mission
                if 't' in np.unique([c[0].lower() for c in self.cads]):
                    if verbose: print('u_star_tess_0',cube[self.cube_indeces['u_star_tess_0']])
                    cube[self.cube_indeces['u_star_tess_0']] = transform_truncated_normal(cube[self.cube_indeces['u_star_tess_0']],
                                                                                    mu=tess_lds[0][0],sigma=tess_lds[1][0])
                    if verbose: print('->',cube[self.cube_indeces['u_star_tess_0']],'u_star_tess_1',cube[self.cube_indeces['u_star_tess_1']])
                    cube[self.cube_indeces['u_star_tess_1']] = transform_truncated_normal(cube[self.cube_indeces['u_star_tess_1']],
                                                                                     mu=tess_lds[0][1],sigma=tess_lds[1][1])
                    if verbose: print('->',cube[self.cube_indeces['u_star_tess_1']])
                if 'k' in np.unique([c[0].lower() for c in self.cads]):
                    cube[self.cube_indeces['u_star_kep_0']] = transform_truncated_normal(cube[self.cube_indeces['u_star_kep_0']],
                                                                                    mu=kep_lds[0][0],sigma=kep_lds[1][0])
                    cube[self.cube_indeces['u_star_kep_1']] = transform_truncated_normal(cube[self.cube_indeces['u_star_kep_1']],
                                                                                     mu=kep_lds[0][1],sigma=kep_lds[1][1])
            # Otherwise the indexes are already uniform from 0->1 so lets leave them

            ######################################
            #     Initialising GP kernel
            ######################################
            for ncad,cad in enumerate(self.cads):
                if verbose: print('logs2_'+cad,cube[self.cube_indeces['logs2_'+cad]])
                cube[self.cube_indeces['logs2_'+cad]]=transform_normal(cube[self.cube_indeces['logs2_'+cad]],
                                                            mu=log_flux_std[ncad],sigma=2.0)
                if verbose: print('->',cube[self.cube_indeces['logs2_'+cad]])

            if self.use_GP:
                # Transit jitter & GP parameters
                #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
                lcrange=self.lc['time'][self.lc['near_trans']][-1]-self.lc['time'][self.lc['near_trans']][0]
                min_cad = np.min([np.nanmedian(np.diff(self.lc['time'][self.lc['near_trans']&(self.lc['cadence']==c)])) for c in self.cads])
                #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
                if verbose: print('logw0',cube[self.cube_indeces['logw0']])
                cube[self.cube_indeces['logw0']] = transform_uniform(cube[self.cube_indeces['logw0']],
                                                                np.log((2*np.pi)/(4*lcrange)),
                                                                np.log((2*np.pi)/min_cad))

                # S_0 directly because this removes some of the degeneracies between
                # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
                maxpower=np.log(np.nanmedian(abs(np.diff(self.lc['flux'][self.lc['near_trans']]))))+1
                if verbose: print('->',cube[self.cube_indeces['logw0']],'logpower',cube[self.cube_indeces['logpower']])
                cube[self.cube_indeces['logpower']] = transform_uniform(cube[self.cube_indeces['logpower']],-20,maxpower)
                if verbose: print('->',cube[self.cube_indeces['logpower']])
            if verbose: print(self.cube_indeces)

            # Creating the second necessary functions:

        def loglike(cube, ndim, nparams):
            pers={}
            for pl in self.duos:
                pers[pl]=(cube[self.cube_indeces['t0_2_'+pl]]-cube[self.cube_indeces['t0_'+pl]])/cube[self.cube_indeces['duo_per_int_'+pl]]
            for pl in self.monos+self.multis:
                pers[pl]=cube[self.cube_indeces['per_'+pl]]

            #Adjusting b here from 0->1 to 0->(1+rp/rs):
            newb=np.array([cube[self.cube_indeces['b_'+pl]] for pl in self.duos+self.monos+self.multis])*(1+np.array([cube[self.cube_indeces['r_pl_'+pl]] for pl in self.duos+self.monos+self.multis])/(109.1*cube[self.cube_indeces['Rs']]))

            if self.assume_circ:
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=cube[self.cube_indeces['Rs']],
                    rho_star=np.exp(cube[self.cube_indeces['logrho_S']]),
                    period=np.array([pers[pl] for pl in self.duos+self.monos+self.multis]),
                    t0=np.array([cube[self.cube_indeces['t0_'+pl]] for pl in self.duos+self.monos+self.multis]),
                    b=newb)
            else:
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=cube[self.cube_indeces['Rs']],
                    rho_star=np.exp(cube[self.cube_indeces['logrho_S']]),
                    period=np.array([pers[pl] for pl in self.duos+self.monos+self.multis]),
                    t0=np.array([cube[self.cube_indeces['t0_'+pl]] for pl in self.duos+self.monos+self.multis]),
                    b=newb,
                    ecc=np.array([cube[self.cube_indeces['ecc_'+pl]] for pl in self.duos+self.monos+self.multis]),
                    omega=np.array([cube[self.cube_indeces['omega_'+pl]] for pl in self.duos+self.monos+self.multis]))

            #TESS:
            i_r=np.array([cube[self.cube_indeces['r_pl_'+pl]] for pl in self.duos+self.monos+self.multis])/(109.1*cube[self.cube_indeces['Rs']])
            mult=(1+np.power(2.511,-1*cube[self.cube_indeces['deltamag_contam']])) if self.useL2 else 1.0

            trans_pred=[]
            cad_index=[]
            if self.use_GP:
                gp_pred=[]
            for cad in self.cads:
                #Looping over each region - Kepler or TESS - and generating lightcurve
                cadmask=self.lc['near_trans']&(self.lc['cadence']==cad)

                #print(self.lc['tele_index'][mask,0].astype(bool),len(self.lc['tele_index'][mask,0]),cadmask[mask],len(cadmask[mask]))

                if cad[0]=='t':

                    if 'u_star_tess_0' in self.cube_indeces:
                        u_tess=np.array([cube[self.cube_indeces['u_star_tess_0']],
                                         cube[self.cube_indeces['u_star_tess_1']]])
                    elif 'q_star_tess_0' in self.cube_indeces:
                        u_tess=np.array([2.*np.sqrt(cube[self.cube_indeces['u_star_tess_0']])*cube[self.cube_indeces['u_star_tess_1']],
                                     np.sqrt(cube[self.cube_indeces['u_star_tess_0']])*(1.-2.*cube[self.cube_indeces['u_star_tess_1']])])

                    #Taking the "telescope" index, and adding those points with the matching cadences to the cadmask
                    cad_index+=[(self.lc['tele_index'][self.lc['near_trans'],0].astype(bool))&cadmask[self.lc['near_trans']]]
                    trans_pred+=[xo.LimbDarkLightCurve(u_tess).get_light_curve(
                                                             orbit=orbit, r=i_r,
                                                             t=self.lc['time'][self.lc['near_trans']],
                                                             texp=np.nanmedian(np.diff(self.lc['time'][cadmask]))
                                                             ).eval()/(self.lc['flux_unit']*mult)]
                elif cad[0]=='k':
                    if 'u_star_kep_0' in self.cube_indeces:
                        u_kep=np.array([cube[self.cube_indeces['u_star_kep_0']],
                                                        cube[self.cube_indeces['u_star_kep_1']]])
                    elif 'q_star_kep_0' in self.cube_indeces:
                        u_kep=np.array([2.*np.sqrt(cube[self.cube_indeces['u_star_kep_0']])*cube[self.cube_indeces['u_star_kep_1']],
                                        np.sqrt(cube[self.cube_indeces['u_star_kep_0']])*(1.-2.*cube[self.cube_indeces['u_star_kep_1']])])

                    cad_index+=[(self.lc['tele_index'][self.lc['near_trans'],1].astype(bool))&cadmask[self.lc['near_trans']]]
                    trans_pred+=[xo.LimbDarkLightCurve(u_kep).get_light_curve(
                                                             orbit=orbit, r=i_r,
                                                             t=self.lc['time'][self.lc['near_trans']],
                                                             texp=np.nanmedian(np.diff(self.lc['time'][cadmask]))
                                                             ).eval()/(self.lc['flux_unit']*mult)]
                if self.use_GP:
                    #Setting GP params and predicting those times for this specific cadence:
                    self.mnest_gps[cad].set_parameter('kernel[0]:log_S0', cube[self.cube_indeces['logpower']] - 4 * cube[self.cube_indeces['logw0']])
                    self.mnest_gps[cad].set_parameter('kernel[0]:log_omega0', cube[self.cube_indeces['logw0']])
                    self.mnest_gps[cad].set_parameter('kernel[1]:log_sigma', cube[self.cube_indeces['logs2_'+cad]])
                    gp_pred+=[np.zeros(np.sum(self.lc['near_trans']))]
                    gp_pred[-1][cadmask]=self.mnest_gps[cad].predict(self.lc['flux'][cadmask] - np.sum(trans_pred[-1][cadmask,:],axis = 1) - cube[self.cube_indeces['mean']],return_cov=False, return_var=False)
            
            #Multiplying lightcurves by "telescope index" 
            model=np.sum(np.stack(trans_pred,axis=2)*np.column_stack(cad_index)[:,np.newaxis,:],axis=(1,2))
            new_yerr_sq = self.lc['flux_err'][self.lc['near_trans']]**2 + \
                          np.dot(self.lc['flux_err_index'][self.lc['near_trans']],
                                 np.exp(np.array([cube[self.cube_indeces['logs2_'+cad]] for cad in self.cads])))
            sum_log_new_yerr = np.sum(-np.sum(self.lc['near_trans'])/2 * np.log(2*np.pi*(new_yerr_sq)))

            if self.use_GP:
                gp_pred=np.sum(np.stack(gp_pred,axis=2)*np.column_stack(cad_index)[:,np.newaxis,:],axis=(1,2))
            else:
                gp_pred=0
            resids = self.lc['flux'][self.lc['near_trans']] - model - gp_pred - cube[self.cube_indeces['mean']]
            loglik = sum_log_new_yerr - np.sum(-0.5*(resids)**2/(2*new_yerr_sq),axis=0)
            print(loglik)
            return loglik

            pymultinest.run(loglike, prior, len(self.cube_indeces), max_iter=max_iter,
                        outputfiles_basename=self.savenames[0]+'_mnest_out/', 
                        **kwargs)
    
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
                            marg_lc[self.lc['near_trans']]=sample['marg_all_light_curve'][self.pseudo_binlc['near_trans']]
                        elif hasattr(self,'lc_near_trans') and len(self.trans_to_plot_i['all']['med'])==len(self.lc_near_trans['time']):
                            marg_lc[self.lc['near_trans']]=sample['marg_all_light_curve'][key1][key2]
                        elif len(self.trans_to_plot_i['all']['med'])==len(self.lc['time']):
                            marg_lc[self.lc['near_trans']]=sample['marg_all_light_curve'][key1][key2][self.lc['near_trans']]

                        
                        #marg_lc[self.lc['near_trans']]=sample['marg_all_light_curve'][self.lc['near_trans']]
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
        if hasattr(self,'trace') and 'marg_all_light_curve' in self.trace.varnames:
            prcnt=np.percentile(self.trace['marg_all_light_curve'],(5,1,50,84,95),axis=0)
            nms=['-2sig','-1sig','med','+1sig','+2sig']
            self.trans_to_plot_i['all']={nms[n]:prcnt[n] for n in range(5)}
        elif 'marg_all_light_curve' in self.init_soln:
            self.trans_to_plot_i['all']['med']=self.init_soln['marg_all_light_curve']
        else:
            print("marg_all_light_curve not in any optimised models")
        for pl in self.planets:
            self.trans_to_plot_i[pl]={}
            if pl in self.multis:
                if hasattr(self,'trace') and 'multi_mask_light_curves' in self.trace.varnames:
                    if len(self.trace['multi_mask_light_curves'].shape)>2:
                        prcnt = np.percentile(self.trace['multi_mask_light_curves'][:,:,self.multis.index(pl)],
                                                  (5,16,50,84,95),axis=0)
                    else:
                        prcnt = np.percentile(self.trace['multi_mask_light_curves'], (5,16,50,84,95), axis=0)

                    nms=['-2sig','-1sig','med','+1sig','+2sig']
                    self.trans_to_plot_i[pl] = {nms[n]:prcnt[n] for n in range(5)}
                elif 'multi_mask_light_curves' in self.init_soln:
                    if len(self.init_soln['multi_mask_light_curves'].shape)==1:
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves']
                    else:    
                        self.trans_to_plot_i[pl]['med'] = self.init_soln['multi_mask_light_curves'][:,self.multis.index(pl)]
                else:
                    print('multi_mask_light_curves not in any optimised models')
            elif pl in self.duos or self.monos:
                if hasattr(self,'trace') and 'marg_light_curve_'+pl in self.trace.varnames:
                    prcnt=np.percentile(self.trace['marg_light_curve_'+pl],(5,16,50,84,95),axis=0)
                    nms=['-2sig','-1sig','med','+1sig','+2sig']
                    self.trans_to_plot_i[pl] = {nms[n]:prcnt[n] for n in range(5)}
                elif 'marg_light_curve_'+pl in self.init_soln:
                    self.trans_to_plot_i[pl]['med'] = self.init_soln['marg_light_curve_'+pl]
                else:
                    print('marg_light_curve_'+pl+' not in any optimised models')
        self.trans_to_plot={'n_samp':n_samp}

        #Adding zeros to other regions where we dont have transits (not in the out of transit mask):
        for key1 in self.trans_to_plot_i:
            self.trans_to_plot[key1]={}
            for key2 in self.trans_to_plot_i[key1]:
                self.trans_to_plot[key1][key2]=np.zeros(len(self.lc['time']))
                if hasattr(self,'pseudo_binlc') and len(self.trans_to_plot_i[pl]['med'])==len(self.pseudo_binlc['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2][self.pseudo_binlc['near_trans']]
                elif hasattr(self,'lc_near_trans') and len(self.trans_to_plot_i[pl]['med'])==len(self.lc_near_trans['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2]
                elif len(self.trans_to_plot_i[pl]['med'])==len(self.lc['time']):
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.trans_to_plot_i[key1][key2][self.lc['near_trans']]

    def init_plot(self,gap_thresh=10):

        #Making sure lc is binned to 30mins
        self.lc=tools.lcBin(self.lc,binsize=1/48.0)
        
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
        masks=[]
    
    def Plot(self, interactive=False, n_samp=None, overwrite=False,return_fig=False, max_gp_len=20000,
             bin_gp=True, plot_loc=None, palette=None, pointcol="k", plottype='png'):
        ################################################################
        #       Varied plotting function for MonoTransit model
        ################################################################
        import seaborn as sns
        if palette is None:
            sns.set_palette('viridis', len(self.planets)+3)
            pal = sns.color_palette('viridis', len(self.planets)+3)
            pal = pal.as_hex()
        else:
            sns.set_palette(palette)
            pal = sns.color_palette(palette)
            pal = pal.as_hex()
        if pointcol=="k":
            sns.set_style('whitegrid')
        #Rasterizing matplotlib files if we have a lot of datapoints:
        raster=True if len(self.lc['time']>8000) else False
        
        if not hasattr(self,'trace'):
            n_samp==1
        
        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        
        if interactive:
            #Plots bokeh figure
            
            from bokeh.plotting import figure, output_file, save, curdoc, show
            from bokeh.models import Band, Whisker, ColumnDataSource, Range1d
            from bokeh.models.arrow_heads import TeeHead
            from bokeh.layouts import gridplot, row, column, layout
            
            if plot_loc is None:
                if not hasattr(self,'savenames'):
                    self.GetSavename(how='save')
                savename=self.savenames[0]+'_model_plot.html'
            else:
                savename=plot_loc
            
            print(savename)

            output_file(savename)

            #Initialising figure:
            p = figure(plot_width=1000, plot_height=600,title=str(self.ID)+" Transit Fit")
        else:
            from iteround import saferound

            #A4 page: 8.27 x 11.69
            fig=plt.figure(figsize=(11.69,8.27))
            gs = fig.add_gridspec(len(self.planets)*4,32,wspace=0.3,hspace=0.001)
        
        #####################################
        #       Initialising figures
        #####################################
        
        if not hasattr(self,'gap_lens') or overwrite:
            self.init_plot()

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
            self.init_trans_to_plot(n_samp)
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
        
        raw_plot_offset = 2.5*abs(self.min_trans) if not self.use_GP else 1.25*abs(self.min_trans) + resid_sd +\
                                                                     abs(np.min(self.gp_to_plot['gp_pred'][self.lc['mask']]))

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
            if pl in self.multis:
                if hasattr(self,'trace'):
                    t0=np.nanmedian(self.trace['multi_t0s'][:,self.multis.index(pl)])
                    per=np.nanmedian(self.trace['multi_pers'][:,self.multis.index(pl)])
                elif hasattr(self,'init_soln'):
                    t0=self.init_soln['multi_t0s'][self.multis.index(pl)]
                    per=self.init_soln['multi_pers'][self.multis.index(pl)]
                #phase-folding onto t0
                self.lc['phase'][pl]=phase=(self.lc['time']-t0-0.5*per)%per-0.5*per
                for ns in range(len(self.lc['limits'])):
                    n_p_sta=np.ceil((t0-self.lc['time'][self.lc['limits'][ns][0]])/per)
                    n_p_end=(t0-self.lc['time'][self.lc['limits'][ns][1]-1])/per
                    
                    if (t0>self.lc['limits'][ns][0])&(t0<self.lc['limits'][ns][0]):
                        #Adding ticks for the position of each planet below the data:
                        f_alls[ns].scatter(t0+np.arange(n_p_sta,n_p_end,1.0)*per,
                                           np.tile(-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets)),
                                                   int(np.ceil(n_p_end-n_p_sta))),
                                           marker="triangle", size=12.5, line_color=pal[2+n], fill_color=col, alpha=0.85)

            elif pl in self.monos:
                #Plotting mono
                if hasattr(self,'trace'):
                    t0=np.nanmedian(self.trace['mono_t0s_'+pl])
                    per=2e3#per=np.nanmedian(self.trace['mono_periods_'+pl][-1])
                elif hasattr(self,'init_soln'):
                    t0=self.init_soln['mono_t0s_'+pl]
                    per=2e3#per=self.init_soln['mono_periods_'+pl][-1]
                #phase-folding onto t0
                self.lc['phase'][pl]=phase=(self.lc['time']-t0-0.5*per)%per-0.5*per
                for ns in range(len(self.lc['limits'])):
                    if (t0>self.lc['limits'][ns][0])&(t0<self.lc['limits'][ns][0]):
                        if interactive:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                               marker="triangle", size=12.5, fill_color=pal[n+2], alpha=0.85)
                        else:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                               "^", markersize=12.5, color=pal[n+2], alpha=0.85, rasterized=raster)
            elif pl in self.duos:
                #Overplotting both transits
                if hasattr(self,'trace'):
                    t0=np.nanmedian(self.trace['duo_t0s_'+pl])
                    t0_2=np.nanmedian(self.trace['duo_t0_2s_'+pl])
                    per=abs(t0_2-t0)
                elif hasattr(self,'init_soln'):
                    t0=self.init_soln['duo_t0s_'+pl]
                    t0_2=self.init_soln['duo_t0_2s_'+pl]
                    per=abs(t0_2-t0)
                #phase-folding onto t0
                self.lc['phase'][pl]=(self.lc['time']-t0-0.5*per)%per-0.5*per
                for ns in range(len(self.lc['limits'])):
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
            if plot_loc is None and plottype=='png':
                plt.savefig(self.savenames[0]+'_model_plot.png',dpi=350,transparent=True)
                #plt.savefig(self.savenames[0]+'_model_plot.pdf')
            elif plot_loc is None and plottype=='pdf':
                plt.savefig(self.savenames[0]+'_model_plot.pdf')
            else:
                plt.savefig(plot_loc)

    def PlotPeriods(self, plot_loc=None,log=True,nbins=25,pmax=None,ymin=None,xlog=False):
        assert hasattr(self,'trace')
        import seaborn as sns
        from scipy.special import logsumexp
        pal=sns.color_palette('viridis_r',7)
        plot_pers=self.duos+self.monos
        if ymin is None:
            ymin=np.min(np.hstack([np.nanmedian(self.trace['logprob_marg_'+pl],axis=0) for pl in self.duos]))/np.log(10)-2.0
        
        if len(plot_pers)>0:
            plt.figure(figsize=(8.5,4.2))
            for npl, pl in enumerate(plot_pers):
                plt.subplot(1,len(plot_pers),npl+1)
                if pl in self.duos:
                    #As we're using the nanmedian log10(prob)s for each period, we need to make sure their sums add to 1.0
                    psum=logsumexp(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0))/np.log(10)
                    #Plotting lines
                    cols=[]
                    coldic={-6:"p<1e-5",-5:"p>1e-5",-4:"p>1e-4",-3:"p>0.1%",-2:"p>1%",-1:"p>10%",0:"p>100%"}
                    probs=np.nanmedian(self.trace['logprob_marg_'+pl],axis=0)/np.log(10)
                    for n in np.arange(len(probs))[np.argsort(probs)][::-1]:
                        # Density Plot and Histogram of all arrival delays        
                        nprob=probs[n]
                        ncol=int(np.floor(np.clip(nprob-psum,-6,0)))
                        
                        if ncol not in cols:
                            cols+=[ncol]
                            plt.plot(np.tile(np.nanmedian(self.trace['duo_pers_'+pl][:,n]),2),
                                     [ymin-psum,nprob-psum],
                                     linewidth=5.0,color=pal[6+ncol],alpha=0.6,label=coldic[ncol])
                        else:
                            plt.plot(np.tile(np.nanmedian(self.trace['duo_pers_'+pl][:,n]),2),
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
                    pmax = np.nanmax(self.trace['mono_pers_'+pl].ravel()) if pmax is None else pmax
                    for ngap in np.arange(len(self.planets[pl]['per_gaps'])):
                        bins=np.arange(np.floor(self.planets[pl]['per_gaps'][ngap,0]),
                                       np.clip(np.ceil(self.planets[pl]['per_gaps'][ngap,0])+1.0,0.0,pmax),
                                       1.0)
                        print(ngap,np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob))
                        plt.hist(self.trace['mono_pers_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                                 weights=np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob),
                                 color=sns.color_palette()[0],histtype="stepfilled")
                        #bins=np.linspace(self.planets[pl]['per_gaps'][ngap,0],self.planets[pl]['per_gaps'][ngap,1],nbins)
                        #weights = np.exp(self.trace['logprob_marg_'+pl][:,ngap]-total_prob) if 'logprob_marg_'+pl in self.trace.varnames else None
                        #plt.hist(self.trace['mono_pers_'+pl][:,ngap], bins=bins, edgecolor=sns.color_palette()[0],
                        #         weights=weights, color=sns.color_palette()[0], histtype="stepfilled")
                        '''
                        if log and float('.'.join(sns.__version__.split('.')[:2]))>=0.11:
                            sns.distplot(self.trace['mono_pers_'+pl][:,ngap], hist=True, kde=True, 
                                         bins=bins, norm_hist=False,
                                         hist_kws={'edgecolor':'None','log':log,
                                                   'weights':weights,'color':sns.color_palette()[0]},
                                         kde_kws={'linewidth':2,'weights':weights,'bw':self.planets[pl]['per_gaps'][ngap,2]*0.1,
                                                  'clip':self.planets[pl]['per_gaps'][ngap,:2],'color':sns.color_palette()[1]})
                            
                        else:
                            sns.distplot(self.trace['mono_pers_'+pl][:,ngap], hist=True, kde=False, 
                                         bins=bins, norm_hist=False,
                                         hist_kws={'edgecolor':'None','log':log,
                                                   'weights':weights,'color':sns.color_palette()[0]},
                                         kde_kws={'linewidth': 2,'clip':self.planets[pl]['per_gaps'][ngap,:2],
                                                  'bw':self.planets[pl]['per_gaps'][ngap,2]*0.1,'color':sns.color_palette()[1]})
                        '''
                    plt.title("Mono - "+str(pl))
                    if log:
                        plt.yscale('log')
                        plt.ylabel("$\log_{10}{\\rm prob}$")
                    else:
                        plt.ylabel("prob")
                    plt.xlim(0,pmax)
                    if xlog:
                        plt.xscale('log')
                        plt.xticks([20,40,60,80,100,150,200,250],np.array([20,40,60,80,100,150,200,250]).astype(str))
                        #plt.xticklabels([20,40,60,80,100,150,200,250])

                    #plt.xlim(60,80)
                    plt.ylim(1e-12,1.0)
                    plt.xlabel("Period [d]")
            if plot_loc is None:
                plt.savefig(self.savenames[0]+'_period_dists.pdf')
            else:
                plt.savefig(plot_loc)
    
    def PlotCorner(self,use_marg=True):
        # Plotting corner for those parameters we're interested in - e.g. orbital parameters
        # If "use_marg" is True - uses the marginalised tdur and period parameters for multis and duos
        # If "use_marg" is False - generates samples for each marginalised distribution and weights by logprob
        import corner
        
        corner_vars=['logrho_S']
        
        for pl in self.duos:
            corner_vars+=['duo_t0s_'+pl,'duo_t0_2s_'+pl]
            corner_vars+=[var+'_marg_'+pl for var in self.marginal_params]
            if 'tdur' not in self.marginal_params:
                corner_vars+=['duo_tdurs_'+pl]
            elif 'b' not in self.marginal_params:
                corner_vars+=['duo_bs_'+pl]
            if 'logror' not in self.marginal_params:
                corner_vars+=['duo_logrors_'+pl]
            if not self.assume_circ and 'ecc_marg_'+pl not in corner_vars:
                corner_vars+=['duo_eccs_'+pl,'duo_omegas_'+pl]
        for pl in self.monos:
            corner_vars+=['mono_t0s_'+pl]
            corner_vars+=[var+'_marg_'+pl for var in self.marginal_params]
            if 'tdur' not in self.marginal_params:
                corner_vars+=['mono_tdurs_'+pl]
            elif 'b' not in self.marginal_params:
                corner_vars+=['mono_bs_'+pl]
            if 'logror' not in self.marginal_params:
                corner_vars+=['mono_logrors_'+pl]
            if not self.assume_circ and 'ecc_marg_'+pl not in corner_vars:
                corner_vars+=['mono_eccs_'+pl,'mono_omegas_'+pl]
        if len(self.multis)>0:
            corner_vars+=['multi_t0s','multi_logrors','multi_bs','multi_pers']
            if not self.assume_circ:
                corner_vars+=['multi_eccs','multi_omegas']

        samples = pm.trace_to_dataframe(self.trace, varnames=corner_vars)
        print(samples.shape,samples.columns)
        assert samples.shape[1]<50
        
        if use_marg: 
            fig = corner.corner(samples)
        else:
            #Not using the marginalised period, and instead using weights:
            logprobs=[]
            
            all_weighted_periods={}
            all_logprobs={}
            
            n_mult=np.product([len(self.planets[mpl]['per_gaps']) for mpl in self.monos]) * \
                   np.product([len(self.planets[dpl]['period_aliases']) for dpl in self.duos])
            print(n_mult,"x samples")
            samples['log_prob']=np.tile(0.0,len(samples))
            samples_len=len(samples)
            samples=pd.concat([samples]*int(n_mult),axis=0)
            print(samples.shape,samples_len)
            
            n_pos=0
            
            for mpl in self.monos:
                for n_gap in np.arange(len(self.planets[mpl]['per_gaps'])):
                    sampl_loc=np.in1d(np.arange(0,len(samples),1),np.arange(n_pos*samples_len,(n_pos+1)*samples_len,1))
                    samples.loc[sampl_loc,'per_marg_'+mpl]=self.trace['mono_pers_'+mpl][:,n_gap]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+mpl]=self.trace['mono_tdurs_'+mpl][:,n_gap]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+mpl]=self.trace['mono_bs_'+mpl][:,n_gap]
                    samples.loc[sampl_loc,'log_prob']=self.trace['logprob_marg_'+mpl][:,n_gap]
                    n_pos+=1
            for dpl in self.duos:
                for n_per in np.arange(len(self.planets[dpl]['period_aliases'])):
                    sampl_loc=np.in1d(np.arange(len(samples)),np.arange(n_pos*samples_len,(n_pos+1)*samples_len))
                    samples.loc[sampl_loc,'per_marg_'+dpl]=self.trace['duo_pers_'+dpl][:,n_per]
                    if 'tdur' in self.marginal_params:
                        samples.loc[sampl_loc,'tdur_marg_'+dpl]=self.trace['duo_tdurs_'+dpl][:,n_per]
                    elif 'b' in self.marginal_params:
                        samples.loc[sampl_loc,'b_marg_'+dpl]=self.trace['duo_bs_'+dpl][:,n_per]
                    samples.loc[sampl_loc,'log_prob'] = self.trace['logprob_marg_'+dpl][:,n_per]
                    n_pos+=1
            weight_samps = np.exp(samples["log_prob"])
            fig = corner.corner(samples[[col for col in samples.columns if col!='log_prob']],weights=weight_samps);
        
        fig.savefig(self.savenames[0]+'_corner.pdf',dpi=400,rasterized=True)
        
        
    def MakeTable(self,short=True,save=True):
        assert hasattr(self,'trace')
        
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
        if save:
            print("Saving sampled model parameters to file with shape: ",df.shape)
            if short:
                df.to_csv(self.savenames[0]+'_mcmc_output_short.csv')
            else:
                df.to_csv(self.savenames[0]+'_mcmc_output.csv')
        return df

    def PlotTable(self,plot_loc=None,return_table=False):
        
        df = MakeTable(short=True)
        
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

    def getLDs(self,n_samples,mission='tess',how='2'):
        Teff_samples = np.random.normal(self.Teff[0],np.average(abs(self.Teff[1:])),n_samples)
        logg_samples = np.random.normal(self.logg[0],np.average(abs(self.logg[1:])),n_samples)
        
        from scipy.interpolate import CloughTocher2DInterpolator as ct2d

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
