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

MonoData_tablepath = os.path.join(os.path.dirname(os.path.dirname( __file__ )),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname(os.path.dirname( __file__ )),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')
if not os.path.isdir(MonoData_savepath):
    os.mkdir(MonoData_savepath)

from . import tools
from . import starpars
from . import MonoSearch
#from . import tools
#from .stellar import starpars
#from . import MonoSearch


#creating new hidden directory for theano compilations:
theano_dir=MonoData_savepath+'/.theano_dir_'+str(np.random.randint(8))
if not os.path.isdir(theano_dir):
    os.mkdir(theano_dir)

if MonoData_savepath=="/Users/hosborn/python/MonoToolsData":
    os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,cxx=/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9,cxxflags = -fbracket-depth=1024,base_compiledir="+theano_dir
else:
    os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,cxxflags = -fbracket-depth=1024,base_compiledir="+theano_dir

import theano.tensor as tt
import pymc3 as pm
import theano
theano.config.print_test_value = True
theano.config.exception_verbosity='high'
#print("theano config:",config)#['device'],config['floatX'],config['cxx'],config['compiledir'],config['base_compiledir'])


class monoModel():
    #The default monoModel class. This is what we will use to build a Pymc3 model
    
    def __init__(self, ID, mission, lc, planets=None, LoadFromFile=False, overwrite=False, savefileloc=None):
        #Initalising MonoModel
                    
        if 'mask' not in lc:
            lc['mask']=np.tile(True,len(x))
            
        if (np.sort(lc['time'])!=lc['time']).all():
            print("#SORTING")
            for key in [k for key in lc if type(k)==np.ndarray and k!='time']:
                if len(lc[key])==len(lc['time']):
                    lc[key]=lc[key][np.argsort(lc['time'])][:]
            lc['time']=np.sort(lc['time'])
        self.ID=ID
        self.lc=lc
        self.planets={}
        self.multis=[];self.monos=[];self.duos=[]

        if planets is not None:
            for pl in planets:
                if planets[pl]['orbit_flag']=='multi':
                    self.add_multi(planets[pl])
                elif planets[pl]['orbit_flag']=='duo':
                    self.add_duo(planets[pl])
                elif planets[pl]['orbit_flag']=='mono':
                    self.add_mono(planets[pl])
        self.mission=mission
        
        self.savefileloc=savefileloc
        self.overwrite=overwrite
        
        #self.
        self.id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
                     'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}

        if not LoadFromFile:
            self.GetSavename(how='save')
        else:
            self.GetSavename(how='load')
    
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

    def add_multi(self, pl_dic, name):
        assert name not in self.planets
        #Adds planet with multiple eclipses
        if not np.isfinite(pl_dic['period_err']):
            pl_dic['period_err'] = 0.5*pl_dic['tdur']/pl_dic['period']

        self.planets[name]=pl_dic
        self.multis+=[name]
        
    def add_mono(self, pl_dic, name):
        #Adds planet with single eclipses
        assert name not in self.planets
        p_gaps=self.compute_period_gaps(pl_dic['tcen'],tdur=pl_dic['tdur'],depth=pl_dic['depth'])
        pl_dic['per_gaps']=np.column_stack((p_gaps,p_gaps[:,1]-p_gaps[:,0]))
        pl_dic['P_min']=p_gaps[0,0]
        self.planets[name]=pl_dic
        self.monos+=[name]
        print(self.planets[name]['per_gaps'])
        
    def compute_rms_series(self,tdur,split_gap_size=2.0,n_steps_per_dur=7):
        # Computing an RMS time series for the lightcurve by binning
        # split_gap_size = Duration at which to cut the lightcurve and compute in loops
        # n_steps_per_dur = number of steps with which to cut up each duration. Odd numbers work most uniformly
        
        if not hasattr(self.lc,'flux_flat'):
            self.lc=tools.lcFlatten(self.lc,winsize=tdur*9,stepsize=tdur/n_steps_per_dur)
        
        rms_series=np.zeros((len(self.lc['time'])))
        binsize=(1/n_steps_per_dur)*tdur
        if np.nanmax(np.diff(self.lc['time']))>split_gap_size:
            loop_blocks=np.array_split(np.arange(len(self.lc['time'])),np.where(np.diff(self.lc['time'])>split_gap_size)[0])
        else:
            loop_blocks=[np.arange(len(self.lc['time']))]
        for sh_time in loop_blocks:
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
        return rms_series

    def compute_period_gaps(self,tcen,tdur,depth,max_per=8000,SNR_thresh=4):
        # Given the time array, the t0 of transit, and the fact that another transit is not observed, 
        #   we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
        # In this case, a list of periods is returned, with all points within 0.5dur to be cut
        dist_from_t0=np.sort(abs(tcen-self.lc['time'][self.lc['mask']]))
        
        rmsseries = depth/self.compute_rms_series(tdur)
        
        #Removing parts of the timeseries where we would not have detected another transit with SNR>thresh
        dist_from_t0=dist_from_t0[rmsseries[self.lc['mask']]<SNR_thresh]
            
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
        #This
        trans=abs(self.lc['time'][self.lc['mask']]-tcen)<0.3*tdur
        if np.sum(trans)==0:
            trans=abs(self.lc['time'][self.lc['mask']]-tcen)<0.5*tdur

        days_in_known_transits = np.sum(trans)*float(self.lc['cadence'][self.lc['mask']][trans][0][1:])/1440
        if tcen_2 is not None:
            trans2=abs(self.lc['time'][self.lc['mask']]-tcen_2)<0.3*tdur
            days_in_known_transits += np.sum(trans2)*float(self.lc['cadence'][self.lc['mask']][trans2][0][1:])/1440
            coverage_thresh*=0.5 #Two transits already in number count, so to compensate we must decrease the thresh
            
        check_pers_ix=[]
        for per in pers:
            phase=(self.lc['time'][self.lc['mask']]-tcen-per*0.5)%per-per*0.5
            intr=abs(phase)<0.3*tdur
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
        
        self.planets[name]=pl_dic
        self.duos+=[name]
    
    def init_starpars(self,Rstar=np.array([1.0,0.08,0.08]),rhostar=np.array([4.1,0.2,0.3]),
                      Teff=np.array([5800,100,100]),logg=np.array([4.3,1.0,1.0]),FeH=0.0):
        #Adds stellar parameters to model
        self.Rstar=np.array(Rstar)
        self.rhostar=np.array(rhostar)
        self.Teff=np.array(Teff)
        self.logg=np.array(logg)
        self.FeH=FeH
    
    def GetSavename(self, how='load', suffix='mcmc.pickle',overwrite=True):
        '''
        # Get unique savename (defaults to MCMC suffic) with format:
        # [savefileloc]/[T/K]IC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
        #
        # INPUTS:
        # - ID
        # - mission - (TESS/K2/Kepler)
        # - how : 'load' or 'save'
        # - suffix : final part of file string. default is _mcmc.pickle
        # - overwrite : if 'save', whether to overwrite past save or not.
        # - savefileloc : file location of files to save (default: 'MonoTools/[T/K]ID[11-number ID]/
        #
        # OUTPUTS:
        # - filepath
        '''
        if self.savefileloc is None or overwrite:
            self.savefileloc=os.path.join(MonoData_savepath,self.id_dic[self.mission]+str(self.ID).zfill(11))
        if not os.path.isdir(self.savefileloc):
            os.system('mkdir '+self.savefileloc)
        pickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"*"+suffix))
        if how == 'load' and len(pickles)>1:
            #finding most recent pickle:
            date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
            datepickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_*_"+suffix))
            if len(datepickles)>1:
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
            datepickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_*_"+suffix))
            if len(datepickles)==0:
                nsim=0
            elif self.overwrite:
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            else:
                #Finding next unused number with this date:
                nsim=1+np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])

        self.savenames=[os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_"+str(int(nsim))+"_"+suffix), os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+'_'+suffix)]

    def init_model(self,assume_circ=False,overwrite=False,
                   use_GP=True,train_GP=True,constrain_LD=True,ld_mult=3,useL2=True,
                   FeH=0.0,LoadFromFile=False,cutDistance=4.5,force_match_input=None,
                   debug=True, pred_all_time=False,marginalise_all=False,
                   use_multinest=False, use_pymc3=True, bin_oot=True, **kwargs):
        # lc - dictionary with arrays:
        #   -  'time' - array of times, (x)
        #   -  'flux' - array of flux measurements (y)
        #   -  'flux_err'  - flux measurement errors (yerr)
        # initdepth - initial depth guess
        # initt0 - initial time guess
        # Rstar - array with radius of star and error/s
        # rhostar - array with density of star and error/s
        # periods - In the case where a planet is already transiting, include the period guess as a an array with length n_pl
        # constrain_LD - Boolean. Whether to use 
        # ld_mult - Multiplication factor on STD of limb darkening]
        # cutDistance - cut out points further than this multiple of transit duration from transit. Default of zero does no cutting
        # force_match_input - add potential multiplied by this to force MCMC to match the input duration & maximise logror
        # bin_oot - in this case, we bin points outside the cutDistance to 30mins
        
        #Adding settings to class - not updating if we already initialised the model with a non-default value:
        self.overwrite=overwrite
        self.assume_circ=assume_circ if not hasattr(self,'assume_circ') or overwrite else assume_circ
        self.use_GP=use_GP if not hasattr(self,'use_GP') or overwrite else use_GP
        self.train_GP=train_GP if not hasattr(self,'train_GP') or overwrite else train_GP
        self.marginalise_all=marginalise_all if not hasattr(self,'marginalise_all') or overwrite else marginalise_all
        self.constrain_LD=constrain_LD if not hasattr(self,'constrain_LD') or overwrite else constrain_LD
        self.ld_mult=ld_mult if not hasattr(self,'ld_mult') or overwrite else ld_mult
        self.useL2=useL2 if not hasattr(self,'useL2') or overwrite else useL2
        self.FeH=FeH if not hasattr(self,'FeH') or overwrite else FeH
        self.LoadFromFile=LoadFromFile if not hasattr(self,'LoadFromFile') or overwrite else LoadFromFile
        self.cutDistance=cutDistance if not hasattr(self,'cutDistance') or overwrite else cutDistance
        self.force_match_input=force_match_input if not hasattr(self,'force_match_input') or overwrite else force_match_input
        self.debug=debug if not hasattr(self,'debug') or overwrite else debug
        self.pred_all_time=pred_all_time if not hasattr(self,'pred_all_time') or overwrite else pred_all_time
        self.use_multinest=use_multinest if not hasattr(self,'use_multinest') or overwrite else use_multinest
        self.use_pymc3=use_pymc3 if not hasattr(self,'use_pymc3') or overwrite else use_pymc3
        self.bin_oot=bin_oot if not hasattr(self,'bin_oot') or overwrite else bin_oot
        
        n_pl=len(self.planets)
        assert n_pl>0

        print(len(self.planets),'planets |','monos:',self.monos,'multis:',self.multis,'duos:',self.duos, "use GP=",self.use_GP)

        ######################################
        #   Masking out-of-transit flux:
        ######################################
        # To speed up computation, here we loop through each planet and add the region around each transit to the data to keep
        if self.cutDistance>0 or not self.use_GP:
            self.lc['near_trans']=np.tile(False, len(self.lc['time']))
            no_trans_mask=np.tile(False, len(self.lc['time']))
            for ipl in self.multis:
                phase=(self.lc['time']-self.planets[ipl]['tcen']-0.5*self.planets[ipl]['period'])%self.planets[ipl]['period']-0.5*self.planets[ipl]['period']
                self.lc['near_trans']+=abs(phase)<self.cutDistance*self.planets[ipl]['tdur']
                no_trans_mask+=abs(phase)<0.5*self.planets[ipl]['tdur']
            for ipl in self.monos:
                self.lc['near_trans']+=abs(self.lc['time']-self.planets[ipl]['tcen'])<self.cutDistance*self.planets[ipl]['tdur']
                no_trans_mask+=abs(self.lc['time']-self.planets[ipl]['tcen'])<0.5*self.planets[ipl]['tdur']
            for ipl in self.duos:
                #speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen'])<cutDistance]=True
                #speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen_2'])<cutDistance]=True
                for per in self.planets[ipl]['period_aliases']:
                    phase=(self.lc['time']-self.planets[ipl]['tcen']-0.5*per)%per-0.5*per
                    self.lc['near_trans']+=abs(phase)<self.cutDistance*self.planets[ipl]['tdur']
                    no_trans_mask+=abs(phase)<0.5*self.planets[ipl]['tdur']
            
            self.lc['no_trans_mask']=~no_trans_mask
            
            if not self.use_GP:
                self.lc=tools.lcFlatten(self.lc,transit_mask=self.lc['no_trans_mask'],
                                             stepsize=0.25*np.min([self.planets[pl]['tdur'] for pl in self.planets]),
                                             winsize=5*np.max([self.planets[pl]['tdur'] for pl in self.planets]))
            if self.bin_oot:
                #Creating a pseudo-binned dataset where out-of-transit LC is binned to 30mins but near-transit is not.
                oot_binsize=1/8 if self.mission.lower()=='kepler' else 1/48
                oot_binlc=tools.lcBin(self.lc,use_flat=~self.use_GP,extramask=~self.lc['near_trans'],
                                      modify_lc=False,binsize=oot_binsize)
                oot_binlc['mask']=np.tile(True,len(oot_binlc['time']))
                oot_binlc['no_trans_mask']=np.tile(True,len(oot_binlc['time']))
                oot_binlc['near_trans']=np.tile(False,len(oot_binlc['time']))
                self.pseudo_binlc={}
                for key in oot_binlc:
                    lckey = key.replace("bin_","") if 'bin_' in key else key
                    lckey = 'flux_err' if lckey=='flux_flat_err' else lckey
                    print(key,lckey,oot_binlc[key],self.lc[lckey])
                    self.pseudo_binlc[lckey]=np.hstack((oot_binlc[key],self.lc[lckey][self.lc['near_trans']]))
                for key in [key for key in self.pseudo_binlc.keys() if key!='time']:
                    if 'bin_' in key:
                        _=self.pseudo_binlc.pop(key)
                    else:
                        print(key,len(self.pseudo_binlc[key]),len(np.argsort(self.pseudo_binlc['time'])))
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
                
            
            print(np.sum(self.lc['near_trans']&self.lc['mask']),"points in new lightcurve, compared to ",np.sum(self.lc['mask'])," in original mask, leaving ",np.sum(self.lc['near_trans']),"points in the lc")

        else:
            #Using all points in the 
            self.lc['near_trans']=np.tile(True,len(self.lc['time']))
        

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
            
            self.lc['tele_index']=np.zeros((len(self.lc['time']),2))
            if bin_oot:
                self.pseudo_binlc['tele_index']=np.zeros((len(self.pseudo_binlc['time']),2))
            else:
                self.lc_near_trans['tele_index']=np.zeros((len(self.lc_near_trans['time']),2))
            for ncad in range(len(self.cads)):
                if self.cads[ncad][0].lower()=='t':
                    self.lc['tele_index'][:,0]+=self.lc['flux_err_index'][:,ncad]
                    print(self.bin_oot,'making tele_index')
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

        if self.use_GP:
            self.gp={}
            if self.train_GP and not hasattr(self,'gp_init_trace'):
                self.GP_training()

        if use_pymc3:
            self.init_pymc3()
        elif use_multinest:
            self.run_multinest(**kwargs)
        
    def GP_training(self,n_draws=900,max_len_lc=25000):
        with pm.Model() as gp_train_model:
            #####################################################
            #     Training GP kernel on out-of-transit data
            #####################################################
            
            mean=pm.Normal("mean",mu=np.median(self.lc['flux'][self.lc['mask']]),
                                  sd=np.std(self.lc['flux'][self.lc['mask']]))

            self.log_flux_std=np.array([np.log(np.nanstd(self.lc['flux'][self.lc['no_trans_mask']][self.lc['cadence'][self.lc['no_trans_mask']]==c])) for c in self.cads]).ravel().astype(np.float32)
            print(self.log_flux_std)
            print(np.sum(self.lc['no_trans_mask']),len(self.lc['no_trans_mask']))
            print(np.unique(self.lc['cadence'][self.lc['no_trans_mask']]))
            print(np.nanstd(self.lc['flux'][self.lc['cadence']==self.cads[0]]),
                  np.nanstd(self.lc['flux'][self.lc['cadence']==self.cads[1]]))
            print(self.cads,np.unique(self.lc['cadence']),self.lc['time'][self.lc['cadence']=='t'],
                  self.log_flux_std,np.sum(self.lc['no_trans_mask']))

            logs2 = pm.Normal("logs2", mu = self.log_flux_std+1, 
                              sd = np.tile(1.0,len(self.log_flux_std)), shape=len(self.log_flux_std))

            # Transit jitter & GP parameters
            #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
            lcrange=self.lc['time'][-1]-self.lc['time'][0]
            #max_cad = np.nanmax([np.nanmedian(np.diff(self.lc['time'][self.lc['near_trans']&(self.lc['cadence']==c)])) for c in self.cads])
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
            print(success, target,lcrange,low,up)
            maxpower=1.0*np.nanstd(self.lc['flux'][(~self.lc['no_trans_mask'])&self.lc['mask']])
            minpower=0.02*np.nanmedian(abs(np.diff(self.lc['flux'][(~self.lc['no_trans_mask'])&self.lc['mask']])))
            print(np.nanmedian(abs(np.diff(self.lc['flux'][self.lc['near_trans']]))),np.nanstd(self.lc['flux'][self.lc['near_trans']]),minpower,maxpower)
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
            print(success, target)
            S0 = pm.Deterministic("S0", power/(w0**4))

            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))
            
            if len(self.lc['time']>max_len_lc):
                mask=(~self.lc['no_trans_mask'])*(np.arange(0,len(self.lc['time']),1)<max_len_lc)
            else:
                mask=~self.lc['no_trans_mask']
            
            self.gp['train'] = xo.gp.GP(kernel, self.lc['time'][~self.lc['no_trans_mask']].astype(np.float32),
                                   self.lc['flux_err'][~self.lc['no_trans_mask']].astype(np.float32)**2 + \
                                   tt.dot(self.lc['flux_err_index'][~self.lc['no_trans_mask']],tt.exp(logs2)),
                                   J=2)
            
            self.gp['train'].log_likelihood(self.lc['flux'][~self.lc['no_trans_mask']]-mean)

            self.gp_init_soln = xo.optimize(start=None, vars=[logs2, power, w0, mean],verbose=True)
            
            self.gp_init_trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=self.gp_init_soln, chains=4,
                                           step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)
        

    def init_pymc3(self,ld_mult=1.5,):
        
        if self.bin_oot:
            lc=self.pseudo_binlc
        elif self.cutDistance>0:
            lc=self.lc_near_trans
        else:
            lc=self.lc
        
        start=None
        with pm.Model() as model:            

            ######################################
            #   Intialising Stellar Params:
            ######################################
            #Using log rho because otherwise the distribution is not normal:
            logrho_S = pm.Normal("logrho_S", mu=np.log(self.rhostar[0]), 
                                 sd=np.average(abs(self.rhostar[1:]/self.rhostar[0])),
                                 testval=np.log(self.rhostar[0]))
            rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S))
            Rs = pm.Normal("Rs", mu=self.Rstar[0], sd=np.average(abs(self.Rstar[1:])),testval=self.Rstar[0],shape=1)
            Ms = pm.Deterministic("Ms",(rho_S/1.408)*Rs**3)

            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if self.useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0
            
            BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)

            print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)
            
            ######################################
            #     Initialising Mono params
            ######################################
            rs=[]
            if len(self.monos)>0:
                # The period distributions of monotransits are tricky as we often have gaps to contend with
                # We cannot sample the full period distribution while some regions have p=0.
                # Therefore, we need to find each possible period region and marginalise over each
                
                min_Ps=np.array([self.planets[pls]['P_min'] for pls in self.monos])
                print(min_Ps)
                #From Dan Foreman-Mackey's thing:
                #log_soft_per = pm.Uniform("log_soft_per", lower=np.log(min_Ps), upper=np.log(100*min_Ps),shape=len(min_Ps))
                #soft_period = pm.Deterministic("soft_period", tt.exp(log_soft_per))
                #pm.Potential("mono_per_prior",-2*log_soft_per) # prior from window function and occurrence rates
                test_ps=np.array([self.planets[pls]['period'] if self.planets[pls]['period']>self.planets[pls]['P_min'] else 1.25*self.planets[pls]['P_min'] for pls in self.monos])
                mono_uniform_index_period={}
                mono_periods={};mono_t0s={};mono_logrors={};mono_bs={};mono_eccs={};mono_omegas={}
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

                    mono_periods[pl]=pm.Deterministic("mono_periods_"+str(pl), tt.power(((1-ind_min)*mono_uniform_index_period[pl]+ind_min),1/per_index)*self.planets[pl]['per_gaps'][:,0])
                    mono_t0s[pl] = pm.Bound(pm.Normal, 
                                            upper=self.planets[pl]['tcen']+self.planets[pl]['tdur']*0.33,
                                            lower=self.planets[pl]['tcen']-self.planets[pl]['tdur']*0.33
                                           )("mono_t0s_"+pl,mu=self.planets[pl]['tcen'],
                                                  sd=self.planets[pl]['tdur']*0.1,
                                                  testval=self.planets[pl]['tcen'])

                    # The Espinoza (2018) parameterization for the joint radius ratio and
                    print(np.log(0.001),"->",np.log(0.25+int(self.useL2)),
                          np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))

                    mono_logrors[pl]=pm.Uniform("mono_logrors_"+pl,lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                            testval=np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))
                    mono_bs[pl] = xo.distributions.ImpactParameter("mono_bs_"+pl,ror=tt.exp(mono_logrors[pl]),
                                                                   testval=self.planets[pl]['b'])
                    if not self.assume_circ:
                        if len(self.planets)==1:
                            mono_eccs[pl] = BoundedBeta("mono_eccs_"+pl, alpha=0.867,beta=3.03,
                                                         testval=0.05)
                        elif len(self.planets)>1:
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            mono_eccs[pl] = pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("mono_eccs_"+pl,alpha= 0.049,beta=2,
                                                                                           testval=0.05)
                    mono_omegas[pl] = xo.distributions.Angle("mono_omegas_"+pl)

                    rs+=[mono_logrors[pl]]
                    
            ######################################
            #     Initialising Duo params
            ######################################
            if len(self.duos)>0:
                #Again, in the case of a duotransit, we have a series of possible periods between two know transits.
                # TO model these we need to compute each and marginalise over them
                duo_periods={};duo_logrors={};duo_bs={};duo_t0s={};duo_eccs={};duo_omegas={};duo_t0_2s={}
                
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
                    duo_periods[pl]=pm.Deterministic("duo_periods_"+pl,
                                                     abs(duo_t0_2s[pl]-duo_t0s[pl])/self.planets[pl]['period_int_aliases'])
                    duo_logrors[pl]=pm.Uniform("duo_logrors_"+pl, lower=np.log(0.001), upper=np.log(0.25+int(self.useL2)),
                                               testval=np.log(self.planets[pl]['r_pl']/(109.1*self.Rstar[0])))
                                               
                    duo_bs[pl] = xo.distributions.ImpactParameter("duo_bs_"+pl,ror=tt.exp(duo_logrors[pl]),
                                                                  testval=self.planets[pl]['b'])
                    if not self.assume_circ:
                        if len(self.planets)==0:
                            duo_eccs[pl] = BoundedBeta("duo_eccs_"+pl, alpha=0.867,beta=3.03,
                                                         testval=0.05,shape=len(self.duos))
                        elif len(self.planets)>1:
                            # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                            duo_eccs[pl] = pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("duo_eccs_"+pl, alpha= 0.049,beta=2,
                                                                                            testval=0.05)
                        duo_omegas[pl] = xo.distributions.Angle("duo_omegas_"+pl)


                    rs+=[duo_logrors[pl]]
            
            #if len(rs)>1:
            #    rs=[tt.vector(rs)]
            ######################################
            #     Initialising Multi params
            ######################################
            
            if len(self.multis)>0:
                #In the case of multitransiting plaets, we know the periods already, so we set a tight normal distribution
                inipers=np.array([self.planets[pls]['period'] for pls in self.multis])
                inipererrs=np.array([self.planets[pls]['period_err'] for pls in self.multis])
                print("init periods:", inipers,inipererrs)
                multi_periods = pm.Normal("multi_periods", 
                                          mu=inipers,
                                          sd=np.clip(inipererrs*0.25,np.tile(0.005,len(inipers)),0.02*inipers),
                                          shape=len(self.multis),
                                          testval=inipers)
                tcens=np.array([self.planets[pls]['tcen'] for pls in self.multis]).ravel()
                tdurs=np.array([self.planets[pls]['tdur'] for pls in self.multis]).ravel()
                multi_t0s = pm.Bound(pm.Normal, upper=tcens+tdurs*0.5, lower=tcens-tdurs*0.5,
                                    )("multi_t0s",mu=tcens, sd=tdurs*0.05,
                                            shape=len(self.multis),testval=tcens)
                print(np.log(0.001),"->",np.log(0.25+float(int(self.useL2))),'rors:',
                      np.log(np.array([self.planets[pls]['r_pl']/(109.1*self.Rstar[0]) for pls in self.multis])))
                multi_logrors=pm.Uniform("multi_logrors",lower=np.log(0.001), upper=np.log(0.25+float(int(self.useL2))),
                                        testval=np.log(np.array([self.planets[pls]['r_pl']/(109.1*self.Rstar[0]) for pls in self.multis])),
                                        shape=len(self.multis))
                multi_bs = xo.distributions.ImpactParameter("multi_bs",ror=tt.exp(multi_logrors),shape=len(self.multis),
                                                            testval=np.array([self.planets[pls]['b'] for pls in self.multis]))
                if not self.assume_circ:
                    if len(self.planets)==0:
                        multi_eccs = BoundedBeta("multi_eccs", alpha=0.867,beta=3.03,testval=0.05,shape=len(self.multis))
                    elif len(self.planets)>1:
                        # The eccentricity prior distribution from Van Eylen for multiplanets (lower-e than single planets)
                        multi_eccs = pm.Bound(pm.Weibull, lower=1e-5, upper=1-1e-5)("multi_eccs", alpha= 0.049,beta=2,
                                                                                    testval=0.05,shape=len(self.multis))
                    multi_omegas = xo.distributions.Angle("multi_omegas",shape=len(self.multis))

                
                rs+=[multi_logrors]
            #print(rs)
            #r_pl = pm.Deterministic("r_pl", tt.exp(tt.stack(rs))* Rs * 109.1)
            #pm.Potential("logr_potential",tt.log(r_pl))

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
                print("LDs",ld_dists)
                u_star_kep = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_kep", 
                                            mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                            sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
            elif np.any([c[0].lower()=='k' for c in self.cads]) and self.constrain_LD:
                u_star_kep = xo.distributions.QuadLimbDark("u_star_kep", testval=np.array([0.3, 0.2]))
            
            mean=pm.Normal("mean",mu=np.median(lc['flux'][lc['mask']]),
                                  sd=np.std(lc['flux'][lc['mask']]))
            if not hasattr(self,'log_flux_std'):
                self.log_flux_std=np.array([np.log(np.nanstd(lc['flux'][lc['no_trans_mask']][lc['cadence'][lc['no_trans_mask']]==c])) for c in self.cads]).ravel().astype(np.float32)
            print(self.log_flux_std,np.sum(lc['no_trans_mask']),"/",len(lc['no_trans_mask']))
            
            logs2 = pm.Normal("logs2", mu = self.log_flux_std+1, 
                              sd = np.tile(2.0,len(self.log_flux_std)), shape=len(self.log_flux_std))

            if self.use_GP:
                ######################################
                #     Initialising GP kernel
                ######################################
                print(np.isnan(lc['time']),np.isnan(lc['flux']),np.isnan(lc['flux_err']))
                if self.train_GP:
                    #Taking trained values from out-of-transit to use as inputs to GP:
                    minmax=np.percentile(self.gp_init_trace["w0"],[0.5,99.5])
                    w0=pm.Interpolated("w0", x_points=np.linspace(minmax[0],minmax[1],201)[1::2],
                                          pdf_points=np.histogram(self.gp_init_trace["w0"],
                                                                  np.linspace(minmax[0],minmax[1],101))[0]
                                         )
                    minmax=np.percentile(self.gp_init_trace["power"],[0.5,99.5])
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
                    print(np.nanmedian(abs(np.diff(lc['flux'][lc['near_trans']]))),np.nanstd(lc['flux'][lc['near_trans']]),minpower,maxpower)
                    success=False;target=0.05
                    while not success and target<0.4:
                        try:
                            power = pm.InverseGamma("power",testval=minpower*5,
                                                    **xo.estimate_inverse_gamma_parameters(lower=minpower, upper=maxpower,target=0.1))
                            success=True
                        except:
                            target+=0.05
                            success=False
                    print(target," after ",int(target/0.05),"attempts")

                    print("input to GP power:",maxpower-1)
                S0 = pm.Deterministic("S0", power/(w0**4))

                # GP model for the light curve
                kernel = xo.gp.terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))
                
                if hasattr(self.lc,'near_trans') and np.sum(lc['near_trans'])!=len(lc['time']):
                    self.gp['use'] = xo.gp.GP(kernel, lc['time'][lc['near_trans']].astype(np.float32),
                                       lc['flux_err'][lc['near_trans']].astype(np.float32)**2 + \
                                       tt.dot(lc['flux_err_index'][lc['near_trans']],tt.exp(logs2)),
                                       J=2)

                    self.gp['all'] = xo.gp.GP(kernel, lc['time'].astype(np.float32),
                                           lc['flux_err'].astype(np.float32)**2 + \
                                           tt.dot(lc['flux_err_index'],tt.exp(logs2)),
                                           J=2)
                else:
                    self.gp['use'] = xo.gp.GP(kernel, lc['time'][lc['mask']].astype(np.float32),
                                           lc['flux_err'][lc['mask']].astype(np.float32)**2 + \
                                           tt.dot(lc['flux_err_index'][lc['mask']],tt.exp(logs2)),
                                           J=2)

            ################################################
            #     Creating function to generate transits
            ################################################
            def gen_lc(i_orbit, i_r, n_pl, mask=None,prefix='',make_deterministic=False):
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
                    
                    if cad[0]=='t':
                        #Taking the "telescope" index, and adding those points with the matching cadences to the cadmask
                        cad_index+=[(lc['tele_index'][mask,0].astype(bool))&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=lc['time'][mask].astype(np.float32),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                    elif cad[0]=='k':
                        cad_index+=[(lc['tele_index'][mask,1]).astype(bool)&cadmask[mask]]
                        trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=lc['time'][mask].astype(np.float32),
                                                                 texp=np.nanmedian(np.diff(lc['time'][cadmask]))
                                                                 )/(lc['flux_unit']*mult)]
                # transit arrays (ntime x n_pls x 2) * telescope index (ntime x n_pls x 2), summed over dimension 2
                if n_pl>1 and make_deterministic:
                    
                    return pm.Deterministic(prefix+"light_curves", 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=2))
                elif n_pl==1 and make_deterministic:
                    return pm.Deterministic(prefix+"light_curves", 
                                        tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2)))
                elif n_pl>1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=2)

                elif n_pl==1 and not make_deterministic:
                    return tt.sum(tt.stack(trans_pred,axis=2).dimshuffle(0,1,2) * tt.stack(cad_index).dimshuffle(1,'x',0),axis=(1,2))

            
            ################################################
            #     Analysing Multiplanets
            ################################################
            if len(self.multis)>0:
                if self.assume_circ:
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S, period=multi_periods, t0=multi_t0s, b=multi_bs)
                else:
                    # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,ecc=multi_eccs, omega=multi_omegas,
                        period=multi_periods, t0=multi_t0s, b=multi_bs)
                print("generating multi lcs:")
                
                multi_mask_light_curves = gen_lc(multi_orbit,tt.exp(multi_logrors),
                                                 len(self.multis),mask=None,
                                                 prefix='multi_mask_',make_deterministic=True)
                
                multi_vx, multi_vy, multi_vz = multi_orbit.get_relative_velocity(multi_t0s)
                multi_tdur=pm.Deterministic("multi_tdurs",
                                            (2 * tt.tile(Rs,len(self.multis)) * tt.sqrt( (1+tt.exp(multi_logrors))**2 - multi_bs**2)) / tt.sqrt(multi_vx**2 + multi_vy**2) )
                if self.force_match_input is not None:
                    #pm.Bound("bounded_tdur_multi",upper=0.5,lower=-0.5,
                    #         tt.log(multi_tdur)-tt.log([self.planets[multi]['tdur'] for multi in self.multis]))
                    pm.Potential("match_input_potential_multi",
                                 self.force_match_input*tt.sum(multi_logrors - \
                                 tt.exp((tt.log(multi_tdur)-tt.log([self.planets[multi]['tdur'] for multi in self.multis]))**2)
                                                              ))
                print("summing multi lcs:")
                if len(self.multis)>1:
                    multi_mask_light_curve = pm.math.sum(multi_mask_light_curves, axis=1) #Summing lightcurve over n planets
                else:
                    multi_mask_light_curve=multi_mask_light_curves
                #Multitransiting planet potentials:
                '''if self.use_GP:
                    pm.Potential("multi_obs",
                                 self.gp['oot'].log_likelihood(self.lc['flux'][self.lc['near_trans']]-(multi_mask_light_curve+ mean)))
                else:
                    new_yerr = self.lc['flux_err'][self.lc['near_trans']].astype(np.float32)**2 + \
                               tt.dot(self.lc['flux_err_index'][self.lc['near_trans']],tt.exp(logs2))
                    pm.Normal("multiplanet_obs",mu=(multi_mask_light_curve + mean),sd=new_yerr,
                              observed=self.lc['flux_flat'][self.lc['near_trans']].astype(np.float32))
                '''
                
            else:
                multi_mask_light_curve = tt.as_tensor_variable(np.zeros(len(lc['time'])))
                #print(multi_mask_light_curve.shape.eval())
                #np.zeros_like(self.lc['flux'][self.lc['near_trans']])
                
            ################################################
            #     Marginalising over Duo periods
            ################################################
            if len(self.duos)>0:
                duo_per_info={}
                duo_vels={}
                duo_tdurs={}
                for nduo,duo in enumerate(self.duos):
                    print("#Marginalising over ",len(self.planets[duo]['period_int_aliases'])," period aliases for ",duo)

                    #Marginalising over each possible period
                    #Single planet with two transits and a gap
                    duo_per_info[duo]={'lc':{}}
                    npers=len(self.planets[duo]['period_int_aliases'])
                    if self.assume_circ:
                        duoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            period=duo_periods[duo], t0=tt.tile(duo_t0s[duo],npers), b=tt.tile(duo_bs[duo],npers))
                    else:
                        duoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            ecc=tt.tile(duo_eccs[duo],npers), omega=tt.tile(duo_omegas[duo],npers),
                            period=duo_periods[duo], t0=tt.tile(duo_t0s[duo],npers), b=tt.tile(duo_bs[duo],npers))
                    duo_vels[duo] = duoorbit.get_relative_velocity(duo_t0s[duo])
                    duo_tdurs[duo]=pm.Deterministic("duo_tdurs_"+duo,(2*Rs*tt.sqrt((1+tt.exp(duo_logrors[duo]))**2-duo_bs[duo]**2))/tt.sqrt(duo_vels[duo][0]**2 + duo_vels[duo][1]**2))
                    if self.force_match_input is not None:
                        #pm.Bound("bounded_tdur_duo_"+str(duo),upper=0.5,lower=-0.5,
                        #         tt.log(duo_tdurs[duo])-tt.log(self.planets[duo]['tdur']))
                        pm.Potential("match_input_potential_duo_"+str(duo),duo_logrors[duo] - 
                             self.force_match_input*tt.exp((tt.log(duo_tdurs[duo])-tt.log(self.planets[duo]['tdur']))**2))

                    duo_per_info[duo]['logpriors'] = tt.log(duoorbit.dcosidb) - 2 * tt.log(duo_periods[duo])
                    duo_per_info[duo]['lcs'] = gen_lc(duoorbit,tt.tile(tt.exp(duo_logrors[duo]),npers),npers,
                                                     mask=None,prefix='duo_mask_'+duo+'_')
                    pm.Deterministic('duo_priors_'+duo,duo_per_info[duo]['logpriors'])

            ################################################
            #     Marginalising over Mono gaps
            ################################################
            if len(self.monos)>0:
                mono_per_info={}
                mono_vels={}
                mono_gap_info={}
                mono_tdurs={}
                for nmono,mono in enumerate(self.monos):
                    print("#Marginalising over ",len(self.planets[mono]['per_gaps'])," period gaps for ",mono)
                    
                    #Single planet with one transits and multiple period gaps
                    mono_gap_info[mono]={'lc':{}}
                    n_gaps=len(self.planets[mono]['per_gaps'][:,0])
                    # Set up a Keplerian orbit for the planets
                    #print(r[mono_ind].ndim,tt.tile(r[mono_ind],len(self.planets[mono]['per_gaps'][:,0])).ndim)
                    if self.assume_circ:
                        monoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            period=mono_periods[mono], 
                            t0=tt.tile(mono_t0s[mono],n_gaps),
                            b=tt.tile(mono_bs[mono],n_gaps))
                    else:
                        monoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            ecc=tt.tile(mono_eccs[mono],n_gaps),
                            omega=tt.tile(mono_omegas[mono],n_gaps),
                            period=mono_periods[mono],
                            t0=tt.tile(mono_t0s[mono],n_gaps),
                            b=tt.tile(mono_bs[mono],n_gaps))
                    
                    mono_vels[mono] = monoorbit.get_relative_velocity(mono_t0s[mono])
                    mono_tdurs[mono]=pm.Deterministic("mono_tdurs_"+mono,(2*Rs*tt.sqrt((1+tt.exp(mono_logrors[mono]))**2-mono_bs[mono]**2))/tt.sqrt(mono_vels[mono][0]**2 + mono_vels[mono][1]**2))
                    if self.force_match_input is not None:
                        #pm.Bound("bounded_tdur_mono_"+str(mono),upper=0.5,lower=-0.5,
                        #         tt.log(mono_tdurs[mono])-tt.log(self.planets[mono]['tdur']))
                        pm.Potential("match_input_potential_mono_"+str(mono),mono_logrors[mono] - self.force_match_input * \
                                     (tt.exp((tt.log(mono_tdurs[mono])-tt.log(self.planets[mono]['tdur']))**2)))

                    mono_gap_info[mono]['logpriors'] = tt.log(monoorbit.dcosidb) - 2*tt.log(per_meds[mono])
                    mono_gap_info[mono]['lcs'] = gen_lc(monoorbit, tt.tile(tt.exp(mono_logrors[mono]),n_gaps),
                                                        n_gaps,mask=None,prefix='mono_mask_'+mono+'_')
                    pm.Deterministic('mono_priors_'+mono,mono_gap_info[mono]['logpriors'])
            #Priors - we have an occurrence rate prior (~1/P), a geometric prior (1/distance in-transit = dcosidb)
            # a window function log(1/P) -> -1*logP and  a factor for the width of the period bin - i.e. log(binsize)
            #mono_gap_info[mono]['logpriors'] = 0.0
            #This is also complicated by the fact that each gap has its own internal gradient
            # but between regions these are not normalised, so we include a factor w.r.t. the median period in the bin
            #I have no idea if we also need to incorporate the *width* of the bin here - I need to test this.
                    
            ################################################
            #            Compute predicted LCs:
            ################################################
            
            # Marginalising together - say we have 3 models to marginalise and [4,6 and 8] regions in each:
            # This means we need to create a sum of each combination (e.g. a loglike)
            if len(self.duos+self.monos)>0:
                iter_models = {}
                n_mod=0
                for duo in self.duos:
                    iter_models[n_mod]={'name':duo,
                                        'n_points':np.sum((abs(lc['time']-self.planets[duo]['tcen'])<0.5*self.planets[duo]['tdur'])|(abs(lc['time']-self.planets[duo]['tcen_2'])<0.5*self.planets[duo]['tdur'])),
                                       'durs':duo_tdurs[duo],
                                       'pers':duo_periods[duo],
                                       'len':len(self.planets[duo]['period_int_aliases']),
                                       'range':np.arange(len(self.planets[duo]['period_int_aliases'])),
                                       'lcs':duo_per_info[duo]['lcs'],
                                       'logpriors':duo_per_info[duo]['logpriors'],
                                       'type':'duo'}
                    n_mod+=1
                for mono in self.monos:
                    iter_models[n_mod]={'name':mono,
                                        'n_points':np.sum(abs(lc['time']-self.planets[mono]['tcen'])<0.5*self.planets[mono]['tdur']),
                                        'durs':mono_tdurs[mono],
                                        'pers':mono_periods[mono],
                                        'len':len(self.planets[mono]['per_gaps']),
                                        'range':np.arange(len(self.planets[mono]['per_gaps'])),
                                        'lcs':mono_gap_info[mono]['lcs'],
                                        'logpriors':mono_gap_info[mono]['logpriors'],
                                        'type':'mono'}
                    n_mod+=1


                #For each combination we will create a combined model and compute the loglik
                new_yerr = lc['flux_err'][lc['mask']].astype(np.float32)**2 + \
                               tt.sum(lc['flux_err_index'][lc['mask']]*tt.exp(logs2).dimshuffle('x',0),axis=1)

                if not self.use_GP:
                    #Calculating some extra info to speed up the loglik calculation
                    new_yerr_sq = new_yerr**2
                    sum_log_new_yerr = tt.sum(-np.sum(lc['mask'])/2 * tt.log(2*np.pi*(new_yerr_sq)))
                    lccheckstack=tt.stack([lc['flux'][lc['mask']],new_yerr_sq],axis=1)

                #NOT marginalising over all models simultaneously, but doing them individually:
                resids={}
                log_prob_margs={}
                for pl in iter_models:
                    resids[pl]={}
                    iter_models[pl]['logprob']={}
                    iter_models[pl]['logliks']={}
                    if iter_models[pl]['len']>1:
                        print(iter_models[pl])
                        for n in range(iter_models[pl]['len']):
                            # For each model we compute residuals (subtract mean and multiplanets)

                            resids[pl][n] = lc['flux'][lc['mask']].astype(np.float32) - \
                                     (iter_models[pl]['lcs'][lc['mask'],n] + multi_mask_light_curve[lc['mask']] + mean.dimshuffle('x'))
                            if self.use_GP:
                                iter_models[pl]['logliks'][n]=self.gp['use'].log_likelihood(y=resids[pl][n])
                            else:
                                iter_models[pl]['logliks'][n] = sum_log_new_yerr - tt.sum(-0.5*(resids[pl][n])**2/(2*new_yerr_sq.dimshuffle(0,'x')),axis=0)

                        # We then compute a marginalised lightcurve from the weighted sum of each model lightcurve:
                        logliks = pm.Deterministic(iter_models[pl]['type']+'_liks_'+str(iter_models[pl]['name']),tt.stack([iter_models[pl]['logliks'][n] for n in range(iter_models[pl]['len'])]))
                        iter_models[pl]['logprob'] = logliks + iter_models[pl]['logpriors']
                        iter_models[pl]['logprob_marg'] = pm.math.logsumexp(iter_models[pl]['logprob'])
                        log_prob_margs[pl] = pm.Deterministic('logprob_marg_'+str(iter_models[pl]['name']), iter_models[pl]['logprob'] - iter_models[pl]['logprob_marg'])
                        pm.Deterministic('period_marg_'+str(iter_models[pl]['name']), tt.sum(iter_models[pl]['pers']*tt.exp(log_prob_margs[pl])))
                        pm.Deterministic('tdur_marg_'+str(iter_models[pl]['name']), tt.sum(iter_models[pl]['durs']*tt.exp(log_prob_margs[pl])))
                        iter_models[pl]['marg_lc'] = pm.Deterministic('marg_light_curve_'+str(iter_models[pl]['name']),
                                                                      tt.sum(iter_models[pl]['lcs']*tt.exp(iter_models[pl]['logprob']-iter_models[pl]['logprob_marg']).dimshuffle('x',0),axis=1))
                    else:
                        iter_models[pl]['marg_lc'] = pm.Deterministic('marg_light_curve_'+str(iter_models[pl]['name']),
                                                                      iter_models[pl]['lcs'])
                        pm.Deterministic('period_marg_'+str(iter_models[pl]['name']),iter_models[pl]['pers'])
                        pm.Deterministic('tdur_marg_'+str(iter_models[pl]['name']),iter_models[pl]['durs'])
                #Now summing over all lcs:
                marg_all_light_curves = tt.stack([iter_models[pl]['marg_lc'] for pl in iter_models], axis=1)
                marg_all_light_curve = pm.Deterministic("marg_all_light_curve",
                                                        tt.sum(marg_all_light_curves,axis=1) + multi_mask_light_curve)
                    
                if self.use_GP:
                    total_llk = pm.Deterministic("total_llk",self.gp['use'].log_likelihood(lc['flux'][lc['mask']] - \
                                                                                    (marg_all_light_curve[lc['mask']] + mean)))
                    gp_pred = pm.Deterministic("gp_pred", self.gp['use'].predict(lc['time'].astype(np.float32),
                                                                                 return_var=False))
                    pm.Potential("llk_gp", total_llk)
                    #pm.Normal("all_obs",mu=(marg_all_light_curve + gp_pred + mean),sd=new_yerr,
                    #          observed=self.lc['flux'][self.lc['near_trans']].astype(np.float32))
                else:
                    pm.Normal("all_obs",mu=(marg_all_light_curve[lc['mask']] + mean),sd=new_yerr,
                              observed=lc['flux'][lc['mask']].astype(np.float32))

            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            print(model.test_point)
            map_soln = xo.optimize(start=start)
            ################################################
            #               Optimizing:
            ################################################

            #Setting up optimization depending on what planet models we have:
            initvars0=[]#r,b
            initvars1=[]#P
            initvars2=[rho_S]#r,b,t0
            initvars3=[logs2]
            initvars4=[]#r,b,P
            if len(self.multis)>0:
                initvars0+=[multi_logrors,multi_bs]
                initvars1+=[multi_periods]
                initvars2+=[multi_logrors,multi_bs,multi_t0s]
                initvars4+=[multi_logrors,multi_bs,multi_periods]
                if not self.assume_circ:
                    initvars2+=[multi_eccs, multi_omegas]

            if len(self.monos)>0:
                for pl in self.monos:
                    initvars0 += [mono_bs[pl]]
                    initvars1 += [mono_logrors[pl]]
                    initvars2 += [mono_logrors[pl],mono_bs[pl],mono_t0s[pl]]
                    initvars4 += [mono_logrors[pl],mono_bs[pl]]
                    for n in range(len(self.planets[pl]['per_gaps'][:,0])):
                        initvars0 += [mono_periods[pl][n]]
                        initvars1 += [mono_periods[pl][n]]
                        initvars4 += [mono_periods[pl][n]]
                    if not self.assume_circ:
                        initvars2+=[mono_eccs[pl], mono_omegas[pl]]

                        #exec("initvars1 += [mono_periods_"+pl+"_"+str(int(n))+"]")
                        #exec("initvars4 += [mono_periods_"+pl+"_"+str(int(n))+"]")
            if len(self.duos)>0:
                #for pl in self.duos:
                #    eval("initvars1+=[duo_period_"+pl+"]")
                for pl in self.duos:
                    initvars0 += [duo_logrors[pl],duo_bs[pl]]
                    initvars1 += [duo_periods[pl]]
                    initvars2 += [duo_logrors[pl],duo_bs[pl],duo_t0s[pl],duo_t0_2s[pl]]
                    initvars4 += [duo_logrors[pl],duo_bs[pl],duo_periods[pl]]
                    if not self.assume_circ:
                        initvars2+=[duo_eccs[pl], duo_omegas[pl]]
            if self.use_GP:
                initvars3+=[logs2, power, w0, mean]
            else:
                initvars3+=[mean]
            initvars5=initvars2+initvars3+[logs2,Rs,Ms]
            if np.any([c[0].lower()=='t' for c in self.cads]):
                initvars5+=[u_star_tess]
            if np.any([c[0].lower()=='k' for c in self.cads]):
                initvars5+=[u_star_kep]

            print("before",model.check_test_point())
            map_soln = xo.optimize(start=start, vars=initvars0,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars1,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars2,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars3,verbose=True)
            map_soln = xo.optimize(start=map_soln, vars=initvars4,verbose=True)
            #Doing everything except the marginalised periods:
            map_soln = xo.optimize(start=map_soln, vars=initvars5)

            print("after",model.check_test_point())

            self.model = model
            self.init_soln = map_soln
    
    def RunMcmc(self, n_draws=1200, plot=True, do_per_gap_cuts=True, overwrite=False,**kwargs):
        if not overwrite:
            self.LoadPickle()
            print("LOADED MCMC")

        if not hasattr(self,'trace') or overwrite:
            assert hasattr(self,'init_soln')
            #Running sampler:
            np.random.seed(int(self.ID))
            with self.model:
                print(type(self.init_soln))
                print(self.init_soln.keys())
                self.trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=self.init_soln, chains=4,
                                       step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

            self.SavePickle()
        
    def Table(self):
        if LoadFromFile and not self.overwrite and os.path.exists(self.savenames[0].replace('mcmc.pickle','results.txt')):
            with open(self.savenames[0].replace('mcmc.pickle','results.txt'), 'r', encoding='UTF-8') as file:
                restable = file.read()
        else:
            restable=self.ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
                                       savename=self.savenames[0].replace('mcmc.pickle','results.txt'), overwrite=False,
                                       savefileloc=None, tracemask=tracemask)
    def run_multinest(self, ld_mult=2.5, verbose=False,max_iter=1500,**kwargs):
        import pymultinest
        
        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        
        if os.path.exists(self.savenames[0].replace('_mcmc.pickle','mnest_out')):
            if not self.overwrite:
                os.system('rm '+self.savenames[0].replace('_mcmc.pickle','mnest_out')+'/*')
        else:
            os.mkdir(self.savenames[0].replace('_mcmc.pickle','mnest_out'))
            out_mnest_folder = self.savenames[0].replace('_mcmc.pickle','mnest_out')


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
                        outputfiles_basename=self.savenames[0].replace('_mcmc.pickle','mnest_out/'), 
                        **kwargs)
    
    
    def Plot(self, interactive=False, n_samp=35, overwrite=False,return_fig=False,max_gp_len=20000, bin_gp=True, plot_loc=None):
        ################################################################
        #       Varied plotting function for MonoTransit model
        ################################################################
        import seaborn as sns
        sns.set_palette('viridis', len(self.planets)+3)
        pal = sns.color_palette('viridis', len(self.planets)+3)
        pal = pal.as_hex()

        #Rasterizing matplotlib files if we have a lot of datapoints:
        raster=True if len(self.lc['time']>8000) else False
        
        if not hasattr(self,'trace'):
            n_samp==1
        
        #Assigning which lc was connected to the transit/gp modelling:
        if self.bin_oot:
            lc=self.pseudo_binlc
        elif self.cutDistance>0:
            lc=self.lc_near_trans
        else:
            lc=self.lc
        

        if interactive:
            #Plots bokeh figure
            
            from bokeh.plotting import figure, output_file, save, curdoc, show
            from bokeh.models import Band, Whisker, ColumnDataSource, Range1d
            from bokeh.models.arrow_heads import TeeHead
            from bokeh.layouts import gridplot, row, column, layout
            
            if plot_loc is None:
                if not hasattr(self,'savenames'):
                    self.GetSavename(how='save')
                savename=self.savenames[0].replace('_mcmc.pickle','_transit_fit.html')
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
        
        self.lc=tools.lcBin(self.lc,binsize=1/48.0)
        
        #Finding if there's a single enormous gap in the lightcurve, and creating time splits for each region
        x_gaps=np.hstack((0, np.where(np.diff(self.lc['time'])>10)[0]+1, len(self.lc['time'])))
        print(x_gaps)
        limits=[]
        binlimits=[]
        if len(lc['time'])!=len(self.lc['time']):
            modlclims=[]
        gap_lens=[]
        for ng in range(len(x_gaps)-1):
            limits+=[[x_gaps[ng],x_gaps[ng+1]]]
            gap_lens+=[self.lc['time'][limits[-1][1]-1]-self.lc['time'][limits[-1][0]]]
            binlimits+=[[np.argmin(abs(self.lc['bin_time']-self.lc['time'][x_gaps[ng]])),
                         np.argmin(abs(self.lc['bin_time']-self.lc['time'][x_gaps[ng+1]-1]))+1]]
            if len(lc['time'])!=len(self.lc['time']):
                modlclims+=[[np.argmin(abs(lc['time']-self.lc['time'][x_gaps[ng]])),
                             np.argmin(abs(lc['time']-self.lc['time'][x_gaps[ng+1]-1]))+1]]
        if len(lc['time'])==len(self.lc['time']):
            modlclims=limits

        gap_lens=np.array(gap_lens)
        all_lens=np.sum(gap_lens)
        limit_mask={}
        modlclim_mask={}
        for n in range(len(gap_lens)):
            modlclim_mask[n]=np.tile(False,len(lc['time']))
            modlclim_mask[n][modlclims[n][0]:modlclims[n][1]][lc['mask'][modlclims[n][0]:modlclims[n][1]]]=True
            limit_mask[n]=np.tile(False,len(self.lc['time']))
            limit_mask[n][limits[n][0]:limits[n][1]][self.lc['mask'][limits[n][0]:limits[n][1]]]=True
        masks=[]
        
        #####################################
        #       Initialising figures
        #####################################
        f_alls=[];f_all_resids=[];f_trans=[];f_trans_resids=[]
        if not interactive:
            #Creating cumulative list of integers which add up to 24 but round to nearest length ratio:
            n_pl_widths=np.hstack((0,np.cumsum(1+np.array(saferound(list((24-len(gap_lens))*gap_lens/all_lens), places=0))))).astype(int)
            #(gs[0, :]) - all top
            #(gs[:, 0]) - all left
            print(gap_lens/all_lens,n_pl_widths,32,range(len(n_pl_widths)-1))
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
            for ng,gaplen in enumerate(gap_lens):
                fwidth=int(np.round(750*gaplen/all_lens)-10)
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
        if not hasattr(self, 'trans_to_plot') or self.trans_to_plot['n_samp']!=n_samp or 'all' not in self.trans_to_plot or overwrite:
            self.init_trans_to_plot={}
            self.init_trans_to_plot['all']={}
            if hasattr(self,'trace') and 'marg_all_light_curve' in self.trace.varnames:
                prcnt=np.percentile(self.trace['marg_all_light_curve'],(5,1,50,84,95),axis=0)
                nms=['-2sig','-1sig','med','+1sig','+2sig']
                self.init_trans_to_plot['all']={nms[n]:prcnt[n] for n in range(5)}
            elif 'marg_all_light_curve' in self.init_soln:
                self.init_trans_to_plot['all']['med']=self.init_soln['marg_all_light_curve']
            else:
                print("marg_all_light_curve not in any optimised models")
            for pl in self.planets:
                self.init_trans_to_plot[pl]={}
                if pl in self.multis:
                    if hasattr(self,'trace') and 'multi_mask_light_curves' in self.trace.varnames:
                        if len(self.trace['multi_mask_light_curves'].shape)>2:
                            prcnt = np.percentile(self.trace['multi_mask_light_curves'][:,:,self.multis.index(pl)],
                                                      (5,16,50,84,95),axis=0)
                        else:
                            prcnt = np.percentile(self.trace['multi_mask_light_curves'], (5,16,50,84,95), axis=0)

                        nms=['-2sig','-1sig','med','+1sig','+2sig']
                        self.init_trans_to_plot[pl] = {nms[n]:prcnt[n] for n in range(5)}
                    elif 'multi_mask_light_curves' in self.init_soln:
                        if len(self.init_soln['multi_mask_light_curves'].shape)==1:
                            self.init_trans_to_plot[pl]['med'] = self.init_soln['multi_mask_light_curves']
                        else:    
                            self.init_trans_to_plot[pl]['med'] = self.init_soln['multi_mask_light_curves'][:,self.multis.index(pl)]
                    else:
                        print('multi_mask_light_curves not in any optimised models')
                elif pl in self.duos or self.monos:
                    if hasattr(self,'trace') and 'marg_light_curve_'+pl in self.trace.varnames:
                        prcnt=np.percentile(self.trace['marg_light_curve_'+pl],(5,16,50,84,95),axis=0)
                        nms=['-2sig','-1sig','med','+1sig','+2sig']
                        self.init_trans_to_plot[pl] = {nms[n]:prcnt[n] for n in range(5)}
                    elif 'marg_light_curve_'+pl in self.init_soln:
                        self.init_trans_to_plot[pl]['med'] = self.init_soln['marg_light_curve_'+pl]
                    else:
                        print('marg_light_curve_'+pl+' not in any optimised models')
            self.trans_to_plot={'n_samp':n_samp}
            
            #Adding zeros to other regions where we dont have transits (not in the out of transit mask):
            for key1 in self.init_trans_to_plot:
                self.trans_to_plot[key1]={}
                for key2 in self.init_trans_to_plot[key1]:
                    self.trans_to_plot[key1][key2]=np.zeros(len(self.lc['time']))
                    self.trans_to_plot[key1][key2][self.lc['near_trans']]=self.init_trans_to_plot[key1][key2][lc['near_trans']]

        #####################################
        #       Initialising GP model
        #####################################
        import celerite
        from celerite import terms
        gp_pred=[]
        gp_sd=[]
        if self.use_GP and (not hasattr(self, 'gp_to_plot') or self.gp_to_plot['n_samp']!=n_samp or 'gp_pred' not in self.gp_to_plot or overwrite):
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
                
                for n in np.arange(len(gap_lens)):

                    #Only creating out-of-transit GP for the binned (e.g. 30min) data
                    cutBools = tools.cutLc(self.lc['time'][limits[n][0]:limits[n][1]],max_gp_len,
                                           transit_mask=self.lc['no_trans_mask'][limits[n][0]:limits[n][1]])

                    limit_mask_bool[n]={}
                    for nc,c in enumerate(cutBools):
                        limit_mask_bool[n][nc]=np.tile(False,len(self.lc['time']))
                        limit_mask_bool[n][nc][limits[n][0]:limits[n][1]][c]=limit_mask[n][limits[n][0]:limits[n][1]][c]
                        i_kernel = terms.SHOTerm(log_S0=np.log(self.meds['S0']), log_omega0=np.log(self.meds['w0']), log_Q=np.log(1/np.sqrt(2)))
                        i_gp = celerite.GP(i_kernel,mean=self.meds['mean'],fit_mean=False)

                        i_gp.compute(self.lc['time'][limit_mask_bool[n][nc]].astype(np.float32),
                                     np.sqrt(self.lc['flux_err'][limit_mask_bool[n][nc]]**2 + \
                                     np.dot(self.lc['flux_err_index'][limit_mask_bool[n][nc]],np.exp(self.meds['logs2']))))
                        #llk=i_gp.log_likelihood(mod.lc['flux'][mod.lc['mask']][limits[n][0]:limits[n][1]][c]-mod.trans_to_plot['all']['med'][mod.lc['mask']][limits[n][0]:limits[n][1]][c]-mod.meds['mean'])
                        #print(llk.eval())
                        i_gp_pred, i_gp_var= i_gp.predict(self.lc['flux'][limit_mask_bool[n][nc]] - \
                                            self.trans_to_plot['all']['med'][limit_mask_bool[n][nc]],
                                            t=self.lc['time'][limits[n][0]:limits[n][1]][c].astype(np.float32),
                                            return_var=True,return_cov=False)
                        gp_pred+=[i_gp_pred]
                        gp_sd+=[np.sqrt(i_gp_var)]
                ''''
                gp_pred=[];gp_sd=[]
                for n in np.arange(len(gap_lens)):
                    with self.model:
                        pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc['time'][limits[n][0]:limits[n][1]],
                                                                return_var=True,return_cov=False),self.meds)                    
                    gp_pred+=[pred]
                    gp_sd+=[np.sqrt(var)]
                    print(n,len(self.lc['time'][limits[n][0]:limits[n][1]]),'->',len(gp_sd[-1]),len(gp_pred[-1]))
                '''
                self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
                self.gp_to_plot['gp_sd']=np.hstack(gp_sd)
                '''
                with self.model:
                    pred,var=xo.eval_in_model(self.gp['use'].predict(self.lc['time'].astype(np.float32),
                                                                     return_var=True,return_cov=False),self.meds)                    
                self.gp_to_plot['gp_pred']=pred
                self.gp_to_plot['gp_sd']=np.sqrt(var)
                '''
            elif n_samp>1:
                assert hasattr(self,'trace')
                #Doing multiple samples and making percentiles:
                for n in np.arange(len(gap_lens)):
                    #Need to break up the lightcurve even further to avoid GP burning memory:
                    cutBools = tools.cutLc(self.lc['time'][limits[n][0]:limits[n][1]],max_gp_len,
                                           transit_mask=self.lc['no_trans_mask'][limits[n][0]:limits[n][1]])
                    i_kernel = terms.SHOTerm(log_S0=np.log(self.meds['S0']), log_omega0=np.log(self.meds['w0']), log_Q=np.log(1/np.sqrt(2)))
                    i_gp = celerite.GP(i_kernel,mean=self.meds['mean'],fit_mean=False)
                    limit_mask_bool[n]={}
                    for nc,c in enumerate(cutBools):
                        limit_mask_bool[n][nc]=np.tile(False,len(self.lc['time']))
                        limit_mask_bool[n][nc][limits[n][0]:limits[n][1]][c]=limit_mask[n][limits[n][0]:limits[n][1]][c]
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
                            marg_lc[self.lc['near_trans']]=sample['marg_all_light_curve'][lc['near_trans']]
                            ii_gp_pred, ii_gp_var= i_gp.predict(self.lc['flux'][limit_mask_bool[n][nc]] - marg_lc[limit_mask_bool[n][nc]],
                                            t=self.lc['time'][limits[n][0]:limits[n][1]][c].astype(np.float32),
                                            return_var=True,return_cov=False)

                            i_gp_pred+=[ii_gp_pred]
                            i_gp_var+=[ii_gp_var]
                        av, std = tools.weighted_avg_and_std(np.vstack(i_gp_pred),np.sqrt(np.vstack(i_gp_var)),axis=0)
                        gp_pred+=[av]
                        gp_sd+=[std]
                self.gp_to_plot['gp_pred']=np.hstack(gp_pred)
                self.gp_to_plot['gp_sd']=np.hstack(gp_sd)
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
                        ii_gp_pred, ii_gp_var = xo.eval_in_model(self.gp['use'].predict(self.lc['time'].astype(np.float32),
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
            resid_sd=np.nanstd(self.lc['flux'] - self.gp_to_plot['gp_pred'] - self.trans_to_plot['all']['med'])
        else:
            resid_sd=np.nanstd(self.lc['flux_flat'] - self.trans_to_plot['all']['med'])
        
        raw_plot_offset = 2.5*abs(self.min_trans) if not self.use_GP else 1.25*abs(self.min_trans) + resid_sd +\
                                                                     abs(np.min(self.gp_to_plot['gp_pred'][self.lc['mask']]))

        for n in np.arange(len(gap_lens)):
            low_lim = self.lc['time'][limits[n][0]]
            upp_lim = self.lc['time'][limits[n][1]-1]
            unmasked_lim_bool = (self.lc['time']>=(low_lim-0.5))&(self.lc['time']<(upp_lim+0.5))
            
            if self.use_GP:
                if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                    bin_detrend=tools.bin_lc_segment(np.column_stack((self.lc['time'][limit_mask[n]],
                                   self.lc['flux'][limit_mask[n]] - \
                                   self.gp_to_plot['gp_pred'][limit_mask[n]],
                                   self.lc['flux_err'][limit_mask[n]])),
                                   binsize=29/1440)
                    bin_resids=tools.bin_lc_segment(np.column_stack((self.lc['time'][limit_mask[n]],
                                   self.lc['flux'][limit_mask[n]] - \
                                   self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                   self.trans_to_plot['all']['med'][limit_mask[n]],
                                   self.lc['flux_err'][limit_mask[n]])),
                                   binsize=29/1440)
            else:
                if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                    bin_resids=tools.bin_lc_segment(np.column_stack((self.lc['time'][limit_mask[n]],
                                   self.lc['flux_flat'][limit_mask[n]] - \
                                   self.trans_to_plot['all']['med'][limit_mask[n]],
                                   self.lc['flux_err'][limit_mask[n]])),
                                   binsize=29/1440)
                mean=np.nanmedian(self.trace['mean']) if hasattr(self,'trace') else self.init_soln['mean']

            
            #Plotting each part of the lightcurve:
            if interactive:
                if self.use_GP:
                    #Plotting GP region and subtracted flux
                    if np.nanmedian(np.diff(self.lc['time']))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux'][limit_mask[n]]+raw_plot_offset,
                                         alpha=0.5,size=0.75,color='black')
                        f_alls[n].circle(self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                         self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] + raw_plot_offset,
                                         alpha=0.65,size=3.5,legend="raw")
                        errors = ColumnDataSource(data=dict(base=self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                                     lower=self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] + \
                                                     raw_plot_offset - self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]],
                                                     upper=self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] + \
                                                     raw_plot_offset + self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]]))

                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux'][limit_mask[n]]+raw_plot_offset,
                                         legend="raw",alpha=0.65,size=3.5)

                    
                    gpband = ColumnDataSource(data=dict(base=self.lc['time'][limit_mask[n]],
                              lower=raw_plot_offset+self.gp_to_plot['gp_pred'][limit_mask[n]]-self.gp_to_plot['gp_sd'][limit_mask[n]], 
                              upper=raw_plot_offset+self.gp_to_plot['gp_pred'][limit_mask[n]]+self.gp_to_plot['gp_sd'][limit_mask[n]]))
                    f_alls[n].add_layout(Band(source=gpband,base='base',lower='lower',upper='upper',
                                              fill_alpha=0.4, line_width=0.0, fill_color=pal[3]))
                    f_alls[n].line(self.lc['time'][limit_mask[n]], self.gp_to_plot['gp_pred'][limit_mask[n]]+raw_plot_offset, 
                                   line_alpha=0.6, line_width=1.0, color=pal[3], legend="GP fit")

                    if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]] ))<1/72:
                        #Here we plot the detrended flux:
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux'][limit_mask[n]]-self.gp_to_plot['gp_pred'][limit_mask[n]], color='black',
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
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux'][limit_mask[n]]-self.gp_to_plot['gp_pred'][limit_mask[n]],
                                         legend="detrended",alpha=0.65,
                                         size=3.5)

                else:
                    if np.nanmedian(np.diff(self.lc['time']))<1/72:
                        #PLOTTING DETRENDED FLUX, HERE WE BIN
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux_flat'][limit_mask[n]]+raw_plot_offset, ".k",
                                         alpha=0.5,size=0.75)
                        f_alls[n].circle(self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                         self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]],
                                         legend="detrended",alpha=0.65,size=3.5)
                        
                        errors = ColumnDataSource(data=dict(base=self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                                     lower=self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] - \
                                                     self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]],
                                                     upper=self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] + \
                                                     self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        
                        #Here we plot the detrended flux:
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux_flat'][limit_mask[n]]+raw_plot_offset, ".k",
                                         alpha=0.5,size=0.75)
                        f_alls[n].circle(self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                         self.lc['bin_flux_flat'][binlimits[n][0]:binlimits[n][1]],
                                         legend="detrended",alpha=0.65,size=3.5)
                        errors = ColumnDataSource(data=dict(base=self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                                     lower=self.lc['bin_flux_flat'][binlimits[n][0]:binlimits[n][1]] - \
                                                     self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]],
                                                     upper=self.lc['bin_flux_flat'][binlimits[n][0]:binlimits[n][1]] + \
                                                     self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                    else:
                        #PLOTTING DETRENDED FLUX, NO BINNING
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux'][limit_mask[n]]+raw_plot_offset,
                                         legend="raw",alpha=0.65,size=3.5)
                        f_alls[n].circle(self.lc['time'][limit_mask[n]],
                                         self.lc['flux_flat'][limit_mask[n]],
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
                    if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                        #HERE WE BIN
                        f_all_resids[n].circle(self.lc['time'][limit_mask[n]],
                                               self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                               self.trans_to_plot['all']['med'][limit_mask[n]], color='black',
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
                        errors = ColumnDataSource(data=dict(base=self.lc['time'][limit_mask[n]],
                                          lower=self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                           self.trans_to_plot['all']['med'][limit_mask[n]] - self.lc['flux_err'][limit_mask[n]],
                                          upper=self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                           self.trans_to_plot['all']['med'][limit_mask[n]] + self.lc['flux_err'][limit_mask[n]]))
                        f_alls[n].add_layout(Whisker(source=errors, base='base', upper='upper',lower='lower',
                                                     line_color='#dddddd', line_alpha=0.5,
                                                     upper_head=TeeHead(line_color='#dddddd',line_alpha=0.5),
                                                     lower_head=TeeHead(line_color='#dddddd',line_alpha=0.5)))
                        f_all_resids[n].circle(self.lc['time'][limit_mask[n]],
                                               self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                               self.trans_to_plot['all']['med'][limit_mask[n]],
                                               alpha=0.65,
                                               size=3.5)

                else:
                    #Plotting detrended:
                    mean=np.nanmedian(self.trace['mean']) if hasattr(self,'trace') else self.init_soln['mean']
                    if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                        f_all_resids[n].circle(self.lc['time'][limit_mask[n]],
                                               self.lc['flux_flat'][limit_mask[n]] - mean - \
                                               self.trans_to_plot["all"]["med"][limit_mask[n]], color='black',
                                               zorder=-1000,alpha=0.5,
                                               size=0.75)
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
                        f_alls[n].circle(self.lc['time'][limit_mask[n]], 
                                         self.lc['flux_flat'][limit_mask[n]] - mean - \
                                         self.trans_to_plot["all"]["med"][limit_mask[n]],
                                         legend="raw data",alpha=0.65,size=3.5)

                f_all_resids[n].xaxis.axis_label = 'Time [BJD-'+str(int(self.lc['jd_base']))+']' #<- x-axis label
                f_alls[n].legend.location = 'bottom_right'
                f_alls[n].legend.background_fill_alpha = 0.1
                f_alls[n].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
                f_alls[n].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
                f_alls[n].xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
                
            else:
                #Matplotlib plot:
                if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                    f_alls[n].plot(self.lc['time'][limit_mask[n]], self.lc['flux'][limit_mask[n]] + raw_plot_offset,
                                   ".k", alpha=0.5,markersize=0.75, rasterized=raster)
                    f_alls[n].errorbar(self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                       self.lc['bin_flux'][binlimits[n][0]:binlimits[n][1]] + raw_plot_offset, 
                                       yerr=self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]],color='C1',
                                       fmt=".", label="raw", ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)
                else:
                    f_alls[n].errorbar(self.lc['time'][limit_mask[n]], self.lc['flux'][limit_mask[n]] + raw_plot_offset, 
                                       yerr=self.lc['flux_err'][binlimits[n][0]:binlimits[n][1]],color='C1',
                                       fmt=".", label="raw", ecolor='#dddddd', 
                                       alpha=0.5,markersize=3.5, rasterized=raster)

                if self.use_GP:
                    if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                        f_alls[n].plot(self.lc['time'][limit_mask[n]], self.lc['flux'][limit_mask[n]] - \
                                       self.gp_to_plot['gp_pred'][limit_mask[n]],
                                       ".k", alpha=0.5,markersize=0.75, rasterized=raster)
                        f_alls[n].errorbar(bin_detrend[:,0], bin_detrend[:,1], yerr=bin_detrend[:,2],color='C2',fmt=".",
                                           label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)
                        
                        #Plotting residuals:
                        f_all_resids[n].plot(self.lc['time'][limit_mask[n]],
                                             self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                             self.trans_to_plot['all']['med'][limit_mask[n]],
                                             ".k", alpha=0.5,markersize=0.75, rasterized=raster)

                        f_all_resids[n].errorbar(bin_resids[:,0],bin_resids[:,1], yerr=bin_resids[:,2], fmt=".", 
                                                 ecolor='#dddddd',alpha=0.5,markersize=0.75, rasterized=raster)
                    
                    else:
                        f_alls[n].errorbar(self.lc['time'][limit_mask[n]], 
                                           self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]], 
                                           yerr=self.lc['flux_err'][limit_mask[n]],color='C2', rasterized=raster,
                                           fmt=".", label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5)
                        
                        f_all_resids[n].errorbar(self.lc['time'][limit_mask[n]], 
                                                 self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                                 self.trans_to_plot['all']['med'][limit_mask[n]],
                                                 yerr=self.lc['flux_err'][limit_mask[n]], fmt=".", rasterized=raster,
                                                 ecolor='#dddddd',label="residuals", alpha=0.5,markersize=0.75)
                    #Plotting GP region and subtracted flux
                    f_alls[n].fill_between(self.lc['time'][limits[n][0]:limits[n][1]],
                                           self.gp_to_plot['gp_pred'][limits[n][0]:limits[n][1]] + raw_plot_offset - \
                                           2*self.gp_to_plot['gp_sd'][limits[n][0]:limits[n][1]],
                                           self.gp_to_plot['gp_pred'][limits[n][0]:limits[n][1]] + raw_plot_offset + \
                                           2*self.gp_to_plot['gp_sd'][limits[n][0]:limits[n][1]], rasterized=raster,
                                           color="C3", alpha=0.2,zorder=10)
                    f_alls[n].fill_between(self.lc['time'][limits[n][0]:limits[n][1]],
                                           self.gp_to_plot['gp_pred'][limits[n][0]:limits[n][1]] + raw_plot_offset - \
                                           self.gp_to_plot['gp_sd'][limits[n][0]:limits[n][1]],
                                           self.gp_to_plot['gp_pred'][limits[n][0]:limits[n][1]] + raw_plot_offset + \
                                           self.gp_to_plot['gp_sd'][limits[n][0]:limits[n][1]], rasterized=raster,
                                           color="C3", label="GP fit",alpha=0.3,zorder=11)


                else:
                    #GP not used.
                    if np.nanmedian(np.diff(self.lc['time'][limit_mask[n]]))<1/72:
                        #Plotting flat flux only
                        f_alls[n].plot(self.lc['time'][limit_mask[n]], self.lc['flux_flat'][limit_mask[n]],
                                       ".k",alpha=0.5,markersize=0.75, rasterized=raster)
                        f_alls[n].errorbar(self.lc['bin_time'][binlimits[n][0]:binlimits[n][1]],
                                           self.lc['bin_flux_flat'][binlimits[n][0]:binlimits[n][1]],
                                           yerr= self.lc['bin_flux_err'][binlimits[n][0]:binlimits[n][1]], rasterized=raster, 
                                           color='C2',fmt=".",label="detrended", ecolor='#dddddd', alpha=0.5,markersize=3.5)
                        #Plotting residuals:
                        f_all_resids[n].plot(self.lc['time'][limit_mask[n]],
                                             self.lc['flux'][limit_mask[n]] - self.gp_to_plot['gp_pred'][limit_mask[n]] - \
                                             self.trans_to_plot['all']['med'][limit_mask[n]],
                                             ".k",label="raw data", alpha=0.5,markersize=0.75, rasterized=raster)

                        f_all_resids[n].errorbar(bin_resids[:,0],bin_resids[:,1],yerr=bin_resids[:,2], fmt=".",
                                                 ecolor='#dddddd',label="residuals", alpha=0.5,
                                                 markersize=0.75, rasterized=raster)
                        
                    else:
                        f_alls[n].errorbar(self.lc['time'][limit_mask[n]], self.lc['flux_flat'][limit_mask[n]],
                                           yerr=self.lc['flux_err'][limit_mask[n]],fmt=".", label="detrended", 
                                           ecolor='#dddddd', alpha=0.5,markersize=3.5, rasterized=raster)
                        
                        f_all_resids[n].errorbar(self.lc['time'][limit_mask[n]], 
                                                 self.lc['flux_flat'][limit_mask[n]] - \
                                                 self.trans_to_plot['all']['med'][limit_mask[n]],
                                                 yerr=self.lc['flux_err'][limit_mask[n]], fmt=".",
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
                f_all_resids[n].set_xlim(self.lc['time'][limits[n][0]]-0.5,self.lc['time'][limits[n][1]-1]+0.5)
                if gap_lens[n]==np.max(gap_lens):
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
            f_all_resids[0].set_ylim(-1.5*resid_sd,1.5*resid_sd)

        #####################################
        #  Plotting individual transits
        #####################################
        maxdur=1.25*np.max([self.planets[ipl]['tdur'] for ipl in self.planets])
        self.lc['phase']={}
        for n,pl in enumerate(self.planets):
            if pl in self.multis:
                if hasattr(self,'trace'):
                    t0=np.nanmedian(self.trace['multi_t0s'][:,self.multis.index(pl)])
                    per=np.nanmedian(self.trace['multi_periods'][:,self.multis.index(pl)])
                elif hasattr(self,'init_soln'):
                    t0=self.init_soln['multi_t0s'][self.multis.index(pl)]
                    per=self.init_soln['multi_periods'][self.multis.index(pl)]
                #phase-folding onto t0
                self.lc['phase'][pl]=phase=(self.lc['time']-t0-0.5*per)%per-0.5*per
                for ns in range(len(limits)):
                    n_p_sta=np.ceil((t0-self.lc['time'][limits[ns][0]])/per)
                    n_p_end=(t0-self.lc['time'][limits[ns][1]-1])/per
                    
                    if (t0>limits[ns][0])&(t0<limits[ns][0]):
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
                for ns in range(len(limits)):
                    if (t0>limits[ns][0])&(t0<limits[ns][0]):
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
                for ns in range(len(limits)):
                    if (t0>limits[ns][0])&(t0<limits[ns][0]):
                        if interactive:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           marker="triangle", size=12.5, fill_color=pal[2+n], alpha=0.85)
                        else:
                            f_alls[ns].scatter([t0],[-1*self.min_trans-0.8*resid_sd-(resid_sd*n/len(self.planets))],
                                           "^", markersize=12.5, color=pal[2+n], alpha=0.85, rasterized=raster)

                    if (t0_2>limits[ns][0])&(t0_2<limits[ns][0]):
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
            
            phasebool=abs(self.lc['phase'][pl])<maxdur
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
                f_trans[n].plot(phaselc[:,0],phaselc[:,1], ".k",label="raw data",alpha=0.5,markersize=0.75, rasterized=raster)
                f_trans[n].errorbar(bin_phase[:,0],bin_phase[:,1], yerr=bin_phase[:,2], fmt='.', 
                                    alpha=0.75, markersize=5, rasterized=raster)
                f_trans_resids[n].plot(phaselc[:,0],
                                       phaselc[:,1]-self.trans_to_plot[pl]['med'][self.lc['mask']&phasebool], ".k",
                                         alpha=0.5,markersize=0.75, rasterized=raster)
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
                f_trans_resids[n].set_ylim(-2*resid_sd,2*resid_sd)
                f_trans[n].yaxis.tick_right()
                f_trans_resids[n].yaxis.tick_right()
                
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
            f_trans_resids[0].set_xlim(-0.8*maxdur,0.8*maxdur)
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
            if return_fig:
                return p
            
        else:
            if plot_loc is None:
                plt.savefig(self.savenames[0].replace('_mcmc.pickle','_period_dists.pdf'))
            else:
                plt.savefig(plot_loc)

    def PlotPeriods(self, plot_loc=None,log=True,nbins=50):
        assert hasattr(self,'trace')
        import seaborn as sns
        plot_pers=self.duos+self.monos
        if len(plot_pers)>0:
            plt.figure(figsize=(8.5,4.2))
            for npl, pl in enumerate(plot_pers):
                plt.subplot(1,len(plot_pers),npl+1)
                if pl in self.duos:
                    #Plotting lines
                    for n in range(len(self.trace['duo_periods_01'][0,:])):
                        # Density Plot and Histogram of all arrival delays                        
                        plt.plot([np.nanmedian(self.trace['duo_periods_'+pl][:,n]),np.nanmedian(self.trace['duo_periods_'+pl][:,n])],
                                 [-20.0,np.nanmedian(self.trace['logprob_marg_'+pl][:,n])/np.log(10)],linewidth=5.0)
                    print(np.min(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0)))
                    plt.title("Duo - ",pl)
                    plt.ylim(np.min(np.nanmedian(self.trace['logprob_marg_'+pl],axis=0))/np.log(10)-2.0,0.1)
                    plt.ylabel("$\log_{10}{p}$")
                    plt.xlabel("Period [d]")
                elif pl in self.monos:
                    plt.title("Mono - "+str(pl))
                    if 'logprob_marg_'+pl in self.trace.varnames:
                        #Checking seaborn version:
                        if log and float('.'.join(sns.__version__.split('.')[:2]))>=0.11:
                            sns.distplot(self.trace['mono_periods_'+pl].ravel(), hist=True, kde=True, bins=nbins, norm_hist=True,
                                         hist_kws={'edgecolor':'None','log':log,
                                                   'weights':np.exp(self.trace['logprob_marg_'+pl].ravel())},
                                         kde_kws={'linewidth': 2,'weights':np.exp(self.trace['logprob_marg_'+pl].ravel())})
                        else:
                            sns.distplot(self.trace['mono_periods_'+pl].ravel(), hist=True, kde=False, bins=nbins, norm_hist=True,
                                         hist_kws={'edgecolor':'None','log':log,
                                                   'weights':np.exp(self.trace['logprob_marg_'+pl].ravel())},
                                         kde_kws={'linewidth': 2})

                        #plt.hist(self.trace['mono_periods_'+pl].ravel(),bins=50,edgecolor='None',
                        #         weights=np.exp(self.trace['logprob_marg_'+pl].ravel()),density=True)
                    else:
                        sns.distplot(self.trace['mono_periods_'+pl].ravel(), hist=True, kde=True, bins=nbins, norm_hist=True,
                                     hist_kws={'edgecolor':'None','log':log},
                                     kde_kws={'linewidth': 3})
                        #plt.hist(self.trace['mono_periods_'+pl].ravel(),bins=50,edgecolor='None',
                        #         density=True)
                    if log:
                        #plt.yscale('log')
                        plt.ylabel("$\log_{10}{\\rm prob. density}$")                        
                        bins = np.histogram(self.trace['mono_periods_'+pl].ravel(),
                                            weights=np.exp(self.trace['logprob_marg_'+pl].ravel()),bins=nbins,density=True)
                        bin_mins=np.nanmin((bins[0]/np.nanmedian(np.diff(bins[1])))[bins[0]>0])
                        print(bin_mins)
                        plt.ylim(np.clip(0.01*bin_mins,1e-50,1.0),1.0)
                    else:
                        plt.ylabel("prob. density")
                    plt.xlabel("Period [d]")
            if plot_loc is None:
                plt.savefig(self.savenames[0].replace('_mcmc.pickle','_period_dists.pdf'))
            else:
                plt.savefig(plot_loc)
    def PlotTable(self,plot_loc=None):
        new_df=True
        assert hasattr(self,'trace')
        cols_to_remove=['gp_', '_gp', 'light_curve','__','mono_uniform_index',
                        'mono_period','mono_tdurs','mono_priors','mono_liks',
                        'duo_period','duo_tdurs','duo_priors','duo_liks',
                        'logprob_marg','mean', 'logrho_S']
        medvars=[var for var in self.trace.varnames if not np.any([icol in var for icol in cols_to_remove])]

        df = pm.summary(self.trace,var_names=medvars,stat_funcs={"5%": lambda x: np.percentile(x, 5),
                                                                 "-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                                 "median": lambda x: np.percentile(x, 50),
                                                                 "+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                                 "95%": lambda x: np.percentile(x, 95)})
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
            fig.savefig(self.savenames[0].replace('_mcmc.pickle','_table.pdf'))
        else:
            plt.savefig(plot_loc)
    
    def PlotCorner(self, varnames=["b", "ecc", "period", "r_pl","u_star","vrel"],
               savename=None, overwrite=False,savefileloc=None,returnfig=False,tracemask=None):
        #Plots Corner plot
        #Plotting corner of the parameters to see correlations
        import corner
        import matplotlib.pyplot as plt
        print("varnames = ",varnames)

        if not hasattr(self,'savenames'):
            self.savenames=self.GetSavename(how='save')
        
        if self.tracemask is None:
            self.tracemask=np.tile(True,len(self.trace['Rs']))
        if "u_star" in varnames:
            #Checking we're not using u_star and instead are using mission-specific params:
            varnames.remove("u_star")
            for u_col in ["u_star_tess","u_star_kep"]:
                if u_col in self.trace.varnames:
                    varnames+=[u_col]
        self.samples = pm.trace_to_dataframe(self.trace, varnames=varnames)
        self.amples=self.samples.loc[self.tracemask]

        plt.figure()
        if 'logprob_class' in self.trace.varnames:
            #If there's a logprob_class, that suggests we had gaps, so we need to do the marginalisation weighting:
            weight_samps = np.exp(self.trace["logprob_class"].flatten())
            fig = corner.corner(self.samples,weights=weight_samps);
        else:
            fig = corner.corner(self.samples)

        fig.savefig(self.savenames[0].replace('_mcmc.pickle','_corner.png'),dpi=250)

        if returnfig:
            return fig
        
    def LoadPickle(self,loadname=None):
        #Pickle file style: folder/TIC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
        if loadname is not None:
            print(self.savenames[0],"exists - loading")
            n_bytes = 2**31
            max_bytes = 2**31 - 1

            ## read
            bytes_in = bytearray(0)
            input_size = os.path.getsize(loadname)
            with open(self.savenames[0], 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            self.trace = pickle.loads(bytes_in)

        if not hasattr(self, 'savenames'):
            self.savenames=self.GetSavename(how='load')
        print(self.savenames[0],os.path.exists(self.savenames[0]))
        if os.path.exists(self.savenames[0]):
            print(self.savenames[0],"exists - loading")
            n_bytes = 2**31
            max_bytes = 2**31 - 1

            ## read
            bytes_in = bytearray(0)
            input_size = os.path.getsize(self.savenames[0])
            with open(self.savenames[0], 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            self.trace = pickle.loads(bytes_in)

    def SavePickle(self):
        if not hasattr(self,'savenames'):
            self.savenames=GetSavename(how='save')
        n_bytes = 2**31
        max_bytes = 2**31 - 1

        ## write
        bytes_out = pickle.dumps(self.trace)
        with open(self.savenames[0], 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])

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
                TessLDs=TessLDs.loc[unq_FeHs['FeH']==unq_FeHs[np.argmin(self.FeH-unq_FeHs.astype(float))]]
                
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
            self.savenames=self.GetSavename(how='save', suffix='mcmc.pickle')
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
        with open(self.savenames[0].replace('_mcmc.pickle','_table.txt'),'w') as file_to_write:
            file_to_write.write(outstring)
        #print("appending to file,",savename,"not yet supported")
        return outstring

def Run(ID, initdepth, initt0, mission='TESS', stellardict=None,n_draws=1200,
        overwrite=False,LoadFromFile=False,savefileloc=None, doplots=True,do_per_gap_cuts=True, **kwargs):
    #, cutDistance=0.0):
    """#PymcSingles - Run model
    Inputs:
    #  * ID - ID of star (in TESS, Kepler or K2)
    #  * initdepth - initial detected depth (for Rp guess)
    #  * initt0 - initial detection transit time
    #  * mission - TESS or Kepler/K2
    #  * stellardict - dictionary of stellar parameters. (alternatively taken from Gaia). With:
    #         Rs, Rs_err - 
    #         rho_s, rho_s_err - 
    #         Teff, Teff_err - 
    #         logg, logg_err - 
    #  * n_draws - number of samples for the MCMC to take
    #  * overwrite - whether to overwrite saved samples
    #  * LoadFromFile - whether to load the last written sample file
    #  * savefileloc - location of savefiles. If None, creates a folder specific to the ID
    # In KWARGS:
    #  * ALL INPUTS TO INIT_MODEL
    
    Outputs:
    # model - the PyMc3 model
    # trace - the samples
    # lc - a 3-column light curve with time, flux, flux_err
    """
    
    if not LoadFromFile:
        savenames=GetSavename(ID, mission, how='save', overwrite=overwrite, savefileloc=savefileloc)
    else:
        savenames=GetSavename(ID, mission, how='load', overwrite=overwrite, savefileloc=savefileloc)
    print(savenames)
    
    if os.path.exists(savenames[1].replace('_mcmc.pickle','.lc')) and os.path.exists(savenames[1].replace('_mcmc.pickle','_hdr.pickle')) and not overwrite:
        print("loading from",savenames[1].replace('_mcmc.pickle','.lc'))
        #Loading lc from file
        df=pd.read_csv(savenames[1].replace('_mcmc.pickle','.lc'))
        lc={col.replace('# ',''):df[col].values for col in df.columns}
        hdr=pickle.load(open(savenames[1].replace('_mcmc.pickle','_hdr.pickle'),'rb'))
    else:
        lc,hdr = openLightCurve(ID,mission,**kwargs)
        print([len(lc[key]) for key in list(lc.keys())])
        pd.DataFrame({key:lc[key] for key in list(lc.keys())}).to_csv(savenames[1].replace('_mcmc.pickle','.lc'))
        pickle.dump(hdr, open(savenames[1].replace('_mcmc.pickle','_hdr.pickle'),'wb'))

    if stellardict is None:
        Rstar, rhostar, Teff, logg, src = starpars.getStellarInfo(ID, hdr, mission, overwrite=overwrite,
                                                             fileloc=savenames[1].replace('_mcmc.pickle','_starpars.csv'),
                                                             savedf=True)
    else:
        if type(stellardict['Rs_err'])==tuple:
            Rstar=np.array([stellardict['Rs'],stellardict['Rs_err'][0],stellardict['Rs_err'][1]])
        else:
            Rstar=np.array([stellardict['Rs'],stellardict['Rs_err'],stellardict['Rs_err']])
        if type(stellardict['rho_s_err'])==tuple:
            rhostar = np.array([stellardict['rho_s'],stellardict['rho_s_err'][0],stellardict['rho_s_err'][1]])
        else:
            rhostar = np.array([stellardict['rho_s'],stellardict['rho_s_err'],stellardict['rho_s_err']])
        if type(stellardict['Teff_err'])==tuple:
            Teff = np.array([stellardict['Teff'],stellardict['Teff_err'][0],stellardict['Teff_err'][1]])
        else:
            Teff = np.array([stellardict['Teff'],stellardict['Teff_err'],stellardict['Teff_err']])
        if type(stellardict['logg_err'])==tuple:
            logg = np.array([stellardict['logg'],stellardict['logg_err'][0],stellardict['logg_err'][1]])
        else:
            logg = np.array([stellardict['logg'],stellardict['logg_err'],stellardict['logg_err']])
    print("Initialising transit model")
    print(lc['time'],type(lc['time']),type(lc['time'][0]))
    model, soln, lcmask, P_gap_cuts = init_model(lc,initdepth, initt0, Rstar, rhostar, Teff,
                                                 logg=logg, **kwargs)
    #initdur=None,n_pl=1,periods=None,per_index=-8/3,
    #assume_circ=False,use_GP=True,constrain_LD=True,ld_mult=1.5,
    #mission='TESS',LoadFromFile=LoadFromFile,cutDistance=cutDistance)
    print("Model loaded")


    #try:
    if LoadFromFile and not overwrite:
        self.LoadPickle(ID, mission, savenames[0])
    else:
        self.trace=None
   
    if self.trace is None:
        #Running sampler:
        np.random.seed(int(self.ID))
        with model:
            print(type(soln))
            print(soln.keys())
            trace = pm.sample(tune=np.clip(int(n_draws*0.66),400,4000), draws=n_draws, start=soln, chains=4,
                                  step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

        self.SavePickle()
            
    if do_per_gap_cuts:
        #Doing Cuts for Period gaps (i.e. where photometry rules out the periods of a planet)
        #Only taking MCMC positions in the trace where either:
        #  - P<0.5dur away from a period gap in P_gap_cuts[:-1]
        #  - OR P is greater than P_gap_cuts[-1]
        tracemask=np.tile(True,len(trace['t0'][:,0]))
        for n in range(len(P_gap_cuts)):
            #for each planet - only use monos
            periods=kwargs.get("periods",None)
            if periods is None or np.isnan(periods[n]) or periods[n]==0.0:
                #Cutting points where P<P_gap_cuts[-1] and P is not within 0.5Tdurs of a gap:
                gap_dists=np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)
                tracemask[(trace['period'][:,n]<P_gap_cuts[n][-1])*(gap_dists>0.5*np.nanmedian(trace['tdur'][:,n]))] = False
            
        #tracemask=np.column_stack([(np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)<0.5*np.nanmedian(trace['tdur'][:,n]))|(trace['period'][:,n]>P_gap_cuts[n][-1]) for n in range(len(P_gap_cuts))]).any(axis=1)
        print(np.sum(~tracemask),"(",int(100*np.sum(~tracemask)/len(tracemask)),") removed due to period gap cuts")
    else:
        tracemask=None
    if doplots:
        print("plotting")
        PlotLC(lc, trace, ID, mission=mission, savename=savenames[0].replace('mcmc.pickle','TransitFit.png'), lcmask=lcmask,tracemask=tracemask)
        PlotCorner(trace, ID, mission=mission, savename=savenames[0].replace('mcmc.pickle','corner.png'),tracemask=tracemask)
    
    if LoadFromFile and not overwrite and os.path.exists(savenames[0].replace('mcmc.pickle','results.txt')):
        with open(savenames[0].replace('mcmc.pickle','results.txt'), 'r', encoding='UTF-8') as file:
            restable = file.read()
    else:
        restable=ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
                              savename=savenames[0].replace('mcmc.pickle','results.txt'), overwrite=False,
                              savefileloc=None, tracemask=tracemask)
    return {'model':model, 'trace':trace, 'light_curve':lc, 'lcmask':lcmask, 'P_gap_cuts':P_gap_cuts, 'tracemask':tracemask,'restable':restable}

def RunFromScratch(ID, mission, tcen, tdur, ra=None, dec=None, 
                   mono_SNRthresh=6.0,
                   other_planet_SNRthresh=6.0, PL_ror_thresh=0.2):
    '''
    # Given only an ID, mission, tcen and tdur, this function will
    # - get the lightcurve and stellar parameters
    # - check if the candidate is a false positive
    # - Search for other transits and/or planets in the lightcurve
    # - Run the required Namaste model for all high-SNR planet candidates
    '''
    
    #Gets stellar info
    Rstar, rhostar, Teff, logg, src = starpars.getStellarInfo(ID, hdr, mission, overwrite=overwrite,
                                                             fileloc=savenames[1].replace('_mcmc.pickle','_starpars.csv'),
                                                             savedf=True)
    
    #Gets Lightcurve
    lc,hdr=openLightCurve(ID,mission,use_ppt=False)
    lc=tools.lcFlatten(lc,winsize=9*tdur,stepsize=0.1*tdur)
    
    #Runs Quick Model fit
    monoparams, interpmodel = MonoSearch.QuickMonoFit(lc,tc,dur,Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)
    
    #Checks to see if dip is due to background asteroid
    asteroidDeltaBIC=MonoSearch.AsteroidCheck(lc, monoparams, interpmodel)
    if asteroidDeltaBIC>6:
        planet_dic_1['01']['flag']='asteroid'
    
    #Checks to see if dip is combined with centroid
    centroidDeltaBIC=MonoSearch.CentroidCheck(lc, monoparams, interpmodel)
    if centroidDeltaBIC>6:
        planet_dic_1['01']['flag']='EB'

    #Searches for other dips in the lightcurve
    planet_dic_1=MonoSearch.SearchForSubsequentTransits(lc, interpmodel, tc, dur, Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)
    
    #Asses whether any dips are significant enough:
    if planet_dic_1['01']['SNR']>mono_SNRthresh:
        #Check if the Depth/Rp suggests we have a very likely EB, we search for a secondary
        if planet_dic_1['01']['rp_rs']<PL_ror_thresh:
            #Searching for other (periodic) planets in the system
            planet_dic_2=MonoSearch.SearchForOtherPlanets(lc, planet_dic_1['01'], SNRthresh=other_planet_SNRthresh)
            if len(planet_dic_2)>1:
                planet_dic_1['01']['flag']='multiplanet'
            else:
                planet_dic_1['01']['flag']='monoplanet'
        else:
            #Likely EB
            planet_dic_1['01']['flag']='EB'
    #If other dips exist, we need to figure out if there are possible integer periods to search between:
    
    #We then do an EB model here
    if planet_dic_1['01']['flag']=='EB':
        #Either doing Namaste model with "third light" switched on.
        print(" ")
    else:
        #If not, we have a planet.
        #Checking if monoplanet is single, double-with-gap, or periodic.
        if planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] == 'singlemono':
            #Monotransit?
            print(" ")
        elif planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] == 'multimono':
            #Two monotransits?
            print(" ")
        elif planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] == 'doublemono':
            #Monotransit with gap?
            print(" ")
        elif planet_dic_1['01']['flag']=='periodic':
            print(" ")

                    
    


def PlotCorner(trace, ID, mission='TESS', varnames=["b", "ecc", "period", "r_pl","u_star","vrel"],
               savename=None, overwrite=False,savefileloc=None,returnfig=False,tracemask=None):
    #Plotting corner of the parameters to see correlations
    import corner
    import matplotlib.pyplot as plt
    print("varnames = ",varnames)
    
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_corner.png', 
                             overwrite=overwrite, savefileloc=savefileloc)[0]
    
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    

    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    samples=samples.loc[tracemask]

    plt.figure()
    if 'logprob_class' in trace.varnames:
        #If there's a logprob_class, that suggests we had gaps, so we need to do the marginalisation weighting:
        weight_samps = np.exp(trace["logprob_class"].flatten())
        fig = corner.corner(samples,weights=weight_samps);
    else:
        weight_samps = np.exp(trace["logprob_class"].flatten())
        fig = corner.corner(samples)

    fig.savefig(savename,dpi=250)
    
    if returnfig:
        return fig
