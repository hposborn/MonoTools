import numpy as np
import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.io import ascii
from scipy.signal import savgol_filter

import pymc3 as pm
import theano.tensor as tt
from theano import printing, function, config
config.exception_verbosity='high'

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

from .stellar import starpars
MonoFit_path = os.path.dirname(os.path.abspath( __file__ ))

class monoModel():
    #The default monoModel class. This is what we will use to build a Pymc3 model

    def __init__(self, ID, lc, planets=None, mission='tess', LoadFromFile=False, overwrite=False, savefileloc=None):
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
                if planets[pl]['orbital_flag']=='multi':
                    self.add_multi(planets[pl])
                elif planets[pl]['orbital_flag']=='duo':
                    self.add_duo(planets[pl])
                elif planets[pl]['orbital_flag']=='mono':
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
        p_gaps=self.compute_period_gaps(pl_dic['tcen'],tdur=pl_dic['tdur'])
        pl_dic['per_gaps']=np.column_stack((p_gaps,p_gaps[:,1]-p_gaps[:,0]))
        self.planets[name]=pl_dic
        self.monos+=[name]
        print(self.planets[name]['per_gaps'])
        
    def compute_period_gaps(self,tcen,tdur,max_per=3000):
        # Given the time array, the t0 of transit, and the fact that another transit is not observed, 
        #   we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
        # In this case, a list of periods is returned, with all points within 0.5dur to be cut
        dist_from_t0=np.sort(abs(tcen-self.lc['time'][self.lc['mask']]))
        
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
    
    def GetSavename(self, how='load', suffix='mcmc.pickle'):
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
        if self.savefileloc is None:
            self.savefileloc=os.path.join(MonoFit_path,'data',self.id_dic[self.mission]+str(self.ID).zfill(11))
        if not os.path.isdir(self.savefileloc):
            os.system('mkdir '+self.savefileloc)
        pickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"*"+suffix))
        if how is 'load' and len(pickles)>1:
            #finding most recent pickle:
            date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
            datepickles=glob.glob(os.path.join(self.savefileloc,self.id_dic[self.mission]+str(self.ID).zfill(11)+"_"+date+"_*_"+suffix))
            if len(datepickles)>1:
                nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
            elif len(datepickles)==1:
                nsim=0
            elif len(datepickles)==0:
                print("problem - no saved mcmc files in correct format")
        elif how is 'load' and len(pickles)==1:
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

    def transit_predict(self,itime,sample):
        #Takes input sample (a dict either from init_soln, or from trace) and produces a transit model
        trans_pred=[]
        if 'mult' not in sample:
            sample['mult']=1.0
        
        if 'ecc' not in sample:
            orbit = xo.orbits.KeplerianOrbit(
                r_star=sample['Rs'], rho_star=sample['rho_S'],
                period=sample['period'], t0=sample['t0'], b=sample['b'])
        else:
            orbit = xo.orbits.KeplerianOrbit(
                r_star=sample['Rs'], rho_star=sample['rho_S'],
                period=sample['period'], t0=sample['t0'], b=sample['b'],
                ecc=sample['ecc'], omega=sample['omega'])

        if 'u_star_tess' in self.init_soln:
            outlc=xo.LimbDarkLightCurve(self.init_soln['u_star_tess']).get_light_curve(
                                                     orbit=orbit, r=sample['r'],
                                                     t=itime,
                                                     texp=np.nanmedian(np.diff(itime))
                                                     )/(self.lc['flux_unit']*sample['mult'])
            trans_pred+=[outlc.eval()]
        else:
            trans_pred+=[np.zeros((len(itime),len(self.planets)))]

        if 'u_star_kep' in self.init_soln:
            outlc = xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                     orbit=orbit, r=sample['r'],
                                                     t=itime,
                                                     texp=np.nanmedian(np.diff(itime))
                                                     )/(self.lc['flux_unit']*sample['mult'])
            trans_pred+=[outlc.eval()]
        else:
            trans_pred+=[np.zeros((len(itime),len(self.planets)))]

        #Here we're making an index for which telescope (kepler vs tess) did the observations,
        # then we multiply the output n_time x n_pl lightcurves by the n_time x n_pl x 2 index and sum along the 3D axis
        
        if not hasattr(self,'cad_index'):
            self.lc['cad_index']=np.zeros((len(self.lc['time']),len(self.planets),2))
            for ncad in range(len(self.cads)):
                if self.cads[ncad][0].lower()=='t':
                    self.lc['cad_index'][:,:,0]+=self.lc['flux_err_index'][:,ncad]
                elif self.cads[ncad][0].lower()=='k':
                    self.lc['cad_index'][:,:,1]+=self.lc['flux_err_index'][:,ncad]

        time_index = np.in1d(self.lc['time'],itime)
        if np.sum(time_index)!=len(itime):
            #We have time points that are not from the lightcurve - need to find nearest cadence type. One liner:
            new_cad_index = self.lc['cad_index'][np.argmin(abs(self.lc['time'][np.newaxis,:]-itime[:,np.newaxis]),axis=0),:]
        else:
            #indexing the cad_index to match the times we're generating data for:
            new_cad_index = self.lc['cad_index'][time_index,:]
        #print(np.dstack(trans_pred).shape)
        #print(new_cad_index.shape)
        return np.sum(np.dstack(trans_pred)*new_cad_index,axis=2)

    def gp_predict(self,itime,sample,return_var=True):
        part_mask=np.in1d(np.round(self.lc['time'],5),np.round(itime,5))
        if part_mask.sum()<50:
            #Not many points - maybe we have a fine grid?
            # in which case we should try to find nearby points and link them in to do the GP computation
            cad=np.nanmedian(np.diff(self.lc['time']))
            part_mask=np.sum(abs(self.lc['time'][np.newaxis,:]-itime[:,np.newaxis])<cad*0.4,axis=0)
        
        trans_pred=self.transit_predict(itime,sample)
        if np.sum(self.lc['mask'][part_mask])>0:
            with pm.Model() as gpmod:
                log_S0=pm.Normal('logS0',mu=float(sample['logS0']),sd=0.0000000001)
                log_w0=pm.Normal('logw0',mu=float(sample['logw0']),sd=0.0000000001)
                logs2=pm.Normal('logs2',mu=sample['logs2'].astype(float),shape=len(sample['logs2']),sd=0.0000000001)
                mean=pm.Normal('mean',mu=float(sample['mean']),sd=0.0000000001)
                    #'mean',upper=sample['mean']+0.001,lower=sample['mean']-0.001,testval=sample['mean'])
                part_gp = xo.gp.GP(xo.gp.terms.SHOTerm(log_S0=log_S0,
                                                       log_w0=log_w0,
                                                       Q=1/np.sqrt(2)),
                                   self.lc['time'][part_mask&self.lc['mask']].astype(np.float32),
                                   self.lc['flux_err'][part_mask&self.lc['mask']]**2 + \
                                   tt.dot(self.lc['flux_err_index'][part_mask&self.lc['mask']],
                                          tt.exp(logs2)),
                                   J=2)
                llk_gp = part_gp.log_likelihood(self.lc['flux'][part_mask&self.lc['mask']]-\
                                                np.sum(trans_pred,axis=-1)[self.lc['mask'][part_mask]]-\
                                                mean)
                #print(llk_gp.eval())
                #gp_pred = part_gp.predict(itime, return_var=return_var).eval()
                gp_pred = xo.eval_in_model(part_gp.predict(itime, return_var=return_var),
                                           {col:sample[col] for col in ['logS0','logw0','logs2',
                                                                        'logpower_interval__','logw0_interval__',
                                                                        'mean']})
            return gp_pred, trans_pred
        else:
            return np.zeros((len(itime),2)), trans_pred
        
    def break_times_up(self,itime,max_len=10000,min_length_part=2500):
        times=[itime]
        if len(itime)>max_len:
            #splitting time up for GP computation - doing so at natural gaps in the lc
            max_time_len=len(itime)
            
            while max_time_len>max_len:
                newtimes=[]
                for n in range(len(times)):
                    if len(times[n])>max_len:
                        middle_boost=5*(0.3-((times[n][:-1]+np.diff(times[n])-np.median(times[n]))/(times[n][-1]-times[n][0]))**2)
                        #Making sure there's no super-small pieces by not snipping near the start & end by setting diffs here to 0.0
                        middle_boost[:min_length_part]*=0.0
                        middle_boost[-1*min_length_part:]*=0.0
                        cut_n=np.argmax(np.diff(times[n])*middle_boost[n])
                        newtimes+=[times[n][:cut_n+1],times[n][cut_n+1:]]
                    else:
                        newtimes+=[times[n]]
                times=newtimes
                max_time_len=np.max([len(t) for t in times])
        return times

    def predict_all(self,itime,use_init=False,N_pred=25,max_len=12500,return_var=True):
        self.part_times = self.break_times_up(itime,max_len)
        
        #extend transit model to full lc time:
        gp_preds, trans_preds = [], []
        if use_init or not hasattr(self,'trace'):
            for t in self.part_times:
                gp_pred, trans_pred=self.gp_predict(t,self.init_soln)
                gp_preds+=[np.column_stack(gp_pred)]
                trans_preds+=[trans_pred]
            gp_preds, trans_preds = np.vstack(gp_preds), np.vstack(trans_preds)
        else:
            #Using samples from trace:
            if not hasattr(self,'tracemask'):
                self.PeriodGapCuts()
            vns=[var for var in self.trace.varnames if 'curve' not in var and 'pred' not in var]
            samples=pm.trace_to_dataframe(self.trace, varnames=vns)
            samples=samples.loc[self.tracemask].iloc[np.random.choice(np.sum(self.tracemask),N_pred,replace=False)]
            for rowid,row in samples.iterrows():
                gp_pred_is, trans_pred_is = [], []
                for t in self.part_times:
                    gp_pred_i, trans_pred_i=self.gp_predict(t,row)
                    gp_pred_is+=[np.column_stack(gp_pred_i)]
                    trans_pred_is+=[trans_pred_i]
                gp_pred_is, trans_pred_is = np.vstack(gp_pred_is), np.vstack(trans_pred_is)
                gp_preds+=[gp_pred_is]
                trans_preds+=[trans_pred_is]
            gp_preds = np.dstack((gp_pred_is))
            trans_preds = np.dstack((trans_pred_is))
        return gp_preds,trans_preds
    '''
        if with_transit:
            with pm.Model() as predict_model:
                if use_init:
                    pass
                else:
                    trans_pred = np.empty((N_pred, len(self.part_times[n])))
                    for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
                        trans_pred[i] = xo.eval_in_model(pm.math.sum(gen_lc(prefix='all_time_'), axis=-1), sample)
        
        return gp_preds,gp_vars
        gp_preds,gp_vars=[],[]
        self.part_gps=[]
        with self.model:
            if use_init:
                for n in range(len(self.part_times)):
                    print(self.lc['flux'][part_mask&self.lc['mask']],np.isnan(self.lc['flux'][part_mask&self.lc['mask']]).sum())
                    llk_gp = self.part_gps[-1].log_likelihood(self.lc['flux'][part_mask&self.lc['mask']]-\
                                                     trans_pred[part_mask&self.lc['mask']]-\
                                                     self.init_soln['mean'])
                    pred_mu, pred_var = self.part_gps[-1].predict(self.part_times[n], return_var=True)+self.init_soln['mean']
                    #pred_mu, pred_var = xo.eval_in_model(pred, self.init_soln)
                    gp_preds+=[pred_mu]
                    gp_vars+=[pred_var]
                gp_preds=np.hstack(gp_preds)
                gp_vars=np.hstack(gp_vars)
            else:
                for n in range(len(self.part_times)):
                    part_mask=np.in1d(self.lc['time'],self.part_times[n])
                    self.part_gps += [xo.gp.GP(self.gp.kernel, self.lc['time'][part_mask&self.lc['mask']].astype(np.float32),
                                           self.lc['flux_err'][part_mask&self.lc['mask']]**2 + \
                                           tt.dot(self.lc['flux_err_index'][part_mask&self.lc['mask']],
                                                  tt.exp(np.nanmedian(self.trace['logs2'],axis=0))),
                                           J=2)]
                    llk_gp = self.part_gps[-1].log_likelihood(self.lc['flux'][part_mask&self.lc['mask']])

                    pred_mu = np.empty((N_pred, len(self.part_times[n])))
                    pred_var = np.empty((N_pred, len(self.part_times[n])))
                    with self.model:
                        pred = self.part_gps[n].predict(self.part_times[n], return_var=True)+mean
                        for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
                            pred_mu[i], pred_var[i] = xo.eval_in_model(pred, sample)

                    gp_preds+=[pred_mu]
                    gp_vars+=[pred_var]
                gp_preds=np.vstack(gp_preds)
                gp_vars=np.vstack(gp_vars)
                
            
        '''
    def init_model(self,assume_circ=False,
                   use_GP=True,constrain_LD=True,ld_mult=3,useL2=True,
                   mission='TESS',FeH=0.0,LoadFromFile=False,cutDistance=4.5,
                   debug=True, pred_all_time=False):
        assert len(self.planets)>0
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
        
        print(len(self.planets),'monos:',self.monos,'multis:',self.multis,'duos:',self.duos)
        
        n_pl=len(self.planets)
        self.cads=np.unique(self.lc['cadence'])
        #In the case of different cadence/missions, we need to separate their respective errors to fit two logs2
        self.lc['flux_err_index']=np.column_stack([np.where(self.lc['cadence']==cad,1.0,0.0) for cad in self.cads])

        ######################################
        #   Creating telescope index func:
        ######################################
        if not hasattr(self,'tele_index'):
            #Here we're making an index for which telescope (kepler vs tess) did the observations,
            # then we multiply the output n_time array by the n_time x 2 index and sum along the 2nd axis

            self.lc['tele_index']=np.zeros((len(self.lc['time']),2))
            for ncad in range(len(self.cads)):
                if self.cads[ncad][0].lower()=='t':
                    self.lc['tele_index'][:,0]+=self.lc['flux_err_index'][:,ncad]
                elif self.cads[ncad][0].lower()=='k':
                    self.lc['tele_index'][:,1]+=self.lc['flux_err_index'][:,ncad]

        ######################################
        #   Masking out-of-transit flux:
        ######################################
        # To speed up computation, here we loop through each planet and add the region around each transit to the data to keep
        if cutDistance>0:
            speedmask=np.tile(False, len(self.lc['time']))
            for ipl in self.multis:
                phase=(self.lc['time']-self.planets[ipl]['tcen']-0.5*self.planets[ipl]['period'])%self.planets[ipl]['period']-0.5*self.planets[ipl]['period']
                speedmask+=abs(phase)<cutDistance*self.planets[ipl]['tdur']
            for ipl in self.monos:
                speedmask+=abs(self.lc['time']-self.planets[ipl]['tcen'])<cutDistance*self.planets[ipl]['tdur']
            for ipl in self.duos:
                #speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen'])<cutDistance]=True
                #speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen_2'])<cutDistance]=True
                for per in self.planets[ipl]['period_aliases']:
                    phase=(self.lc['time']-self.planets[ipl]['tcen']-0.5*per)%per-0.5*per
                    speedmask+=abs(phase)<cutDistance*self.planets[ipl]['tdur']
            self.lc['oot_mask']=self.lc['mask']&speedmask
            print(np.sum(speedmask),"points in new lightcurve, compared to ",np.sum(self.lc['mask'])," in original mask, leaving ",np.sum(self.lc['oot_mask']),"points in the lc")

        else:
            #Using all points in the 
            self.lc['oot_mask']=self.lc['mask']

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

            # The baseline flux
            mean=pm.Normal("mean",mu=np.median(self.lc['flux'][self.lc['mask']]),
                                  sd=np.std(self.lc['flux'][self.lc['mask']]))

            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0
            
            print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)

            ######################################
            #     Initialising Periods & tcens
            ######################################
            tcens=np.array([self.planets[pls]['tcen'] for pls in self.multis+self.monos+self.duos])
            tdurs=np.array([self.planets[pls]['tdur'] for pls in self.multis+self.monos+self.duos])
            print(tcens,tdurs)
            t0 = pm.Bound(pm.Normal, upper=tcens+tdurs*0.5, lower=tcens-tdurs*0.5)("t0",mu=tcens, sd=tdurs*0.05,
                                        shape=len(self.planets),testval=tcens)

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
                mono_periods={}
                mono_log_periods={}
                mono_ix_periods={}
                for pl in self.monos:
                    mono_log_periods[pl]=pm.Uniform("mono_uniforms_"+str(pl),0.0,1.0,
                                                    shape=len(self.planets[pl]['per_gaps'][:,0]))
                    mono_log_periods[pl]=pm.Deterministic("mono_logp_"+str(pl),mono_log_periods[pl] * \
                                                          self.planets[pl]['per_gaps'][:,2] * self.planets[pl]['per_gaps'][:,0])
                    mono_periods[pl]=pm.Deterministic("mono_period_"+str(pl),tt.exp(mono_log_periods[pl]))
                    '''
                    mono_log_periods[pl]=pm.Uniform("mono_logp_"+str(pl),
                                                    lower=self.planets[pl]['per_gaps'][ngap,0],
                                                    upper=self.planets[pl]['per_gaps'][ngap,1],
                                                    shape=len(self.planets[pl]['per_gaps'][:,0]))
                    mono_periods[pl]=pm.Deterministic("mono_period_"+str(pl),tt.exp(mono_log_periods[pl]))
                    
                    for ngap in range(len(self.planets[pl]['per_gaps'][:,0])):
                        #Using pareto with alpha=1.0 as p ~ -1*(alpha+1)
                        #ie prior on period is prop to 1/P (window function) x 1/P (occurrence flat in LnP) x Rs/a (added later)
                        mono_periods[pl][ngap]=pm.Bound(pm.Pareto,
                                                        lower=self.planets[pl]['per_gaps'][ngap,0],
                                                        upper=self.planets[pl]['per_gaps'][ngap,1]
                                                        )("mono_period_"+pl+'_'+str(int(ngap)), 
                                                          m=self.planets[pl]['per_gaps'][0,0],
                                                          alpha=1.0)
                    '''
            if len(self.duos)>0:
                #Again, in the case of a duotransit, we have a series of possible periods between two know transits.
                # TO model these we need to compute each and marginalise over them
                duo_periods={}
                tcens=np.array([self.planets[pls]['tcen'] for pls in self.duos])
                tcens2=np.array([self.planets[pls]['tcen_2'] for pls in self.duos])
                tdurs=np.array([self.planets[pls]['tdur'] for pls in self.duos])
                t0_second_trans = pm.Bound(pm.Normal, 
                                           upper=tcens2+tdurs*0.5, 
                                           lower=tcens2-tdurs*0.5)("t0_second_trans",mu=tcens2,
                                                                  sd=np.tile(0.2,len(self.duos)),
                                                                  shape=len(self.duos),testval=tcens2)
                for npl,pl in enumerate(self.duos):
                    duo_periods[pl]=pm.Deterministic("duo_period_"+pl,
                                                     abs(t0_second_trans-t0[-1*(len(self.duos)+npl)])/self.planets[pl]['period_int_aliases'])
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

                
            ######################################
            #     Initialising R_p & b
            ######################################
            # The Espinoza (2018) parameterization for the joint radius ratio and
            # impact parameter distribution
            rpls=np.array([self.planets[pls]['r_pl'] for pls in self.multis+self.monos+self.duos])/(109.1*self.Rstar[0])
            bs=np.array([self.planets[pls]['b'] for pls in self.multis+self.monos+self.duos])
            if useL2:
                #EB case as second light needed:
                r, b = xo.distributions.get_joint_radius_impact(
                    min_radius=0.001, max_radius=1.25,
                    testval_r=rpls, testval_b=bs)
            else:
                r, b = xo.distributions.get_joint_radius_impact(
                    min_radius=0.001, max_radius=0.25,
                    testval_r=rpls, testval_b=bs)

            r_pl = pm.Deterministic("r_pl", r * Rs * 109.1)
            pm.Potential("logr_potential",tt.log(r_pl))

            ######################################
            #     Initialising Limb Darkening
            ######################################
            # Here we either constrain the LD params given the stellar info, OR we let exoplanet fit them
            if len(np.unique([c[0] for c in self.cads]))==1:
                if constrain_LD:
                    n_samples=1200
                    # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.
                
                    #Single mission
                    if np.unique([c[0] for c in self.cads])[0].lower()=='t':
                        ld_dists=self.getLDs(n_samples=3000,mission='tess')
                        u_star_tess = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_tess", 
                                                    mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                                    sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
                    elif np.unique([c[0] for c in self.cads])[0].lower()=='k':
                        ld_dists=self.getLDs(n_samples=3000,mission='kepler')
                        u_star_kep = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_kep", 
                                                    mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                                    sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))

                else:
                    if self.cads[0][0].lower()=='t':
                        u_star_tess = xo.distributions.QuadLimbDark("u_star_tess", testval=np.array([0.3, 0.2]))
                    elif self.cads[0][0].lower()=='k':
                        u_star_kep = xo.distributions.QuadLimbDark("u_star_kep", testval=np.array([0.3, 0.2]))

            else:
                if constrain_LD:
                    n_samples=1200
                    #Multiple missions - need multiple limb darkening params:
                    ld_dist_tess=self.getLDs(n_samples=3000,mission='tess')

                    u_star_tess = pm.Bound(pm.Normal, 
                                           lower=0.0, upper=1.0)("u_star_tess", 
                                                                 mu=np.clip(np.nanmedian(ld_dist_tess,axis=0),0,1),
                                                                 sd=np.clip(ld_mult*np.nanstd(ld_dist_tess,axis=0),0.05,1.0), 
                                                                 shape=2,
                                                                 testval=np.clip(np.nanmedian(ld_dist_tess,axis=0),0,1))
                    ld_dist_kep=self.getLDs(n_samples=3000,mission='tess')

                    u_star_kep = pm.Bound(pm.Normal, 
                                           lower=0.0, upper=1.0)("u_star_kep", 
                                                                 mu=np.clip(np.nanmedian(ld_dist_kep,axis=0),0,1),
                                                                 sd=np.clip(ld_mult*np.nanstd(ld_dist_kep,axis=0),0.05,1.0), 
                                                                 shape=2,
                                                                 testval=np.clip(np.nanmedian(ld_dist_kep,axis=0),0,1))
                else:
                    # The Kipping (2013) parameterization for quadratic limb darkening paramters
                    u_star_tess = xo.distributions.QuadLimbDark("u_star_tess", testval=np.array([0.3, 0.2]))
                    u_star_kep = xo.distributions.QuadLimbDark("u_star_kep", testval=np.array([0.3, 0.2]))
            
            ######################################
            #     Initialising GP kernel
            ######################################
            log_flux_std=np.array([np.log(np.std(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==c)])) for c in self.cads]).ravel()
            logs2 = pm.Normal("logs2", mu = 2*log_flux_std, sd = np.tile(2.0,len(log_flux_std)), shape=len(log_flux_std))

            if use_GP:
                # Transit jitter & GP parameters
                #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
                lcrange=self.lc['time'][self.lc['oot_mask']][-1]-self.lc['time'][self.lc['oot_mask']][0]
                min_cad = np.min([np.nanmedian(np.diff(self.lc['time'][self.lc['oot_mask']&(self.lc['cadence']==c)])) for c in self.cads])
                #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
                logw0 = pm.Uniform("logw0",lower=np.log((2*np.pi)/(4*lcrange)), 
                                   upper=np.log((2*np.pi)/min_cad),testval=np.log((2*np.pi)/(lcrange)))

                # S_0 directly because this removes some of the degeneracies between
                # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
                maxpower=np.log(np.nanmedian(abs(np.diff(self.lc['flux'][self.lc['oot_mask']]))))+1
                logpower = pm.Uniform("logpower",lower=-20,upper=maxpower,testval=maxpower-6)
                print("input to GP power:",maxpower-1)
                logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

                # GP model for the light curve
                kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))

            if not assume_circ:
                # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, shape=n_pl,
                                  testval=np.tile(0.05,n_pl))
                omega = xo.distributions.Angle("omega", shape=n_pl, testval=np.tile(0.5,n_pl))

            if use_GP:
                self.gp = xo.gp.GP(kernel, self.lc['time'][self.lc['oot_mask']].astype(np.float32),
                                   self.lc['flux_err'][self.lc['oot_mask']]**2 + \
                                   tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)),
                                   J=2)
            
            ################################################
            #     Creating function to generate transits
            ################################################
            def gen_lc(i_orbit,i_r,n_pl,mask=None,prefix=''):
                # Short method to create stacked lightcurves, given some input time array and some input cadences:
                # This function is needed because we may have 
                #   -  1) multiple cadences and 
                #   -  2) multiple telescopes (and therefore limb darkening coefficients)
                trans_pred=[]
                mask = ~np.isnan(self.lc['time']) if mask is None else mask
                if np.sum(self.lc['tele_index'][:,0])>0:
                    trans_pred+=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                             orbit=i_orbit, r=i_r,
                                                             t=self.lc['time'][mask],
                                                             texp=np.nanmedian(np.diff(self.lc['time'][mask]))
                                                             )/(self.lc['flux_unit']*mult)]
                else:
                    trans_pred+=[tt.zeros(( len(self.lc['time'][mask]),n_pl ))]

                if np.sum(self.lc['tele_index'][:,1])>0:
                    trans_pred+=[xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                             orbit=i_orbit, r=i_r,
                                                             t=self.lc['time'][mask],
                                                             texp=30/1440
                                                             )/(self.lc['flux_unit']*mult)]
                else:
                    trans_pred+=[tt.zeros(( len(self.lc['time'][mask]),n_pl ))]
                # transit arrays (ntime x n_pls x 2) * telescope index (ntime x n_pls x 2), summed over dimension 2
                return pm.Deterministic(prefix+"light_curves", 
                                        tt.sum(tt.stack(trans_pred,axis=-1) * \
                                               self.lc['tele_index'][self.lc['oot_mask']][:,np.newaxis,:],
                                               axis=-1))
            
            ################################################
            #     Analysing Multiplanets
            ################################################
            if len(self.multis)>0:
                multi_inds=np.array([pl in self.multis for pl in self.multis+self.monos+self.duos])
                if assume_circ:
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        period=multi_periods, t0=t0[multi_inds], b=b[multi_inds])
                else:
                    # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        ecc=ecc[multi_inds], omega=omega[multi_inds],
                        period=multi_periods, t0=t0[multi_inds], b=b[multi_inds])
                #Generating lightcurves using pre-defined gen_lc function:
                multi_mask_light_curves = gen_lc(multi_orbit,r[multi_inds],
                                                 len(self.multis),mask=self.lc['oot_mask'],prefix='mask_')
                multi_mask_light_curve = pm.math.sum(multi_mask_light_curves, axis=-1) #Summing lightcurve over n planets
            else:
                multi_mask_light_curve = tt.alloc(0.0,np.sum(self.lc['oot_mask']))
                print(multi_mask_light_curve.shape.eval())
                #np.zeros_like(self.lc['flux'][self.lc['oot_mask']])
                
            ################################################
            #     Marginalising over Duo periods
            ################################################
            if len(self.duos)>0:
                duo_per_info={}
                for nduo,duo in enumerate(self.duos):
                    print("#Marginalising over ",len(self.planets[duo]['period_int_aliases'])," period aliases for ",duo)

                    #Marginalising over each possible period
                    #Single planet with two transits and a gap
                    duo_per_info[duo]={'logpriors':[],
                                        'logliks':[],
                                        'lcs':[]}
                    
                    duo_ind=np.where([pl==duo for pl in self.multis+self.monos+self.duos])[0][0]

                    for i,p_int in enumerate(self.planets[duo]['period_int_aliases']):
                        with pm.Model(name="duo_"+duo+"_per_{0}".format(i), model=model) as submodel:
                            # Set up a Keplerian orbit for the planets
                            if assume_circ:
                                duoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    period=duo_periods[duo][i], t0=t0[duo_ind], b=b[duo_ind])
                            else:
                                duoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    ecc=ecc[duo_ind], omega=omega[duo_ind],
                                    period=duo_periods[duo][i], t0=t0[duo_ind], b=b[duo_ind])
                            print(self.lc['time'][self.lc['oot_mask']],np.sum(self.lc['oot_mask']))
                            
                            # Compute the model light curve using starry
                            duo_mask_light_curves_i = gen_lc(duoorbit,r[duo_ind],1,
                                                           mask=self.lc['oot_mask'],prefix='duo_mask_'+duo+'_')
                            
                            #Summing lightcurve over n planets
                            duo_per_info[duo]['lcs'] += [tt.sum(duo_mask_light_curves_i,axis=1)]
                                #pm.math.sum(duo_mask_light_curves_i, axis=-1)]
                            
                            duo_per_info[duo]['logpriors'] +=[tt.log(duoorbit.dcosidb) - 2 * tt.log(duo_periods[duo][i])]
                            #print(duo_mask_light_curves_i.shape.eval({}))
                            #print(duo_per_info[duo]['lcs'][-1].shape.eval({}))

                            #sum_lcs = (duo_mask_light_curve+multi_mask_light_curve) + mean
                            other_models = multi_mask_light_curve + mean
                            comb_models = duo_per_info[duo]['lcs'][-1] + other_models
                            resids = self.lc['flux'][self.lc['oot_mask']] - comb_models
                            if use_GP:
                                duo_per_info[duo]['logliks']+=[self.gp.log_likelihood(resids)]
                            else:
                                new_yerr = self.lc['flux_err'][self.lc['oot_mask']]**2 + \
                                           tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)),
                                duo_per_info[duo]['logliks']+=[tt.sum(pm.Normal.dist(mu=0.0,
                                                                                     sd=new_yerr
                                                                                    ).logp(resids))]
                    print(tt.stack(duo_per_info[duo]['logliks']))
                    print(tt.stack(duo_per_info[duo]['logpriors']))
                    # Compute the marginalized probability and the posterior probability for each period
                    logprobs = tt.stack(duo_per_info[duo]['logpriors']).squeeze() + \
                               tt.stack(duo_per_info[duo]['logliks']).squeeze()
                    print(logprobs.shape)
                    logprob_marg = pm.math.logsumexp(logprobs)
                    print(logprob_marg.shape)
                    duo_per_info[duo]['logprob_class'] = pm.Deterministic("logprob_class_"+duo, logprobs - logprob_marg)
                    pm.Potential("logprob_"+duo, logprob_marg)
                    
                    print(len(duo_per_info[duo]['lcs']))
                    
                    # Compute the marginalized light curve
                    duo_per_info[duo]['marg_lc']=pm.Deterministic("light_curve_"+duo,
                                                                  pm.math.dot(tt.stack(duo_per_info[duo]['lcs']).T,
                                                                              tt.exp(duo_per_info[duo]['logprob_class'])))
                #Stack the marginalized lightcurves for all duotransits:
                duo_mask_light_curves=pm.Deterministic("duo_mask_light_curves",
                                                  tt.stack([duo_per_info[duo]['marg_lc'] for duo in self.duos]))
                duo_mask_light_curve=pm.Deterministic("duo_mask_light_curve",tt.sum(duo_mask_light_curves,axis=0))
            else:
                duo_mask_light_curve = tt.alloc(0.0,np.sum(self.lc['oot_mask']))

            ################################################
            #     Marginalising over Mono gaps
            ################################################
            if len(self.monos)>0:
                mono_gap_info={}
                for nmono,mono in enumerate(self.monos):
                    print("#Marginalising over ",len(self.planets[mono]['per_gaps'])," period gaps for ",mono)
                    
                    #Single planet with one transits and multiple period gaps
                    mono_gap_info[mono]={'logliks':[]}
                    mono_ind=np.where([pl==mono for pl in self.multis+self.monos+self.duos])[0][0]
                    
                    # Set up a Keplerian orbit for the planets
                    print(r[mono_ind].ndim,tt.tile(r[mono_ind],len(self.planets[mono]['per_gaps'][:,0])).ndim)

                    if assume_circ:
                        monoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            period=mono_periods[mono], 
                            t0=tt.tile(t0[mono_ind],len(self.planets[mono]['per_gaps'][:,0])),
                            b=tt.tile(b[mono_ind],len(self.planets[mono]['per_gaps'][:,0])))
                    else:
                        monoorbit = xo.orbits.KeplerianOrbit(
                            r_star=Rs, rho_star=rho_S,
                            ecc=tt.tile(ecc[mono_ind],len(self.planets[mono]['per_gaps'][:,0])),
                            omega=tt.tile(omega[mono_ind],len(self.planets[mono]['per_gaps'][:,0])),
                            period=mono_periods[mono],
                            t0=tt.tile(t0[mono_ind],len(self.planets[mono]['per_gaps'][:,0])),
                            b=tt.tile(b[mono_ind],len(self.planets[mono]['per_gaps'][:,0])))
                    
                    # Compute the model light curve using starry
                    mono_gap_info[mono]['lc'] = gen_lc(monoorbit, tt.tile(r[mono_ind],len(self.planets[mono]['per_gaps'][:,0])),
                                                    len(self.planets[mono]['per_gaps'][:,0]),
                                                    mask=self.lc['oot_mask'],prefix='mono_mask_'+mono+'_')
                    
                    #Priors - we have an occurrence rate prior (~1/P), a geometric prior (1/distance in-transit = dcosidb)
                    # a window function log(1/P) -> -1*logP and  a factor for the width of the period bin - i.e. log(binsize)
                    #mono_gap_info[mono]['logpriors'] = 0.0
                    mono_gap_info[mono]['logpriors'] = tt.log(monoorbit.dcosidb) - \
                                                        2 * mono_log_periods[mono] + \
                                                        tt.log(self.planets[mono]['per_gaps'][:,2])
                    
                    other_models = duo_mask_light_curve + multi_mask_light_curve + mean
                    
                    #Looping over each period gap to produce loglik:
                    for i,gap_pers in enumerate(self.planets[mono]['per_gaps']):
                        with pm.Model(name="mono_"+mono+"_per_{0}".format(i), model=model) as submodel:
                            comb_models = mono_gap_info[mono]['lc'][:,i] + other_models
                            resids = self.lc['flux'][self.lc['oot_mask']] - comb_models
                            if use_GP:
                                mono_gap_info[mono]['logliks']+=[self.gp.log_likelihood(resids)]
                            else:
                                new_yerr = self.lc['flux_err'][self.lc['oot_mask']]**2 + \
                                           tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)),
                                mono_gap_info[mono]['logliks']+=[tt.sum(pm.Normal.dist(mu=0.0,sd=new_yerr).logp(resids))]
                    
                    # Compute the marginalized probability and the posterior probability for each period gap
                    logprobs = mono_gap_info[mono]['logpriors'] + tt.stack(mono_gap_info[mono]['logliks'])
                    logprob_marg = pm.math.logsumexp(logprobs)
                    mono_gap_info[mono]['logprob_class'] = pm.Deterministic("logprob_class_"+mono, logprobs - logprob_marg)
                    pm.Potential("logprob_"+mono, logprob_marg)

                    # Compute the marginalized light curve
                    mono_gap_info[mono]['marg_lc']=pm.Deterministic("light_curve_"+mono,
                                                                    pm.math.dot(mono_gap_info[mono]['lc'],
                                                                                tt.exp(mono_gap_info[mono]['logprob_class'])))
                #Stack the marginalized lightcurves for all monotransits:
                mono_mask_light_curves_all=pm.Deterministic("mono_mask_light_curves_all",
                                                   tt.stack([mono_gap_info[mono]['marg_lc'] for mono in self.monos]))
                mono_mask_light_curve=pm.Deterministic("mono_mask_light_curve",tt.sum(mono_mask_light_curves_all,axis=0))

            else:
                mono_mask_light_curve = tt.alloc(0.0,np.sum(self.lc['oot_mask']))

            ################################################
            #            Compute predicted LCs:
            ################################################
            #Now we have lightcurves for each of the possible parameters we want to marginalise, we need to sum them
            print(tt.stack((mono_mask_light_curve,multi_mask_light_curve)))
            print(tt.stack((mono_mask_light_curve,duo_mask_light_curve)))
            mask_light_curve = pm.Deterministic("mask_light_curve", tt.sum(tt.stack((duo_mask_light_curve,
                                                                                    multi_mask_light_curve,
                                                                                    mono_mask_light_curve)),axis=0))
            if use_GP:
                total_llk = pm.Deterministic("total_llk",self.gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - \
                                                                                mask_light_curve - mean))
                llk_gp = pm.Potential("llk_gp", total_llk)
                mask_gp_pred = pm.Deterministic("mask_gp_pred", self.gp.predict(return_var=False))
                
                if pred_all_time:
                    gp_pred = pm.Deterministic("gp_pred", self.gp.predict(self.lc['time'][self.lc['mask']],
                                                                          return_var=False))
            else:
#gp = GP(kernel, t, tt.dot(newyerr,(1+tt.exp(ex_errs)))**2)
                pm.Normal("obs", mu=mask_light_curve + mean, 
                          sd=tt.sqrt(tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)) + \
                                     self.lc['flux_err_index'][self.lc['oot_mask']]**2),
                          observed=self.lc['flux'][self.lc['oot_mask']])

            tt.printing.Print('r_pl')(r_pl)
            #tt.printing.Print('t0')(t0)
            '''
            print(P_min,t0,type(x[self.lc['oot_mask']]),x[self.lc['oot_mask']][:10],np.nanmedian(np.diff(x[self.lc['oot_mask']])))'''
            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point
            print(model.test_point)
            
            ################################################
            #               Optimizing:
            ################################################

            #Setting up optimization depending on what planet models we have:
            initvars0=[r, b]
            initvars1=[logs2]
            initvars2=[r, b, t0, rho_S]
            initvars3=[]
            initvars4=[r, b]
            if len(self.multis)>1:
                initvars1+=[multi_periods]
                initvars4+=[multi_periods]
            if len(self.monos)>1:
                for pl in self.monos:
                    for n in range(len(self.planets[pl]['per_gaps'][:,0])):
                        initvars1 += [mono_periods[pl][n]]
                        initvars4 += [mono_periods[pl][n]]
                        #exec("initvars1 += [mono_period_"+pl+"_"+str(int(n))+"]")
                        #exec("initvars4 += [mono_period_"+pl+"_"+str(int(n))+"]")
            if len(self.duos)>1:
                #for pl in self.duos:
                #    eval("initvars1+=[duo_period_"+pl+"]")
                for pl in self.duos:
                    initvars1 += [duo_periods[pl]]
                    initvars4 += [duo_periods[pl]]
                    #exec("initvars1 += [duo_period_"+pl+"]")
                    #exec("initvars4 += [duo_period_"+pl+"]")
                initvars2+=['t0_second_trans']
                initvars4+=['t0_second_trans']
            if len(self.multis)>1:
                initvars1 += [multi_periods]
                initvars4 += [multi_periods]
            if not assume_circ:
                initvars2+=[ecc, omega]
            if use_GP:
                initvars3+=[logs2, logpower, logw0, mean]
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
    
    def RunMcmc(self, n_draws=250, plot=True, do_per_gap_cuts=True, LoadFromFile=False, **kwargs):
        if LoadFromFile and not self.overwrite:
            self.LoadPickle()
            print("LOADED MCMC")

        if not hasattr(self,'trace'):
            assert hasattr(self,'init_soln')
            #Running sampler:
            np.random.seed(int(self.ID))
            with self.model:
                print(type(self.init_soln))
                print(self.init_soln.keys())
                self.trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=self.init_soln, chains=4,
                                       step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

            self.SavePickle()
        if do_per_gap_cuts:
            self.PeriodGapCuts()
        
        if plot:
            print("plotting")
            self.PlotMono()
            self.PlotCorner()
            self.PlotMonoInteractive()
        
        if LoadFromFile and not self.overwrite and os.path.exists(savenames[0].replace('mcmc.pickle','results.txt')):
            with open(savenames[0].replace('mcmc.pickle','results.txt'), 'r', encoding='UTF-8') as file:
                restable = file.read()
        else:
            restable=self.ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
                                       savename=savenames[0].replace('mcmc.pickle','results.txt'), overwrite=False,
                                       savefileloc=None, tracemask=tracemask)

            #tracemask=np.column_stack([(np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)<0.5*np.nanmedian(trace['tdur'][:,n]))|(trace['period'][:,n]>P_gap_cuts[n][-1]) for n in range(len(P_gap_cuts))]).any(axis=1)
            #print(np.sum(~tracemask),"(",int(100*np.sum(~tracemask)/len(tracemask)),") removed due to period gap cuts")
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
        '''
        if not LoadFromFile:
            savenames=GetSavename(how='save')
        else:
            savenames=GetSavename(how='load')
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
        init_model(lc,initdepth, initt0, Rstar, rhostar, Teff,
                                                     logg=logg, **kwargs)
        #initdur=None,n_pl=1,periods=None,per_index=-8/3,
        #assume_circ=False,use_GP=True,constrain_LD=True,ld_mult=1.5,
        #mission='TESS',LoadFromFile=LoadFromFile,cutDistance=cutDistance)
        print("Model loaded")


        if trace is None:
            #Running sampler:
            np.random.seed(int(self.ID))
            with model:
                print(type(soln))
                print(soln.keys())
                trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=soln, chains=4,
                                      step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

            self.SavePickle(trace, ID, mission, savenames[0])

        if do_per_gap_cuts:
            #Doing Cuts for Period gaps (i.e. where photometry rules out the periods of a planet)
            #Only taking MCMC positions in the trace where either:
            #  - P<0.5dur away from a period gap in P_gap_cuts[:-1]
            #  - OR P is greater than P_gap_cuts[-1]
            if not hasattr(self,'tracemask'):
                self.tracemask=np.tile(True,len(self.trace['t0'][:,0]))
            
            for npl,pl in enumerate(self.multis+self.monos+self.duos):
                if pl in self.monos:
                    per_gaps=compute_period_gaps(np.nanmedian(trace['t0'][:,npl]),lc['time']['mask'])
                    #for each planet - only use monos
                    if len(per_gaps)>1:
                        #Cutting points where P<P_gap_cuts[-1] and P is not within 0.5Tdurs of a gap:
                        gap_dists=np.nanmin(abs(trace['period'][:,npl][:,np.newaxis]-per_gaps[:-1][np.newaxis,:]),axis=1)
                        self.tracemask[(trace['period'][:,npl]<per_gaps[-1])*(gap_dists>0.5*np.nanmedian(trace['tdur'][:,npl]))] = False
        if doplots:
            print("plotting")
            PlotLC(lc, trace, ID, mission=mission, savename=savenames[0].replace('mcmc.pickle','TransitFit.png'), lcmask=lcmask,tracemask=tracemask)
            PlotCorner(trace, ID, mission=mission, savename=savenames[0].replace('mcmc.pickle','corner.png'),tracemask=tracemask)

            #tracemask=np.column_stack([(np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)<0.5*np.nanmedian(trace['tdur'][:,n]))|(trace['period'][:,n]>P_gap_cuts[n][-1]) for n in range(len(P_gap_cuts))]).any(axis=1)
            print(np.sum(~tracemask),"(",int(100*np.sum(~tracemask)/len(tracemask)),") removed due to period gap cuts")
        elif not hasattr(self,'tracemask'):
            self.tracemask=None
            
        
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
        return {'model':model, 'trace':trace, 'light_curve':lc, 'lcmask':lcmask, 'P_gap_cuts':P_gap_cuts, 'tracemask':tracemask,'restable':restable}'''
        
    def PlotMonoInteractive(self):
        #Plots bokeh figure
        from bokeh.plotting import figure, output_file, save

        if not hasattr(self,'savenames'):
            self.GetSavename(how='save')
        savename=self.savenames[0].replace('_mcmc.pickle','_transit_fit.html')
        print(savename)

        output_file(savename)

        #Initialising figure:
        p = figure(plot_width=1000, plot_height=600,title=str(self.ID)+" Transit Fit")

        #Finding if there's a single enormous gap in the lightcurve:
        x_gap=np.max(np.diff(self.lc['time']))>10
        if x_gap:
            print(" GAP IN X OF ",np.argmax(np.diff(lc['time'])))
            f_all_1=figure(width=360, plot_height=400, title=None)
            f_all_2=figure(width=360, plot_height=400, title=None)
            f_all_resid_1=figure(width=360, plot_height=150, title=None)
            f_all_resid_2=figure(width=360, plot_height=150, title=None)
        else:
            f_all=figure(width=720, plot_height=400, title=None)
            f_all_resid=figure(width=720, plot_height=150, title=None)
        
        #In the "duo" case, we have a complicated trace, so we need to remedy this:
        if "light_curves" in self.trace:
            pred = self.trace["light_curves"][self.tracemask,:,:]
        elif len(self.duos)==1:
            pred=[]
            for n in range(len(self.planets[self.duos[0]]['period_int_aliases'])):
                pred+=[self.trace["per_"+str(n)+"_light_curves"][self.tracemask,:,:]]
            pred=np.vstack(pred)
        elif len(self.duos)==2:
            pred=[]
            for n1 in range(len(self.planets[self.duos[0]]['period_int_aliases'])):
                for n2 in range(len(self.planets[self.duos[1]]['period_int_aliases'])):
                    pred+=[self.trace["per_"+str(n1)+"_"+str(n2)+"_light_curves"][self.tracemask,:,:]]
            pred=np.vstack(pred)
        #Need to check how many planets are here:
        pred = np.percentile(pred, [16, 50, 84], axis=0)

        # Compute the GP prediction
        if 'gp_pred' in self.trace:
            gp_pred = np.median(self.trace["gp_pred"][self.tracemask,:] + self.trace["mean"][self.tracemask, None], axis=0)
        elif type(self.gp)==dict:
            gp_pred=np.zeros(len(self.lc['time']))
            for nc,cad in enumerate(self.cads):
                pred[self.lc['cadence']==cad] = self.gp[cad].predict(self.lc['flux'][self.lc['cadence']==cad]-\
                                                                      pred[self.lc['cadence']==cad,1],
                                                                     self.lc['time'][self.lc['cadence']==cad],return_cov=False)
            gp_pred += np.nanmedian(self.trace["mean"][self.tracemask])
        elif type(self.gp)==xo.gp.GP:
            gp_pred = self.gp.predict(self.lc['flux']-pred[1,:],
                                      self.lc['time'],return_cov=False) + np.nanmedian(self.trace["mean"][self.tracemask])

        #if self.lc['oot_mask']!=self.lc['mask']:
        #Initialising GP was cut away from transits, so maybe we need to re-train with all the points


        '''if 'mult' in trace.varnames:
            pred = trace["light_curves"][tracemask,:,:]/np.tile(trace['mult'],(1,len(trace["light_curves"][0,:,0]),1)).swapaxes(0,2)
        else:
            pred = trace["light_curves"][tracemask,:,:]'''

        gp_pred = np.percentile(gp_pred, [16, 50, 84], axis=0)

        #Plotting model with GPs:
        min_trans=abs(np.min(np.sum(pred[1,:,:],axis=1)))
        if x_gap:
            gap_pos=np.average(self.lc['time'][np.argmax(np.diff(self.lc['time'])):(1+np.argmax(np.diff(self.lc['time'])))])
            before_gap_lc,before_gap_gp=(self.lc['time']<gap_pos),(self.lc['time']<gap_pos)
            after_gap_lc,after_gap_gp=(self.lc['time']>gap_pos),(self.lc['time']>gap_pos)

            print(np.sum(before_gap_lc),len(self.lc['time'][before_gap_lc]),np.sum(before_gap_gp),len(gp_mod[before_gap_gp]))

            f_all_1.circle(self.lc['time'][before_gap_lc], self.lc['flux'][before_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_2.circle(self.lc['time'][after_gap_lc], self.lc['flux'][after_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            f_all_1.line(self.lc['time'][before_gap_lc], gp_mod[before_gap_gp]+2.5*min_trans, color="C3", label="GP fit")
            f_all_2.line(self.lc['time'][after_gap_lc], gp_mod[after_gap_gp]+2.5*min_trans, color="C3", label="GP fit")

            f_all_1.circle(self.lc['time'][before_gap_lc], self.lc['flux'][before_gap_lc] - gp_mod[before_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_2.circle(self.lc['time'][after_gap_lc], self.lc['flux'][after_gap_lc] - gp_mod[after_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            #Plotting residuals at the bottom:
            f_all_resid_1.circle(self.lc['time'][before_gap_lc], 
                             self.lc['flux'][before_gap_lc] - gp_mod[before_gap_gp] - np.sum(pred[1,before_gap_gp,:],axis=1), 
                             ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid_1.set_xlabel('Time (BJD-245700)')
            f_all_resid_1.set_xlim(self.lc['time'][before_gap_lc][0],self.lc['time'][before_gap_lc][-1])

            f_all_resid_2.circle(self.lc['time'][after_gap_lc], 
                             self.lc['flux'][after_gap_lc] - gp_mod[after_gap_gp] - np.sum(pred[1,after_gap_gp,:],axis=1),
                             ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid_2.set_xlabel('Time (BJD-245700)')
            f_all_resid_2.set_xlim(self.lc['time'][after_gap_lc][0],self.lc['time'][after_gap_lc][-1])
            #print(len(lc[:,0]),len(lc[lcmask,0]),len(gp_mod))
            f_all_resid_1.set_ylim(2*np.percentile(self.lc['flux'] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            f_all_resid_2.set_ylim(2*np.percentile(self.lc['flux'] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            for n_pl in range(len(pred[1,0,:])):
                f_all_1.plot(self.lc['time'][before_gap_lc], pred[1,before_gap_gp,n_pl], color="C1", label="model")

                art = f_all_1.patch(np.append(self.lc['time'][before_gap_lc], self.lc['time'][before_gap_lc][::-1]),
                                    np.append(pred[0,before_gap_gp,n_pl], pred[2,before_gap_gp,n_pl][::-1]),
                                           color="C1", alpha=0.5, zorder=1000)
                f_all_1.set_xlim(self.lc['time'][before_gap_lc][0],self.lc['time'][before_gap_lc][-1])

                f_all_2.line(self.lc['time'][after_gap_lc], pred[1,after_gap_gp,n_pl], color="C1", label="model")
                art = f_all_2.patch(np.append(self.lc['time'][after_gap_lc], self.lc['time'][after_gap_lc][::-1]),
                                    np.append(pred[0,after_gap_gp,n_pl], pred[2,after_gap_gp,n_pl][::-1]),
                                    color="C1", alpha=0.5, zorder=1000)
                f_all_2.set_xlim(self.lc['time'][after_gap_lc][0],self.lc['time'][after_gap_lc][-1])

                f_all_1.set_ylim(np.percentile(self.lc['flux']-gp_mod,0.25),np.percentile(self.lc['flux']+2.5*min_trans,99))
                f_all_2.set_ylim(np.percentile(self.lc['flux']-gp_mod,0.25),np.percentile(self.lc['flux']+2.5*min_trans,99))

            f_all_1.get_xaxis().set_ticks([])
            f_all_2.get_yaxis().set_ticks([])
            f_all_2.get_xaxis().set_ticks([])

            f_all_resid_2.get_yaxis().set_ticks([])
            f_all_1.spines['right'].set_visible(False)
            f_all_resid_1.spines['right'].set_visible(False)
            f_all_2.spines['left'].set_visible(False)
            f_all_resid_2.spines['left'].set_visible(False)
            #
            #spines['right'].set_visible(False)
            #
            #f_all_2.set_yticks([])
            #f_all_2.set_yticklabels([])
            #f_all_1.tick_params(labelright='off')
            #f_all_2.yaxis.tick_right()

            f_zoom=fig.add_subplot(gs[:3, 6:])
            f_zoom_resid=fig.add_subplot(gs[3, 6:])

        else:
            #No gap in x, plotting normally:
            print(len(self.lc['time']),len(self.lc['flux']),len(self.lc['time']),len(self.lc['flux']),len(gp_mod),len(np.sum(pred[1,:,:],axis=1)))
            f_all.circle(self.lc['time'], self.lc['flux']+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all.line(self.lc['time'], gp_mod+2.5*min_trans, color="C3", label="GP fit")

            # Plot the data
            f_all.plot(self.lc['time'], self.lc['flux'] - gp_mod, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            #Plotting residuals at the bottom:
            f_all_resid.circle(self.lc['time'], self.lc['flux'] - gp_mod - np.sum(pred[1,:,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid.set_xlabel('Time (BJD-245700)')
            f_all_resid.set_ylim(2*np.percentile(self.lc['flux'] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            f_all_resid.set_xlim(self.lc['time'][0],self.lc['time'][-1])

            for n_pl in range(len(pred[1,0,:])):
                f_all.line(self.lc['time'], pred[1,:,n_pl], color="C1", label="model")
                art = f_all.patch(np.append(self.lc['time'],self.lc['time'][::-1]),
                                  np.append(pred[0,:,n_pl], pred[2,:,n_pl][::-1]),
                                  color="C1", alpha=0.5, zorder=1000)
                f_all.set_xlim(self.lc['time'][0],self.lc['time'][-1])

            f_all.set_xticks([])
            f_zoom=figure(width=250, plot_height=400, title=None)
            f_zoom_resid=figure(width=250, plot_height=150, title=None)

        min_trans=0;min_resid=0
        for n_pl in range(len(pred[1,0,:])):
            # Get the posterior median orbital parameters
            p = np.median(self.trace["period"][tracemask,n_pl])
            t0 = np.median(self.trace["t0"][tracemask,n_pl])
            tdur = np.nanmedian(self.trace['tdur'][tracemask,n_pl])
            #tdurs+=[(2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2))/np.nanmedian(trace['vrel'][tracemask,n_pl])]
            #print(min_trans,tdurs[n_pl],2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2),np.nanmedian(trace['vrel'][tracemask, n_pl]))

            phase=(self.lc['time']-t0+p*0.5)%p-p*0.5
            zoom_ind=abs(phase)<tdur

            resids=self.lc['flux'][zoom_ind] - gp_mod[zoom_ind] - np.sum(pred[1,zoom_ind,:],axis=1)

            if zoom_plot_time:
                #Plotting time:
                f_zoom.plot(phase[zoom_ind], min_trans+self.lc['flux'][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom.plot(phase[zoom_ind], min_trans+pred[1,zoom_ind,n_pl], color="C"+str(n_pl+1), label="model")
                art = f_zoom.patch(phase[zoom_ind], min_trans+pred[0,zoom_ind,n_pl], min_trans+pred[2,zoom_ind,n_pl],
                                          color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
                f_zoom_resid.plot(phase[zoom_ind],min_resid+resids,
                                  ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                                  "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

            else:
                print("#Normalising to transit duration",min_trans)
                f_zoom.plot(phase[zoom_ind]/tdur, min_trans+self.lc['flux'][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)

                f_zoom.plot(np.sort(phase[zoom_ind]/tdur), min_trans+pred[1,zoom_ind,n_pl][np.argsort(phase[zoom_ind])], color="C"+str(n_pl+1), label="model")
                art = f_zoom.patch(np.sort(phase[zoom_ind])/tdur,
                                          min_trans+pred[0,zoom_ind,n_pl][np.argsort(phase[zoom_ind])],
                                          min_trans+pred[2,zoom_ind,n_pl][np.argsort(phase[zoom_ind])],
                                          color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
                f_zoom_resid.plot(phase[zoom_ind]/tdur,min_resid+resids,
                                  ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                                  "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

            min_resid+=np.percentile(resids,99)
            min_trans+=abs(1.25*np.min(pred[1,:,n_pl]))

        f_zoom.set_xticks([])
        if zoom_plot_time:
            f_zoom_resid.set_xlabel('Time - t_c')
            f_zoom_resid.set_xlim(-1*tdur,tdur)
            f_zoom.set_xlim(-1*tdur,tdur)
        else:
            f_zoom_resid.set_xlabel('normalised time [t_dur]')
            f_zoom_resid.set_xlim(-1,1)
            f_zoom.set_xlim(-1,1)
        if x_gap:
            p = gridplot([[f_all_1, f_all_2, f_zoom], [f_all_resid_1, f_all_resid_2, f_zoom_resid]])
        else:
            p = gridplot([[f_all, f_zoom], [f_all_resid, f_zoom_resid]])

        save(p)
        if returnfig:
            return p
        
    def PlotMono(self, overwrite=False, savefileloc=None, 
           returnfig=False, tracemask=None, zoom_plot_time=False):

        #Plots MPL figure
        # Plot LC with best-fit transit and GP-model with residuals and transit zooms
        # INPUTS:
        # - lc (dictionary of lightcurve with time & flux arrays)
        # - trace (output of MCMC with model params and 
        # - ID
        # - mission
        # - savename
        # - overwrite
        # - savefileloc
        # - returnfig
        # - lcmask
        # - tracemask
        # - zoom_plot_time (plot zoom plot as time. Default is normalised by tdur)

        #The tracemask is a mask used to remove samples where the period is inconsistent with the presence of photometry:
        if self.tracemask is None:
            self.tracemask=np.tile(True,len(self.trace['Rs']))

        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(14,6))
        
        lcmask=self.lc['oot_mask'][:]
        
        # Compute the GP prediction
        if 'gp_pred' in self.trace:
            gp_mod = np.median(self.trace["gp_pred"][tracemask,:] + self.trace["mean"][tracemask, None], axis=0)
            assert len(self.lc['time'][lcmask])==len(gp_mod)
        elif type(self.gp)==dict:
            pred=np.zeros(len(self.lc['time']))
            for nc,cad in enumerate(self.cads):
                for col in ['logw0','logpower','logS0']:
                        self.gp[cad].set_parameter(col,np.nanmedian(self.trace[col][tracemask]))
                self.gp[cad].set_parameter('logs2',np.nanmedian(self.trace['logs2'][tracemask,nc]))
                pred[self.lc['cadence']==cad] = self.gp[cad].predict(self.lc['time'][self.lc['cadence']==cad])
            pred += np.nanmedian(self.trace["mean"][tracemask])
        elif type(self.gp)==xo.gp.GP:
            for col in ['logw0','logpower','logS0','logs2']:
                self.gp.set_parameter(col,np.nanmedian(self.trace[col][tracemask]))
            pred = self.gp.predict(self.lc['time']) + np.nanmedian(self.trace["mean"][tracemask])

        #if self.lc['oot_mask']!=self.lc['mask']:
        #Initialising GP was cut away from transits, so maybe we need to re-train with all the points

        #Finding if there's a single enormous gap in the lightcurve:
        x_gap=np.max(np.diff(self.lc['time'][lcmask]))>10
        if x_gap:
            print(" GAP IN X OF ",np.argmax(np.diff(self.lc['time'])))
            gs = fig.add_gridspec(4,8,wspace=0.3,hspace=0.001)
            f_all_1=fig.add_subplot(gs[:3, :3])
            f_all_2=fig.add_subplot(gs[:3, 3:6])
            f_all_resid_1=fig.add_subplot(gs[3, :3])#, sharey=f_all_2)
            f_all_resid_2=fig.add_subplot(gs[3, 3:6])#, sharey=f_all_resid_1)
        else:
            gs = fig.add_gridspec(4,4,wspace=0.3,hspace=0.001)
            f_all=fig.add_subplot(gs[:3, :3])
            f_all_resid=fig.add_subplot(gs[3, :3])

            
        '''if 'mult' in trace.varnames:
            pred = trace["light_curves"][tracemask,:,:]/np.tile(trace['mult'],(1,len(trace["light_curves"][0,:,0]),1)).swapaxes(0,2)
        else:
            pred = trace["light_curves"][tracemask,:,:]
        '''
        pred = self.trace["light_curves"][self.tracemask,:,:]
        #Need to check how many planets are here:
        pred = np.percentile(pred, [16, 50, 84], axis=0)

        gp_pred = np.percentile(pred, [16, 50, 84], axis=0)

        #Plotting model with GPs:
        min_trans=abs(np.min(np.sum(pred[1,:,:],axis=1)))
        if x_gap:
            gap_pos=np.average(self.lc['time'][np.argmax(np.diff(self.lc['time'])):(1+np.argmax(np.diff(self.lc['time'])))])
            before_gap_lc,before_gap_gp=(self.lc['time']<gap_pos)&lcmask,(self.lc['time'][lcmask]<gap_pos)
            after_gap_lc,after_gap_gp=(self.lc['time']>gap_pos)&lcmask,(self.lc['time'][lcmask]>gap_pos)

            print(np.sum(before_gap_lc),len(self.lc['time'][before_gap_lc]),np.sum(before_gap_gp),len(gp_mod[before_gap_gp]))

            f_all_1.plot(self.lc['time'][before_gap_lc], self.lc['flux'][before_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_2.plot(self.lc['time'][after_gap_lc], self.lc['flux'][after_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            f_all_1.plot(self.lc['time'][before_gap_lc], gp_mod[before_gap_gp]+2.5*min_trans, color="C3", label="GP fit")
            f_all_2.plot(self.lc['time'][after_gap_lc], gp_mod[after_gap_gp]+2.5*min_trans, color="C3", label="GP fit")

            f_all_1.plot(self.lc['time'][before_gap_lc], self.lc['flux'][before_gap_lc] - gp_mod[before_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_2.plot(self.lc['time'][after_gap_lc], self.lc['flux'][after_gap_lc] - gp_mod[after_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            #Plotting residuals at the bottom:
            f_all_resid_1.plot(self.lc['time'][before_gap_lc], 
                             self.lc['flux'][before_gap_lc] - gp_mod[before_gap_gp] - np.sum(pred[1,before_gap_gp,:],axis=1), 
                             ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid_1.set_xlabel('Time (BJD-245700)')
            f_all_resid_1.set_xlim(self.lc['time'][before_gap_lc][0],self.lc['time'][before_gap_lc][-1])

            f_all_resid_2.plot(self.lc['time'][after_gap_lc], 
                             self.lc['flux'][after_gap_lc] - gp_mod[after_gap_gp] - np.sum(pred[1,after_gap_gp,:],axis=1),
                             ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid_2.set_xlabel('Time (BJD-245700)')
            f_all_resid_2.set_xlim(self.lc['time'][after_gap_lc][0],self.lc['time'][after_gap_lc][-1])
            #print(len(lc[:,0]),len(lc[lcmask,0]),len(gp_mod))
            f_all_resid_1.set_ylim(2*np.percentile(self.lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            f_all_resid_2.set_ylim(2*np.percentile(self.lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            for n_pl in range(len(pred[1,0,:])):
                f_all_1.plot(self.lc['time'][before_gap_lc], pred[1,before_gap_gp,n_pl], color="C1", label="model")
                art = f_all_1.fill_between(self.lc['time'][before_gap_lc], pred[0,before_gap_gp,n_pl], pred[2,before_gap_gp,n_pl],
                                           color="C1", alpha=0.5, zorder=1000)
                f_all_1.set_xlim(self.lc['time'][before_gap_lc][0],self.lc['time'][before_gap_lc][-1])

                f_all_2.plot(self.lc['time'][after_gap_lc], pred[1,after_gap_gp,n_pl], color="C1", label="model")
                art = f_all_2.fill_between(self.lc['time'][after_gap_lc], pred[0,after_gap_gp,n_pl], pred[2,after_gap_gp,n_pl],
                                           color="C1", alpha=0.5, zorder=1000)
                f_all_2.set_xlim(self.lc['time'][after_gap_lc][0],self.lc['time'][after_gap_lc][-1])

                f_all_1.set_ylim(np.percentile(self.lc['flux'][lcmask]-gp_mod,0.25),np.percentile(self.lc['flux'][lcmask]+2.5*min_trans,99))
                f_all_2.set_ylim(np.percentile(self.lc['flux'][lcmask]-gp_mod,0.25),np.percentile(self.lc['flux'][lcmask]+2.5*min_trans,99))

            f_all_1.get_xaxis().set_ticks([])
            f_all_2.get_yaxis().set_ticks([])
            f_all_2.get_xaxis().set_ticks([])

            f_all_resid_2.get_yaxis().set_ticks([])
            f_all_1.spines['right'].set_visible(False)
            f_all_resid_1.spines['right'].set_visible(False)
            f_all_2.spines['left'].set_visible(False)
            f_all_resid_2.spines['left'].set_visible(False)
            #
            #spines['right'].set_visible(False)
            #
            #f_all_2.set_yticks([])
            #f_all_2.set_yticklabels([])
            #f_all_1.tick_params(labelright='off')
            #f_all_2.yaxis.tick_right()

            f_zoom=fig.add_subplot(gs[:3, 6:])
            f_zoom_resid=fig.add_subplot(gs[3, 6:])

        else:
            #No gap in x, plotting normally:
            print(len(self.lc['time']),len(self.lc['flux']),len(self.lc['time'][lcmask]),len(self.lc['flux'][lcmask]),len(gp_mod),len(np.sum(pred[1,:,:],axis=1)))
            f_all.plot(self.lc['time'][lcmask], self.lc['flux'][lcmask]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all.plot(self.lc['time'][lcmask], gp_mod+2.5*min_trans, color="C3", label="GP fit")

            # Plot the data
            f_all.plot(self.lc['time'][lcmask], self.lc['flux'][lcmask] - gp_mod, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

            #Plotting residuals at the bottom:
            f_all_resid.plot(self.lc['time'][lcmask], self.lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
            f_all_resid.set_xlabel('Time (BJD-245700)')
            f_all_resid.set_ylim(2*np.percentile(self.lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
            f_all_resid.set_xlim(self.lc['time'][lcmask][0],lc['time'][lcmask][-1])

            for n_pl in range(len(pred[1,0,:])):
                f_all.plot(self.lc['time'][lcmask], pred[1,:,n_pl], color="C1", label="model")
                art = f_all.fill_between(lc['time'][lcmask], pred[0,:,n_pl], pred[2,:,n_pl], color="C1", alpha=0.5, zorder=1000)
                f_all.set_xlim(self.lc['time'][lcmask][0],self.lc['time'][lcmask][-1])

            f_all.set_xticks([])
            f_zoom=fig.add_subplot(gs[:3, 3])
            f_zoom_resid=fig.add_subplot(gs[3, 3])

        min_trans=0;min_resid=0
        for n_pl in range(len(pred[1,0,:])):
            # Get the posterior median orbital parameters
            p = np.median(self.trace["period"][self.tracemask,n_pl])
            t0 = np.median(self.trace["t0"][self.tracemask,n_pl])
            tdur = np.nanmedian(self.trace['tdur'][self.tracemask,n_pl])
            #tdurs+=[(2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2))/np.nanmedian(trace['vrel'][tracemask,n_pl])]
            #print(min_trans,tdurs[n_pl],2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2),np.nanmedian(trace['vrel'][tracemask, n_pl]))

            phase=(self.lc['time'][lcmask]-t0+p*0.5)%p-p*0.5
            zoom_ind=abs(phase)<tdur

            resids=self.lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind] - np.sum(pred[1,zoom_ind,:],axis=1)

            if zoom_plot_time:
                #Plotting time:
                f_zoom.plot(phase[zoom_ind], min_trans+self.lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom.plot(phase[zoom_ind], min_trans+pred[1,zoom_ind,n_pl], color="C"+str(n_pl+1), label="model")
                art = f_zoom.fill_between(phase[zoom_ind], min_trans+pred[0,zoom_ind,n_pl], min_trans+pred[2,zoom_ind,n_pl],
                                          color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
                f_zoom_resid.plot(phase[zoom_ind],min_resid+resids,
                                  ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                                  "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

            else:
                print("#Normalising to transit duration",min_trans)
                f_zoom.plot(phase[zoom_ind]/tdur, min_trans+self.lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)

                f_zoom.plot(np.sort(phase[zoom_ind]/tdur), min_trans+pred[1,zoom_ind,n_pl][np.argsort(phase[zoom_ind])], color="C"+str(n_pl+1), label="model")
                art = f_zoom.fill_between(np.sort(phase[zoom_ind])/tdur,
                                          min_trans+pred[0,zoom_ind,n_pl][np.argsort(phase[zoom_ind])],
                                          min_trans+pred[2,zoom_ind,n_pl][np.argsort(phase[zoom_ind])],
                                          color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
                f_zoom_resid.plot(phase[zoom_ind]/tdur,min_resid+resids,
                                  ".k", label="data", zorder=-1000,alpha=0.5)
                f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                                  "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

            min_resid+=np.percentile(resids,99)
            min_trans+=abs(1.25*np.min(pred[1,:,n_pl]))

        f_zoom.set_xticks([])
        if zoom_plot_time:
            f_zoom_resid.set_xlabel('Time - t_c')
            f_zoom_resid.set_xlim(-1*tdur,tdur)
            f_zoom.set_xlim(-1*tdur,tdur)
        else:
            f_zoom_resid.set_xlabel('normalised time [t_dur]')
            f_zoom_resid.set_xlim(-1,1)
            f_zoom.set_xlim(-1,1)

        if savename is None:
            savename=GetSavename(ID, mission, how='save', suffix='_TransitFit.png', 
                                 overwrite=overwrite, savefileloc=savefileloc)[0]
            print(savename)

        plt.savefig(savename,dpi=250)

        if returnfig:
            return fig

        
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
        Teff_samples = np.random.normal(self.Teff[0],self.Teff[1],n_samples)
        logg_samples = np.random.normal(self.logg[0],self.logg[1],n_samples)
        
        from scipy.interpolate import CloughTocher2DInterpolator as ct2d

        if mission[0].lower()=="t":
            import pandas as pd
            from astropy.io import ascii
            TessLDs=ascii.read(os.path.join(MonoFit_path,'data','tables','tessLDs.txt')).to_pandas()
            TessLDs=TessLDs.rename(columns={'col1':'logg','col2':'Teff','col3':'FeH','col4':'L/HP','col5':'a',
                                               'col6':'b','col7':'mu','col8':'chi2','col9':'Mod','col10':'scope'})
            a_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.a.values.astype(float))
            b_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.b.values.astype(float))

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

            arr = np.genfromtxt(os.path.join(MonoFit_path,"data","KeplerLDlaws.txt"),skip_header=2)
            #Selecting FeH manually:
            feh_ix=arr[:,2]==find_nearest_2D(self.FeH,np.unique(arr[:, 2]))
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
        if varnames is None or varnames is 'all':
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
            trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=soln, chains=4,
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
    lc=lcFlatten(lc,winsize=9*tdur,stepsize=0.1*tdur)
    
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
        if planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] is 'singlemono':
            #Monotransit?
            print(" ")
        elif planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] is 'multimono':
            #Two monotransits?
            print(" ")
        elif planet_dic_1['01']['flag']=='monoplanet' and planet_dic_1['01']['orbit_flag'] is 'doublemono':
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
