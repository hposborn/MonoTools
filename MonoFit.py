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
        self.planets[name]=pl_dic
        self.multis+=[name]
        
    def add_mono(self, pl_dic, name):
        #Adds planet with single eclipses
        assert name not in self.planets
        pl_dic['per_gaps']=self.compute_period_gaps(pl_dic['tcen'],dur=pl_dic['tdur'])
        pl_dic['P_min']=pl_dic['per_gaps'][0]
        #print("P_min",pl_dic['P_min'],pl_dic['per_gaps'],pl_dic['tcen'],pl_dic['tdur'])
        self.planets[name]=pl_dic
        self.monos+=[name]
        #print("P_min after",self.planets[name]['P_min'])
        
    def compute_period_gaps(self,t0,dur=0.5):
        # Given the time array, the t0 of transit, and the fact that another transit is not observed, 
        #   we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
        # In this case, a list of periods is returned, with all points within 0.5dur to be cut
        dist_from_t0=np.sort(abs(t0-self.lc['time'][self.lc['mask']]))
        gaps=np.where(np.diff(dist_from_t0)>(0.9*dur))[0]
        listgaps=[]
        for ng in range(len(gaps)):
            start,end=dist_from_t0[gaps[ng]],dist_from_t0[gaps[ng]+1]
            listgaps+=[np.linspace(start,end,int(np.ceil(2*(end-start)/dur)))]
        listgaps+=[np.max(dist_from_t0)]
        return np.hstack(listgaps)
    
    def compute_duo_period_aliases(self,duo,dur=0.5):
        # Given the time array, the t0 of transit, and the fact that two transits are observed, 
        #   we want to calculate a distribution of impossible periods to remove from the period alias list
        #finding the longest unbroken observation for P_min guess
        P_min = np.max([self.compute_period_gaps(duo['tcen'],dur=duo['tdur']),
                        self.compute_period_gaps(duo['tcen2'],dur=duo['tdur'])])
        
        check_pers_ints = np.arange(1,np.ceil(duo['period']/P_min),1.0)
        check_pers_ix=np.tile(False,len(check_pers_ints))
        Npts_from_known_transits=np.sum(abs(self.lc['time']-duo['tcen'])<0.3*duo['tdur']) + \
                                 np.sum(abs(self.lc['time']-duo['tcen_2'])<0.3*duo['tdur'])
        #Looping through potential periods and counting points in-transit
        for nper,per_int in enumerate(check_pers_ints):
            per=duo['period']/per_int
            phase=(lc_time-duo['tcen']-per*0.5)%per-per*0.5
            Npts_in_tr=np.sum(abs(phase)<0.3*duo['tdur'])
            check_pers_ix[nper]=Npts_in_tr<1.075*Npts_from_known_transits #Less than 15% of another eclipse is covered
            #print(per,Npts_in_tr/Npts_from_known_transits)
        duo['period_int_aliases']=check_pers_ints[check_pers_ix]
        duo['period_aliases']=duo['period']/duo['period_int_aliases']
        duo['P_min']=np.min(duo['period_aliases'])
        return duo
    
    def add_duo(self, pl_dic,name):
        assert name not in self.planets
        #Adds planet with two eclipses and unknown period between these
        assert pl_dic['period']==abs(pl_dic['tcen']-pl_dic['tcen_2'])
        #Calculating P_min and the integer steps
        duo=self.compute_duo_period_aliases(duo)
        
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
        # - savefileloc : file location of files to save (default: 'NamastePymc3/[T/K]ID[11-number ID]/
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

    def init_model(self,assume_circ=False,
                   use_GP=True,constrain_LD=True,ld_mult=3,useL2=True,
                   mission='TESS',FeH=0.0,LoadFromFile=False,cutDistance=5.0,
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
        # cutDistance - cut out points further than this from transit. Default of zero does no cutting
        
        print(len(self.planets),self.planets,'monos:',self.monos,'multis:',self.multis,'duos:',self.duos)
        
        n_pl=len(self.planets)
        self.cads=np.unique(self.lc['cadence'])

        start=None
        with pm.Model() as model:

            # We're gonna need a bounded normal:
            #BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)

            #Stellar parameters (although these aren't really fitted.)
            #Using log rho because otherwise the distribution is not normal:
            logrho_S = pm.Normal("logrho_S", mu=np.log(self.rhostar[0]), sd=np.average(abs(self.rhostar[1:]/self.rhostar[0])),testval=np.log(self.rhostar[0]))
            rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S))
            Rs = pm.Normal("Rs", mu=self.Rstar[0], sd=np.average(abs(self.Rstar[1:])),testval=self.Rstar[0],shape=1)
            Ms = pm.Deterministic("Ms",(rho_S/1.408)*Rs**3)

            # The baseline flux
            mean = pm.Normal("mean", mu=0.0, sd=1.0,testval=0.0)

            # The 2nd light (not third light as companion light is not modelled) 
            # This quantity is in delta-mag
            if useL2:
                deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
                mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
            else:
                mult=1.0
            
            print("Forming Pymc3 model with: monos:",self.monos,"multis:",self.multis,"duos:",self.duos)
            # The time of a reference transit for each planet
            # If multiple times are given that means multiple transits *without* a set period - a "complex" t0
            if len(self.monos+self.multis)>0:
                #Normal transit fit - no complex gaps
                init_t0 = pm.Normal("init_t0", mu=[self.planets[pls]['tcen'] for pls in self.multis+self.monos], sd=0.3, shape=len(self.multis+self.monos))
            if len(self.duos)>0:
                # In cases where we have two transits with a gap between, we will have two t0s for a single planet (and N periods)
                t0_first_trans = pm.Normal("t0_first_trans",
                                            mu=np.array([self.planets[pls]['tcen_2'] for pls in self.duos]), 
                                            sd=np.tile(0.2,len(self.duos)),
                                            shape=len(self.duos))
                t0_second_trans = pm.Normal("t0_second_trans",
                                            mu=np.array([self.planets[pls]['tcen_2'] for pls in self.duos]), 
                                            sd=np.tile(0.2,len(self.duos)),
                                            shape=len(self.duos))

            #Cutting points for speed of computation:
            if len(self.duos)==0 and cutDistance is not None and cutDistance>0:
                speedmask=np.tile(False, np.sum(self.lc['mask']))
                for ipl in self.multis:
                    phase=(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen']-0.5*self.planets[ipl]['period'])%self.planets[ipl]['period']-0.5*self.planets[ipl]['period']
                    speedmask[abs(phase)<cutDistance]=True
                for ipl in self.monos:
                    speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen'])<cutDistance]=True
                for ipl in self.duos:
                    speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen'])<cutDistance]=True
                    speedmask[abs(self.lc['time'][self.lc['mask']]-self.planets[ipl]['tcen_2'])<cutDistance]=True
                self.lc['oot_mask']=self.lc['mask'][:]
                self.lc['oot_mask'][self.lc['oot_mask']]=speedmask[:]
                print(np.sum(~self.lc['oot_mask']),"points cut from lightcurve, compared to ",np.sum(self.lc['mask'])," in original mask, leaving ",np.sum(self.lc['oot_mask']),"points in the lc")

            else:
                #Using all points in the 
                self.lc['oot_mask']=self.lc['mask']
            
            if len(self.monos)>0:
                min_Ps=np.array([self.planets[pls]['P_min'] for pls in self.planets if self.planets[pls]['orbit_flag']=='mono']).ravel()
                print(min_Ps)
                #From Dan Foreman-Mackey's thing:
                soft_period = pm.Bound(pm.Pareto, lower=0.0, upper=1.0)("soft_period", m=min_Ps, alpha=2/3.,shape=len(min_Ps))
            if len(self.multis)>0:
                known_period = pm.Normal("known_period", mu=np.array([self.planets[pls]['period'] for pls in self.multis]),sd=0.1,shape=len(self.multis))
            
            #P_index = pm.Bound("P_index", upper=1.0, lower=0.0)("P_index", mu=0.5, sd=0.33, shape=n_pl)

            # The Espinoza (2018) parameterization for the joint radius ratio and
            # impact parameter distribution
            if useL2:
                #EB case as second light needed:
                RpRs, b = xo.distributions.get_joint_radius_impact(
                    min_radius=0.001, max_radius=1.25,
                    testval_r=np.array([self.planets[pls]['depth'] for pls in self.multis+self.monos+self.duos])**0.5,
                    testval_b=np.random.rand(len(self.planets))
                )
            else:
                RpRs, b = xo.distributions.get_joint_radius_impact(
                    min_radius=0.001, max_radius=0.25,
                    testval_r=np.array([self.planets[pls]['depth'] for pls in self.multis+self.monos+self.duos])**0.5,
                    testval_b=np.random.rand(len(self.planets))
                )

            r_pl = pm.Deterministic("r_pl", RpRs * Rs * 109.1)

            #Initialising Limb Darkening:
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

            #Initialising GP kernel (i.e. without initialising Potential)
            if use_GP:
                # Transit jitter & GP parameters
                #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
                print("all cadences",self.cads)
                #Transit jitter will change if observed by Kepler/K2/TESS - need multiple logs^2
                if len(self.cads)==1:
                    logs2 = pm.Uniform("logs2", upper=np.log(np.std(self.lc['flux'][self.lc['oot_mask']]))+4,
                                       lower=np.log(np.std(self.lc['flux'][self.lc['oot_mask']]))-4)
                    min_cad=np.nanmedian(np.diff(self.lc['time']))#Limiting to <1 cadence
                else:
                    logs2 = pm.Uniform("logs2", upper=[np.log(np.std(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==c)]))+4 for c in self.cads],
                                       lower=[np.log(np.std(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==c)]))-4 for c in self.cads], shape=len(self.cads))
                    min_cad=np.min([np.nanmedian(np.diff(self.lc['time'][self.lc['cadence']==c])) for c in self.cads])

                logw0_guess = np.log(2*np.pi/10)

                lcrange=self.lc['time'][self.lc['oot_mask']][-1]-self.lc['time'][self.lc['oot_mask']][0]

                #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
                logw0 = pm.Uniform("logw0",lower=np.log((2*np.pi)/(4*lcrange)), 
                                   upper=np.log((2*np.pi)/min_cad))

                # S_0 directly because this removes some of the degeneracies between
                # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
                logpower = pm.Uniform("logpower",lower=-20,upper=np.log(np.nanmedian(abs(np.diff(self.lc['flux'][self.lc['oot_mask']])))))
                logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

                # GP model for the light curve
                kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))

            if not assume_circ:
                # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, shape=n_pl,
                                  testval=np.tile(0.1,n_pl))
                omega = xo.distributions.Angle("omega", shape=n_pl, testval=np.tile(0.1,n_pl))

            if use_GP:
                if len(self.cads)==1:
                    #Observed by a single telescope
                    gp = xo.gp.GP(kernel, self.lc['time'][self.lc['oot_mask']].astype(np.float32), tt.exp(logs2) + self.lc['flux_err'][self.lc['oot_mask']].astype(np.float32), J=2)
                else:
                    #We have multiple logs2 terms due to multiple telescopes:
                    gp_i=[]
                    for n in range(len(self.cads)):
                        cad_ix=self.lc['oot_mask']&(self.lc['cadence']==self.cads[n])
                        gp_i += [xo.gp.GP(kernel, 
                                          self.lc['time'][cad_ix].astype(np.float32), 
                                          tt.exp(logs2[n]) + self.lc['flux_err'][cad_ix].astype(np.float32), J=2)]
            
            def gen_lc(mask=None,prefix=''):
                # Short method to create stacked lightcurves, given some input time array and some input cadences:
                # This function is needed because we may have 
                #   -  1) multiple cadences and 
                #   -  2) multiple telescopes (and therefore limb darkening coefficients)
                lc_c=[]
                lc_cad_xs=[]
                mask = ~np.isnan(self.lc['time']) if mask is None else mask
                for cad in self.cads:
                    t_cad=self.lc['time'][mask][self.lc['cadence'][mask]==cad]
                    lc_cad_xs+=[t_cad]
                    if cad[0]=='t':
                        lc_c +=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=orbit, r=r_pl,
                                                                 t=t_cad,
                                                                 texp=float(int(cad[1:]))/1440.0
                                                             )/(self.lc['flux_unit']*mult)]
                        
                    elif cad[0]=='k':
                        lc_c += [xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=orbit, r=r_pl,
                                                                 t=t_cad,
                                                                 texp=float(int(cad[1:]))/1440.0
                                                             )/(self.lc['flux_unit']*mult)]
                print(lc_c[-1].shape)
                #Sorting by time so that it's in the correct order here:
                lc_cad_xs=np.hstack(lc_cad_xs)
                print(len(lc_cad_xs),np.min(lc_cad_xs),np.max(lc_cad_xs),len(self.lc['time'][mask]),self.lc['time'][mask][0],self.lc['time'][mask][-1])
                assert (np.sort(lc_cad_xs)==self.lc['time'][mask]).all()
                return pm.Deterministic(prefix+"light_curves", tt.concatenate(lc_c,axis=0)[np.argsort(lc_cad_xs)])

            # Complex T requires special treatment of orbit (i.e. period) and potential:
            if len(self.duos)==1:
                print("DUO - making orbital arrays","pls=",len(self.planets),"monos=",len(self.monos),"multis=",len(self.multis))
                #Marginalising over each possible period
                #Single planet with two transits and a gap
                logprobs = []
                all_lcs = []
                
                per_int_steps=pm.Deterministic("per_int_steps", 
                                               tt.as_tensor_variable(self.planets[self.duos[0]]['period_int_aliases'])
                                              )
                print(p_int_steps,range(len(p_int_steps[self.duos[0]])))
                for i in range(len(p_int_steps[self.duos[0]])):
                    with pm.Model(name="per_{0}".format(i), model=model) as submodel:
                        if len(self.planets)>1:
                            #Have other planets - need to concatenate periods and t0s
                            if len(self.monos)>0 and len(self.multis)>0:
                                print("1 duo, plus monos, plus multis")
                                period=pm.Deterministic("period",tt.concatenate((soft_period,known_period,(t0_second_trans-t0_first_trans)/p_int_steps[self.duos[0]][i])))
                            elif len(self.monos)>0 and len(self.multis)==0:
                                print("1 duo, plus monos")
                                period=pm.Deterministic("period",tt.concatenate((soft_period,(t0_second_trans-t0_first_trans)/p_int_steps[self.duos[0]][i])))
                            elif len(self.multis)>0 and len(self.monos)==0:
                                print("1 duo, plus multis")
                                period=pm.Deterministic("period",tt.concatenate((known_period,(t0_second_trans-t0_first_trans)/p_int_steps[self.duos[0]][i])))
                            t0=pm.Deterministic("t0",tt.concatenate((init_t0,t0_first_trans)))
                        else:
                            print("1 duo no others")
                            period=pm.Deterministic("period",(t0_second_trans-t0_first_trans)/p_int_steps[self.duos[0]][i])
                            t0=pm.Deterministic("t0",t0_first_trans)
                        
                        logp = pm.Deterministic("logp", tt.log(period))
                        
                        # Set up a Keplerian orbit for the planets
                        if assume_circ:
                            orbit = xo.orbits.KeplerianOrbit(
                                r_star=Rs, rho_star=rho_S,
                                period=period, t0=t0, b=b)
                        else:
                            orbit = xo.orbits.KeplerianOrbit(
                                r_star=Rs, rho_star=rho_S,
                                ecc=ecc, omega=omega,
                                period=period, t0=t0, b=b)
                        print("orbit set up")
                        vx, vy, vz = orbit.get_relative_velocity(t0_2use)
                        #vsky[self.lc['oot_mask']] 
                        if n_pl>1:
                            vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
                        else:
                            vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)
                        tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)

                        print(self.lc['time'][self.lc['oot_mask']])
                        if debug:
                            tt.printing.Print('r_pl')(r_pl)
                            tt.printing.Print('mult')(mult)
                            tt.printing.Print('tdur')(tdur)
                            tt.printing.Print('t0')(t0)
                            tt.printing.Print('b')(b)
                            tt.printing.Print('p2use')(p2use)
                            tt.printing.Print('rho_S')(rho_S)
                            tt.printing.Print('Rs')(Rs)
                        # Compute the model light curve using starry
                        light_curves = gen_lc(mask=self.lc['oot_mask'])
                        #light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,
                        #                                     t=x[self.lc['oot_mask']])/(self.lc['flux_unit']*mult)
                        
                        light_curve = pm.math.sum(light_curves, axis=-1) + mean #Summing lightcurve over n planets
                        all_lcs.append(light_curve)
                        
                        if use_GP:
                            if len(self.cads)==1:
                                #Observed by a single telescope
                                loglike = tt.sum(gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - light_curve))
                                if pred_all_time:
                                    gp_pred = pm.Deterministic("gp_pred", gp.predict(self.lc['time']))
                            else:
                                #We have multiple logs2 terms due to multiple telescopes:
                                llk_gp_i = []
                                gp_pred_i= []
                                for n in range(len(self.cads)):
                                    llk_gp_i += [gp_i[n].log_likelihood(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==self.cads[n])] - light_curve[self.lc['cadence'][self.lc['oot_mask']]==self.cads[n]])]
                                    if pred_all_time:
                                        gp_pred_i += [gp_i[n].predict(self.lc['time'][self.lc['cadence']==self.cads[n]])]
                                loglike = tt.sum(llk_gp_i)
                                if pred_all_time:
                                    gp_pred = pm.Deterministic("gp_pred", tt.concatenate(gp_pred_i))

                            #chisqs = pm.Deterministic("chisqs", (self.lc['flux'] - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                            #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                            #llk = pm.Deterministic("llk", model.logpt)
                        else:
                            loglike = tt.sum(pm.Normal.dist(mu=light_curve, sd=self.lc['flux_err'][self.lc['oot_mask']]).logp(self.lc['flux'][self.lc['oot_mask']]))

                        logprior = tt.log(orbit.dcosidb) - 2 * tt.log(complex_pers[0][i])
                        logprobs.append(loglike + logprior)

            elif len(self.duos)==2:
                #Two planets with two transits... This has to be the max.
                for i_1, p_int_1 in enumerate(self.planets[self.duos[0]]['period_int_aliases']):
                    for i_2, p_int_2 in enumerate(self.planets[self.duos[1]]['period_int_aliases']):
                        with pm.Model(name="per_{0}_{1}".format(i_1,i_2), model=model) as submodel:
                            submodel_per_ints=pm.Deterministic("submodel_per_ints", tt.as_tensor_variable([p_int_1,p_int_2]))
                            if len(self.multis+self.monos)>0:
                                #Have other planets - need to concatenate periods and t0s
                                if len(self.monos)>0 and len(self.multis)>0:
                                    period=pm.Deterministic("period",tt.concatenate((soft_period, known_period, (t0_second_trans-t0_first_trans)/submodel_per_ints)))
                                elif len(self.monos)>0 and len(self.multis)==0:
                                    period=pm.Deterministic("period",tt.concatenate((soft_period,(t0_second_trans-t0_first_trans)/submodel_per_ints)))
                                elif len(self.multis)>0 and len(self.monos)==0:
                                    period=pm.Deterministic("period",tt.concatenate((known_period,(t0_second_trans-t0_first_trans)/submodel_per_ints)))
                                t0=pm.Deterministic("t0",tt.concatenate((init_t0,t0_first_trans)))
                            else:
                                period=pm.Deterministic("period",(t0_second_trans-t0_first_trans)/submodel_per_ints)
                                t0=pm.Deterministic("t0",t0_first_trans)
                            logp = pm.Deterministic("logp", tt.log(period))
                            # Set up a Keplerian orbit for the planets
                            if assume_circ:
                                orbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    period=period, t0=t0, b=b)
                            else:
                                orbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    ecc=ecc, omega=omega,
                                    period=period, t0=t0, b=b)
                            vx, vy, vz = orbit.get_relative_velocity(t0)
                            #vsky = 
                            if n_pl>1:
                                vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
                            else:
                                vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)
                            tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)

                            # Compute the model light curve using starry
                            light_curves = gen_lc(mask=self.lc['oot_mask'])
                            #light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,
                            #                      t=x[self.lc['oot_mask']])/(self.lc['flux_unit']*mult)
                            light_curve = pm.math.sum(light_curves, axis=-1) + mean     
                            all_lcs.append(light_curve)

                            if use_GP:
                                if len(self.cads)==1:
                                    #Observed by a single telescope
                                    loglike = tt.sum(gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - light_curve))
                                    if pred_all_time:
                                        gp_pred = pm.Deterministic("gp_pred", gp.predict(self.lc['time']))
                                else:
                                    #We have multiple logs2 terms due to multiple telescopes:
                                    for n in range(len(self.cads)):
                                        llk_gp_i += [gp_i[n].log_likelihood(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==self.cads[n])] - light_curve[self.lc['cadence'][self.lc['oot_mask']]==self.cads[n]])]
                                        if pred_all_time:
                                            gp_pred_i += [gp_i[n].predict(self.lc['time'])]

                                    loglike = tt.sum(tt.stack(llk_gp_i))
                                    if pred_all_time:
                                        gp_pred = pm.Deterministic("gp_pred", tt.stack(gp_pred_i))

                                #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                                #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                                #llk = pm.Deterministic("llk", model.logpt)
                            else:
                                loglike = tt.sum(pm.Normal.dist(mu=light_curve, 
                                                                sd=self.lc['flux_err'][self.lc['oot_mask']]
                                                               ).logp(self.lc['flux'][self.lc['oot_mask']]))
                            
                            logprior = tt.sum(tt.log(orbit.dcosidb)) -\
                                       2 * tt.log((t0_second_trans-t0_first_trans)/submodel_per_ints)
                            logprobs.append(loglike + logprior)

                # Compute the marginalized probability and the posterior probability for each period
                logprobs = tt.stack(logprobs)
                logprob_marg = pm.math.logsumexp(logprobs)
                logprob_class = pm.Deterministic("logprob_class", logprobs - logprob_marg)
                pm.Potential("logprob", logprob_marg)

                # Compute the marginalized light curve
                pm.Deterministic("light_curve", tt.sum(tt.stack(all_lcs) * tt.exp(logprob_class)[:, None], axis=0))
            else:
                #No complex periods - i.e. no gaps:
                t0=pm.Deterministic("t0",init_t0)
                if len(self.monos)>0 and len(self.multis)>0:
                    tt.printing.Print('t0')(t0)
                    tt.printing.Print('soft_period')(soft_period)
                    tt.printing.Print('known_period')(known_period)
                    period=pm.Deterministic("period",tt.concatenate([soft_period,known_period]))
                elif len(self.monos)>0 and len(self.multis)==0:
                    period=pm.Deterministic("period",soft_period)
                elif len(self.multis)>0 and len(self.monos)==0:
                    period=pm.Deterministic("period",known_period)

                logp = pm.Deterministic("logp", tt.log(period))
                
                if assume_circ:
                    orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        period=period, t0=t0, b=b)
                else:
                    # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                    orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        ecc=ecc, omega=omega,
                        period=period, t0=t0, b=b)

                vx, vy, vz = orbit.get_relative_velocity(t0)
                #vsky = 
                if n_pl>1:
                    vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
                else:
                    vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)

                tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)
                
                #Generating lightcurves using pre-defined gen_lc function:
                
                if pred_all_time:
                    light_curves = gen_lc()
                    light_curve = pm.math.sum(light_curves, axis=-1)
                mask_light_curves = gen_lc(mask=self.lc['oot_mask'],prefix='mask_')
                mask_light_curve = pm.math.sum(mask_light_curves, axis=-1)

                # Compute the model light curve using starry
                if use_GP:
                    if len(self.cads)==1:
                        llk_gp = pm.Potential("llk_gp", gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - mask_light_curve))
                        if pred_all_time:
                            gp_pred = pm.Deterministic("gp_pred", gp.predict(self.lc['time']))
                    else:
                        #We have multiple logs2 terms due to multiple telescopes:
                        llk_gp_i = []
                        gp_pred_i = []
                        for n in range(len(self.cads)):
                            llk_gp_i += [gp_i[n].log_likelihood(self.lc['flux'][self.lc['oot_mask']&(self.lc['cadence']==self.cads[n])] - mask_light_curve[self.lc['cadence'][self.lc['oot_mask']]==self.cads[n]])]
                            if pred_all_time:
                                gp_pred_i += [gp_i[n].predict(self.lc['time'][self.lc['cadence']==self.cads[n]])]
                        #print(gp_pred_i[0].shape,gp_pred_i[1].shape,np.hstack((gp_pred_i)))
                        llk_gp = pm.Potential("llk_gp", tt.stack(llk_gp_i,axis=0))
                        if pred_all_time:
                            gp_pred = pm.Deterministic("gp_pred", tt.join(gp_pred_i))
                    #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                    #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                    #llk = pm.Deterministic("llk", model.logpt)
                else:
                    pm.Normal("obs", mu=mask_light_curve, 
                              sd=self.lc['flux_err'][self.lc['oot_mask']],
                              observed=self.lc['flux'][self.lc['oot_mask']])

            tt.printing.Print('period')(period)
            tt.printing.Print('r_pl')(r_pl)
            #tt.printing.Print('t0')(t0)
            '''
            print(P_min,t0,type(x[self.lc['oot_mask']]),x[self.lc['oot_mask']][:10],np.nanmedian(np.diff(x[self.lc['oot_mask']])))'''
            self.model=model
            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = self.model.test_point
            print(model.test_point)
            if not LoadFromFile:
                print("before",model.check_test_point())
                map_soln = xo.optimize(start=start, vars=[RpRs, b],verbose=True)
                map_soln = xo.optimize(start=map_soln, vars=[logs2],verbose=True)
                map_soln = xo.optimize(start=map_soln)
                #map_soln = xo.optimize(start=map_soln, vars=[period, t0])
                map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower])
                map_soln = xo.optimize(start=map_soln, vars=[logw0])
                #if not assume_circ:
                #    map_soln = xo.optimize(start=map_soln, vars=[ecc, omega, period, t0])
                map_soln = xo.optimize(start=map_soln, vars=[RpRs, b],verbose=True)
                map_soln = xo.optimize(start=map_soln)
                print("after",model.check_test_point())
                
                self.model = model
                self.init_soln = map_soln
    
    def RunMcmc(self, n_draws=1200, plot=True, do_per_gap_cuts=True, LoadFromFile=True, **kwargs):
        if LoadFromFile and not self.overwrite:
            self.LoadPickle()

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
            #Doing Cuts for Period gaps (i.e. where photometry rules out the periods of a planet)
            #Only taking MCMC positions in the trace where either:
            #  - P<0.5dur away from a period gap in P_gap_cuts[:-1]
            #  - OR P is greater than P_gap_cuts[-1]
            if not hasattr(self,'tracemask'):
                self.tracemask=np.tile(True,len(self.trace['t0'][:,0]))
            
            for npl,pl in enumerate(self.multis+self.monos+self.duos):
                if pl in self.monos:
                    per_gaps=self.compute_period_gaps(np.nanmedian(self.trace['t0'][:,npl]),np.nanmedian(self.trace['tdur'][:,npl]))
                    #for each planet - only use monos
                    if len(per_gaps)>1:
                        #Cutting points where P<P_gap_cuts[-1] and P is not within 0.5Tdurs of a gap:
                        gap_dists=np.nanmin(abs(self.trace['period'][:,npl][:,np.newaxis]-per_gaps[:-1][np.newaxis,:]),axis=1)
                        self.tracemask[(self.trace['period'][:,npl]<per_gaps[-1])*(gap_dists>0.5*np.nanmedian(self.trace['tdur'][:,npl]))] = False
        elif not hasattr(self,'tracemask'):
            self.tracemask=None
        
        if plot:
            print("plotting")
            self.PlotMono()
            self.PlotCorner()
            self.PlotMonoInteractive()
        
        if LoadFromFile and not self.overwrite and os.path.exists(savenames[0].replace('mcmc.pickle','results.txt')):
            with open(savenames[0].replace('mcmc.pickle','results.txt'), 'r', encoding='UTF-8') as file:
                restable = file.read()
        else:
            restable=ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
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

        if not hasattr(self,'savename'):
            savename=self.GetSavename(ID, mission, how='save', suffix='_TransitFit.png', 
                                 overwrite=overwrite, savefileloc=savefileloc)[0]
            print(savename)

        output_file(savename)

        #Initialising figure:
        p = figure(plot_width=1000, plot_height=600,title=str(ID)+" Transit Fit")

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

        # Compute the GP prediction
        if 'gp_pred' in self.trace:
            gp_mod = np.median(self.trace["gp_pred"][tracemask,:] + self.trace["mean"][tracemask, None], axis=0)

        '''if 'mult' in trace.varnames:
            pred = trace["light_curves"][tracemask,:,:]/np.tile(trace['mult'],(1,len(trace["light_curves"][0,:,0]),1)).swapaxes(0,2)
        else:
            pred = trace["light_curves"][tracemask,:,:]'''
        pred = trace["light_curves"][tracemask,:,:]
        #Need to check how many planets are here:
        pred = np.percentile(pred, [16, 50, 84], axis=0)

        gp_pred = np.percentile(pred, [16, 50, 84], axis=0)

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
        
        lcmask=self.lc['oot_mask']
        assert len(self.lc['time'][lcmask])==len(self.trace['gp_pred'][0,:])

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

        # Compute the GP prediction
        gp_mod = np.median(self.trace["gp_pred"][self.tracemask,:] + self.trace["mean"][self.tracemask, None], axis=0)

        '''if 'mult' in trace.varnames:
            pred = trace["light_curves"][tracemask,:,:]/np.tile(trace['mult'],(1,len(trace["light_curves"][0,:,0]),1)).swapaxes(0,2)
        else:
            pred = trace["light_curves"][tracemask,:,:]'''
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
            self.savenames=self.GetSavename(self, how='save')
        
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
