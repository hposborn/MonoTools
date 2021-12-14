from posix import times_result
import httplib2
import numpy as np
import pandas as pd
import pickle
import os
import gzip

from copy import deepcopy
from datetime import datetime

from scipy.signal import savgol_filter
import exoplanet as xo
import scipy.interpolate as interp
import scipy.optimize as optim
import matplotlib.pyplot as plt
import matplotlib

from astropy import units
from astropy.coordinates.sky_coordinate import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits

from . import tools, starpars

import seaborn as sns
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger("httplib2").setLevel(logging.WARNING)


class lc():
    """Base lightcurve class which allows for standard binning, flattening, loading, etc. 
    Typically for objects observed by multiple sectors, campaigns, missions, etc. the `multilc` daughter class should be used instead."""

    def __init__(self):
        """Initialising lightcurve. To add fluxes and info, `load_lc` should be called. 
        """
        
        self.flx_unit_dic = {'ppm':[1e-6,0],'ppt':[1e-3,0],'norm0':[1.0,0],'norm1':[1.0,1]}
        self.lc_dic={'tess':'ts','kepler':'k1','k2':'k2','corot':'co','cheops':'ch'}
        self.timeseries=[]
        self.cadence_list=[]

    def load_lc(self, time, fluxes, flux_errs, src, mission, jd_base, flx_system, sect=None, cadence=None, default_flx_system='ppt', *args, **kwargs):
        """Loading lightcuve given vital infomation.

        Args:
            time (np.ndarray): time array
            fluxes (dict): Dictionary of various flux arrays
            flux_errs (dict): Dictionary of various flux error arrays. This should only have fluxes as seen in the fluxes dict with "_err" as suffixes
            src (str): String describing source of lightcurve (e.g. pipeline)
            mission (str): Photometric mission/telescope of lightcurve (e.g. k2, tess, kepler, etc)
            jd_base (float): JD epoch for time=0
            flx_system (str): Type of flux - either "ppm", "ppt", "norm0", "norm1" or "elec" (see `change_flx_system`)
            cadence (float OR int OR np.ndarray): Either - cadence in seconds OR 
                                                         - a 1D array of strings in format [telescope ID k1/k2/co/te/ch]_[cadence in secs]_[pipeline source]_[Sector/Q/camp]
                                                  If left None this is estimated from time array. Default is None
            default_flx_system (str): Type of flux - either "ppm", "ppt", "norm0", "norm1" or "elec" (see `change_flx_system `)
        """

        #This sector value, conjoined with the mission, becomes the info and is added to the cadence information
        #If we do not have a sector/Q/campaign, we make one up using 'abc' (no clash with other systems)
        sect = sect if sect is not None else 'a' if not hasattr(self,'info') else chr(97+len(getattr(self,'info').keys()))
        if cadence is None:
            cad=86400*np.nanmedian(np.diff(time))
            cad=int(np.round(cad,np.clip(int(-1*np.floor(np.log10(cad)-1)),-100,0))) #Round cadence to nearest 1%, i.e. 1798 -> 1800. 121 -> 120. 42 -> 42
            cadence_str=self.lc_dic[mission.lower()]+"_"+str(cad)+"_"+src+'_'+str(sect)
        elif cadence in [float,int,np.float64,np.int64]:
            #Cadence in seconds
            cadence_str=self.lc_dic[mission.lower()]+"_"+str(int(cad))+"_"+src+'_'+str(sect)
        elif type(cadence)==str and len(cadence[0].split('_'))==4:
            cadence_str=cadence
        else:
            #Cadence in array
            assert type(cadence)==np.ndarray and type(cadence[0])==str and len(cadence[0].split('_'))==4 and len(cadence)==len(time)
            cadence_str=cadence[0]
        #Initialising dictionary of parameters specific to that cadence:        
        self.info = {cadence_str:{}}
        self.info[cadence_str]['src']=src
        self.info[cadence_str]['mission']=mission
        self.info[cadence_str]['sect']=sect

        #Global info that must be maintained across all sectors:
        self.jd_base=jd_base
        self.flx_system = flx_system
        self.flx_unit   = self.flx_unit_dic[flx_system][0] if flx_system!='elec' else 1/np.nanmedian(fluxes['flux'])

        #Adding timeseries:
        finitetimemask=[np.isfinite(time)]
        self.time = time[finitetimemask]
        self.timeseries+=['time']
        self.cadence = np.tile(cadence_str,np.sum(finitetimemask))
        self.cadence_list+=[cadence_str]
        self.timeseries+=['cadence']

        for iflux in fluxes:
            setattr(self,iflux,fluxes[iflux][finitetimemask])
            self.timeseries+=[iflux]
            if iflux+'_err' in flux_errs:
                setattr(self,iflux+"_err",flux_errs[iflux+'_err'][finitetimemask])
                self.timeseries+=[iflux+"_err"]
        if 'flux_err' not in self.timeseries:
            setattr(self,'flux_err',np.tile(1.06*np.nanmedian(abs(np.diff(self.flux))),np.sum(finitetimemask)))

        # Adding "extra" timeseries and values from kwargs
        for key, value in kwargs.items():
            if type(value)==np.ndarray and len(value)==len(self.time) and key not in self.timeseries:
                setattr(self,key,value[finitetimemask])
                self.timeseries+=[key]
            elif type(value)!=np.ndarray:
                self.info[cadence_str][key]=value

        #Changing flux to "default" system
        self.change_flx_system(default_flx_system)
        self.loaded=True

    def remove_binned_arrs(self):
        """Removing binned timeseries arrays. This is necessary to stack lightcurves
        """
        rem=[]
        for ikey in self.timeseries:
            if 'bin_' in ikey:
                if hasattr(self,ikey):
                    delattr(self,ikey)
                rem+=[ikey]
        for ikey in rem:
            self.timeseries.remove(ikey)

    def remove_flattened_arrs(self):
        """Removing flattened timeseries arrays. This reduces size when saving
        """
        for ikey in self.timeseries:
            if '_flat' in ikey:
                if hasattr(self,ikey):
                    delattr(self,ikey)
                self.timeseries.remove(ikey)

    def change_jd_base(self, new_jd_base):
        """Change timing epoch

        Args:
            new_jd_base (float): New base for timing array
        """
        if self.jd_base!=new_jd_base:
            self.time+=(self.jd_base - new_jd_base)
            self.jd_base = new_jd_base

    def sort_timeseries(self):
        """Loop through the timeseries associated with this lightcurve (`timeseries`) and sort by time.
        """
        ixsort=np.argsort(self.time)[:]
        if 'bin_time' in self.timeseries:
            binixsort=np.argsort(self.bin_time)[:]
        cuts=[]
        for ts in self.timeseries:
            if 'bin_' in ts:
                if not len(getattr(self,ts))==len(self.bin_time):
                    delattr(self,ts)
                    cuts+=[ts]
                else:
                    setattr(self,ts,getattr(self,ts)[binixsort])
            else:
                assert len(getattr(self,ts))==len(self.time)
                setattr(self,ts,getattr(self,ts)[ixsort])
        for c in cuts:
            self.timeseries.remove(c)

    def change_flx_system(self, new_flx_system, mask=None):
        """
        Convert lightcurves between flux unit/systems, e.g. to a lightcurve in ppm normalised to 0 ('ppm'), or to a flux ratio normalised to 1 ('norm1')

        Args:
            new_flx_system (str): One of "ppm", "ppt", "norm0", "norm1" or "elec". These denote:
                                   - `ppm`: normalised lightcurve with median at 0.0 with units in parts per million
                                   - `ppt`: normalised lightcurve with median at 0.0 with units in parts per thousand
                                   - `norm0`: normalised lightcurve with median at 0.0 with units as a ratio (0->1)
                                   - `norm1`: normalised lightcurve with median at 1.0 with units as a ratio (0->1)
                                   - `elec`: un-normalised lightcurve with units of pure electrons and median proportional to stellar magnitude
            mask (np.ndarray, optional): Boolean array masking bad points to create median. Default is None
        """ 
        #Only updating if we need to...
        if self.flx_system!=new_flx_system:
            #Initialising mask:
            if mask is None and hasattr(self,'mask') and hasattr(self,'in_trans'):
                mask=self.mask*~self.in_trans
            elif mask is None and hasattr(self,'mask'):
                mask=self.flux_mask
            else:
                mask=np.isfinite(self.flux)
            if not hasattr(self,'cadence_list'):
                self.cadence_list=np.unique(self.cadence)
            for unq_cad in self.cadence_list:
                for iflux in [ser for ser in self.timeseries if 'flux' in ser and '_err' not in ser]:
                    mu_old = np.nanmedian(getattr(self,iflux)[mask*(self.cadence==unq_cad)])
                    #In the case of "electron" fluxes, we have to make sure we store the old median values for each sector/unique cadences:
                    if self.flx_system is 'elec':
                        self.info[unq_cad][iflux+'_elec_flux'] = mu_old
                        self.flx_unit_dic['elec']=[1/mu_old,mu_old]
                    elif new_flx_system == 'elec':
                        assert hasattr(self.info[unq_cad],iflux+'_elec_flux') #Must have raw electron flux to convert back to that system
                        self.flx_unit_dic['elec']=[1/self.info[unq_cad][iflux+'_elec_flux'],self.info[unq_cad][iflux+'_elec_flux']]
                    getattr(self,iflux)[self.cadence==unq_cad] = (getattr(self,iflux)[self.cadence==unq_cad]-self.flx_unit_dic[self.flx_system][1])*self.flx_unit_dic[self.flx_system][0]/self.flx_unit_dic[new_flx_system][0]+self.flx_unit_dic[new_flx_system][1]
                    if hasattr(self,iflux+'_err'):
                        getattr(self,iflux+'_err')[self.cadence==unq_cad] = getattr(self,iflux+'_err')[self.cadence==unq_cad]*self.flx_unit_dic[self.flx_system][0]/self.flx_unit_dic[new_flx_system][0]
                    #print("converting cadence:",unq_cad,"and flux:",iflux,"with median:",mu_old,"to new median:",np.nanmedian(getattr(self,iflux)[self.cadence==unq_cad]))
            #Updating universal flux system and unit 
            self.flx_system=new_flx_system
            self.flx_unit=self.flx_unit_dic[self.flx_system][0]

    def make_mask(self,overwrite=False):
        """Making mask for timeseries. Requires both a "flux" mask (i.e. bad/anomalous points) and a "cadence" mask (i.e. duplicate cadences due to multiple lightcurve sources)

        Args:
            overwrite (bool, optional): [description]. Defaults to False.
        """
        #Making mask
        if 'flux_mask' not in self.timeseries or overwrite:
            self.make_fluxmask()
        self.mask=self.flux_mask
        self.timeseries+=['mask']

    def make_fluxmask(self,flux_arr_name='flux',cut_all_anom_lim=5.0,end_of_orbit=True,use_flat=False,mask_islands=False,input_mask=None,in_transit=None,extreme_anom_limit=0.25):
        """mask bad points in lightcurve

        Args:
            cut_all_anom_lim (float, optional): Cut all single points above AND below lightcurve further away than this anomalous value. Defaults to 5.0.
            end_of_orbit (bool, optional): Check if masking lightcurve segments before/after a gap vastly reduces rms? Defaults to True.
            use_flat (bool, optional): Run mask on flattened flux array? Defaults to False.
            mask_islands (bool, optional): Mask small islands of data in time series?  Defaults to False.
            input_mask (np.ndarray, optional): A previously-defined mask. Defaults to None.
            in_transit (np.ndarray, optional): A mask where all in-transit points are flagged as True. Defaults to None
            cut_periodic_high_regions (bool, optional): Using the orbital period, should we cut the periodic high SAA regions? Defaults to False
            extreme_anom_limit (float, optional): Flux variation in ratio above/below which, we will out. Defaults to 0.25 (i.e. points above/below 25% are missing)
        Returns:
            [type]: [description]
        """
        # Mask bad data (nans, infs and negatives)
        self.flux_mask = np.isfinite(getattr(self,flux_arr_name)) & np.isfinite(self.time) & np.isfinite(getattr(self,flux_arr_name+'_err'))
        if 'flux_mask' not in self.timeseries:
            self.timeseries+=['flux_mask']
        #For corot cadences, we'll cut regions from the SAA
        for corotcad in [cad[:2] for cad in self.cadence_list if 'co_' in cad]:
            ix=np.in1d(self.cadence,corotcad)
            self.flux_mask[ix] = tools.CutHighRegions(self.flux[ix],self.flux_mask[ix],std_thresh=4.5,n_pts=25,n_loops=2)
        
        if np.sum(self.flux_mask)>0:
            # & (lc[prefix+'flux'+suffix]>0.0)
            #print(np.sum(mask))
            if input_mask is not None:
                self.flux_mask=self.flux_mask&input_mask
            # Mask data if it's 4.2-sigma from its points either side (repeating at 7-sigma to get any points missed)
            # print(np.sum(~lc['mask']),"points before quality flags")
            if hasattr(self, 'quality') and len(self.quality)==len(getattr(self,flux_arr_name)):
                qs=[1,2,3,4,6,7,8,9,13,15,16,17]#worst flags to cut - for the moment just using those in the archive_manual
                #if np.max(self.quality)>(2**18-1):
                #    _=qs.pop(9)
                #    #qs+=[23] #Everest files need extra quality flags (and do not need 15 apparently)...
                #    # 23 sometimes is very far-from-median flux islands, and somtimes transit points :/ So let's ignore 23
                self.flux_mask=self.flux_mask&(np.sum(np.vstack([self.quality.astype(int) & 2 ** (q - 1) for q in qs]),axis=0)==0)
            
            if in_transit is not None:
                out_of_trans=~in_transit
            elif hasattr(self, 'in_trans'):
                out_of_trans=~self.in_trans
            else:
                out_of_trans=np.tile(True, len(self.flux_mask))

            #Also must be between 0.75 and 1.25
            self.flux_mask[out_of_trans] *= ((self.flux[out_of_trans]<(self.flx_unit_dic[self.flx_system][1]+extreme_anom_limit/self.flx_unit_dic[self.flx_system][0])) & \
                                             (self.flux[out_of_trans]>(self.flx_unit_dic[self.flx_system][1]-extreme_anom_limit/self.flx_unit_dic[self.flx_system][0])))

            #print(np.sum(~lc['mask']),"points after quality flags")
            if cut_all_anom_lim>0:
                #Stacking 20 point-shifted lightcurves on top of each other for quick median filter: is (flux - median of 20pts)<threshold*MAD of 20pts
                stack_shitfed_flux=np.column_stack([getattr(self,flux_arr_name)[self.flux_mask][n:(-20+n)] for n in range(20)])
                self.flux_mask[self.flux_mask][10:-10]=abs(getattr(self,flux_arr_name)[self.flux_mask][10:-10] - np.nanmedian(stack_shitfed_flux,axis=1))<cut_all_anom_lim*np.nanmedian(abs(np.diff(stack_shitfed_flux,axis=1)),axis=1)
                #Now doing difference
                self.flux_mask[self.flux_mask*out_of_trans]=tools.CutAnomDiff(getattr(self,flux_arr_name)[self.flux_mask*out_of_trans],cut_all_anom_lim)
                '''
                #Doing this a second time with more stringent limits to cut two-point outliers:
                self.flux_mask[self.flux_mask]=CutAnomDiff(lc[prefix+'flux'+suffix][self.flux_mask],cut_all_anom_lim+3.5)
                '''
                #print(np.sum(~lc['mask']),"after before CutAnomDiff")

            if mask_islands:
                #Masking islands of data which are <12hrs long and >12hrs from other data
                time_regions=tools.find_time_regions(self.time)
                xmask=np.tile(True,np.sum(self.flux_mask))
                for j in range(len(time_regions)):
                    jump_before = 100 if j==0 else time_regions[j][0]-time_regions[j-1][1]
                    jump_after = 100 if j==(len(time_regions)-1) else time_regions[j+1][0]-time_regions[j][1]
                    if (time_regions[j][1]-time_regions[j][0])<0.5 and jump_before>0.5 and jump_after>0.5:
                        #ISLAND! NEED TO MASK
                        xmask[(self.time[self.flux_mask]>time_regions[j][0])*(self.time[self.flux_mask]<time_regions[j][1])]=False
                self.flux_mask[self.flux_mask]*=xmask

            #End-of-orbit cut
            # Arbritrarily looking at the first/last 15 points and calculating STD of first/last 300 pts.
            # We will cut the first/last points if the lightcurve STD is drastically better without them
            if end_of_orbit:
                stds=np.array([np.nanstd(getattr(self,flux_arr_name)[self.flux_mask][n:(300+n)]) for n in np.arange(0,17)])
                stds/=np.min(stds)
                newmask=np.tile(True,np.sum(self.flux_mask))
                for n in np.arange(15):
                    if stds[n]>1.05*stds[-1]:
                        newmask[n]=False
                        newmask[n+1]=False
                stds=np.array([np.nanstd(getattr(self,flux_arr_name)[self.flux_mask][(-300+n):n]) for n in np.arange(-17,0)])
                stds/=np.min(stds)
                for n in np.arange(-15,0):
                    if stds[n]>1.05*stds[0]:
                        newmask[n]=False
                        newmask[n-1]=False
                self.flux_mask[self.flux_mask]=newmask

            # Identify outliers
            m2 = self.flux_mask[:]

            for i in range(10):
                try:
                    y_prime = np.interp(self.time, self.time[m2], getattr(self,flux_arr_name)[m2])
                    smooth = savgol_filter(y_prime, 101, polyorder=3)
                    resid = getattr(self,flux_arr_name) - smooth
                    sigma = np.sqrt(np.nanmean(resid**2))
                    #m0 = abs(resid) < cut_all_anom_lim*sigma
                    # Making this term less likely to cut low-flux points...
                    m0 = (resid < 0.66*cut_all_anom_lim*sigma)&(resid > -1*cut_all_anom_lim*sigma)
                    #print(np.sum((y_prime/y_prime)!=1.0),np.sum(m2),np.sum(m0))
                    if m2.sum() == m0.sum():
                        m2 = m0
                        break
                    m2 = m0+m2
                except:
                    resid = np.zeros(len(getattr(self,flux_arr_name)))
                    sigma = 1.0
                    pass
        self.timeseries+=['flux_mask']
        self.make_mask()

    def flatten(self, timeseries=['flux'], knot_dist=1.25, maxiter=10, sigmaclip = 3., flattype='bspline', 
                stepsize=0.15, reflect=True, polydegree=3, transit_mask=None):
        """AI is creating summary for flatten

        Args:
            timeseries (list, optional): List of timeseries to flatten. Defaults to ['flux'].
            knot_dist (float, optional): Typical distance between spline "knots" OR polynomial fitting lengths. Defaults to 1.25.
            maxiter (int, optional): Maximum iterations to perform to find/remove anomalies. Defaults to 10.
            sigmaclip (float, optional): Significance of anomalies from the mean above which we clip. Defaults to 3..
            flattype (str, optional): Either use a 'bspline' (which fits smooth splines while iterating away anomalies/transits/etc)
                                      Or a 'polystep' (which uses out-of-box polynomials to smooth data without influencing transit depth). Defaults to 'bspline'.
            stepsize (float, optional): [description]. Defaults to 0.15.
            reflect (bool, optional): [description]. Defaults to True.
            polydegree (int, optional): [description]. Defaults to 3.
            transit_mask (np.ndarray, optional): Mask of transits where 0 = in transit and 1 = out of transit. Defaults to None.
        """
        self.sort_timeseries() #Requires timeseries sorted in time

        if flattype=='bspline':
            if transit_mask is None and hasattr(self,'in_trans') and np.sum(self.in_trans)>0 and type(transit_mask)!=np.ndarray:
                transit_mask = ~self.in_trans[:]
            for its in timeseries:
                timearr=self.bin_time[:] if 'bin_' in its else self.time[:]

                maskarr=self.mask[:] if 'bin_' not in its else None
                spline = tools.kepler_spline(timearr,getattr(self,its),flux_mask=maskarr,
                                             maxiter=maxiter,bk_space=knot_dist,transit_mask=transit_mask,reflect=reflect)[0]
                setattr(self, its+'_spline', spline)
                self.timeseries+=[its+'_spline']
                setattr(self, its+'_flat', getattr(self,its) - spline)
                self.timeseries+=[its+'_flat']

        elif flattype=='polystep':
            for its in timeseries:
                timearr=self.bin_time[:] if 'bin_' in its else self.time[:]

                if its+'_flat' not in self.timeseries:
                    setattr(self,its+'_flat',np.zeros(len(getattr(self,its))))

                if hasattr(self,its+'err'):
                    uselc=np.column_stack((timearr,getattr(self,its)[:],getattr(self,its+'_err')[:]))
                else:
                    uselc=np.column_stack((timearr,getattr(self,its)[:],np.tile(np.nanmedian(abs(np.diff(getattr(self,its)))),len(timearr))))
                if 'bin_' not in its:
                    initmask=(self.mask[:]&(self.flux[:]/self.flux==1.0)&(self.flux_err[:]/self.flux_err==1.0)).astype(int)[:]
                    if hasattr(self,'in_trans') and np.sum(self.in_trans)>0 and type(transit_mask)!=np.ndarray:
                        transit_mask = ~self.in_trans
                    if type(transit_mask)==np.ndarray:
                        if debug: print("transit mask:",type(initmask),len(initmask),
                            initmask[0],type(transit_mask),len(transit_mask),transit_mask[0])
                        initmask=(initmask.astype(bool)&transit_mask).astype(int)
                else:
                    initmask=(np.isfinite(uselc[:,1])&np.isfinite(uselc[:,2])).astype(int)
                uselc=np.column_stack((uselc,initmask))
                region_starts=uselc[1+np.hstack((-1,np.where(np.diff(np.sort(uselc[:,0]))>knot_dist)[0])),0]
                region_ends  =uselc[np.hstack((np.where(np.diff(np.sort(uselc[:,0]))>knot_dist)[0],len(uselc[:,0])-1)),0]
                
                stepcentres=[]
                uselc_w_reflect=[]
                for n in range(len(region_starts)):
                    useix=(uselc[:,0]>=region_starts[n])*(uselc[:,0]<=region_ends[n])
                    stepcentres+=[np.arange(region_starts[n],region_ends[n],stepsize) + 0.5*stepsize]
                    if reflect and (uselc[useix,0][-1]-uselc[useix,0][0])>0.8*knot_dist:
                        partlc=uselc[useix]
                        incad=np.nanmedian(np.diff(partlc[:,0]))
                        xx=[np.arange(np.nanmin(partlc[:,0])-knot_dist*0.4,np.nanmin(partlc[:,0])-incad,incad),
                            np.arange(np.nanmax(partlc[:,0])+incad,np.nanmax(partlc[:,0])+knot_dist*0.4,incad)]
                        #Adding the lc, plus a reflected region either side of each part.
                        # Also adding a boolean array to show where the reflected parts are
                        refl_t=np.hstack((xx[0],partlc[:,0],xx[1]))
                        refl_flux=np.vstack((partlc[:len(xx[0]),1:][::-1],
                                            partlc[:,1:],
                                            partlc[-1*len(xx[1]):,1:][::-1]  ))
                        refl_bool=np.hstack((np.zeros(len(xx[0])),np.tile(1.0,len(partlc[:,0])),np.zeros(len(xx[1]))))
                        #print(partlc.shape,len(xx[0]),len(xx[1]),refl_t.shape,refl_flux.shape,refl_bool.shape)
                        uselc_w_reflect+=[np.column_stack((refl_t,refl_flux,refl_bool))]
                    elif (uselc[useix,0][-1]-uselc[useix,0][0])<0.8*knot_dist:
                        uselc_w_reflect+=[np.column_stack((uselc[useix],np.tile(1.0,len(uselc[useix,0])) ))]
                stepcentres=np.hstack(stepcentres)
                if reflect:
                    uselc=np.vstack(uselc_w_reflect)
                else:
                    uselc=np.column_stack((uselc,np.ones(len(uselc[:,0])) ))
                #uselc[:,2]=np.clip(uselc[:,2],np.nanmedian(uselc[:,2])*0.8,100)
                #print(len(uselc),np.sum(uselc[:,3]),np.sum(uselc[:,4]))
                #now for each step centre we perform the flattening:
                #actual flattening
                for stepcent in stepcentres:
                    win,box = tools.formwindow(uselc,stepcent, knot_dist, stepsize,0.8*knot_dist)  #should return window around box not including box
                    newbox=box[uselc[:,4].astype(bool)] # Excluding from our box any points which are actually part of the "reflection"
                    #Checking that we have points in the box where the window is not entirely junk/masked
                    if np.sum(newbox)>0 and np.sum(win&uselc[:,3].astype(bool))>0:
                        #Forming the polynomial fit from the window around the box:
                        baseline = tools.dopolyfit(uselc[win,:3],mask=uselc[win,3].astype(bool),
                                            stepcent=stepcent,d=polydegree,ni=maxiter,sigclip=sigmaclip)
                        getattr(self, its+'_flat')[newbox] = getattr(self,its)[newbox] - np.polyval(baseline,timearr[newbox]-stepcent)
            self.timeseries+=[its+'_flat']
    
    def OOTbin(self,near_transit_mask,use_flat=False,binsize=1/48):
        """Out-Of-Transit binning of the lightcurve (to speed up computation)

        Args:
            near_transit_mask ([type]): [description]
            use_flat (bool, optional): [description]. Defaults to False.
            binsize (float, optional): [description]. Defaults to 1/48.
        """
        flux_name='flux' if not use_flat else 'flux_flat'

        #Binning the timeseries while masking the near-transit regions
        self.bin(timeseries=[flux_name], binsize=binsize, extramask=~near_transit_mask)
        if use_flat:
            self.bin_flux_err = self.bin_flux_flat_err
        
        #Stacking the binned out-of-transit regions with the unbinned near-transit data:
        ootlcdict={"ootbin_"+key:np.hstack((getattr(self,'bin_'+key)[np.isfinite(getattr(self,"bin_"+flux_name))],
                                            getattr(self,key)[near_transit_mask*self.mask])) for key in ['time',flux_name,'flux_err','cadence']}
        ootlcdict['ootbin_near_trans']=np.hstack((np.tile(False,np.sum(np.isfinite(getattr(self,"bin_"+flux_name)))),
                                                  np.tile(True,np.sum(near_transit_mask*self.mask)) ))
        assert hasattr(self,'in_trans')
        ootlcdict['ootbin_in_trans']=np.hstack((np.tile(False,np.sum(np.isfinite(getattr(self,"bin_"+flux_name)))),
                                        self.in_trans[near_transit_mask*self.mask] ))

        #Sorting these timeseries by the stacked time
        for key in ["ootbin_"+flux_name,'ootbin_flux_err','ootbin_cadence','ootbin_near_trans','ootbin_in_trans']:
            setattr(self,key,ootlcdict[key][np.argsort(ootlcdict['ootbin_time'])])
        setattr(self,"ootbin_time",np.sort(ootlcdict['ootbin_time']))

        #Rebinning without the near-transit mask
        self.bin(timeseries=['flux','flux_flat'],binsize=binsize)

    def bin(self,timeseries=['flux'],binsize=1/48,split_gap_size=0.8,use_masked=True, extramask=None, overwrite=False, **kwargs):
        """Binning lightcurve to e.g. 30-min cadence for planet search

        Args:
            timeseries (list, optional): List of timeseries which to bin. Defaults to ['flux']
            binsize (float, optional): Size of bins in units matching lightcurve time. Defaults to 1/48.
            split_gap_size (float, optional): Size to count as a "gap" in the lightcurve. Defaults to 0.8.
            use_masked (bool, optional): whether to bin using only the masked flux array. Defaults to True.
            extramask (np.ndarray, optional): Added mask to use when binning, otherwise either only `mask` or nothing is used. Defaults to None.
            overwrite (bool, optional): Whether to overwrite already-stored binned data array. Defaults to False
        """
        
        if not np.all(np.sort(self.time)==self.time):
            self.sort_timeseries()

        if np.any(['_flat' in its and its not in self.timeseries for its in timeseries]):
            self.flatten(timeseries=timeseries)
            
        #setattr(self, 'bin_cadence',binlc['flux'][:,0])

        #Initialising mask to use:
        if extramask is not None and type(extramask)==np.ndarray and (type(extramask[0])==bool)|(type(extramask[0])==np.bool_) and use_masked:
            mask=(self.mask&extramask).astype(bool)
        elif use_masked:
            mask=self.mask.astype(bool)
        else:
            #Must use cad_mask as this excludes duplicates
            mask=self.cad_mask.astype(bool)
        
        bintime=[]
        time_bools=np.zeros(len(self.time))
        bintime_bools=[]

        #Found lightcurve gaps - making shorter blocks to loop through.
        if np.nanmax(np.diff(np.sort(self.time)))>split_gap_size:
            #We have gaps in the lightcurve, so we'll find the bins by looping through those gaps
            time_regions=tools.find_time_regions(self.time)
            for j in range(len(time_regions)):
                time_bools[mask*(self.time>=time_regions[j][0])*(self.time<=time_regions[j][1])]=int(j+1)
                cad=np.nanmedian(np.diff(self.time[time_bools==j+1]))
                if binsize>(cad*1.5):
                    bintime+=[np.arange(time_regions[j][0],time_regions[j][1]+binsize,binsize)]
                else:
                    bintime+=[self.time[time_bools==(j+1)]]
                bintime_bools+=[np.tile(j+1,len(bintime[-1]))]
        else:
            time_bools[mask]=1
            cad=np.nanmedian(np.diff(self.time))
            if binsize>(cad*1.5):
                bintime+=[np.arange(np.nanmin(self.time),np.nanmax(self.time)+binsize,binsize)]
            else:
                bintime+=[self.time[mask]]
            bintime_bools+=[np.ones(len(bintime[-1]))]

        setattr(self,'bin_time', np.hstack((bintime)))
        bintime_bools=np.hstack((bintime_bools))

        #For each of the seprated lightcurve blocks:
        setattr(self,'bin_cadence',np.empty(len(bintime_bools), dtype='<U16'))
        for fkey in timeseries:
            #Initialising binned arrays
            setattr(self,'bin_'+fkey,np.zeros(len(bintime_bools)))
            setattr(self,'bin_'+fkey+'_err',np.zeros(len(bintime_bools)))
            self.timeseries+=['bin_time','bin_'+fkey,'bin_'+fkey+'_err','bin_cadence']
            #print(fkey,self.bin_time.shape,getattr(self,'bin_'+fkey).shape,getattr(self,'bin_'+fkey+'_err').shape)
            for j in np.arange(1,1+np.max(time_bools)).astype(int):
                #For each of the flux arrays (binned and normal):
                ierrs=getattr(self,fkey+'_err')[time_bools==j] if hasattr(self,fkey+'_err') and not np.all(np.isnan(getattr(self,'flux_err')[time_bools==j])) else getattr(self,'flux_err')[time_bools==j]
                
                #Using the pre-computed "bintime_bools" and "time_bools" to index the (empty) binned array and the self.lc.time one...
                if np.sum(bintime_bools==j)==np.sum(time_bools==j):
                    #Cadence is ~binsize, so we can just take the raw flux values - i.e. no binning
                    getattr(self,'bin_'+fkey)[bintime_bools==j]=getattr(self,fkey)[time_bools==j]
                    getattr(self,'bin_'+fkey+'_err')[bintime_bools==j]=ierrs
                    if np.all(getattr(self,'bin_cadence')[bintime_bools==j]=='##-####-####-##'):
                        getattr(self,'bin_cadence')[bintime_bools==j]=self.cadence[time_bools==j]
                else:
                    #Only doing the binning if the cadence involved is >> the cadence
                    digi=np.digitize(self.time[time_bools==j],bintime[j-1])
                    binnedlc = np.vstack([[tools.weighted_avg_and_std(getattr(self,fkey)[time_bools==j][digi==d],ierrs[digi==d])] for d in np.arange(len(bintime[j-1]))])
                    getattr(self,'bin_'+fkey)[bintime_bools==j]=binnedlc[:,0]
                    getattr(self,'bin_'+fkey+'_err')[bintime_bools==j]=binnedlc[:,1]
                    if np.all(getattr(self,'bin_cadence')[bintime_bools==j]==''):
                        getattr(self,'bin_cadence')[bintime_bools==j] = np.array([self.cadence[time_bools==j][digi==d][-1] if d in digi else '' for d in np.arange(len(bintime[j-1]))])
        self.timeseries=list(np.unique(self.timeseries))

class multilc(lc):
    """A flexible lightcurve class built from multiple individual lightcurves
       - i.e. a combination of TESS sectors, or an ensemble of both CoRoT, K2 and TESS data

    Example:
    mylc=lc(203311200,'k2')
    mylc.
    """
    def __init__(self,id,mission,radec=None,load=True, do_search=True,flx_system='ppt',jd_base=2457000.,savefileloc=None,extralc=None):
        """AI is creating summary for __init__

        Args:
            id (int): Identifier
            mission (str): Mission associated with ID
            radec (astropy.coordinates.SkyCoord, optional): Ra/Dec object for . Defaults to None.
            load (bool,optional): Whether to load lightcurve object from file. Defauls to True
            do_search (bool,optional): Whether to call `get_all_lightcurves` when initialising. Defauls to True
            flx_system (str,optional): Default flux system to use. Defaults to 'ppt'
            jd_base (float,optional): Base of timing system in julian date. Defaults to 2457000
            savefileloc (str,optional): File location to load and/or save lightcurve. Defaults to a new target-specific folder in `tools.MonoData_savepath`
        """
        self.savefileloc = os.path.join(tools.MonoData_savepath,tools.id_dic[mission]+str(id).zfill(11),tools.id_dic[mission]+str(id).zfill(11)+'_lc.pkl.gz') if savefileloc is None else savefileloc
        if load and os.path.exists(self.savefileloc):
            with gzip.open(self.savefileloc, "rb") as f:
                pick = pickle.load(f)
                assert not isinstance(pick, multilc)
                #print(In this case, unpickle your object separately)
                for key in pick:
                    setattr(self,key,pick[key])
        else:
            if radec is not None:
                self.radec=radec
            self.id=int(id)
            self.mission=mission.lower()
            super(lc, self).__init__()
            #The above initalisation of an lc class does not copy over the default arrays/lists, so we'll do that here:

            self.cadence_list=[]
            self.mask_cadences=[] # List of cadences to mask (i.e. when there are multiple timeseries together)

            self.flx_unit_dic = {'ppm':[1e-6,0],'ppt':[1e-3,0],'norm0':[1.0,0],'norm1':[1.0,1]}
            self.lc_dic={'tess':'ts','kepler':'k1','k2':'k2','corot':'co','cheops':'ch'}
            self.timeseries=[]
            self.flx_system=flx_system
            self.jd_base=jd_base
            
            self.info={}
            self.all_ids={'tess':{},'k2':{},'kepler':{},'corot':{}}
            self.all_ids[mission]={'id':id}
        if do_search:
            self.get_all_lightcurves(extralc=extralc)

    def stack(self, newlcs, priorities=None):
        """Stacks lightcurves onto the multilc object

        Args:
            newlcs (list): A list of lightcurve classes (either multilc or lc) to add together
            priorities (list, optional): Which lightcurve sources take priority in the case that they overlap in timing?
                                         This is required in the case that lightcurves are overlapping... Defaults to None.
        """
        if priorities is None:
            #Here we can list the priorities for Kepler/K2 and TESS:
            priorities=["k1_120_pdc","k1_1800_pdc","k2_120_ev","k2_120_vand","k2_120_pdc","k2_1800_ev","k2_1800_vand","k2_1800_pdc","ts_20_pdc","ts_120_pdc","ts_1800_pdc","ts_1800_qlp","ts_1800_el"]

        # Tidying up before stacking:
        if newlcs is not None and len(newlcs)>0:
            #Initialising the flux system and jd base based on the first new lightcurve we add
            if not hasattr(self,'flx_system'):
                self.flx_system=newlcs[0].flx_system
            if not hasattr(self,'flx_unit'):
                self.flx_unit=self.flx_unit_dic[self.flx_system][0]
            if not hasattr(self,'jd_base'):
                self.jd_base=newlcs[0].jd_base
            uniform_flux_sys=self.flx_system if hasattr(self,'flx_system') else newlcs[0].flx_system
            uniform_jd_base=self.jd_base if hasattr(self,'jd_base') else newlcs[0].jd_base
            #print([newlc.jd_base for newlc in newlcs],'->',uniform_jd_base)

        for newlc in newlcs:
            # Making sure new flux system matches this flux system:
            newlc.change_flx_system(uniform_flux_sys)
            # Making sure new timing matches this timing:
            newlc.change_jd_base(uniform_jd_base)
            # Removing binned arrays when stacking
            newlc.remove_binned_arrs()
            if not hasattr(newlc,'mask') or np.isnan(newlc.mask).sum()>0:
                newlc.make_mask()
                #print(newlc.mask,np.sum(newlc.mask))
        # Making sure we just have pure timeseries (no binned timeseries)
        self.remove_binned_arrs()

        # Stacking each timeseries on top of each other
        for newlc in newlcs:
            # Looping over all the unique cadences (i.e. unique timeseries/sectors/etc) in newlc
            for unq_cad in np.unique(newlc.cadence):
                #print(unq_cad,newlc.mask,np.sum(newlc.mask),self.mask,np.sum(self.mask))
                # Detecting if there is a matching mission/sector in the current class:
                matching_cads = [cad for cad in self.cadence_list if unq_cad[:2]==cad[:2] and unq_cad.split('_')[-1]==cad.split('_')[-1]]
                if unq_cad in matching_cads:
                    #Lightcurves have the exact same mission/cadence/pipeline/sector - we will not stack them
                    continue
                if len(matching_cads)>0:
                    for matching_cad in matching_cads:
                        #print(matching_cads,priorities)
                        #We have an overlap. Adding one lightcurve cadence to the list of those to mask, depending on priorities:
                        if unq_cad in np.unique(self.cadence):
                            #Here we appear to have exactly matching datasets... So we'll skip this whole stacking step and keep the version already in the self.lc
                            continue
                        elif priorities.index('_'.join(matching_cad.split('_')[:3]))<priorities.index('_'.join(unq_cad.split('_')[:3])):
                            # In this case, the new lightcurve is lower priority, so gets added to the mask_cadences list:
                            self.mask_cadences+=[unq_cad]
                        else:
                            # newlc takes priority
                            self.mask_cadences+=[matching_cad]
                # Now we can stack necessary info:
                self.info[unq_cad] = newlc.info[unq_cad]
                self.cadence_list+=[unq_cad]
                
                # Looping over and stacking all timeseries:
                for ikey in np.unique(newlc.timeseries+self.timeseries):
                    if ikey!='time':
                        if ikey not in self.timeseries:
                            if hasattr(self,'time'):
                                #print(ikey,"not in self.timeseries which has length",len(self.time))
                                # New timeseries for the current class. Adding nans to make up lengths
                                setattr(self,ikey,np.hstack((np.tile(np.nan,len(self.time)),getattr(newlc,ikey)[newlc.cadence==unq_cad])))
                            else:
                                #print(ikey,"not in self.timeseries which has 0 length",getattr(newlc,ikey)[newlc.cadence==unq_cad])
                                setattr(self,ikey,getattr(newlc,ikey)[newlc.cadence==unq_cad])
                            self.timeseries+=[ikey]
                        elif ikey not in newlc.timeseries:
                            # Timeseries in current class but not in new one. Adding nans to make up lengths
                            setattr(self,ikey,np.hstack((getattr(self,ikey),np.tile(np.nan,np.sum(newlc.cadence==unq_cad)))))
                        else:
                            # Timeseries in both classes:
                            setattr(self,ikey,np.hstack((getattr(self,ikey),getattr(newlc,ikey)[newlc.cadence==unq_cad])))
                
                #Finally updating the time array:
                if hasattr(self,'time'):
                    self.time = np.hstack((self.time,newlc.time[newlc.cadence==unq_cad]))
                else:
                    self.time = newlc.time[newlc.cadence==unq_cad]
                    self.timeseries+=['time']
                
                #print(unq_cad,newlc.mask,np.sum(newlc.mask),self.mask,np.sum(self.mask))
            
    def get_radec(self):
        """Get radec from ID by searching the input catalogue
        """
        print("Accessing online catalogues to match ID to RA/Dec (may be slow)","mission=",self.mission)
        if self.mission=='k2':
            k2_cat = Vizier(catalog=['J/ApJS/224/2']).query_object("EPIC"+str(int(self.id)), catalog=['J/ApJS/224/2'],radius=1*u.arcsec)
            print(k2_cat)
            if 'J/ApJS/224/2/table5' in k2_cat.keys():
                k2_cat=k2_cat[0].to_pandas()
                k2_cat=k2_cat.loc[k2_cat['EPIC']==self.id]
                k2_cat=k2_cat.iloc[0] if type(k2_cat)==pd.DataFrame else k2_cat
                self.radec = SkyCoord(k2_cat['RAJ2000']*u.deg,k2_cat['DEJ2000']*u.deg)
                self.all_ids['k2']['data']=k2_cat
        elif self.mission=='tess':
            tess_cat=Catalogs.query_object("TIC"+str(int(self.id)), catalog="TIC", radius=1*u.arcsec).to_pandas()
            tess_cat=tess_cat.loc[tess_cat['ID'].values.astype(int)==self.id]
            self.radec = SkyCoord(tess_cat['ra']*u.deg,tess_cat['dec']*u.deg)
            self.all_ids['tess']['data']=tess_cat
        elif self.mission=='kepler':
            res=Vizier(catalog=['V/133/kic']).query_object("KIC"+str(int(self.id)),radius=1*u.arcsec)
            if 'V/133/kic' in res.keys():
                kep_cat=res[0].to_pandas()
                kep_cat=kep_cat.loc[kep_cat['KIC']==self.id]
                kep_cat=kep_cat.iloc[0] if type(kep_cat)==pd.DataFrame else kep_cat
                self.radec=SkyCoord(kep_cat['RAJ2000']*u.deg,kep_cat['DEJ2000']*u.deg)
                self.all_ids['kepler']['data']=kep_cat
        if type(self.radec.ra.deg)==np.ndarray:
            self.radec=self.radec[0]
    
    def get_all_survey_ids(self,search=['all'],overwrite=False,**kwargs):
        """Given the source ID and mission, search all photometric missions for both the coplementary mission IDs, and extra catalogue data/observability

        Args:
            search (list, optional): A list of missions to search. Defaults to ['all'].
            overwrite (bool, optional): Whether to overwrite the info already stored in `all_ids`. Defaults to False.
        """
        if not hasattr(self,'radec') and search!=[self.mission]:
            #OK - we need a coordinate. Hacking one from the ID...
            self.get_radec()
        
        # K2 ID and data:
        if (overwrite or self.all_ids['k2']=={}) and ('all' in search or 'k2' in search):
            v = Vizier(catalog=['J/ApJS/224/2'])
            res = v.query_region(self.radec, radius=5*u.arcsec,cache=False)
            print(res)
            if len(res)>0 and len(res[0])>0:
                self.all_ids['k2']={'id':res[0]['EPIC'][0]}
        if (self.all_ids['k2']!={} and 'id' in self.all_ids['k2']) and (overwrite or 'search' not in self.all_ids['k2'] or self.all_ids['k2']['search'] is None or len(self.all_ids['k2']['search'])==0) and ('all' in search or 'k2' in search):
            self.all_ids['k2']['search']=self.get_K2_campaigns()
        
        # TESS ID and data:
        if (overwrite or self.all_ids['tess']=={}) and ('all' in search or 'tess' in search):
           
            tess_id = Catalogs.query_criteria(coordinates=self.radec,radius=12*u.arcsec,catalog="TIC",
                                                objType="STAR",columns=['ID','KIC','Tmag']).to_pandas()
            if tess_id is not None and len(tess_id)>0:
                tess_id=tess_id.iloc[np.argmin(tess_id['Tmag'])] if type(tess_id)==pd.DataFrame else tess_id
                self.all_ids['tess']={'id':tess_id['ID']}
        if (self.all_ids['tess']!={} and 'id' in self.all_ids['tess']) and (overwrite or 'search' not in self.all_ids['tess'] or self.all_ids['tess']['search'] is None or len(self.all_ids['tess']['search'])==0) and ('all' in search or 'tess' in search):
            self.all_ids['tess']['search']=self.get_tess_sectors()

        # Kepler ID and data:
        if (overwrite or self.all_ids['kepler']=={}) and ('all' in search or 'kepler' in search):
            v = Vizier(catalog=['V/133/kic'])
            res=v.query_region(self.radec, radius=5*u.arcsec, catalog=['V/133/kic'])
            if 'V/133/kic' in res.keys():
                if len(res['V/133/kic'])>1:
                    #print(res['V/133/kic'][['KIC','kepmag']], "MULTIPLE KICS FOUND")
                    self.all_ids['kepler'] = {'id':res['V/133/kic']['KIC'][np.argmin(res['V/133/kic']['kepmag'])]}
                elif len(res['V/133/kic'])==1:
                    #print(res['V/133/kic'][['KIC','kepmag']], "ONE KIC FOUND")
                    self.all_ids['kepler'] = {'id':res['V/133/kic']['KIC'][0]}
        if (self.all_ids['kepler']!={} and 'id' in self.all_ids['kepler']) and (overwrite or 'search' not in self.all_ids['kepler'] or self.all_ids['kepler']['search'] is None or len(self.all_ids['kepler']['search'])==0) and ('all' in search or 'kepler' in search):
            self.all_ids['kepler']['search']=np.arange(18)

        # CoRoT ID and data:
        if (overwrite or self.all_ids['corot']=={}) and ('all' in search or 'corot' in search):
            #We need to do this process to access the fits files as to find the ID, we can do this at a later point
            df=self.get_corot_campaigns()
            if df is not None and len(df)>0:
                self.all_ids['corot']={'id':df['ID'].values[0],
                                       'search':df}

    def get_all_lightcurves(self,all_pipelines=False,extralc=None,**kwargs):
        """
        Download all available space photometry for a target. This uses the info stored in `all_ids`

        Args:
            all_pipelines (bool, optional): Whether to download (i.e. simultaneous) photometry from multiple pipelines? Defaults to False.
        """
        #Checking we have lightcurve locations to search:
        if not np.any(['search' in iddic and iddic['search'] is not None for iddic in self.all_ids]) or ('overwrite' in kwargs and kwargs['overwrite']):
            print("Getting all IDs")
            self.get_all_survey_ids(**kwargs)
            #print(self.all_ids)
        all_lcs = [] if extralc is None else [extralc]

        if self.all_ids['tess'] is not None and self.all_ids['tess'] is not {} and 'search' in self.all_ids['tess'] and self.all_ids['tess']['search'] is not None:
            for sector in self.all_ids['tess']['search']:
                searchlist=['20_spoc','120_spoc','1800_spoc','1800_qlp'] if all_pipelines else ['all']
                for search in searchlist:
                    all_lcs+=[self.get_tess_lc(sector,search=search,**kwargs)]
        if self.all_ids['k2'] is not None and self.all_ids['k2'] is not {} and 'search' in self.all_ids['k2'] and self.all_ids['k2']['search'] is not None:
            for campaign in self.all_ids['k2']['search']:
                searchlist=['ev','vand','pdc'] if all_pipelines else ['all']
                for search in searchlist:
                    all_lcs+=[self.get_k2_lc(campaign,**kwargs)]
        if self.all_ids['kepler'] is not None and self.all_ids['kepler'] is not {} and 'search' in self.all_ids['kepler'] and self.all_ids['kepler']['search'] is not None:
            for q in self.all_ids['kepler']['search']:
                all_lcs+=[self.get_kepler_lc(q,**kwargs)]
        if self.all_ids['corot'] is not None and self.all_ids['corot'] is not {} and 'search' in self.all_ids['corot'] and self.all_ids['corot']['search'] is not None:
            for info in self.all_ids['corot']['search'].iterrows():
                all_lcs+=[self.get_corot_lc(info[1]["fits_link"],info[1]["Run Code"],**kwargs)]
        #print([newlc.cadence for newlc in all_lcs if newlc is not None])
        #all_lcs=[lc for lc in all_lcs]
        assert len(all_lcs)>0
        #print(all_lcs)
        #print(hasattr(self,'mask'),self.mask,self.mask.shape,np.sum(self.mask))
        #print([hasattr(ilc,'mask') for ilc in all_lcs])
        self.stack([newlc for newlc in all_lcs if newlc is not None])
        #print(hasattr(self,'mask'),self.mask,self.mask.shape,np.sum(self.mask))
        self.make_mask()
        #hasattr(self,'mask'),self.mask,self.mask.shape,np.sum(self.mask))
        self.save()
    
    def get_K2_campaigns(self,id=None):
        """See which K2 campaigns observed a given target?

        Args:
            id (int, optional): EPIC ID. Defaults to None, in which case it is taken from `all_ids`

        Returns:
            list: List of campaigns in which the target was observed with K2
        """
        if id is None and self.mission.lower()=='k2':
            id = self.id
        elif id is None and 'k2' in self.all_ids:
            id = self.all_ids['k2']['id']
        assert id is not None

        if len(str(int(id)))==8 and str(int(id))[:2]=='60':
            #Engineering campaign, so we don't have a proper EPIC here.
            df=None
            v = Vizier(catalog=['J/ApJS/224/2'])
            res = v.query_region(self.radec, radius=5*u.arcsec, catalog=['J/ApJS/224/2'])
            if len(res)>0 and len(res[0])>0:
                other_epic=res[0]['EPIC'][0]
                obs_table = Observations.query_object("EPIC "+str(int(other_epic)),timeout=120)
                cands=list(np.unique(obs_table[obs_table['obs_collection']=='K2']['sequence_number'].data.data).astype(int).astype(str))
            else:
                cands=[]
            cands+=['E']
        else:
            from astroquery.mast import Observations
            #Normal K2 observation:
            df,_=starpars.GetExoFop(id,"k2")
            obs_table = Observations.query_object("EPIC "+str(int(id)))
            cands=list(np.unique(obs_table[obs_table['obs_collection']=='K2']['sequence_number'].data.data).astype(str))
        
        if df is None:
            df={'campaign':None}
        if df['campaign'] is None or (type(df['campaign']) in [str,list] and len(df['campaign'])==0):
            df['campaign']=','.join(cands)
        else:
            df['campaign']=','.join(cands+str(df['campaign']).split(','))
        df['campaign']=df['campaign'].replace('.0','')
        df['campaign']=df['campaign'].replace('102','10')
        return [c for c in np.unique(np.array(str(df['campaign']).split(','))) if c!='']

    def get_tess_sectors(self,id=None,sectors='all', **kwargs):
        """See which sectors are observable with TESS

        Args:
            id (int, optional): TIC ID. Defaults to None, in which case it is taken from `all_ids`
            sectors (str,list of np.ndarray, optional): Either 'all' (in which case we search all sectors), or a list/array of sectors to search. Defaults to 'all'.

        Returns:
            list: List of sectors observed with TESS for the specfied ID
        """
        if id is None and self.mission.lower()=='tess':
            id = self.id 
        elif id is None and 'tess' in self.all_ids:
            id = self.all_ids['tess']['id']
        assert id is not None

        #Using the JD today and the JD start of the first sector (TJD=1325.29278) to estimate most recent sector
        from astropy.time import Time
        most_recent_sect = int(np.ceil((Time(datetime.now().strftime("%Y-%m-%d")).jd-2458325.29278)/27.295))
        epoch=pd.read_csv(tools.MonoData_tablepath+"/tess_lc_locations.csv",index_col=0)
        if most_recent_sect not in epoch.index:
            epoch=tools.update_lc_locs(epoch,most_recent_sect)
        
        
        if sectors == 'all':
            if hasattr(self,'radec'):
                sect_obs=tools.observed(int(id),self.radec)
            else:
                sect_obs=tools.observed(int(id))
            epochs=[key for key in epoch.index if sect_obs[key]]

            if epochs==[]:
                #NO EPOCHS OBSERVABLE APPARENTLY. USING THE EPOCHS ON EXOFOP/TIC8
                toi_df=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
                if id in toi_df['TIC ID'].values:
                    print("FOUND TIC IN TOI LIST")
                    epochs=list(np.array(toi_df.loc[toi_df['TIC ID']==id,'Sectors'].values[0].split(',')).astype(int))
        elif type(sectors)==list or type(sectors)==np.ndarray:
            epochs=[s for s in sectors if s<=np.max(epoch.index)]
        else:
            epochs=[sectors]
        return epochs

    def get_corot_campaigns(self,id=None,search_radius=25.):
        """AI is creating summary for corot_observability

        Args:
            search_radius (float, optional): Square search "radius" (i.e. half boxlength) in arcseconds. Defaults to 15.

        Returns:
            pandas DataFrame: A table with each holding data for a CoRoT field
        """
        if id is None and self.mission.lower()=='corot':
            id = self.id 
        elif id is None and self.all_ids['corot'] is not None and self.all_ids['corot'] is not {} and 'id' in self.all_ids['corot']:
            #print(self.all_ids['corot'])
            id = self.all_ids['corot']['id']
        if self.radec is None:
            self.get_radec()

        if id is None:
            #Searching by RA/Dec
            if (abs(self.radec.ra.deg-97.5)>8 and abs(self.radec.dec.deg-2.5)>8.5) and (abs(self.radec.ra.deg-285)>8 and abs(self.radec.dec.deg-2.5)>9):
                #Not in either exoplanet field... No point searching
                return pd.DataFrame([])
            else:
                url="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec?bgApp=%2FETSS%2Fnph-etss&etss_dataset=corot_exo&chk_runcode=LRc01&chk_runcode=LRc02&chk_runcode=LRc03&chk_runcode=LRc04&chk_runcode=LRc05&chk_runcode=LRc06&chk_runcode=LRc07&chk_runcode=LRc08&chk_runcode=LRc09&chk_runcode=SRc01&chk_runcode=SRc02&chk_runcode=SRc03&chk_runcode=IRa01&chk_runcode=LRa01&chk_runcode=LRa02&chk_runcode=LRa03&chk_runcode=LRa04&chk_runcode=LRa05&chk_runcode=LRa06&chk_runcode=SRa01&chk_runcode=SRa02&chk_runcode=SRa03&chk_runcode=SRa04&chk_runcode=SRa05&" + \
                            "min_alpha="+str(self.radec.ra.deg-search_radius/3600)+"&max_alpha="+str(self.radec.ra.deg+search_radius/3600)+"&display_alpha=on&" + \
                            "min_delta="+str(self.radec.dec.deg-search_radius/3600)+"&max_delta="+str(self.radec.dec.deg+search_radius/3600)+"&display_delta=on&min_magnit_b=&max_magnit_b=&display_magnit_b=on&min_magnit_v=&max_magnit_v=&display_magnit_v=on&min_magnit_r=&max_magnit_r=&display_magnit_r=on&min_magnit_i=&max_magnit_i=&display_magnit_i=on&min_startdat=&max_startdat=&display_startdat=on&min_end_date=&max_end_date=&display_end_date=on&min_lc_mean_b=&max_lc_mean_b=&display_lc_mean_b=on&min_lc_rms_b=&max_lc_rms_b=&display_lc_rms_b=on&min_lc_mean_g=&max_lc_mean_g=&display_lc_mean_g=on&min_lc_rms_g=&max_lc_rms_g=&display_lc_rms_g=on&min_lc_mean_r=&max_lc_mean_r=&display_lc_mean_r=on&min_lc_rms_r=&max_lc_rms_r=&display_lc_rms_r=on&min_lc_mean=&max_lc_mean=&display_lc_mean=on&min_lc_rms=&max_lc_rms=&display_lc_rms=on&chk_exptime=32&chk_exptime=512&chk_exptime=-1&display_exptime=on&min_npts=&max_npts=&display_npts=on&min_median=&max_median=&display_median=on&min_median_unc=&max_median_unc=&display_median_unc=on&min_mean_value=&max_mean_value=&display_mean_value=on&min_median_abs_dev=&max_median_abs_dev=&display_median_abs_dev=on&min_dispersion=&max_dispersion=&display_dispersion=on&min_chisquared=&max_chisquared=&display_chisquared=on&min_n5sigma=&max_n5sigma=&display_n5sigma=on&min_f5sigma=&max_f5sigma=&display_f5sigma=on&min_teff=&max_teff=&display_teff=on&min_contfact=&max_contfact=&display_contfact=on&min_spectral_type=&max_spectral_type=&display_spectral_type=on&chk_lumclass=I&chk_lumclass=II&chk_lumclass=II-III&chk_lumclass=III&chk_lumclass=IV&chk_lumclass=IV%2B&chk_lumclass=IV-V&chk_lumclass=V-&chk_lumclass=V&chk_lumclass=V%2B&display_lumclass=on&chk_lumclass=null&etssquery=Submit"
        else:
            url="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec?bgApp=%2FETSS%2Fnph-etss&etss_dataset=corot_exo&etssdetail="+str(id)+"&etssfind=View"
        h=httplib2.Http()
        resp = h.request(url)
        import bs4 as bs
        sp = bs.BeautifulSoup(resp[1], 'lxml')
        tb = sp.find_all('table')[0]
        df = pd.read_html(str(tb),encoding='utf-8', header=0)[0].iloc[1:]
        if df.shape[0]==0:
            return pd.DataFrame([])
        else:
            df["ID"] = [int(k.split(' ')[0]) for k in df['CoRoT ID']] #CoRoT ID string has extra info, so we need to parse that correctly
            df["plot_link"] = [str(tag)[9:-38] for tag in tb.find_all('a') if "Plot Time Series" in tag]
            df["fits_link"] = ["https://exoplanetarchive.ipac.caltech.edu/"+k[k.find("exodata")+3:k.find("_lc.tbl")].replace("//","/").replace('lightcurve','FITSfiles')+'.fits' for k in df["plot_link"].values]
            if id is None:
                # Making sure we dont have multiple stars here (taking the brightest)
                self.all_ids["corot"] = df.iloc[np.argmin(df["Vmag"])]["ID"]
                df=df.loc[df["ID"]==self.all_ids["corot"]]
            return df
    
    def get_tess_lc(self,sector,search=['all'],use_fast=False,use_eleanor=False,**kwargs):
        """Access TESS ligthcurve for given sector

        Args:
            sector (int): sector to search
            search (list, optional): list of lightcurve sources to search. Defaults to ['all']. 
                                    This will iterate through 'spoc_20', 'spoc_120', 'spoc_1800'/'spoc_600', 'qlp_1800'/'qlp_600' and 'eleanor_1800'/'eleanor_600' until a lightcurve is found
            use_fast (bool, optional): Search for 20-second data? Defaults to False.
            use_eleanor (bool, optional): Search eleanor? Defaults to False.

        Returns:
            lightcurve.lc: TESS lightcurve
        """

        #['all','spoc_20','spoc_120','spoc_1800','qlp_1800','eleanor_1800']
        #use_ppt=True, coords=None, use_qlp=None, use_eleanor=None, data_loc=None, search_fast=False, **kwargs):

        #2=minute cadence data from MAST
        searched=[]
        h = httplib2.Http()
        strtid=str(int(self.id)).zfill(16)
        epoch=pd.read_csv(tools.MonoData_tablepath+"/tess_lc_locations.csv",index_col=0)
        if ('all' in search or 'spoc_20' in search) and use_fast:
            searched+=['te_120_spoc_'+str(sector)]
            type='fast-lc'
            fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(sector).zfill(4)+"/"+strtid[:4]+"/"+strtid[4:8] + \
                    "/"+strtid[-8:-4]+"/"+strtid[-4:]+"/tess"+str(epoch.loc[sector,'date'])+"-s"+str(sector).zfill(4)+"-" + \
                    strtid+"-"+str(epoch.loc[sector,'runid']).zfill(4)+"-a_"+type+".fits"
            resp = h.request(fitsloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    return self.read_from_file(hdus,fitsloc,mission='tess',sect=str(sector),src='spoc',**kwargs)
        if ('all' in search or 'spoc_120' in search):
            type='lc'
            fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(sector).zfill(4)+"/"+strtid[:4]+"/"+strtid[4:8] + \
                    "/"+strtid[-8:-4]+"/"+strtid[-4:]+"/tess"+str(epoch.loc[sector,'date'])+"-s"+str(sector).zfill(4)+"-" + \
                    strtid+"-"+str(epoch.loc[sector,'runid']).zfill(4)+"-s_"+type+".fits"
            searched+=['te_120_spoc_'+str(sector)]
            resp = h.request(fitsloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    return self.read_from_file(hdus,fitsloc,mission='tess',sect=str(sector),src='spoc',**kwargs)
        cad='1800' if sector<=26 else '600'
        if ('all' in search or 'spoc_1800' in search or 'spoc_600' in search):
            #Getting spoc 30min data:
            fitsloc='https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc/s'+str(int(sector)).zfill(4) + \
                    "/target/"+strtid[:4]+"/"+strtid[4:8]+"/"+strtid[8:12]+"/"+strtid[12:] + \
                    "/hlsp_tess-spoc_tess_phot_"+strtid+"-s"+str(int(sector)).zfill(4)+"_tess_v1_lc.fits"
            resp = h.request(fitsloc, 'HEAD')
            
            searched+=['te_'+cad+'_spoc_'+str(sector)]
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    return self.read_from_file(hdus,fitsloc,mission='tess',sect=str(sector),src='spoc',**kwargs)

        if ('all' in search or 'qlp_1800' in search or 'qlp_600' in search):
            #QLP orbit files stored in folder:
            orbits=[7+sector*2,8+sector*2]
            qlpfiles=['/'.join(self.savefileloc.split('/')[:-1])+"/orbit-"+str(int(orbits[n]))+"_qlplc.h5" for n in range(2)]
            import h5py
            if os.path.isfile(qlpfiles[0]) and os.path.isfile(qlpfiles[1]):
                f1=h5py.File(qlpfiles[0])
                f2=h5py.File(qlpfiles[1])
                return self.read_from_file([f1,f2],qlpfiles[0],mission='tess',sect=str(sector),src='qlp',**kwargs)
            else:
                fitsloc='https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/qlp/s'+str(int(sector)).zfill(4) + \
                        "/"+strtid[:4]+"/"+strtid[4:8]+"/"+strtid[8:12]+"/"+strtid[12:] + \
                        "/hlsp_qlp_tess_ffi_s"+str(int(sector)).zfill(4)+"-"+strtid+"_tess_v01_llc.fits"
                #print("QLP:",fitsloc)
                resp = h.request(fitsloc, 'HEAD')
                if int(resp[0]['status']) < 400:
                    with fits.open(fitsloc,show_progress=False) as hdus:
                        return self.read_from_file(hdus,fitsloc,mission='tess',src='qlpfts',sect=str(sector),**kwargs)
            searched+=['te_'+cad+'_qlp_'+str(sector)]
        if ('all' in search or 'eleanor_1800' in search or 'eleanor_600' in search) and use_eleanor:
            import eleanor
            print("Loading Eleanor Lightcurve")
            try:
                #Getting eleanor lightcurve:
                try:
                    star = eleanor.Source(tic=id, sector=sector)
                except:
                    star = eleanor.Source(coords=self.radec, sector=sector)
                try:
                    elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True,save_postcard=False)
                except:
                    try:
                        elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=False,save_postcard=False)
                    except:
                        elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False,save_postcard=False)
                elen_hdr={'ID':star.tic,'GaiaID':star.gaia,'Tmag':star.tess_mag,
                        'RA':star.coords[0],'dec':star.coords[1],'mission':'TESS','campaign':sector,'source':'eleanor',
                        'ap_masks':elen_obj.all_apertures,'ap_image':np.nanmedian(elen_obj.tpf[50:80],axis=0)}
                return self.read_from_file(elen_obj,elen_hdr,mission='tess',src='el',sect=str(sector),**kwargs)
            except Exception as e:
                print(e)
            searched+=['te_'+cad+'_eleanor_'+str(sector)]
        return None
    

    def get_k2_lc(self,camp,id=None,search=['all'],**kwargs):
        """Get K2 lightcurve for a single campaign

        Args:
            camp (str or int): K2 Campaign
            id (str or int,optional): EPIC ID of target. Defaults to None, in which case the ID is taken from `all_ids`
            search (list, optional): Which pipelines to search. Defaults to ['all'] which will iterate through 'everest','vand' and then 'pdc' until it finds a lightcurve

        Returns:
            lightcurve.lc: Detrended K2 lightcurve
        """

        if id is None and self.mission.lower()=='k2':
            id = self.id 
        elif id is None and 'k2' in self.all_ids:
            id = self.all_ids['k2']['id']
        assert id is not None
        if ('all' in search or 'everest' in search) and camp!='E':
            ilc = self.get_everest_k2_lc(int(id), int(float(camp)),**kwargs)
            if ilc is not None:
                return ilc
        if ('all' in search or 'vand' in search):
            ilc = self.get_vanderburg_k2_lc(int(id), camp)
            if ilc is not None:
                return ilc
        if ('all' in search or 'pdc' in search):
            ilc = self.get_pdc_k2_lc(int(id),int(float(camp)),**kwargs)
            if ilc is not None:
                return ilc

    def get_kepler_lc(self, q, id=None, get_short_cadence=True,**kwargs):
        """Get Kepler lightcurve for single quarter

        Args:
            q ([type]): [description]
            id ([type], optional): [description]. Defaults to None.
            get_short_cadence (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if id is None and self.mission.lower()=='kepler':
            id = self.id 
        elif id is None and self.all_ids['kepler'] is not None and self.all_ids['kepler']!={}:
            id = self.all_ids['kepler']['id']
        assert id is not None
        llcs = {0: ["2009131105131"],1: ["2009166043257"],2: ["2009259160929"],3: ["2009350155506"],4: ["2010078095331", "2010009091648"],
                5: ["2010174085026"],6: ["2010265121752"],7: ["2010355172524"],8: ["2011073133259"],9: ["2011177032512"],10: ["2011271113734"],
                11: ["2012004120508"],12: ["2012088054726"],13: ["2012179063303"],14: ["2012277125453"],15: ["2013011073258"],16: ["2013098041711"],17: ["2013131215648"]}

        # Quarter index to filename prefix for short cadence Kepler data.
        # Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
        slcs = {0: ["2009131110544"],1: ["2009166044711"],2: ["2009201121230", "2009231120729", "2009259162342"],3: ["2009291181958", "2009322144938", "2009350160919"],
                4: ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],5: ["2010111051353", "2010140023957", "2010174090439"],
                6: ["2010203174610", "2010234115140", "2010265121752"],7: ["2010296114515", "2010326094124", "2010355172524"],
                8: ["2011024051157", "2011053090032", "2011073133259"],9: ["2011116030358", "2011145075126", "2011177032512"],
                10: ["2011208035123", "2011240104155", "2011271113734"],11: ["2011303113607", "2011334093404", "2012004120508"],
                12: ["2012032013838", "2012060035710", "2012088054726"],13: ["2012121044856", "2012151031540", "2012179063303"],
                14: ["2012211050319", "2012242122129", "2012277125453"],15: ["2012310112549", "2012341132017", "2013011073258"],
                16: ["2013017113907", "2013065031647", "2013098041711"],17: ["2013121191144", "2013131215648"]}

        keplcs=[]
        if get_short_cadence:
            lclocs=['http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(id)).zfill(9)[0:4]+'/'+str(int(id)).zfill(9)+'/kplr'+str(int(id)).zfill(9)+'-'+str(iq)+'_slc.fits' for iq in slcs[q]]
            h = httplib2.Http()
            resps = np.array([int(h.request(lcloc, 'HEAD')[0]['status'])<400 for lcloc in lclocs])
            if np.any(resps):
                return self.read_from_file([fits.open(lcloc,show_progress=False) for lcloc in np.array(lclocs)[resps]],fname=lclocs[0],sect=q,src='pdc',mission='kepler',**kwargs)
        lclocs=['http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(id)).zfill(9)[0:4]+'/'+str(int(id)).zfill(9)+'/kplr'+str(int(id)).zfill(9)+'-'+str(iq)+'_llc.fits' for iq in llcs[q]]
        #print(lclocs)
        h = httplib2.Http()
        resps = np.array([int(h.request(lcloc, 'HEAD')[0]['status'])<400 for lcloc in lclocs])
        if np.any(resps):
            return self.read_from_file([fits.open(lcloc,show_progress=False) for lcloc in np.array(lclocs)[resps]],fname=lclocs[0],sect=q,src='pdc',mission='kepler',**kwargs)

    def get_corot_lc(self,fitslink,campaign,**kwargs):
        """Get single CoRoT lightcurve from link to fits file

        Args:
            fitslink (str): HTTP path to fits file on MAST/exoarchive
            campaign (str): CoRoT campaign ID

        Returns:
            lightcurve.lc: CoRoT lightcurve
        """
        return self.read_from_file(fits.open(fitslink,show_progress=False),fitslink,src='esa',mission='corot',sect=campaign,**kwargs)

    def get_vanderburg_k2_lc(self,id,camp,v=1,**kwargs):
        """Get single K2SFF (Vanderburg et al 2013) K2 lightcurve for selected campaign

        Args:
            id (str): EPIC ID of target
            camp (str): K2 Campaign
            v (int, optional): Version of pipeline. Defaults to 1.

        Returns:
            lightcurve.lc: Detrended K2 lightcurve
        """
        #camp=camp.split(',')[0] if len(camp)>3
        if camp=='et' or camp=='E' or camp=='e':
            camp='e'
        else:
            camp=str(int(float(camp))).zfill(2)
        h = httplib2.Http()
        if camp in ['09','10','11']:
            lcvand=[]
            url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(id)[:4]+'00000/'+str(id)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(id)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
            resp1 = h.request(url1, 'HEAD')
            if int(resp1[0]['status']) < 400:
                lcvand+=[fits.open(url1,show_progress=False)]
            url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(id)[:4]+'00000/'+str(id)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(id)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
            resp2 = h.request(url2, 'HEAD')
            if int(resp2[0]['status']) < 400:
                lcvand+=[fits.open(url2,show_progress=False)]
            return self.read_from_file(lcvand,url1,mission='k2',src='vand',sect=camp,**kwargs)
        elif camp=='e':
            url='https://www.cfa.harvard.edu/~avanderb/k2/ep'+str(id)+'alldiagnostics.csv'
            return self.read_from_file(pd.read_csv(url,index_col=False),url,mission='k2',src='vand',sect=camp,jd_base=2454833,flx_system='norm1',**kwargs)
        else:
            urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(id)[:4]+'00000/'+str(id)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(id)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
            resp = h.request(urlfitsname, 'HEAD')
            if int(resp[0]['status']) < 400:
                return self.read_from_file(fits.open(urlfitsname,show_progress=False),urlfitsname,mission='k2',src='vand',sect=camp,**kwargs)

    def get_everest_k2_lc(self,id,camp):
        """Get single Everest (Luger et al 2016) K2 lightcurve for selected campaign

        Args:
            id (int): EPIC ID of target
            camp (str): K2 Campaign

        Returns:
            lightcurve.lc: Detrended K2 lightcurve
        """
        import everest
        if camp in [10,11,10.0,11.0,'10','11','10.0','11.0']:
            camps=[int(str(int(float(camp)))+'1'),int(str(int(float(camp)))+'2')]
        else:
            camps=[int(float(camp))]
        lcev={}
        camps=np.unique(np.array(camps))
        hdr=None
        for c in camps:
            try:
                st1=everest.Everest(int(id),season=c,show_progress=False)
                if lcev=={}:
                    lcev={'time':st1.time,
                        'flux':st1.flux,
                        'flux_err':st1.fraw_err,
                        'raw_flux':st1.fraw,
                        'raw_flux_err':st1.fraw_err,
                        'quality':st1.quality}
                else:
                    lcev={'time':np.hstack((lcev['time'],st1.time)),
                        'flux':np.hstack((lcev['flux'],st1.flux)),
                        'flux_err':np.hstack((lcev['flux_err'],st1.fraw_err)),
                        'raw_flux':np.hstack((lcev['raw_flux'],st1.fraw)),
                        'raw_flux_err':np.hstack((lcev['raw_flux_err'],st1.fraw_err)),
                        'quality':np.hstack((lcev['quality'],st1.quality))}
                hdr={'cdpp':st1.cdpp,'ID':st1.ID,'Tmag':st1.mag,'mission':'K2','name':st1.name,'campaign':camp,'lcsource':'everest','jd_base':2454833,'flx_system':'elec'}
            except:
                print(c,"not possible to load")
                return None
        if hdr is not None:
            lcev.update(hdr)
            out=self.read_from_file(lcev,hdr,mission='k2',sect=camp,src='ev')
            return out
    
    def get_pdc_k2_lc(self,id,camp,**kwargs):
        """Get single PDC K2 lightcurve for selected campaign

        Args:
            id (int): EPIC ID of target
            camp (str): K2 Campaign
            v (int, optional): Version of pipeline. Defaults to 1.

        Returns:
            lightcurve.lc: Detrended K2 lightcurve
        """
        if camp == '10':
            urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(id)[:4]+'00000/'+str(id)[4:6]+'000/ktwo'+str(id)+'-c102_llc.fits'
        else:
            urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(id)[:4]+'00000/'+str(id)[4:6]+'000/ktwo'+str(id)+'-c'+str(camp).zfill(2)+'_llc.fits'
        resp = httplib2.Http().request(urlfilename1, 'HEAD')
        if int(resp[0]['status']) < 400:
            
            return self.read_from_file(fits.open(urlfilename1,show_progress=False),urlfilename1,mission='kepler',src='pdc',sect=camp,**kwargs)

    
    def make_cadmask(self):
        #Masking any cadences we don't want:
        self.cad_mask=~np.in1d(self.cadence, self.mask_cadences)
        if 'cad_mask' not in self.timeseries:
            self.timeseries+=['cad_mask']

    def make_mask(self,overwrite=False):
        """Making mask for timeseries. Requires both a "flux" mask (i.e. bad/anomalous points) and a "cadence" mask (i.e. duplicate cadences due to multiple lightcurve sources)

        Args:
            overwrite (bool, optional): [description]. Defaults to False.
        """
        #Making mask
        if 'cad_mask' not in self.timeseries or overwrite or np.nan in self.cad_mask:
            self.make_cadmask()
        if 'flux_mask' not in self.timeseries or overwrite:
            self.make_fluxmask()
        self.mask=self.flux_mask*self.cad_mask
    
    def save(self,savename=None):
        """AI is creating summary for save

        Args:
            savename (str, optional): Savename to save to. Defaults to None.
        """
        assert hasattr(self,'savefileloc') or savename is not None

        #We don't want to save any binned or flattened arrays - we can re-do those later
        if hasattr(self,'bin_time'):
            self.remove_binned_arrs()
        if hasattr(self,'flux_flat'):
            self.remove_flattened_arrs()

        saveloc = savename if savename is not None else self.savefileloc
        if not os.path.isdir(os.path.dirname(saveloc)):
            os.mkdir(os.path.dirname(saveloc))
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(self.__dict__)
        #bytes_out = pickle.dumps(self)
        with gzip.open(saveloc, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
        del bytes_out
    
    def save_csv(self,savename=None):
        """AI is creating summary for save_csv

        Args:
            savename (str, optional): Filename to save under Defaults to None.
        """
        assert hasattr(self,'savefileloc') or savename is not None
        saveloc = savename if savename is not None else self.savefileloc.replace(".pkl.gz",".csv")

        df=pd.DataFrame()
        for ts in self.timeseries:
            df[ts]=getattr(self,ts)
        df['time']+=self.jd_base
        #print(saveloc,df.shape)
        df.to_csv(saveloc)

    def read_from_file(self, f, fname, sect, **kwargs):
        """opens and processes all lightcurve files (especially, but not only, fits files).

        Args:
            f (various types): opened file. This could be:
                                - astropy.fits format from kepler, k2 (everest, k2sc, pdc, vand) or tess
            fname (str): string of filename. Can be blank for certain file types.
            sect (str): string of sector. Can be left blank (i.e. None), but this may make stitching lightcurves more difficult...
        
        Kwargs:
            src (str): String describing source of lightcurve (e.g. pipeline)
            mission (str): Photometric mission/telescope of lightcurve (e.g. k2, tess, kepler, etc)
            jd_base (float): JD epoch for time=0
            flx_system (str): Type of flux used in lightcurve - either "ppm", "ppt", "norm0", "norm1" or "elec" (see `change_flx_system`)
            cut_all_anom_lim (float, optional): Perform cuts on all anomalies when masking. Defaults to 4.0.
            default_flx_system (str, optional): Default flux system to use. Defaults to 'ppt'.
            force_raw_flux (bool, optional): Force the lightcurve to swap 'flux' for 'raw_flux' (i.e. due to bad PDC detrending). Defaults to False.
            end_of_orbit (bool, optional): Cut/fix the end-of-orbit flux? Defaults to False.
        """
        if type(f)!=fits.hdu.hdulist.HDUList and type(f)!=fits.fitsrec.FITS_rec:
            #Making sure we have the tools needed to identify possible file types:
            import eleanor
            import lightkurve
            import h5py
        else:
            #f is fits file, but to make this section easier, we'll add it into a list:
            f=[f]
        ilc=lc()
        if type(f)==list and (type(f[0])==fits.hdu.hdulist.HDUList or type(f[0])==fits.fitsrec.FITS_rec):
            #print(f[0][0].header['TELESCOP'])
            if f[0][0].header['TELESCOP']=='Kepler' or fname.find('kepler')!=-1 or fname.find('kplr')!=-1:
                if fname.find('k2sff')!=-1:
                    bgf=f[1+np.argmax([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['FRAW'] - \
                        f[1+np.argmin([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['FRAW']
                    ilc.load_lc(np.hstack([fi[1].data['T'] for fi in f]), 
                                 fluxes={'flux':np.hstack([fi[1].data['FCOR'] for fi in f]),
                                         'bg_flux':bgf},
                                 flux_errs={'flux_err':np.tile(np.median(abs(np.diff(np.hstack([fi[1].data['FCOR'] for fi in f])))),len(np.hstack([fi[1].data['T'] for fi in f]))),
                                             'bg_flux_err':np.tile(1.06*np.nanmedian(abs(np.diff(bgf))),len(bgf))},
                                 src='vand', mission='k2', jd_base=2454833, flx_system='norm1', sect=sect)
                    return ilc
                elif fname.find('everest')!=-1:
                    #logging.debug('Everest file')#Everest (Luger et al) detrending:
                    ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                 fluxes={'flux':np.hstack([fi[1].data['FCOR'] for fi in f]),
                                         'raw_flux':np.hstack([fi[1].data['fraw'] for fi in f]),
                                         'bg_flux':np.hstack([fi[1].data['BKG'] for fi in f])},
                                 flux_errs={'flux_err':np.hstack([fi[1].data['RAW_FERR'] for fi in f]),
                                            'bg_flux_err':np.sqrt(np.hstack([fi[1].data['BKG'] for fi in f]))},
                                 src='ev', mission='k2', jd_base=2454833, flx_system='norm1', sect=sect,quality=np.hstack([fi[1].data['QUALITY'] for fi in f]))
                    return ilc
                elif fname.find('k2sc')!=-1:
                    #logging.debug('K2SC file')#K2SC (Aigraine et al) detrending:
                    ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                 fluxes={'flux':np.hstack([fi[1].data['flux'] for fi in f])},
                                 flux_errs={'flux_err':np.hstack([fi[1].data['error'] for fi in f])},
                                 src='k2sc',mission='k2', jd_base=2454833, flx_system='norm1', sect=sect)
                    return ilc

                elif fname.find('kplr')!=-1 or fname.find('ktwo')!=-1:
                    #logging.debug('kplr/ktwo file')
                    miss='k2' if fname.find('ktwo')!=-1 else 'kepler'
                    if fname.find('llc')!=-1 or fname.find('slc')!=-1:
                        #logging.debug('NASA/Ames file')#NASA/Ames Detrending:
                        if ~np.isnan(np.nanmedian(np.hstack([fi[1].data['PSF_CENTR2'] for fi in f]))):
                            c1=np.hstack([fi[1].data['PSF_CENTR1'] for fi in f]);c2=np.hstack([fi[1].data['PSF_CENTR2'] for fi in f])
                        else:
                            c1=np.hstack([fi[1].data['MOM_CENTR1'] for fi in f]);c2=np.hstack([fi[1].data['MOM_CENTR2'] for fi in f])
                        ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                     fluxes={'flux':np.hstack([fi[1].data['PDCSAP_FLUX'] for fi in f]),
                                             'raw_flux':np.hstack([fi[1].data['SAP_FLUX'] for fi in f]),
                                             'bg_flux':np.hstack([fi[1].data['SAP_BKG'] for fi in f])},
                                     flux_errs={'flux_err':np.hstack([fi[1].data['PDCSAP_FLUX_ERR'] for fi in f]),
                                                 'bg_flux_err':np.hstack([fi[1].data['SAP_BKG_ERR'] for fi in f])},
                                    src='pdc',mission=miss, jd_base=2454833, flx_system='elec', sect=sect, cent1=c1, cent2=c2)
                        return ilc
                    elif fname.find('XD')!=-1 or fname.find('X_D')!=-1:
                        #logging.debug('Armstrong file')#Armstrong detrending:
                        ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                     fluxes={'flux':np.hstack([fi[1].data['DETFLUX'] for fi in f])},
                                     flux_errs={'flux_err':np.hstack([fi[1].data['APTFLUX_ERR'] for fi in f])/np.hstack([fi[1].data['APTFLUX'] for fi in f])},
                                    src='arm',mission='k2', jd_base=2454833, flx_system='elec', sect=sect)
                        return ilc

                else:
                    print("unidentified file type")
                    #logging.debug("no file type for "+str(f))
                    return None
            elif f[0][0].header['TELESCOP'].lower()=='tess':
                if 'ORIGIN' in f[0][0].header and f[0][0].header['ORIGIN']=='MIT/QLP':
                    ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                fluxes={'flux':np.hstack([fi[1].data['SAP_FLUX'] for fi in f]),
                                        'xl_ap_flux':np.hstack([fi[1].data['KSPSAP_FLUX_LAG'] for fi in f]),
                                        'xs_ap_flux':np.hstack([fi[1].data['KSPSAP_FLUX_SML'] for fi in f])},
                                flux_errs={'flux_err':np.hstack([fi[1].data['KSPSAP_FLUX_ERR'] for fi in f])},
                                src='qlp',mission='tess', jd_base=2457000, flx_system='elec', sect=sect, 
                                cent1=np.hstack([fi[1].data['SAP_X'] for fi in f]), cent2=np.hstack([fi[1].data['SAP_Y'] for fi in f]))
                    return ilc

                else:
                    if ~np.isnan(np.nanmedian(np.hstack([fi[1].data['PSF_CENTR2'] for fi in f]))):
                        c1=np.hstack([fi[1].data['PSF_CENTR1'] for fi in f]);c2=np.hstack([fi[1].data['PSF_CENTR2'] for fi in f])
                    else:
                        c1=np.hstack([fi[1].data['MOM_CENTR1'] for fi in f]);c2=np.hstack([fi[1].data['MOM_CENTR2'] for fi in f])
                    ilc.load_lc(np.hstack([fi[1].data['TIME'] for fi in f]), 
                                        fluxes={'flux':np.hstack([fi[1].data['PDCSAP_FLUX'] for fi in f]),
                                                'raw_flux':np.hstack([fi[1].data['SAP_FLUX'] for fi in f]),
                                                'bg_flux':np.hstack([fi[1].data['SAP_BKG'] for fi in f])},
                                        flux_errs= {'flux_err':np.hstack([fi[1].data['PDCSAP_FLUX_ERR'] for fi in f]),
                                                    'bg_flux_err':np.hstack([fi[1].data['SAP_BKG'] for fi in f])},
                                        src='pdc',mission='tess', jd_base=2457000, flx_system='elec', sect=sect, cent1=c1, cent2=c2,
                                        quality=np.hstack([fi[1].data['QUALITY'] for fi in f]))
                    return ilc

            elif f[0][0].header['TELESCOP'].lower()=='corot':
                if f[0][0].header['FILENAME'][9:12]=='MON':
                    ilc.load_lc(np.hstack([fi[1].data['DATEHEL'] for fi in f]), 
                                 fluxes={'flux':np.hstack([fi[1].data['WHITEFLUX'] for fi in f])},
                                 flux_errs={'flux_err':np.hstack([fi[1].data['WHITEFLUXDEV'] for fi in f])},
                                 src='corot',mission='corot', jd_base=2451545, flx_system='norm1', sect=sect, quality=np.hstack([fi[1].data['STATUS'] for fi in f]))
                    return ilc
                elif f[0][0].header['FILENAME'][9:12]=='CHR':
                    ilc.load_lc(np.hstack([fi[1].data['DATEHEL'] for fi in f]), 
                                 fluxes={'flux':np.hstack([fi[1].data['WHITEFLUX'] for fi in f]),
                                         'blue_flux':np.hstack([fi[1].data['BLUEFLUX'] for fi in f]),
                                         'green_flux':np.hstack([fi[1].data['GREENFLUX'] for fi in f]),
                                         'red_flux':np.hstack([fi[1].data['REDFLUX'] for fi in f])},
                                 flux_errs={'flux_err':np.hstack([fi[1].data['WHITEFLUXDEV'] for fi in f]),
                                            'blue_flux_err':np.hstack([fi[1].data['BLUEFLUXDEV'] for fi in f]),
                                            'green_flux_err':np.hstack([fi[1].data['GREENFLUXDEV'] for fi in f]),
                                            'red_flux_err':np.hstack([fi[1].data['REDFLUXDEV'] for fi in f])},
                                 src='corot',mission='corot', jd_base=2451545, flx_system='norm1', sect=sect, quality=np.hstack([fi[1].data['STATUS'] for fi in f]))
                    return ilc
        elif type(f).__name__=='TessLightCurve':
            ilc.load_lc(f.time, fluxes={'flux':f.flux},flux_errs={'flux_err':f.flux_err},
                         src='lk',mission='tess', jd_base=2457000, flx_system='norm1', sect=sect, cent_1 = f.centroid_col, cent_2=f.centroid_row,quality=f.quality)
            return ilc
        elif type(f)==list and type(f[0])==h5py._hl.files.File:
            def mag2flux(mags):
                flux = np.power(10,(mags-np.nanmedian(mags))/-2.5)
                return flux
            def magerr2flux(flux,magerrs):
                return flux*(magerrs/(2.5/np.log(10)))
            #QLP .h5py file
            bkf=np.hstack((f[0]['LightCurve']['Background']['Value'][:], f[1]['LightCurve']['Background']['Value'][:]))
            ilc.load_lc(np.hstack((f[0]['LightCurve']['BJD'],f[1]['LightCurve']['BJD'])),
                         fluxes={'flux':mag2flux(np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_003']['KSPMagnitude'][:],
                                                            f[1]['LightCurve']['AperturePhotometry']['Aperture_003']['KSPMagnitude'][:]))),
                                 'raw_flux':mag2flux(np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_003']['RawMagnitude'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_003']['RawMagnitude'][:]))),
                                 'bg_flux':bkf/np.nanmedian(bkf),
                                 'xs_ap_flux':mag2flux(np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_000']['KSPMagnitude'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_000']['KSPMagnitude'][:]))),
                                 'xl_ap_flux':mag2flux(np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_004']['KSPMagnitude'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_004']['KSPMagnitude'][:])))},
                         flux_errs={'flux_err':magerr2flux(mag2flux(np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitude'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitude'][:]))),
                                                           np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitudeError'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitudeError'][:]))),
                                    'bg_flux_err':np.sqrt(bkf)/np.nanmedian(bkf)},
                         src='qlph5',mission='tess', jd_base=2457000, flx_system='norm1', sect=sect,
                         cent_1=np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_003']['X'][:],
                                            f[1]['LightCurve']['AperturePhotometry']['Aperture_003']['X'][:])),
                         cent_2=np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_003']['Y'][:],
                                            f[1]['LightCurve']['AperturePhotometry']['Aperture_003']['Y'][:])),
                         quality=np.array([np.power(2,15) if c=='G' else 0.0 for c in np.hstack((f[0]['LightCurve']['AperturePhotometry']['Aperture_003']['QualityFlag'][:],
                                                                f[1]['LightCurve']['AperturePhotometry']['Aperture_003']['QualityFlag'][:]))]).astype(int))
            return ilc
        elif type(f)==h5py._hl.files.File:
            #QLP is defined in mags, so lets
            def mag2flux(mags):
                flux = np.power(10,(mags-np.nanmedian(mags))/-2.5)
                return flux
            def magerr2flux(flux,magerrs):
                return flux*(magerrs/(2.5/np.log(10)))
            #QLP .h5py file
            ilc.load_lc(f['LightCurve']['BJD'],
                         fluxes={'flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_003']['KSPMagnitude'][:]),
                                 'raw_flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_003']['RawMagnitude'][:]),
                                 'bg_flux':f['LightCurve']['Background']['Value'][:],
                                 'xs_ap_flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_000']['KSPMagnitude'][:]),
                                 'xl_ap_flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_004']['KSPMagnitude'][:])},
                         flux_errs={'flux_err':magerr2flux(mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitude'][:]),
                                                                    f['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitudeError'][:]),
                                    'bg_flux_err':np.sqrt(f['LightCurve']['Background']['Value'][:])},
                         src='qlpfts',mission='tess', jd_base=2457000, flx_system='norm1', sect=sect,
                         cent_1=f['LightCurve']['AperturePhotometry']['Aperture_003']['X'][:],
                         cent_2=f['LightCurve']['AperturePhotometry']['Aperture_003']['Y'][:],
                         quality=np.array([np.power(2,15) if c=='G' else 0.0 for c in f['LightCurve']['AperturePhotometry']['Aperture_003']['QualityFlag'][:]]).astype(int))
            return ilc
        elif type(f)==eleanor.TargetData:
            # Eleanor TESS object
            q=f.quality
            # Fixing negative quality values as 2^15
            q[q<0.0]=np.power(2,15)
            q=q.astype(int)
            ilc.load_lc(f.time,fluxes={'flux':f.corr_flux,'raw_flux':f.raw_flux,'xs_ap_flux':f.all_corr_flux[1],'xl_ap_flux':f.all_corr_flux[16]},
                         flux_errs={'flux_err':f.flux_err}, src='el',mission='tess', jd_base=2457000, flx_system='elec', sect=sect,
                         cent_1=f.centroid_xs,cent_2=f.centroid_ys,quality=q)
            return ilc
        elif type(f)==np.ndarray and np.shape(f)[1]==3:
            #Already opened lightcurve file
            ilc.load_lc(f[:,0],fluxes={'flux':f[:,1]},flux_errs={'flux_err':f[:,2]},
                         src='csv',mission='corot', jd_base=2454833, flx_system='norm1', sect=None)
            return ilc
        elif type(f)==pd.DataFrame:
            #Already opened lightcurve file
            assert 'mission' in kwargs and 'jd_base' in kwargs and 'flx_system' in kwargs
            ilc.load_lc(f[:,0],fluxes={k:f[k].values for k in f.columns if 'flux' in k and '_err' not in k},flux_errs={k:f[k].values for k in f.columns if 'flux_err' in k},
                         src='df',mission=kwargs['mission'], jd_base=kwargs['jd_base'], flx_system=kwargs['flx_system'], sect=sect)
            return ilc
        elif type(f)==dict:
            f.update(**kwargs)
            if 'quality' not in f:
                f['quality']=np.zeros(len(f['time']))
            ilc.load_lc(f['time'],fluxes={k:f[k] for k in f if 'flux' in k and '_err' not in k},flux_errs={k:f[k] for k in f if 'flux_err' in k},
                         src=f['src'], mission=f['mission'], jd_base=f['jd_base'], flx_system=f['flx_system'], sect=sect, quality=f['quality'])
            return ilc
        else:
            print('cannot identify fits type to identify with')
            #logging.debug('Found fits file but cannot identify fits type to identify with')

        if hasattr(self,'flux'):
            self.make_mask()
    
    def plot(self, plot_rows=None, timeseries=['flux'], jump_thresh=10, ylim=None, xlim=None, yoffset=0, savepng=True, savepdf=False):
        """Plot the lightcurve using Matplotlib.

        In the default case, either data that is extremely long (i.e Kepler), or data that has a large gap (i.e. TESS Y1/3) will be split into two rows.
        Gaps between quarters, sectors, etc will result in horizontally-split plot frames.
        Typically ~30min cadence data is plotted unbinned and shorter-cadence ddata is plotted raw and with 30-minute bins

        Args:
            plot_rows (int, optional): Number of rows for the plot to have. Defaults to None, in which case this is automatically guessed between 1 and 4
            timeseries (list, optional): List of which timeseries to plot (i.e. enables plotting of e.g. background flux or flattened flux. Defaults to ['flux']
            jump_thresh (int, optional): Threshold beyond which we call a gap/jump between observations a major difference. Defaults to 10.
        """
        # Step 1 - count total time. Divide by plot_rows, or estimate ideal plot_rows given data duration
        # Step 2 - Loop through cadences and round/cut into 3. 
        # Step 3 - calculate gaps between cadences, cut up plot to hide gaps.

        from iteround import saferound

        xlim_mask=self.mask.astype(bool) if xlim is None else self.mask&(self.time>xlim[0])&(self.time<xlim[1])

        fine_cuts={}
        big_cuts={}
        total_time=0
        init_ordered_cadences =  np.array(self.cadence_list)[np.argsort([np.nanmedian(self.time[self.cadence==cad]) for cad in self.cadence_list])]
        ordered_cadences=[]
        for cad in init_ordered_cadences:
            ix=(self.cadence==cad)&xlim_mask
            if np.sum(ix)>0:
                fine_cuts[cad]={'cadence':float(cad.split('_')[1])/86400,
                                'start':np.min(self.time[ix]),'end':np.max(self.time[ix])}
                fine_cuts[cad]['total_dur']=np.sum(self.cadence==cad)*fine_cuts[cad]['cadence']
                fine_cuts[cad]['start_end_dur']=fine_cuts[cad]['end']-fine_cuts[cad]['start']
                fine_cuts[cad]['mad']=1.06*np.nanmedian(abs(np.diff(self.flux[ix])))
                fine_cuts[cad]['minmax']=np.nanpercentile(self.flux[ix],[0.25,99.75])
                total_time+=fine_cuts[cad]['start_end_dur']
                ordered_cadences+=[cad]
        ordered_cadences=np.array(ordered_cadences)
        time_regions=tools.find_time_regions(self.time[xlim_mask])
        for nj in range(len(time_regions)):
            big_cuts[nj]={'start':time_regions[nj][0],'end':time_regions[nj][1],
                          'all_cads':np.unique(self.cadence[(self.time>time_regions[nj][0])*(self.time<time_regions[nj][1])])}
            big_cuts[nj]['total_dur']=np.sum([fine_cuts[cad]['start_end_dur'] for cad in big_cuts[nj]['all_cads']])
            #Let's just estimate the number of rows we want per big 
            big_cuts[nj]['n_ideal_split'] = np.log10(np.clip(3.333*(total_time-80),10,100000))*big_cuts[nj]['total_dur']/total_time
        #Automatic 
        plot_rows = int(np.clip(np.round(np.sum([big_cuts[nj]['n_ideal_split'] for nj in big_cuts])),1,4)) if plot_rows is None else plot_rows

        subplots_ix = tools.partition_list(np.array([fine_cuts[cad]['start_end_dur'] for cad in fine_cuts]), plot_rows)
        #np.array(saferound(-0.5+np.cumsum(np.array([fine_cuts[cad]['start_end_dur'] for cad in fine_cuts])/(total_time/(plot_rows))), places=0)).astype(int)

        #print(plot_rows,fine_cuts,subplots_ix,[r for r in range(int(plot_rows))])
        for irow in range(plot_rows):
            assert np.sum(subplots_ix==irow)>0
            plots_in_this_row = np.array(list(fine_cuts.keys()))[subplots_ix==irow]
            durs = [fine_cuts[cad2]['start_end_dur'] for cad2 in plots_in_this_row]
            plot_cols = np.hstack((0,np.cumsum(saferound(24*np.array(durs)/np.sum(durs), places=0))))
            for icol,key in enumerate(plots_in_this_row):
                fine_cuts[key]['n_plot_row']=irow
                fine_cuts[key]['n_plot_col']=(int(plot_cols[icol]),int(plot_cols[icol+1]))
        
        fig=plt.figure(figsize=(11.69,8.27)) #A4 page: 8.27 x 11.69
        gs = fig.add_gridspec(plot_rows,24,wspace=0.07,hspace=0.18)
        subplots={}
        
        minmax_global = (np.min([fine_cuts[cad]['minmax'][0]-0.5*fine_cuts[cad]['mad'] for cad in fine_cuts]),np.max([fine_cuts[cad]['minmax'][1]+0.5*fine_cuts[cad]['mad'] for cad in fine_cuts]))
        import seaborn as sns
        sns.set_palette('viridis')
        if 'flux_flat' in timeseries and not hasattr(self,'flux_flat'):
            self.flatten()

        
        for cad in ordered_cadences:
            #print(fine_cuts[cad]['n_plot_col'])
            subplots[cad]=fig.add_subplot(gs[fine_cuts[cad]['n_plot_row'],fine_cuts[cad]['n_plot_col'][0]:fine_cuts[cad]['n_plot_col'][1]])
        for it, itimeseries in enumerate(timeseries):
            if not hasattr(self,'bin_'+itimeseries):
                self.bin(timeseries=[itimeseries])

            for cad in ordered_cadences:
                #subplots[cad]=fig.add_subplot(gs[fine_cuts[cad]['n_plot_row'],fine_cuts[cad]['n_plot_col'][0]:fine_cuts[cad]['n_plot_col'][1]])
                ix=(self.cadence==cad)*self.mask
                bin_ix=(self.bin_cadence==cad)*np.isfinite(getattr(self,"bin_"+itimeseries))
                if (fine_cuts[cad]['cadence']*1440)>20 and total_time<500:
                    #Plotting only real points as "binned points" style:
                    subplots[cad].plot(self.time[ix],yoffset*it+getattr(self,itimeseries)[ix],'.',alpha=0.8,markersize=3.0,color='C'+str(it),label=itimeseries)
                elif (fine_cuts[cad]['cadence']*1440)>20 and total_time>500:
                    #So much data that we should bin it back down (to 2-hour bins)
                    self.bin(timeseries=[itimeseries], binsize=1/12)
                    subplots[cad].plot(self.time[ix],yoffset*it+getattr(self,itimeseries)[ix],'.k',markersize=0.75,alpha=0.25)
                    subplots[cad].plot(self.bin_time[bin_ix],yoffset*it+getattr(self,"bin_"+itimeseries)[bin_ix],'.',alpha=0.8,markersize=3.0,color='C'+str(it),label=itimeseries)
                else:
                    #Plotting real points as fine scatters and binned points above:
                    subplots[cad].plot(self.time[ix],yoffset*it+getattr(self,itimeseries)[ix],'.k',markersize=0.75,alpha=0.25)
                    subplots[cad].plot(self.bin_time[bin_ix],yoffset*it+getattr(self,"bin_"+itimeseries)[bin_ix],'.',alpha=0.8,markersize=3.0,color='C'+str(it),label=itimeseries)
                if it==(len(timeseries)-1):
                    if fine_cuts[cad]['n_plot_col'][0]!=0.0:
                        subplots[cad].set_yticklabels([])
                    else:
                        subplots[cad].set_ylabel("Relative Flux ["+self.flx_system+"]")
                    if fine_cuts[cad]['n_plot_row']==plot_rows-1:
                        subplots[cad].set_xlabel("Time [BJD-"+str(int(self.jd_base))+"]")
                    if ylim is None:
                        subplots[cad].set_ylim(minmax_global[0],minmax_global[1]+yoffset*len(itimeseries))
                    else:
                        subplots[cad].set_ylim(ylim)
                    if xlim is None:
                        subplots[cad].set_xlim(fine_cuts[cad]['start']-0.25,fine_cuts[cad]['end']+0.25)
                    else:
                        subplots[cad].set_xlim(xlim)
        subplots[cad].legend()
        if savepng:
            plt.savefig(self.savefileloc.replace('_lc.pkl.gz','_lc.png'))
        if savepdf:
            plt.savefig(self.savefileloc.replace('_lc.pkl.gz','_lc.pdf'))

            