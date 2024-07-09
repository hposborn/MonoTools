import warnings
warnings.filterwarnings("ignore")

import numpy as np
import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.io import ascii
from scipy.signal import savgol_filter
from scipy.interpolate import LSQUnivariateSpline, BSpline,splev,splrep

import h5py

import pymc as pm
import astropy.units as u
from astropy.units import cds
from astropy import constants as c

from astropy import units
from astropy.coordinates.sky_coordinate import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.units import Quantity
from astroquery.gaia import Gaia

import pickle
import os.path
from datetime import datetime
import requests
import httplib2
from lxml import html
import importlib

import glob


MonoData_tablepath = os.path.join(os.path.dirname(__file__),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname( __file__ ),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')

from . import starpars

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}
lc_dic={'tess':'ts','kepler':'k1','k2':'k2','corot':'co','cheops':'ch'}

#goto='/Users/hosborn' if 'Users' in os.path.dirname(os.path.realpath(__file__)).split('/') else '/home/hosborn'

def open_fits(f,fname,mission,cut_all_anom_lim=4.0,use_ppt=True,force_raw_flux=False,end_of_orbit=False,mask=None,**kwargs):
    """opens and processes all lightcurve files (especially, but not only, fits files).

    Args:
        f ([type]): [description]
        fname ([type]): [description]
        mission ([type]): [description]
        cut_all_anom_lim (float, optional): [description]. Defaults to 4.0.
        use_ppt (bool, optional): [description]. Defaults to True.
        force_raw_flux (bool, optional): [description]. Defaults to False.
        end_of_orbit (bool, optional): Cut/fix the end-of-orbit flux? Defaults to False.
        mask ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    if type(f)==fits.hdu.hdulist.HDUList or type(f)==fits.fitsrec.FITS_rec:
        if f[0].header['TELESCOP']=='Kepler' or fname.find('kepler')!=-1:
            if fname.find('k2sff')!=-1:
                lc={'time':f[1].data['T'],
                    'flux':f[1].data['FCOR'],
                    'flux_err':np.tile(np.median(abs(np.diff(f[1].data['FCOR']))),len(f[1].data['T'])),
                    'flux_raw':f[1].data['FRAW'],
                    'bg_flux':f[1+np.argmax([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['FRAW']-
                              f[1+np.argmin([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['FRAW']}
                    #'rawflux':,'rawflux_err':,}
            elif fname.find('everest')!=-1:
                #logging.debug('Everest file')#Everest (Luger et al) detrending:
                print("everest file")
                lc={'time':f[1].data['TIME'],'flux':f[1].data['FCOR'],'flux_err':f[1].data['RAW_FERR'],
                    'raw_flux':f[1].data['fraw'],'bg_flux':f[1].data['BKG'],'quality':f[1].data['QUALITY']}
            elif fname.find('k2sc')!=-1:
                print("K2SC file")
                #logging.debug('K2SC file')#K2SC (Aigraine et al) detrending:
                lc={'time':f[1].data['time'],'flux':f[1].data['flux'],'flux_err':f[1].data['error']}
            elif fname.find('kplr')!=-1 or fname.find('ktwo')!=-1:
                #logging.debug('kplr/ktwo file')
                if fname.find('llc')!=-1 or fname.find('slc')!=-1:
                    #logging.debug('NASA/Ames file')#NASA/Ames Detrending:
                    print("Kepler file")
                    lc={'time':f[1].data['TIME'],'flux':f[1].data['PDCSAP_FLUX'],
                        'flux_err':f[1].data['PDCSAP_FLUX_ERR'],'raw_flux':f[1].data['SAP_FLUX'],
                        'bg_flux':f[1].data['SAP_BKG']}
                    if ~np.isnan(np.nanmedian(f[1].data['PSF_CENTR2'])):
                        lc['cent_1']=f[1].data['PSF_CENTR1'];lc['cent_2']=f[1].data['PSF_CENTR2']
                    else:
                        lc['cent_1']=f[1].data['MOM_CENTR1'];lc['cent_2']=f[1].data['MOM_CENTR2']
                    if 'SAP_BKG' in f[1].columns.names:
                        lc['bg_flux']=f[1].data['SAP_BKG']
                elif fname.find('XD')!=-1 or fname.find('X_D')!=-1:
                    #logging.debug('Armstrong file')#Armstrong detrending:
                    lc={'time':f[1].data['TIME'],'flux':f[1].data['DETFLUX'],
                        'flux_err':f[1].data['APTFLUX_ERR']/f[1].data['APTFLUX']}
            else:
                print("unidentified file type")
                #logging.debug("no file type for "+str(f))
                return None
        elif f[0].header['TELESCOP']=='TESS':
            lc   = {'time':f[1].data['TIME'],
                    'raw_flux':f[1].data['SAP_FLUX']/np.nanmedian(f[1].data['SAP_FLUX']),
                    'bg_flux':f[1].data['SAP_BKG'],
                    'quality':f[1].data['QUALITY']}
            if 'ORIGIN' in f[0].header and f[0].header['ORIGIN']=='MIT/QLP':
                lc['flux']=f[1].data['KSPSAP_FLUX']/np.nanmedian(f[1].data['KSPSAP_FLUX'])
                lc['flux_err']= f[1].data['KSPSAP_FLUX_ERR']/np.nanmedian(f[1].data['KSPSAP_FLUX'])
                lc['cent_1']  = f[1].data['SAP_X']
                lc['cent_2']  = f[1].data['SAP_Y']
                lc['flux_xl_ap']=f[1].data['KSPSAP_FLUX_LAG']
                lc['flux_sm_ap']=f[1].data['KSPSAP_FLUX_SML']
            else:
                lc['flux'] = f[1].data['PDCSAP_FLUX']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
                lc['flux_err'] = f[1].data['PDCSAP_FLUX_ERR']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
                if ~np.isnan(np.nanmedian(f[1].data['PSF_CENTR2'])):
                    lc['cent_1']=f[1].data['PSF_CENTR1'];lc['cent_2']=f[1].data['PSF_CENTR2']
                else:
                    lc['cent_1']=f[1].data['MOM_CENTR1'];lc['cent_2']=f[1].data['MOM_CENTR2']
        elif f[0].header['TELESCOP']=='COROT':
            if 'MAGNIT_V' in f[0].header and type(f[0].header['MAGNIT_V'])==float and not np.isnan(f[0].header['MAGNIT_V']):
                mag=f[0].header['MAGNIT_V']
            else:
                mag=f[0].header['MAGNIT_R']
            lc={'time':f[1].data['DATEHEL'],'quality':f[1].data['STATUS'],'mask':np.tile(True,len(f[1].data['DATEHEL']))}

            if f[0].header['FILENAME'][9:12]=='MON':
                lc['flux']=f[1].data['WHITEFLUX']
                lc['flux_err']=f[1].data['WHITEFLUXDEV']

            elif f[0].header['FILENAME'][9:12]=='CHR':
                #logging.debug('3-colour CoRoT lightcurve')
                #Adding colour fluxes together...
                lc['flux'] = f[1].data['REDFLUX']+f[1].data['BLUEFLUX']+f[1].data['GREENFLUX']
                lc['flux_err'] = f[1].data['GREENFLUXDEV']+f[1].data['BLUEFLUXDEV']+f[1].data['REDFLUXDEV']
            if np.isnan(lc['flux_err']).all():
                mederr=np.nanmedian(1.06*abs(np.diff(lc['flux'])))
            else:
                mederr=np.nanmedian(lc['flux_err'][(lc['flux_err']>0.0)*(~np.isnan(lc['flux_err']))])
            lc['flux_err'][(lc['flux_err']>0.0)*(~np.isnan(lc['flux_err']))]=mederr
            lc['flux_err']/=np.nanmedian(lc['flux'])
            lc['flux']/=np.nanmedian(lc['flux'])
            lc['mask']=cut_high_regions(lc['flux'],np.isfinite(lc['flux']),std_thresh=4.5,n_pts=25,n_loops=2)
    elif type(f).__name__=='TessLightCurve':
        import lightkurve
        lc={'time':f.time,'flux':f.flux,'flux_err':f.flux_err,'quality':f.quality,
            'cent_1':f.centroid_col,'cent_2':f.centroid_row}
        #Adjusting flux_err for abs mag diff:
        lc['flux_err']*=np.nanmedian(abs(np.diff(lc['time'])))/np.nanmedian(lc['flux_err'])
    elif type(f)==h5py._hl.files.File:
        #QLP is defined in mags, so lets
        def mag2flux(mags):
            flux = np.power(10,(mags-np.nanmedian(mags))/-2.5)
            return flux/np.nanmedian(flux)
        def magerr2flux(magerrs,mags):
            flux=np.power(10,(mags-np.nanmedian(mags))/-2.5)
            return flux*(magerrs/(2.5/np.log(10)))/np.nanmedian(flux)
        #QLP .h5py file
        lc={'time':f['LightCurve']['BJD'],
            'flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_003']['KSPMagnitude'][:]),
            'raw_flux':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_003']['RawMagnitude'][:]),
            'bg_flux':f['LightCurve']['Background']['Value'][:],
            'cent_1':f['LightCurve']['AperturePhotometry']['Aperture_003']['X'][:],
            'cent_2':f['LightCurve']['AperturePhotometry']['Aperture_003']['Y'][:],
            'quality':np.array([np.power(2,15) if c=='G' else 0.0 for c in f['LightCurve']['AperturePhotometry']['Aperture_003']['QualityFlag'][:]]).astype(int),
            'flux_sm_ap':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_000']['KSPMagnitude'][:]),
            'flux_xl_ap':mag2flux(f['LightCurve']['AperturePhotometry']['Aperture_004']['KSPMagnitude'][:])}
        #    'flux_err':magerr2flux(f['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitudeError'][:],
        #                           f['LightCurve']['AperturePhotometry']['Aperture_002']['RawMagnitude'][:]),
        lc['flux_err']=np.tile(np.nanmedian(abs(np.diff(lc['raw_flux']))),len(lc['time']))
        if force_raw_flux:
            end_of_orbit=True
    elif type(f)==np.ndarray and np.shape(f)[1]==3:
        #Already opened lightcurve file
        lc={'time':f[:,0],'flux':f[:,1],'flux_err':f[:,2]}
        if mission.lower()=='corot':
            #Need to remove SAA spikes
            stacked_arr=np.vstack((lc['flux'][(15-n):(len(lc['time'])-(n+15))] for n in np.hstack((np.arange(-15,-4),
                                                                                                   np.arange(4,15),0))))
            errs=np.nanmedian(abs(np.diff(stacked_arr[:-1],axis=0)),axis=0)
            stacked_arr=np.vstack((stacked_arr[-1]-np.nanmedian(stacked_arr[:11],axis=0),
                                   stacked_arr[-1]-np.nanmedian(stacked_arr[11:-1],axis=0)))/errs
            anom_high=np.hstack((np.tile(0,15),np.sum(stacked_arr,axis=0),np.tile(0,15)))
            mask=anom_high<4
    elif type(f)==dict:
        lc=f
    else:
        try:
            from eleanor import eleanor
            if type(f)==eleanor.TargetData:
                #Eleanor TESS object
                lc={'time':f.time,'flux':f.corr_flux,'flux_err':f.flux_err,'raw_flux':f.raw_flux,
                    'cent_1':f.centroid_xs,'cent_2':f.centroid_ys,'quality':f.quality,
                    'flux_sm_ap':f.all_corr_flux[1],
                    'flux_xl_ap':f.all_corr_flux[16]}
        except:
            print("May be eleanor file but we cannot import it.")
        #Fixing negative quality values as 2^15
        lc['quality'][lc['quality']<0.0]=np.power(2,15)
        lc['quality']=lc['quality'].astype(int)
        end_of_orbit=True

        print('cannot identify fits type to identify with')
        #logging.debug('Found fits file but cannot identify fits type to identify with')
        return None
    if 'raw_flux' in lc:
        if np.nanmedian(lc['raw_flux'])>0.1:
            lc['raw_flux']/=np.nanmedian(lc['raw_flux'])
            lc['raw_flux']-=1.0

    if force_raw_flux and 'raw_flux' in lc:
        #Here we'll force ourselves to use raw flux, and not the detrended flux, if it exists:
        lc['detrended_flux']=lc.pop('flux')
        lc['flux']=lc['raw_flux'][:]

    lc['mask']=mask_lc(lc,fname,cut_all_anom_lim=cut_all_anom_lim,use_ppt=use_ppt,end_of_orbit=end_of_orbit,input_mask=mask)

    #Including the cadence in the lightcurve as ["t2","t30","k1","k30"] mission letter + cadence
    lc['cadence']=np.tile(mission[0]+str(np.round(np.nanmedian(np.diff(lc['time']))*1440).astype(int)),len(lc['time']))

    # Only discard positive outliers

    #print(np.sum(~lc['mask']),"points masked in lc of",len(lc['mask']))
    '''
    # Make sure that the data type is consistent
    lc['time'] = np.ascontiguousarray(x[m2], dtype=np.float64)
    lc['flux'] = np.ascontiguousarray(y[m2], dtype=np.float64)
    lc['flux_err'] = np.ascontiguousarray(yerr[m2], dtype=np.float64)
    lc['trend_rem'] = np.ascontiguousarray(smooth[m2], dtype=np.float64)

    for key in lc:
        if key not in ['time','flux','flux_err','trend_rem']:
            lc[key]=np.ascontiguousarray(lc[key][m][m2], dtype=np.float64)
    '''
    #Make sure no nanned times get through here:
    for key in [key for key in lc if key!='time' and type(lc[key])==np.ndarray and len(lc[key])==len(lc['time'])]:
        lc[key] = lc[key][np.isfinite(lc['time'])]
    lc['time'] = lc['time'][np.isfinite(lc['time'])]

    lc['flux_unit']=0.001 if use_ppt else 1.0

    return lc

def find_time_regions(time,split_gap_size=1.5,min_region_dur=0.25,**kwargs):
    if np.nanmax(np.diff(np.sort(time)))>split_gap_size:
        #We have gaps in the lightcurve, so we'll find the bins by looping through those gaps
        time_starts = np.hstack((np.nanmin(time),np.sort(time)[1+np.where(np.diff(np.sort(time))>split_gap_size)[0]]))
        time_ends   = np.hstack((time[np.where(np.diff(np.sort(time))>split_gap_size)[0]],np.nanmax(time)))
        reg_durs=np.array([time_ends[i]-time_starts[i] for i in np.arange(len(time_starts))])
        return [(time_starts[i],time_ends[i]) for i in np.arange(len(time_starts))[reg_durs>min_region_dur]]
    else:
        return [(np.nanmin(time),np.nanmax(time))]

def mask_lc(lc,fhead,cut_all_anom_lim=5.0,use_ppt=False,end_of_orbit=True,mask=None,
           use_binned=False,use_flat=False,mask_islands=True,input_mask=None,**kwargs):
    # Mask bad data (nans, infs and negatives)

    prefix= 'bin_' if use_binned else ''
    suffix='_flat' if use_flat else ''
    #if 'flux_unit' in lc:
    #    lc['flux']*=lc['flux_unit']
    #    lc['flux_err']*=lc['flux_unit']
    #    if lc['flux_unit']==0.001:
    #        use_ppt=True
    mask = np.ones(len(lc['time'])) if mask is None else mask
    mask = np.isfinite(lc[prefix+'flux'+suffix]) & np.isfinite(lc[prefix+'time']) & np.isfinite(lc[prefix+'flux_err'])
    if np.sum(mask)>0:
        # & (lc[prefix+'flux'+suffix]>0.0)
        #print(np.sum(mask))
        if input_mask is not None:
            mask=mask&input_mask
        # Mask data if it's 4.2-sigma from its points either side (repeating at 7-sigma to get any points missed)
        #print(np.sum(~lc['mask']),"points before quality flags")
        if 'quality' in lc and len(lc['quality'])==len(lc[prefix+'flux'+suffix]):
            qs=[1,2,3,4,6,7,8,9,13,15,16,17]#worst flags to cut - for the moment just using those in the archive_manual
            #if type(fhead)==dict and 'lcsource' in fhead.keys() and fhead['lcsource']=='everest':
            #    qs+=[23]
            #    print("EVEREST file with ",np.log(np.max(lc['quality']))/np.log(2)," max quality")
            mask=mask&(np.sum(np.vstack([lc['quality'] & 2 ** (q - 1) for q in qs]),axis=0)==0)
        #print(np.sum(~lc['mask']),"points after quality flags")
        if cut_all_anom_lim>0:
            #Stacking 20 point-shifted lightcurves on top of each other for quick median filter: is (flux - median of 20pts)<threshold*MAD of 20pts
            stack_shitfed_flux=np.column_stack([lc[prefix+'flux'+suffix][mask][n:(-20+n)] for n in range(20)])
            mask[mask][10:-10]=abs(lc[prefix+'flux'+suffix][mask][10:-10] - np.nanmedian(stack_shitfed_flux,axis=1))<cut_all_anom_lim*np.nanmedian(abs(np.diff(stack_shitfed_flux,axis=1)),axis=1)
            #Now doing difference
            mask[mask]=cut_anom_diff(lc[prefix+'flux'+suffix][mask],cut_all_anom_lim)
            '''
            #Doing this a second time with more stringent limits to cut two-point outliers:
            mask[mask]=cut_anom_diff(lc[prefix+'flux'+suffix][mask],cut_all_anom_lim+3.5)
            '''
            #print(np.sum(~lc['mask']),"after before cut_anom_diff")
        mu = np.median(lc[prefix+'flux'+suffix][mask])
        if mu<1e-3:
            #In this case we have an already zero-divided flux array:
            mu+=1
            lc[prefix+'flux'+suffix]+=1
        if use_ppt:
            # Convert to parts per thousand
            lc[prefix+'flux'+suffix] = (lc[prefix+'flux'+suffix] / mu - 1) * 1e3
            lc[prefix+'flux'+suffix+'_err'] *= 1e3/mu
        else:
            lc[prefix+'flux'+suffix] = (lc[prefix+'flux'+suffix] / mu - 1)
            lc[prefix+'flux'+suffix+'_err'] /= mu

        if mask_islands:
            #Masking islands of data which are <12hrs long and >12hrs from other data
            jumps=np.where(np.diff(lc['time'][mask])>0.5)[0]
            jumps=np.column_stack((np.hstack(([0,jumps+1])),
                                   np.hstack(([jumps,len(lc['time'][mask])-1]))))
            xmask=np.tile(True,np.sum(mask))
            for j in range(len(jumps[:,0])):
                t0=lc['time'][mask][jumps[j,0]]
                t1=lc['time'][mask][jumps[j,1]]
                jump_before = 100 if j==0 else t0-(lc['time'][mask][jumps[j-1,1]])
                jump_after = 100 if j==(len(jumps[:,0])-1) else (lc['time'][mask][jumps[j+1,0]])-t1
                if (t1-t0)<0.5 and jump_before>0.5 and jump_after>0.5:
                    #ISLAND! NEED TO MASK
                    xmask[jumps[j,0]:(jumps[j,1]+1)]=False
            mask[mask]*=xmask

        #End-of-orbit cut
        # Arbritrarily looking at the first/last 15 points and calculating STD of first/last 300 pts.
        # We will cut the first/last points if the lightcurve STD is drastically better without them
        if end_of_orbit:
            stds=np.array([np.nanstd(lc[prefix+'flux'+suffix][mask][n:(300+n)]) for n in np.arange(0,17)])
            stds/=np.min(stds)
            newmask=np.tile(True,np.sum(mask))
            for n in np.arange(15):
                if stds[n]>1.05*stds[-1]:
                    newmask[n]=False
                    newmask[n+1]=False
            stds=np.array([np.nanstd(lc[prefix+'flux'+suffix][mask][(-300+n):n]) for n in np.arange(-17,0)])
            stds/=np.min(stds)
            for n in np.arange(-15,0):
                if stds[n]>1.05*stds[0]:
                    newmask[n]=False
                    newmask[n-1]=False
            mask[mask]=newmask

        # Identify outliers
        m2 = mask[:]

        for i in range(10):
            try:
                y_prime = np.interp(lc[prefix+'time'], lc[prefix+'time'][m2], lc[prefix+'flux'+suffix][m2])
                smooth = savgol_filter(y_prime, 101, polyorder=3)
                resid = lc[prefix+'flux'+suffix] - smooth
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
                resid = np.zeros(len(lc[prefix+'flux'+suffix]))
                sigma = 1.0
                pass
        return mask*m2
    else:
        return mask


def cut_high_regions(flux, mask, std_thresh=3.2,n_pts=25,n_loops=2):
    # Masking anomalous high region using a running 25-point median and std comparison
    # This is best used for e.g. Corot data which has SAA crossing events.

    digi=np.vstack([np.arange(f,len(flux)-(n_pts-f)) for f in range(n_pts)])
    stacked_fluxes=np.vstack([flux[digi[n]] for n in range(n_pts)])

    std_threshs=np.linspace(std_thresh-1.5,std_thresh,n_loops)

    for n in range(n_loops):
        stacked_masks=np.vstack([mask[digi[n]] for n in range(n_pts)])
        stacked_masks=stacked_masks.astype(int).astype(float)
        stacked_masks[stacked_masks==0.0]=np.nan

        meds=np.nanmedian(stacked_fluxes*stacked_masks,axis=0)
        stds=np.nanstd(stacked_fluxes*stacked_masks,axis=0)
        #Adding to the mask any points identified in 80% of these passes:
        #print(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]))
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0))
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0).shape)
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0)[1:-1].shape)
        #print(mask.shape)

        mask*=np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),
                                              stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
                                              np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)])
                           ,axis=0)[1:-1]<20
    return mask

def open_PDC(epic,camp,use_ppt=True,**kwargs):
    if camp == '10':
    #https://archive.stsci.edu/missions/k2/lightcurves/c1/201500000/69000/ktwo201569901-c01_llc.fits
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c102_llc.fits'
    else:
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp).zfill(2)+'_llc.fits'
    if requests.get(urlfilename1, timeout=600).status_code==200:
        with fits.open(urlfilename1,show_progress=False) as hdus:
            lc=open_fits(hdus,urlfilename1,mission='kepler',use_ppt=use_ppt,**kwargs)
            lc['src']='K2_pdc'
        return lc
    else:
        return None

def open_Vand(epic,camp,v=1,use_ppt=True,**kwargs):
    lcvand=[]
    #camp=camp.split(',')[0] if len(camp)>3
    if camp=='et' or camp=='E' or camp=='e':
        camp='e'
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
    else:
        camp=str(int(float(camp))).zfill(2)
    if camp in ['09','10','11']:
        #C91: https://archive.stsci.edu/missions/hlsp/k2sff/c91/226200000/35777/hlsp_k2sff_k2_lightcurve_226235777-c91_kepler_v1_llc.fits
        url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
        print("Vanderburg LC at ",url1)
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1,show_progress=False) as hdus:
                lcvand+=[open_fits(hdus,url1,mission='k2',use_ppt=use_ppt,**kwargs)]
        url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
        if requests.get(url2, timeout=600).status_code==200:
            with fits.open(url2,show_progress=False) as hdus:
                lcvand+=[open_fits(hdus,url2,mission='k2',use_ppt=use_ppt)]
    elif camp=='e':
        print("Engineering data")
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
        url='https://www.cfa.harvard.edu/~avanderb/k2/ep'+str(epic)+'alldiagnostics.csv'
        print("Vanderburg LC at ",url)
        df=pd.read_csv(url,index_col=False)
        lc={'time':df['BJD - 2454833'].values,
            'flux':df[' Corrected Flux'].values,
            'flux_err':np.tile(np.median(abs(np.diff(df[' Corrected Flux'].values))),df.shape[0])}
        lcvand+=[open_fits(lc,url,mission='k2',use_ppt=use_ppt,**kwargs)]
    else:
        urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
        if requests.get(urlfitsname, timeout=600).status_code==200:
            with fits.open(urlfitsname,show_progress=False) as hdus:
                lcvand+=[open_fits(hdus,urlfitsname,mission='k2',use_ppt=use_ppt,**kwargs)]
            print("Extracted vanderburg LC from ",urlfitsname)
        else:
            print("Cannot find vanderburg LC at ",urlfitsname)
    #Cutting Nones:
    lcvand=[lc for lc in lcvand if lc is not None]
    if lcvand is not None and len(lcvand)>0:
        lc=lc_stack(lcvand)
        lc['src']='K2_vand'
        return lc
    else:
        return None

def open_Everest(epic,camp,pers=None,durs=None,t0s=None,use_ppt=True,**kwargs):
    import everest
    if camp in [10,11,10.0,11.0,'10','11','10.0','11.0']:
        camp=[int(str(int(float(camp)))+'1'),int(str(int(float(camp)))+'2')]
    else:
        camp=[int(float(camp))]
    lcs=[]
    lcev={}
    camp=np.unique(np.array(camp))
    hdr=None
    for c in camp:
        try:
            st1=everest.Everest(int(epic),season=c,show_progress=False)
            if pers is not None and durs is not None and t0s is not None:
                #Recomputing lightcurve given planets
                for pl in range(len(pers)):
                    p2mask=pers[pl] if pers[pl] is not None and not np.isnan(pers[pl]) and pers[pl]!=0.0 else 200
                    dur2mask=durs[pl] if durs[pl] is not None and not np.isnan(durs[pl]) and durs[pl]!=0.0 else 0.6
                    st1.mask_planet(t0s[pl], p2mask, dur2mask)
                st1.compute()

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
            hdr={'cdpp':st1.cdpp,'ID':st1.ID,'Tmag':st1.mag,'mission':'K2','name':st1.name,'campaign':camp,'lcsource':'everest'}
        except:
            print(c,"not possible to load")
            continue
    if hdr is not None:
        lc=open_fits(lcev,hdr,mission='k2',use_ppt=use_ppt)
        #elif int(camp)>=14:
        #    lcloc='https://archive.stsci.edu/hlsps/everest/v2/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_everest_k2_llc_'+str(epic)+'-c'+str(int(camp))+'_kepler_v2.0_lc.fits'
        #    lcev=open_fits(fits.open(lcloc),lcloc)
        #lc=lc_stack(lcs)
        lc['src']='K2_ev'
        return lc
    else:
        return None


def get_k2_lc(epic,camp,saveloc=None,pers=None,durs=None,t0s=None,use_ppt=True):
    '''
    Gets (or tries to get) all LCs from K2 sources. Order is Everest > Vanderburg > PDC.
    '''
    from urllib.request import urlopen
    import everest
    lcs={}
    lcs['vand']={camp:open_Vand(int(epic), camp, use_ppt=use_ppt)}
    if camp!='E':
        lcs['ev']={camp:open_Everest(int(epic), int(float(camp)), pers=pers, durs=durs, t0s=t0s, use_ppt=use_ppt)}
        lcs['pdc']={camp:open_PDC(int(epic),int(float(camp)),use_ppt=use_ppt)}
    lcs={ilc:lcs[ilc] for ilc in lcs if lcs[ilc][camp] is not None}
    if len(lcs.keys())>1:
        lens = {l:len(lcs[l][camp]['flux'][lcs[l][camp]['mask']]) for l in lcs}
        stds = {l:np.nanmedian(abs(np.diff(lcs[l][camp]['flux'][lcs[l][camp]['mask']])))/(lens[l]/np.nanmax(lens[l]))**3 for l in lcs}
        #Making a metric from std and length - std/len_norm**3. i.e. a lc 75% as long as the longest is downweighted by 0.42 (e.g. std increased by 2.4
        ordered_keys = [k for k, v in sorted(stds.items(), key=lambda item: item[1])]
        list(np.array(list(lcs.keys()))[np.argsort(stds)])
        lc=lc_stackDicts(lcs,ordered=ordered_keys)
        return lc
    elif len(lcs.keys())==1:
        return lcs[list(lcs.keys())[0]][camp]
    elif len(lcs.keys())==0:
        return None

def K2_lc(epic,coor=None,pers=None,durs=None,t0s=None, use_ppt=True):
    '''
    # Opens K2 lc
    '''
    #if df is None or (type(df['campaign']) in [str,list] and len(df['campaign'])==0):
    from astroquery.mast import Observations
    if len(str(int(epic)))==8 and str(int(epic))[:2]=='60':
        #Engineering campaign, so we don't have a proper EPIC here.
        df=None
        v = Vizier(catalog=['J/ApJS/224/2'])
        res = v.query_region(coor, radius=5*units.arcsec, catalog=['J/ApJS/224/2'])
        if len(res)>0 and len(res[0])>0:
            other_epic=res[0]['EPIC'][0]
            obs_table = Observations.query_object("EPIC "+str(int(other_epic)))
            cands=list(np.unique(obs_table[obs_table['obs_collection']=='K2']['sequence_number'].data.data).astype(int).astype(str))
        else:
            cands=[]
        cands+=['E']
    else:
        #Normal K2 observation:
        df,_=starpars.GetExoFop(epic,"k2")
        obs_table = Observations.query_object("EPIC "+str(int(epic)))
        cands=list(np.unique(obs_table[obs_table['obs_collection']=='K2']['sequence_number'].data.data).astype(str))
    if df is None:
        df={'campaign':None}
    if df['campaign'] is None or (type(df['campaign']) in [str,list] and len(df['campaign'])==0):
        df['campaign']=','.join(cands)
    else:
        df['campaign']=','.join(cands+str(df['campaign']).split(','))
    df['campaign']=df['campaign'].replace('.0','')
    df['campaign']=df['campaign'].replace('102','10')
    lcs=[]
    print("K2 campaigns to search:",np.unique(np.array(str(df['campaign']).split(','))))
    for camp in np.unique(np.array(str(df['campaign']).split(','))):
        if camp!='':
            lcs+=[get_k2_lc(epic,camp,pers=pers,durs=durs,t0s=t0s, use_ppt=use_ppt)]
    lcs=[lc for lc in lcs if lc is not None]
    if len(lcs)>1:
        lcs=lc_stack(lcs)
        return lcs,df
    elif len(lcs)==1:
        return lcs[0],df
    else:
        return None,df


def get_kepler_lc(kic,cadence='long',use_ppt=True,**kwargs):
    '''
    This module uses the KIC of a planet candidate to download lightcurves

    Args:
        kic: EPIC (K2) or KIC (Kepler) id number

    Returns:
        lightcurve
    '''
    qcodes=["2009131105131_llc","2009131110544_slc","2009166043257_llc","2009166044711_slc","2009201121230_slc",
            "2009231120729_slc","2009259160929_llc","2009259162342_slc","2009291181958_slc","2009322144938_slc",
            "2009350155506_llc","2009350160919_slc","2010019161129_slc","2010049094358_slc","2010078095331_llc",
            "2010078100744_slc","2010111051353_slc","2010140023957_slc","2010174085026_llc","2010174090439_slc",
            "2010203174610_slc","2010234115140_slc","2010265121752_llc","2010265121752_slc","2010296114515_slc",
            "2010326094124_slc","2010355172524_llc","2010355172524_slc","2011024051157_slc","2011053090032_slc",
            "2011073133259_llc","2011073133259_slc","2011116030358_slc","2011145075126_slc","2011177032512_llc",
            "2011177032512_slc","2011208035123_slc","2011240104155_slc","2011271113734_llc","2011271113734_slc",
            "2011303113607_slc","2011334093404_slc","2012004120508_llc","2012004120508_slc","2012032013838_slc",
            "2012060035710_slc","2012088054726_llc","2012088054726_slc","2012121044856_slc","2012151031540_slc",
            "2012179063303_llc","2012179063303_slc","2012211050319_slc","2012242122129_slc","2012277125453_llc",
            "2012277125453_slc","2012310112549_slc","2012341132017_slc","2013011073258_llc","2013011073258_slc",
            "2013017113907_slc","2013065031647_slc","2013098041711_llc","2013098041711_slc","2013121191144_slc",
            "2013131215648_llc"]
    #qcodes=[2009131105131,2009166043257,2009259160929,2009350155506,2010009091648,2010078095331,2010174085026,
    #        2010265121752,2010355172524,2011073133259,2011177032512,2011271113734,2012004120508,2012088054726,
    #        2012179063303,2012277125453,2013011073258,2013098041711,2013131215648]
    lcs=[]
    if cadence=='long':
        for q in [qc for qc in qcodes if qc[-4:]=='_llc']:
            lcloc='http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/kplr'+str(int(kic)).zfill(9)+'-'+str(q)+'.fits'
            h = httplib2.Http()
            resp = h.request(lcloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(lcloc,show_progress=False) as hdu:
                    ilc=open_fits(hdu,lcloc,mission='kepler',use_ppt=use_ppt,**kwargs)
                    if ilc is not None:
                        lcs+=[ilc]
                    hdr=hdu[1].header
    elif cadence == 'short' and 'slc' in q:
        for q in [qc for qc in qcodes if qc[-4:]=='_slc']:
            lcloc='http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/kplr'+str(int(kic)).zfill(9)+'-'+str(q)+'.fits'
            h = httplib2.Http()
            resp = h.request(lcloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(lcloc,show_progress=False) as hdu:
                    ilc=open_fits(hdu,lcloc,mission='kepler',use_ppt=use_ppt,**kwargs)
                    if ilc is not None:
                        lcs+=[ilc]
                    hdr=hdu[1].header
    if len(lcs)>0:
        lc=lc_stack(lcs)
        return lc,hdr
    else:
        return None,None

def lc_stackDicts(lcdicts, ordered=None):
    #Stacks multiple lcs together while keeping info from secondary data sources.
    #lcdicts must be in form {'src1':{'camp1':{'time':[],'flux:[], ...},'sect2':{'time':...}},'src2':{'camp1':...}}}

    # Ordered should be ordered

    outlc_by_sect=[]
    #Getting all sectors/campaigns across all lightcurve extractions:
    allsects=np.unique([key_i for lcsrc in lcdicts for key_i in lcdicts[lcsrc]])
    #allkeys=np.unique(np.hstack([list(lcs[nlc].keys()) for nlc in range(len(lcs)) if lcs[nlc] is not None]))
    #allkeys=allkeys[allkeys not in ['flux_format','flux_unit','src']] #This is the only non-timeseries keyword
    ordered=list(lcdicts.keys()) if ordered is None else ordered
    #Removing keys if they are in ordered but not in lcdicts
    for key in ordered:
        if key not in lcdicts:
            ordered.remove(key)

    #print(allsects, lcdicts.keys())
    #Stacking each timeseries on top of each other
    for sect in allsects:
        sec_lc={}
        #print(allsects,allkeys,lcdicts.keys())
        for lcsrc in ordered:
            if sec_lc=={} and sect in lcdicts[lcsrc]:
                sec_lc.update(lcdicts[lcsrc][sect])
                sec_lc['src']=lcsrc
            elif sect in lcdicts[lcsrc]:
                #Section already created - adding under secondary name:
                sec_lc.update({lcsrc+'_'+key:lcdicts[lcsrc][sect][key] for key in lcdicts[lcsrc][sect] if key not in ['flux_format','flux_unit','src']})
        if sec_lc=={}:
            outlc_by_sect+=[None]
        else:
            outlc_by_sect+=[sec_lc]
    '''
    for sec in allsects:
        if sec in spoclcs:
            sec_lc=spoclcs[sec]
            if sec in qlplcs:
                sec_lc.update({'qlp_'+key:qlplcs[sec][key] for key in qlplcs[sec] if key not in ['flux_format','flux_unit']})
            elif sec in elenlcs:
                sec_lc.update({'elen_'+key:elenlcs[sec][key] for key in elenlcs[sec] if key not in ['flux_format','flux_unit']})
            fu=spoclcs[sec]['flux_unit']
        elif sec in qlplcs:
            sec_lc=qlplcs[sec]
            if sec in elenlcs:
                sec_lc.update({'elen_'+key:elenlcs[sec][key] for key in elenlcs[sec] if key not in ['flux_format','flux_unit']})
            fu=qlplcs[sec]['flux_unit']

        elif sec in elenlcs:
            sec_lc=elenlcs[sec]
            fu=elenlcs[sec]['flux_unit']
        else:
            sec_lc=None

        outlc_by_sect+=[sec_lc]
    '''
    lc=lc_stack(outlc_by_sect)
    return lc

def lc_stack(lcs):
    if len(lcs)==1:
        return lcs[0]
    else:
        #Stacks multiple lcs together
        outlc={}
        allkeys=np.unique(np.hstack([list(lcs[nlc].keys()) for nlc in range(len(lcs)) if lcs[nlc] is not None]))
        allkeys=allkeys[allkeys!='flux_format'] #This is the only non-timeseries keyword
        #Checking we dont have matching lcs:
        if len(lcs)>1:
            matching=[]
            for n1 in range(len(lcs)):
                for n2 in range(n1+1,len(lcs)):
                    if n1==n2 or (np.all(np.isin(lcs[n1]['time'],lcs[n2]['time'])) and np.all(np.isin(lcs[n1]['flux'],lcs[n2]['flux']))):
                        match_deletes+=[n1,n2]
            if len(matching)==1:
                lcs=lcs.remove(matching[0])
            elif len(matching)>1:
                for match in matching[:-1]:
                    lcs=lcs.remove(match)
            #print("matching pairs = ",matching)


        #Stacking each timeseries on top of each other
        for nlc in range(len(lcs)):
            if lcs[nlc] is not None:
                #In the case of a "parallel" lc, we put this into the dict, but with e.g. time_1 or flux_err_2:
                for key in allkeys:
                    if key in lcs[nlc]:
                        newarr=lcs[nlc][key]
                    else:
                        newarr=np.tile(np.nan,len(lcs[nlc]['time']))
                    if key in outlc:
                        outlc[key]=np.hstack([outlc[key],newarr])
                    else:
                        outlc[key]=newarr
                outlc['flux_unit']=lcs[nlc]['flux_unit']
        return outlc

def cut_anom_diff(flux,thresh=4.2):
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

def observed(tic,radec=None,maxsect=84):
    # Using either "webtess" page or Chris Burke's tesspoint to check if TESS object was observed:
    # Returns dictionary of each sector and whether it was observed or not
    
    tess_stars2px = importlib.import_module("tess-point.tess_stars2px")
    #from tesspoint import tess_stars2px_function_entry as tess_stars2px
    result = tess_stars2px.tess_stars2px_function_entry(tic, radec.ra.deg, radec.dec.deg)
    sectors = result[3]
    out_dic={s:True if s in sectors else False for s in np.arange(maxsect)}
    #print(out_dic)
    return out_dic

def get_corot_lc(corid,use_ppt=True,**kwargs):
    #These are pre-computed CoRoT LCs I have lying around. There is no easy API as far as I can tell.
    initstring="https://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_exo/FITSfiles/"
    corotlclocs={102356770:["LRa03/EN2_STAR_MON_0102356770_20091003T223149_20100301T055642.fits"],
                 102387834:["LRa03/EN2_STAR_CHR_0102387834_20091003T223149_20100301T055610.fits"],
                 102574444:["LRa01/EN2_STAR_CHR_0102574444_20071023T223035_20080303T093534.fits"],
                 102582649:["LRa06/EN2_STAR_MON_0102582649_20120112T183055_20120329T092714.fits",
                            "LRa01/EN2_STAR_CHR_0102582649_20071023T223035_20080303T093534.fits"],
                 102586624:["LRa06/EN2_STAR_MON_0102586624_20120112T183055_20120329T092714.fits",
                            "LRa01/EN2_STAR_CHR_0102586624_20071023T223035_20080303T093534.fits"],
                 102647266:["LRa01/EN2_STAR_CHR_0102647266_20071023T223035_20080303T093534.fits"],
                 102709133:["LRa01/EN2_STAR_CHR_0102709133_20071023T223035_20080303T093502.fits"],
                 102723949:["LRa06/EN2_STAR_CHR_0102723949_20120112T183055_20120329T092714.fits",
                            "LRa01/EN2_STAR_CHR_0102723949_20071023T223035_20080303T093502.fits",
                            "IRa01/EN2_STAR_CHR_0102723949_20070203T130553_20070401T235518.fits"],
                 102765275:["LRa06/EN2_STAR_MON_0102765275_20120112T183055_20120329T092714.fits",
                            "LRa01/EN2_STAR_CHR_0102765275_20071023T223035_20080303T093534.fits",
                            "IRa01/EN2_STAR_MON_0102765275_20070203T130553_20070401T235518.fits"],
                 102801672:["IRa01/EN2_STAR_MON_0102801672_20070206T133547_20070401T235654.fits"],
                 102802996:["IRa01/EN2_STAR_MON_0102802996_20070206T133547_20070401T235654.fits"],
                 102822869:["IRa01/EN2_STAR_MON_0102822869_20070206T133547_20070401T235654.fits"],
                 102829388:["IRa01/EN2_STAR_MON_0102829388_20070206T133547_20070401T235654.fits"],
                 102855409:["IRa01/EN2_STAR_CHR_0102855409_20070206T133547_20070401T235654.fits"],
                 102868004:["IRa01/EN2_STAR_MON_0102868004_20070206T133547_20070401T235654.fits"],
                 102874481:["IRa01/EN2_STAR_CHR_0102874481_20070206T133547_20070401T235654.fits"],
                 102895957:["IRa01/EN2_STAR_CHR_0102895957_20070203T130553_20070401T235934.fits"],
                 102919036:["IRa01/EN2_STAR_MON_0102919036_20070203T130553_20070401T235518.fits"],
                 102973379:["IRa01/EN2_STAR_MON_0102973379_20070206T133547_20070401T235654.fits"],
                 211616889:["SRc01/EN2_STAR_MON_0211616889_20070413T180030_20070509T065744.fits"],
                 211621528:["SRc01/EN2_STAR_CHR_0211621528_20070413T180030_20070509T065744.fits"],
                 211631779:["SRc01/EN2_STAR_MON_0211631779_20070413T180206_20070509T065920.fits"],
                 211634383:["SRc01/EN2_STAR_MON_0211634383_20070413T180206_20070509T065920.fits"],
                 211647475:["SRc01/EN2_STAR_CHR_0211647475_20070413T180030_20070509T065744.fits"],
                 211649312:["SRc01/EN2_STAR_MON_0211649312_20070413T180030_20070509T065744.fits"],
                 211650063:["SRc01/EN2_STAR_MON_0211650063_20070413T180206_20070509T065920.fits"],
                 211666578:["SRc01/EN2_STAR_MON_0211666578_20070413T180030_20070509T065744.fits"],
                 310190466:["LRc03/EN2_STAR_MON_0310190466_20090403T220030_20090702T022725.fits"],
                 315188649:["SRa03/EN2_STAR_CHR_0315188649_20100305T001525_20100329T065610.fits"],
                 629951504:["LRc08/EN2_STAR_MON_0629951504_20110708T153829_20110930T045022.fits"]}
    if int(corid) in corotlclocs:
        lcs=[]
        for loc in corotlclocs[int(corid)]:
            with fits.open(initstring+loc,show_progress=False,timeout=120) as hdus:
                lci=open_fits(hdus,initstring+loc,mission='corot',use_ppt=use_ppt,**kwargs)
                lci['src']='corot'
                lcs+=[lci]
        lc=lc_stack(lcs)
        lc['jd_base']=2451545
        return lc
    else:
        return None

def update_tce_timeseres(maxsect=None,overwrite=False):
    """Update the TCE timeseries file using newly detected TCEs

    Args:
        maxsect (int, optional): Maximum sector to use. Defaults to None which loads the last sector in the `tess_lc_locations.csv` file
        overwrite (bool, optional): Restart the file from scratch? Defaults to False.
    """
    from astropy.time import Time
    import httplib2
    
    #Getting the latest sector from the tess_lc_locations file
    maxsect = np.max(pd.read_csv(MonoData_tablepath+"/tess_lc_locations.csv",index_col=0).index.values.astype(int)) if maxsect is None else maxsect

    #Getting timing info for each sector from the TESS data release notes
    h=httplib2.Http()
    resp = h.request("https://heasarc.gsfc.nasa.gov/docs/tess/data_release_notes.html", 'HEAD')
    if int(resp[0]['status']) < 400:
        dat=h.request("https://heasarc.gsfc.nasa.gov/docs/tess/data_release_notes.html", 'GET')[1].decode()
        dat_no_space=dat.replace(" ","").replace("<td>","").replace("</td>","").split("\n")
        dat=dat.split("\n")
        jds={}
        for sect in range(1,maxsect+1):
            if "<p>"+str(8+sect*2)+"</p>" in dat_no_space:
                i_startstop=np.max([dat_no_space.index("<p>"+str(7+sect*2)+"</p>"),dat_no_space.index("<p>"+str(8+sect*2)+"</p>")])
                startstops=[]
                for i_s in [i_startstop+1,i_startstop+2]:
                    purestr=dat[i_s].replace("<td>","").replace("<p>","").replace("</p>","").replace("--","-").replace("</td>","").replace(" ","")
                    startstops+=["-".join(purestr.split("-")[:3]),"-".join(purestr.split("-")[3:])]
                jds[sect]=[Time(startstops[0].replace(" ","")+"T12:00:00",format="isot").jd,Time(startstops[3].replace(" ","")+"T12:00:00",format="isot").jd]    
            else:
                print(sect, "doesn't yet have TESS DRN data available.")    
    
    #Loading the old file and checking up to where we are complete
    tcefile=pd.read_csv(MonoData_tablepath+"/tess_tce_fractions.csv.gz",index_col=0,compression="gzip") if not overwrite else pd.DataFrame()
    ends=np.array([jds[j][1] for j in np.sort(list(jds.keys()))])
    
    #The sect to start from is therefore the index of the sorted last "end" plus one (to account for starting at 1) plus one (as we need the next available sector)
    minsect=np.argmin(abs(ends-np.max(tcefile.index.values)))+2 if not overwrite else 1

    #Getting the tables of single-sector TCE data:
    h=httplib2.Http()
    resp = h.request("https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html", 'HEAD')
    if int(resp[0]['status']) < 400:
        dat=h.request("https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html", 'GET')[1].decode()
        start_row=".csv File</th>\n</tr>\n</thead>\n<tbody>\n<tr>"
        end_row="\n</tbody>\n</table>"
        tce_singlesect_tables=pd.read_html("<tbody>\n<table>\n"+dat[dat.find(start_row)+len(start_row)+1:dat.find(end_row)]+"\n</tbody>\n</table>",index_col=0)[0]
        dat=dat[dat.find(end_row)+20:]
        tce_multisect_tables=pd.read_html("<tbody>\n<table>\n"+dat[dat.find(start_row)+len(start_row)+1:dat.find(end_row)]+"\n</tbody>\n</table>",index_col=0)[0]
        tce_multisect_tables['s_start']=[int(s.split(' ')[1]) for s in tce_multisect_tables.index.values]
        tce_multisect_tables['s_end']=[int(s.split(' ')[1]) for s in tce_multisect_tables[1].values]
    #print(tce_multisect_tables)
    #Looping through all sectors, computing TCE per cadence, adding to a list
    from astropy.io import ascii
    times=[]
    frac_in_tce=[]
    maxsect=70
    if minsect>=maxsect:
        return None
    for s in np.arange(minsect,maxsect+1):
        times+=[np.arange(jds[s][0]-0.25,jds[s][1]+0.25,0.05)]
        h=httplib2.Http()
        fileloc="https://archive.stsci.edu/missions/tess/catalogs/tce/"+tce_singlesect_tables.loc["Sector "+str(s),2]
        resp=h.request(fileloc,"HEAD")
        total=0
        print("Sector",s)
        frac_in_tce+=[np.zeros(len(times[-1]))]
        if int(resp[0]['status'])<400:
            df=ascii.read(fileloc).to_pandas()
            df=df.loc[(df['tce_period']>7)&(df['tce_model_snr']<25)]
            total+=df.shape[1]
            frac_in_tce[-1]+=np.sum(abs((times[-1][:,None]-0.5*df['tce_period'][None,:]-df['tce_time0bt'][None,:])%df['tce_period'][None,:]-0.5*df['tce_period'][None,:])<(0.02+df['tce_duration'][None,:]/48),axis=1)
        #Looping through multis
        for multis in tce_multisect_tables.loc[(s>=tce_multisect_tables['s_start'])&(s<=tce_multisect_tables['s_end'])].iterrows():
            h=httplib2.Http()
            fileloc="https://archive.stsci.edu/missions/tess/catalogs/tce/"+multis[1][2]
            resp=h.request(fileloc,"HEAD")
            if int(resp[0]['status'])<400:
                df=ascii.read(fileloc).to_pandas()
                df=df.loc[(df['tce_period']>14)&(df['tce_model_snr']<30)]
                total+=df.shape[1]
                frac_in_tce[-1]+=np.sum(abs((times[-1][:,None]-0.5*df['tce_period'][None,:]-df['tce_time0bt'][None,:])%df['tce_period'][None,:]-0.5*df['tce_period'][None,:])<(0.02+df['tce_duration'][None,:]/48),axis=1)
        frac_in_tce[-1]/=float(total)
        #Normalising:
        frac_in_tce[-1]/=np.nanmedian(frac_in_tce[-1])
        frac_in_tce[-1]-=1.0
        frac_in_tce[-1]/=np.nanstd(frac_in_tce[-1])
    
    #Stacking new TCE lists with the old ones
    newtcefile=pd.DataFrame()
    newtcefile['TJD']=np.hstack(times)
    newtcefile['TCE_fraction']=np.hstack(frac_in_tce)
    newtcefile=newtcefile.sort_values('TJD')
    newtcefile=newtcefile.set_index(newtcefile['TJD']).loc[:,['TCE_fraction']]
    
    #There may be overlapping values due to date issues... averaging the values where there are overlaps
    overlaps=np.where(np.diff(newtcefile['TJD'])<0.045)[0]
    oldtimes=newtcefile.loc[~np.isin(range(len(newtcefile)),np.unique(np.hstack((overlaps,overlaps+1)))),'TJD']
    oldtces=newtcefile.loc[~np.isin(range(len(newtcefile)),np.unique(np.hstack((overlaps,overlaps+1)))),'TCE_fraction']
    newertcefile=pd.DataFrame()
    newertcefile['TJD']=np.hstack((oldtimes,0.5*(newtcefile.loc[overlaps,'TJD'].values+newtcefile.loc[overlaps+1,'TJD'].values)))
    newertcefile['TCE_fraction']=np.hstack((oldtces,0.5*(newtcefile.loc[overlaps,'TCE_fraction'].values+newtcefile.loc[overlaps+1,'TCE_fraction'].values)))
    newertcefile=newertcefile.sort_values('TJD')
    newertcefile=newertcefile.set_index(newertcefile['TJD']).loc[:,['TCE_fraction']]
    newertcefile.to_csv(MonoData_tablepath+"/tess_tce_fractions.csv.gz",compression="gzip")

def saferound2(durations,n_plot=24):
    """Custom function to round the plot durations such that all values are always equal to or greater than 1."""
    from iteround import saferound
    from copy import deepcopy
    minroundedsize=0
    durs=deepcopy(durations[:])
    while minroundedsize<1:
        sizes=saferound(n_plot*np.array(durs)/np.sum(durs),places=0)
        minroundedsize=np.min(sizes)
        if minroundedsize==0:
            durs[np.argmin(sizes)]*=1.5
            #Going again with larger sizes
    return sizes

def update_lc_locs(epoch,most_recent_sect):
    #Updating the table of lightcurve locations using the scripts on the MAST/TESS "Bulk Downloads" page.
    all_sects=np.arange(np.max(epoch.index.values),most_recent_sect).astype(int)+1
    for sect in all_sects:
        fitsloc="https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_"+str(sect).zfill(2)+"_lc.sh"
        h = httplib2.Http()
        resp, content = h.request(fitsloc)
        if int(resp['status']) < 400:
            filename=content.split(b'\n')[1].decode().split(' ')[-2].split('-')
            epoch.loc[sect, ['date', 'runid']] = [int(filename[0][4:]), int(filename[3])]
        else:
            print("Sector "+str(sect)+" not (yet) found on MAST | RESPONCE:"+resp['status'])
    epoch.to_csv(MonoData_tablepath+"/tess_lc_locations.csv")
    return epoch

def TESS_lc(tic, sectors='all',use_ppt=True, coords=None, use_qlp=None, use_eleanor=None, data_loc=None, search_fast=False, update_logs=True, **kwargs):
    #Downloading TESS lc
    if data_loc is None:
        data_loc=MonoData_savepath+"/TIC"+str(int(tic)).zfill(11)
    #Using the JD today and the JD start of the first sector (TJD=1325.29278) to estimate most recent sector
    from astropy.time import Time
    most_recent_sect = np.floor((Time(datetime.now().strftime("%Y-%m-%d")).jd-2458325.29278)/27.295)-1
    epoch=pd.read_csv(MonoData_tablepath+"/tess_lc_locations.csv",index_col=0)
    if most_recent_sect not in epoch.index:
        epoch=update_lc_locs(epoch,most_recent_sect)
    sect_to_orbit={sect+1:[9+sect*2,10+sect*2] for sect in range(np.max(epoch.index))}
    lcs=[];lchdrs=[]
    if sectors == 'all':
        if coords is not None and type(coords)==SkyCoord:
            sect_obs=observed(coords)
        else:
            sect_obs=observed(tic)
        epochs=[key for key in epoch.index if sect_obs[key]]

        if epochs==[]:
            #NO EPOCHS OBSERVABLE APPARENTLY. USING THE EPOCHS ON EXOFOP/TIC8
            toi_df=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
            if tic in toi_df['TIC ID'].values:
                print("FOUND TIC IN TOI LIST")
                epochs=list(np.array(toi_df.loc[toi_df['TIC ID']==tic,'Sectors'].values[0].split(',')).astype(int))
    elif type(sectors)==list or type(sectors)==np.ndarray:
        epochs=[s for s in sectors if s<=np.max(epoch.index)]
    else:
        epochs=[sectors]

    lchdrs=[]
    qlplcs={}
    elenorlcs={}
    spoclcs={}
    for key in epochs:
        #2=minute cadence data from tess website
        ntype=0
        types=['fast-lc','lc'] if search_fast else ['lc']
        while key not in spoclcs and ntype<len(types):
            fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(key).zfill(4)+"/"+str(tic).zfill(16)[:4]+"/"+str(tic).zfill(16)[4:8] + \
                    "/"+str(tic).zfill(16)[-8:-4]+"/"+str(tic).zfill(16)[-4:]+"/tess"+str(epoch.loc[key,'date'])+"-s"+str(key).zfill(4)+"-" + \
                    str(tic).zfill(16)+"-"+str(epoch.loc[key,'runid']).zfill(4)+"-s_"+types[ntype]+".fits"
            h = httplib2.Http()
            strtid=str(int(tic)).zfill(16)
            resp = h.request(fitsloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    spoclcs[key]=open_fits(hdus,fitsloc,mission='tess',use_ppt=use_ppt,**kwargs)
                    lchdrs+=[hdus[0].header]
            else:
                print("LC type",types[ntype],"is not accessible")
            ntype+=1
    
        if key not in spoclcs:
            #Getting spoc 30min data:
            fitsloc='https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc/s'+str(int(key)).zfill(4) + \
                    "/target/"+strtid[:4]+"/"+strtid[4:8]+"/"+strtid[8:12]+"/"+strtid[12:] + \
                    "/hlsp_tess-spoc_tess_phot_"+strtid+"-s"+str(int(key)).zfill(4)+"_tess_v1_lc.fits"
            resp = h.request(fitsloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    spoclcs[key]=open_fits(hdus,fitsloc,mission='tess')
                    lchdrs+=[hdus[0].header]

        if use_qlp is None or use_qlp is True:
            qlpfiles=[data_loc+"/orbit-"+str(int(sect_to_orbit[key][n]))+"_qlplc.h5" for n in range(2)]
            #print(key,
            #      qlpfiles[0],os.path.isfile(qlpfiles[0]),
            #      qlpfiles[1],os.path.isfile(qlpfiles[1]))
            if os.path.isfile(qlpfiles[0]) and os.path.isfile(qlpfiles[1]):

                f1=h5py.File(qlpfiles[0])
                f2=h5py.File(qlpfiles[1])
                qlplcs[key]=lc_stack([open_fits(f1,sect_to_orbit[key][0],mission='tess',use_ppt=use_ppt,**kwargs),
                                     open_fits(f2,sect_to_orbit[key][1],mission='tess',use_ppt=use_ppt,**kwargs)])
                lchdrs+=[{'source':'qlp'}]
            else:
                fitsloc='https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/qlp/s'+str(int(key)).zfill(4) + \
                        "/"+strtid[:4]+"/"+strtid[4:8]+"/"+strtid[8:12]+"/"+strtid[12:] + \
                        "/hlsp_qlp_tess_ffi_s"+str(int(key)).zfill(4)+"-"+strtid+"_tess_v01_llc.fits"
                #print("QLP:",fitsloc)
                resp = h.request(fitsloc, 'HEAD')
                if int(resp[0]['status']) < 400:
                    with fits.open(fitsloc,show_progress=False) as hdus:
                        qlplcs[key]=open_fits(hdus,fitsloc,mission='tess',use_ppt=use_ppt,**kwargs)
                        lchdrs+=[hdus[0].header]
        elif use_eleanor is None or use_eleanor is True:
            from eleanor import eleanor
            print("Loading Eleanor Lightcurve")
            try:
                #Getting eleanor lightcurve:
                try:
                    star = eleanor.Source(tic=tic, sector=key)
                except:
                    star = eleanor.Source(coords=coords, sector=key)
                try:
                    elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True,save_postcard=False)
                except:
                    try:
                        elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=False,save_postcard=False)
                    except:
                        elen_obj=eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False,save_postcard=False)
                elen_hdr={'ID':star.tic,'GaiaID':star.gaia,'Tmag':star.tess_mag,
                          'RA':star.coords[0],'dec':star.coords[1],'mission':'TESS','campaign':key,'source':'eleanor',
                          'ap_masks':elen_obj.all_apertures,'ap_image':np.nanmedian(elen_obj.tpf[50:80],axis=0)}
                elenorlcs[key]=open_fits(elen_obj,elen_hdr,mission='tess',use_ppt=use_ppt,**kwargs)
                lchdrs+=[elen_hdr]
            except Exception as e:
                print(e, tic,"not observed by TESS in sector",key)

    if len(spoclcs)+len(qlplcs)+len(elenorlcs)>0:
        lc=lc_stackDicts({'spoc':spoclcs,'qlp':qlplcs,'elen':elenorlcs},['spoc','qlp','elen'])
        return lc,lchdrs[0]
        #elif len(lcs)==1:
        #    #print(lcs,lchdrs)
        #    return lcs[0],lchdrs[0]
    else:
        return None,None

def open_light_curve(ID,mission,coor=None,use_ppt=True,other_data=True,
                   jd_base=2457000,save=True,**kwargs):
    #from ..stellar import tess_stars2px_mod
    if coor is None:
        #Doing this to get coordinates:
        df,_=starpars.GetExoFop(ID,mission)
        #Getting coordinates from df in order to search other surveys for ID/data:
        ra,dec=df['ra'],df['dec']
        if type(ra)==str and (ra.find(':')!=-1)|(ra.find('h')!=-1):
            coor=SkyCoord(ra,dec,unit=(units.hourangle,units.deg))
        elif (type(ra)==float)|(type(ra)==np.float64) or (type(ra)==str)&(ra.find(',')!=-1):
            coor=SkyCoord(ra,dec,unit=units.deg)

    #Finding IDs for other missions:
    IDs={mission.lower():ID}
    if not other_data:
        for other in ['tess','k2','kepler','corot']:
            if other!=mission.lower():
                IDs[other]=None
    else:
        if mission.lower()!='k2':
            v = Vizier(catalog=['J/ApJS/224/2'])
            res = v.query_region(coor, radius=5*units.arcsec, catalog=['J/ApJS/224/2'])
            if len(res)>0 and len(res[0])>0:
                IDs['k2']=res[0]['EPIC'][0]
            else:
                IDs['k2']=None
        if mission.lower()!='tess':
            # Let's search for associated TESS lightcurve:


            tess_id = Catalogs.query_criteria("TIC",coordinates=coor,radius=12*units.arcsec,
                                              objType="STAR",columns=['ID','KIC','Tmag']).to_pandas()
            #print(tess_id)
            #
            '''tess_id, tess_dat, sects = tess_stars2px_mod.SectFromCoords(coor,tic=None)
            if tess_id is not None:
                IDs['tess']=tess_id
            '''
            if tess_id is not None and len(tess_id)>0:
                tess_id=tess_id.iloc[np.argmin(tess_id['Tmag'])] if type(tess_id)==pd.DataFrame else tess_id
                IDs['tess']=tess_id['ID']
            else:
                IDs['tess']=None
        #else:
        #    tess_id, tess_dat, sects = tess_stars2px_mod.SectFromCoords(coor,tic=ID)
        if mission.lower()!='kepler':
            v = Vizier(catalog=['V/133/kic'])
            res=v.query_region(coor, radius=5*units.arcsec, catalog=['V/133/kic'])
            if 'V/133/kic' in res.keys():
                if len(res['V/133/kic'])>1:
                    #print(res['V/133/kic'][['KIC','kepmag']], "MULTIPLE KICS FOUND")
                    IDs['kepler'] = res['V/133/kic']['KIC'][np.argmin(res['V/133/kic']['kepmag'])]
                elif len(res['V/133/kic'])==1:
                    #print(res['V/133/kic'][['KIC','kepmag']], "ONE KIC FOUND")
                    IDs['kepler'] = res['V/133/kic']['KIC'][0]
                elif len(res['V/133/kic'])==0:
                    IDs['kepler'] = None
                    #print(res['V/133/kic'], "NO KICS FOUND")
            else:
                IDs['kepler'] = None

    #Opening using url search:
    lcs={};hdrs={}
    if IDs['tess'] is not None:
        lcs['tess'],hdrs['tess'] = TESS_lc(IDs['tess'], use_ppt=use_ppt, coords=coor, **kwargs)
        if lcs['tess'] is not None:
            lcs['tess']['time']-=(jd_base-2457000)
    if IDs['k2'] is not None:
        lcs['k2'],hdrs['k2'] = K2_lc(IDs['k2'],coor,pers=kwargs.get('periods',None),
                                     durs=kwargs.get('initdurs',None),
                                     t0s=kwargs.get('initt0',None),
                                     use_ppt=use_ppt)
        if lcs['k2'] is not None:
            for key in lcs['k2']:
                if 'time' in key:
                    lcs['k2'][key]-=(jd_base-2454833)
    if IDs['kepler'] is not None:
        lcs['kepler'],hdrs['kepler'] = get_kepler_lc(IDs['kepler'],use_ppt=use_ppt)
        if lcs['kepler'] is not None:
            lcs['kepler']['time']-=(jd_base-2454833)
    if mission.lower() == 'corot':
        lcs['corot'] = getCorotLC(ID,use_ppt=use_ppt)
        lcs['corot']['time']-=(jd_base-lcs['corot']['jd_base'])
        lcs['corot']['jd_base']=jd_base
        hdrs['corot'] = None
    #print(IDs,lcs)
    if len(lcs.keys())>=1:
        lc=lc_stack([lcs[lc] for lc in lcs if lcs[lc] is not None])
    elif not other_data:
        lc=lcs[mission.lower()]
    else:
        lc=lcs[list(lcs.keys())[0]]

    if lc is not None:
        lc['jd_base']=jd_base

        #Maing sure lightcurve is sorted by time, and that there are no nans in the time array:
        for key in lc:
            if key!='time' and type(lc[key])==np.ndarray and len(lc[key])==len(lc['time']):
                lc[key]=lc[key][~np.isnan(lc['time'])]
                lc[key]=lc[key][:][np.argsort(lc['time'][~np.isnan(lc['time'])])]
        lc['time']=np.sort(lc['time'][~np.isnan(lc['time'])])

    if save:
        ID_string=id_dic[mission]+str(ID).zfill(11)
        if not os.path.isdir(MonoData_savepath+'/'+ID_string):
            os.system("mkdir "+MonoData_savepath+'/'+ID_string)
        pickle.dump(lc,open(MonoData_savepath+'/'+ID_string+'/'+ID_string+'_lc.pickle','wb'))

    return lc,hdrs[mission.lower()]

def load_lc(lcid,mission='tess',file_loc=None):
    # Quick tool to load pickled lightcurve dict.
    # lcid = ID
    # mission = 'tess'. mission string (TESS, K2, Kepler, etc)
    # file_loc = None ; loction of pickle. If None defaults to $MONOTOOLSPATH
    ID_string=id_dic[mission]+str(lcid).zfill(11)
    file_loc=MonoData_savepath+'/'+ID_string if file_loc is not None else file_loc
    return pickle.load(open(MonoData_savepath+'/'+ID_string+'/'+ID_string+'_lc.pickle','rb'))

def cut_lc(lctimes,max_len=10000,return_bool=True,transit_mask=None):
    # Naturally cut the lightcurve time into chunks smaller than max_len (e.g. for GP computations)
    assert(np.isnan(lctimes).sum()==0)
    if return_bool:
        bools=[np.tile(True,len(lctimes))]
        max_time_len=np.sum(bools[0])
        if np.sum(bools[0])>max_len:
            while max_time_len>max_len:
                newbools=[]
                for n in range(len(bools)):
                    if np.sum(bools[n])>max_len:
                        middle_boost=4*(0.3-((lctimes[bools[n]][:-1]+np.diff(lctimes[bools[n]]) - \
                                              np.median(lctimes[bools[n]]))/(lctimes[bools[n]][-1]-lctimes[bools[n]][0]))**2)
                        #And then cut along the maximum value into two new times:
                        if transit_mask is not None:
                            #Making sure we dont do the cuts on transits
                            middle_boost*=(transit_mask[bools[n]][1:]|transit_mask[bools[n]][:-1]).astype(float)
                        maxloc=np.argmax(np.diff(lctimes[bools[n]])*middle_boost)
                        cut_time=0.5*(lctimes[bools[n]][maxloc]+lctimes[bools[n]][maxloc+1])
                        newbools+=[bools[n]&(lctimes<=cut_time),
                                   bools[n]&(lctimes>cut_time)]
                    else:
                        newbools+=[bools[n]]
                bools=newbools
                max_time_len=np.max([np.sum(b) for b in bools])
            return bools
        else:
            return bools
    else:
        if len(lctimes)>max_len:
            times=[lctimes]
            max_time_len=len(times[0])
            while max_time_len>max_len:
                newtimes=[]
                for n in range(len(times)):
                    if len(times[n])>max_len:
                        #For chunks larger than max_len we create a*boost* for how central they are w.r.t the full lc
                        middle_boost=4*(0.3-((times[n][:-1]+np.diff(times[n])-np.median(times[n]))/(times[n][-1]-times[n][0]))**2)
                        #And then cut along the maximum value of this boost multiplied by the lightcurve gapsinto two new times:
                        cut_n=np.argmax(np.diff(times[n])*middle_boost[n])
                        newtimes+=[times[n][:cut_n+1],times[n][cut_n+1:]]
                    else:
                        newtimes+=[times[n]]
                times=newtimes
                max_time_len=np.max([len(t) for t in times])
            return times
        else:
            return [lctimes]

def med_and_std(values):
    return [np.nanmedian(values),np.nanstd(values)]

def weighted_avg_and_std(values, errs, masknans=True, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if len(values)>1:
        average = np.average(values, weights=1/errs**2,axis=axis)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=1/errs**2,axis=axis)
        binsize_adj = np.sqrt(len(values)) if axis is None else np.sqrt(values.shape[axis])
        return [average, np.sqrt(variance)/binsize_adj]
    elif len(values)==1:
        return [values[0], errs[0]]
    else:
        return [np.nan, np.nan]

def lc_bin(lc,binsize=1/48,split_gap_size=0.8,use_flat=True,use_masked=True, use_raw=False,extramask=None,modify_lc=True):
    #Binning lightcurve to e.g. 30-min cadence for planet search
    # Can optionally use the flatted lightcurve
    binlc={}

    #Using flattened lightcurve as well as normal one:
    if use_flat and 'flux_flat' not in lc:
        lc=lc_flatten(lc)
    if use_flat:
        flux_dic=['flux_flat','flux']
        binlc['flux_flat']=[]
        binlc['flux']=[]
    else:
        flux_dic=['flux']
        binlc['flux']=[]
    binlc['bin_cadence']=[]

    if use_raw:
        flux_dic+=['raw_flux']
        binlc['raw_flux']=[]

    if np.nanmax(np.diff(lc['time']))>split_gap_size:
        loop_blocks=np.array_split(np.arange(len(lc['time'])),np.where(np.diff(lc['time'])>2.0)[0])
    else:
        loop_blocks=[np.arange(len(lc['time']))]
    if extramask is not None and type(extramask)==np.ndarray and (type(extramask[0])==bool)|(type(extramask[0])==np.bool_):
        mask=lc['mask']&extramask
    else:
        mask=lc['mask']
    #For each of the seprated lightcurve blocks:
    for sh_time in loop_blocks:
        nodata=False
        #For each of the flux arrays (binned and normal):
        cads=None;digi=None
        for fkey in flux_dic:
            if use_masked:
                if len(lc[fkey][sh_time][mask[sh_time]])>0:
                    lc_segment=np.column_stack((lc['time'][sh_time][mask[sh_time]],
                                                lc[fkey][sh_time][mask[sh_time]],
                                                lc['flux_err'][sh_time][mask[sh_time]]))
                    cads=lc['cadence'][sh_time][mask[sh_time]]
                else:
                    nodata=True
            else:
                lc_segment=np.column_stack((lc['time'][sh_time],lc[fkey][sh_time],lc['flux_err'][sh_time]))
                cads=lc['cadence'][sh_time]
            if binsize>(1.66*np.nanmedian(np.diff(lc['time'][sh_time]))) and not nodata:
                #Only doing the binning if the cadence involved is >> the cadence
                binlc[fkey]+=[bin_lc_segment(lc_segment, binsize)]
                digi=np.digitize(lc_segment[:,0],
                                 np.arange(np.min(lc_segment[:,0])-0.5*binsize,np.max(lc_segment[:,0])+0.5*binsize,binsize))
            elif not nodata:
                binlc[fkey]+=[lc_segment]
        if binsize>(1.66*np.nanmedian(np.diff(lc['time'][sh_time]))) and digi is not None:
            binlc['bin_cadence']+=[np.array([cads[digi==d] if type(cads[digi==d])==str else cads[digi==d][0] for d in np.unique(digi)])[:,np.newaxis]]
        else:
            if cads is not None:
                binlc['bin_cadence']+=[cads[:,np.newaxis]]
    binlc={fkey:np.vstack(binlc[fkey]) for fkey in binlc}
    if modify_lc:
        lc['bin_time']=binlc['flux'][:,0]
        for fkey in flux_dic:
            lc['bin_'+fkey]=binlc[fkey][:,1]
            #Need to clip error here as tiny (and large) errors from few points cause problems down the line.
            lc['bin_'+fkey+'_err']=np.clip(binlc[fkey][:,2],0.9*np.nanmedian(binlc[fkey][:,2]),20*np.nanmedian(binlc[fkey][:,2]))
        return lc
    else:
        ret_lc={}
        ret_lc['time']=binlc['flux'][:,0]
        for fkey in flux_dic:
            ret_lc['bin_'+fkey]=binlc[fkey][:,1]
            #Need to clip error here as tiny (and large) errors from few points cause problems down the line.
            ret_lc['bin_'+fkey+'_err']=np.clip(binlc[fkey][:,2],0.9*np.nanmedian(binlc[fkey][:,2]),20*np.nanmedian(binlc[fkey][:,2]))
        ret_lc['bin_cadence']=binlc['bin_cadence'].ravel()
        return ret_lc

def old_bin_lc_given_new_x(lc_segment, new_x):
    binsize=np.nanmedian(np.diff(new_x))
    #Making bin divisions half way between each defined x point here
    new_bins=np.hstack((new_x[0]-0.5*binsize,0.5*(new_x[:-1]+new_x[1:]),new_x[-1]+0.5*binsize))
    digi=np.digitize(lc_segment[:,0],new_bins)
    #print(len(np.unique(digi)),np.max(digi),np.min(digi))
    fluxes=np.vstack([[weighted_avg_and_std(lc_segment[digi==d,1],lc_segment[digi==d,2])] for d in range(1,len(new_bins))])
    #print(len(new_x),fluxes.shape, len(new_bins), new_x[:2],new_bins[:3],new_x[-3:],new_bins[-4:],)
    binlc=np.column_stack((new_x,fluxes))
    return binlc

def bin_lc_given_new_x(lc_segment, new_x):
    return np.column_stack((bin_light_curve(time=lc_segment[:,0],flux=lc_segment[:,1],flux_err=lc_segment[:,2],bin_time=binsize,return_bin_indices=return_digi,new_time_bins=new_x)))

def old_bin_lc_segment(lc_segment, binsize,return_digi=False):
    if len(lc_segment)>0:
        digi=np.digitize(lc_segment[:,0],np.arange(np.min(lc_segment[:,0])-0.5*binsize,np.max(lc_segment[:,0])+0.5*binsize,binsize))
        binlc=np.vstack([[[np.nanmedian(lc_segment[digi==d,0])]+\
                                weighted_avg_and_std(lc_segment[digi==d,1],lc_segment[digi==d,2])] for d in np.unique(digi)])
        if return_digi:
            return binlc, digi
        else:
            return binlc
    else:
        return lc_segment

def bin_lc_segment(lc_segment, binsize, return_digi=False):
    return np.column_stack((bin_light_curve(time=lc_segment[:,0],flux=lc_segment[:,1],flux_err=lc_segment[:,2],bin_time=binsize,return_bin_indices=return_digi)))

def bin_light_curve(time, flux, flux_err= None,
    bin_time= 1 / 24, return_std=True, return_bin_indices= False,new_time_bins = None,
):
    """Bins a light curve into time intervals and calculates the weighted mean
    and optionally the standard deviation of the flux in each bin.

    Parameters
    ----------
    time : np.ndarray
        The 1D array containing the time points of the light curve.
    flux : np.ndarray
        The 1D array containing the flux values of the light curve at the
        corresponding time points.
    flux_err : Optional[np.ndarray], optional
        The 1D array containing the error (uncertainty) on the flux values.
        If not provided, a weight of 1 is used for each data point.
    bin_time : float, optional
        The width of the bins in units of time. Defaults to 1/24 (days).
    return_std : bool, optional
        If True (default), calculate and return the weighted standard deviation
        of the flux in each bin. If False, only the bin centers and weighted
        mean flux are returned.
    return_bin_indices : bool, optional
        If True (default), return the indexes for each bin derived by `np.digitize`.
    new_time_bins : Optional[np.ndarray], optional
        An array of potential new times to bin into.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        A tuple containing the bin centers and the weighted mean flux in each bin.
        If `return_std` is True, the tuple also includes the weighted standard
        deviation of the flux in each bin.

    Notes
    -----
    This function removes any NaN values from the input arrays before binning.
    For bins with only one data point, the standard deviation is set to zero.

    See Also
    --------
    bin_light_curve_slow
    """
    if np.any(np.isnan(time)):
        mask = ~np.isnan(time)
        time = time[mask]
        flux = flux[mask]
        if flux_err is not None:
            flux_err = flux_err[mask]

    if new_time_bins is None:
        # Create bins based on the specified bin_time
        bins = np.arange(np.nanmin(time), np.nanmax(time) + 2 * bin_time, bin_time)
    else:
        avbinsize=np.nanmedian(np.diff(new_x))
        #Making bin divisions half way between each defined x point here
        bins=np.hstack((new_x[0]-0.5*avbinsize,0.5*(new_x[:-1]+new_x[1:]),new_x[-1]+0.5*avbinsize))

    # Digitize the time array into the created bins
    bin_indices = np.digitize(time, bins) - 1

    # Find non-empty bins
    points_per_bin = np.bincount(bin_indices)
    non_empty_bins = np.where(points_per_bin > 0)
    bin_centers = bins[non_empty_bins] + bin_time / 2

    weights = 1.0 if flux_err is None else 1 / flux_err**2
    weighted_sum = np.bincount(bin_indices, weights=flux * weights)[non_empty_bins]
    sum_of_weights = np.bincount(bin_indices, weights=None if flux_err is None else weights)[
        non_empty_bins
    ]
    weighted_mean = weighted_sum / sum_of_weights

    if return_std:
        mean_repeated = np.repeat(weighted_mean, points_per_bin[non_empty_bins])
        variance = (
            np.bincount(bin_indices, weights=(flux - mean_repeated) ** 2 * weights)[non_empty_bins]
            / sum_of_weights
        )
        weighted_std = np.where(points_per_bin[non_empty_bins] == 1, 0, np.sqrt(variance))
        if return_bin_indices:
            return bin_centers, weighted_mean, weighted_std/np.sqrt(points_per_bin[non_empty_bins]), bin_indices
        else:
            return bin_centers, weighted_mean, weighted_std/np.sqrt(points_per_bin[non_empty_bins])
    else:
        if return_bin_indices:
            return bin_centers, weighted_mean, bin_indices
        else:
            return bin_centers, weighted_mean

def create_transit_mask(t,tcens,tdurs,maskdist=1.1):
    in_trans=np.zeros_like(t).astype(bool)
    for n in range(len(tcens)):
        in_trans+=abs(t-tcens[n])<0.5*maskdist*tdurs[n]
    return ~in_trans

def do_polyfit(win,mask=None,stepcent=0.0,d=3,ni=10,sigclip=3):
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
        randmask = abs(np.random.normal(0.0,1.0,len(maskedwin)))<best_offset
        randmask = np.tile(True,len(maskedwin)) if np.sum(randmask)==0 else randmask

        new_base = np.polyfit(maskedwin[randmask,0]-stepcent,maskedwin[randmask,1],
                              w=1.0/np.power(maskedwin[randmask,2],2),deg=d)
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        new_offset = (maskedwin[:,1]-np.polyval(new_base,maskedwin[:,0]))**2/maskedwin[:,2]**2
        new_llk=-0.5 * np.sum(new_offset)
        if new_llk>best_llk:
            #If that fit is better than the last one, we update the offsets and the llk:
            best_llk=new_llk
            best_offset=new_offset[:]
            best_base=new_base[:]
    return best_base

def form_window(dat,cent,size,boxsize,gapthresh=1.0):

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

def lc_flatten(lc, winsize = 3.5, stepsize = 0.15, polydegree = 2,
              niter = 10, sigmaclip = 3., gapthreshold = 1.0,
              use_binned=False, use_mask=True, reflect=True, transit_mask=None, debug=False):
    '''#Flattens any lightcurve while maintaining in-transit depth.

    Args:
    lc.           # dictionary with time,flux,flux_err, flux_unit (1.0 or 0.001 [ppt]) and mask
    winsize = 2   #days, size of polynomial fitting region
    stepsize = 0.2  #days, size of region within polynomial region to detrend
    polydegree = 3  #degree of polynomial to fit to local curve
    niter = 20      #number of iterations to fit polynomial, clipping points significantly deviant from curve each time.
    sigmaclip = 3.   #significance at which points are clipped (as niter)
    gapthreshold = 1.0  #days, threshold at which a gap in the time series is detected and the local curve is adjusted to not run over it
    use_binned = False. #Using the binned values in the lc dict
    use_mask = True.    #Use the lightcurve mask to remove pre-determined anomalous values from fitting
    reflect = True      #Whether to use end-of-lightcurve reflection to remove poor end-of-lc detrending
    transit_mask = None #bolean array masking known transits so that their depressed flux wont influence the polynomial fitting
    '''
    winsize=3.9 if np.isnan(winsize) else winsize
    stepsize=0.15 if np.isnan(stepsize) else stepsize

    prefix='bin_' if use_binned else ''

    lc[prefix+'flux_flat']=np.zeros(len(lc[prefix+'time']))
    #general setup
    uselc=np.column_stack((lc[prefix+'time'][:],lc[prefix+'flux'][:],lc[prefix+'flux_err'][:]))
    if len(lc['mask'])==len(uselc[:,0]) and use_mask:
        initmask=(lc['mask']&(lc['flux']/lc['flux']==1.0)&(lc['flux_err']/lc['flux_err']==1.0)).astype(int)[:]
        if type(transit_mask)==np.ndarray:
            if debug: print("transit mask:",type(initmask),len(initmask),
                  initmask[0],type(transit_mask),len(transit_mask),transit_mask[0])
            initmask=(initmask.astype(bool)&transit_mask).astype(int)

    else:
        initmask=(np.isfinite(uselc[:,1])&np.isfinite(uselc[:,2])).astype(int)
    uselc=np.column_stack((uselc,initmask))
    uselc[:,1:3]/=lc['flux_unit']
    uselc[:,1]-=np.nanmedian(lc[prefix+'flux'])

    jumps=np.hstack((0,np.where(np.diff(uselc[:,0])>winsize*0.8)[0]+1,len(uselc[:,3]) )).astype(int)
    stepcentres=[]
    uselc_w_reflect=[]

    for n in range(len(jumps)-1):
        stepcentres+=[np.arange(uselc[jumps[n],0],
                                uselc[np.clip(jumps[n+1],0,len(uselc)-1),0],
                                stepsize) + 0.5*stepsize]
        if reflect and (uselc[jumps[n]:jumps[n+1],0][-1]-uselc[jumps[n]:jumps[n+1],0][0])>0.8*winsize:
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
            #print(partlc.shape,len(xx[0]),len(xx[1]),refl_t.shape,refl_flux.shape,refl_bool.shape)
            uselc_w_reflect+=[np.column_stack((refl_t,refl_flux,refl_bool))]
        elif (uselc[jumps[n]:jumps[n+1],0][-1]-uselc[jumps[n]:jumps[n+1],0][0])<0.8*winsize:
            uselc_w_reflect+=[np.column_stack((uselc[jumps[n]:jumps[n+1]],np.tile(1.0,len(uselc[jumps[n]:jumps[n+1],0])) ))]
    stepcentres=np.hstack(stepcentres)
    if reflect:
        uselc=np.vstack(uselc_w_reflect)
    else:
        uselc=np.column_stack((uselc,np.ones(len(uselc[:,0])) ))
    uselc[:,2]=np.clip(uselc[:,2],np.nanmedian(uselc[:,2])*0.8,100)
    #print(len(uselc),np.sum(uselc[:,3]),np.sum(uselc[:,4]))
    #now for each step centre we perform the flattening:
    #actual flattening
    for s,stepcent in enumerate(stepcentres):
        win,box = form_window(uselc,stepcent,winsize,stepsize,gapthreshold)  #should return window around box not including box
        newbox=box[uselc[:,4].astype(bool)] # Excluding from our box any points which are actually part of the "reflection"
        #Checking that we have points in the box where the window is not entirely junk/masked
        if np.sum(newbox)>0 and np.sum(win&uselc[:,3].astype(bool))>0:
            #Forming the polynomial fit from the window around the box:
            if debug: print("window size:",np.sum(win),"masked points:",np.sum(uselc[win,3]))
            if debug: print("window lc:",uselc[win,:3])
            baseline = do_polyfit(uselc[win,:3],mask=uselc[win,3].astype(bool),
                                 stepcent=stepcent,d=polydegree,ni=niter,sigclip=sigmaclip)
            lc[prefix+'flux_flat'][newbox] = lc[prefix+'flux'][newbox] - np.polyval(baseline,lc[prefix+'time'][newbox]-stepcent)*lc['flux_unit']
            #Here we have

    return lc

def run_from_scratch(ID, mission, tcen, tdur, ra=None, dec=None,
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
    lc,hdr=open_light_curve(ID,mission,use_ppt=False)
    lc=lc_flatten(lc,winsize=9*tdur,stepsize=0.1*tdur)

    #Runs Quick Model fit
    monoparams, interpmodel = search.QuickMonoFit(lc,tc,dur,Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)

    #Checks to see if dip is due to background asteroid
    asteroidDeltaBIC=search.AsteroidCheck(lc, monoparams, interpmodel)
    if asteroidDeltaBIC>6:
        planet_dic_1['01']['flag']='asteroid'

    #Checks to see if dip is combined with centroid
    centroidDeltaBIC=search.CentroidCheck(lc, monoparams, interpmodel)
    if centroidDeltaBIC>6:
        planet_dic_1['01']['flag']='EB'

    #Searches for other dips in the lightcurve
    planet_dic_1=search.SearchForSubsequentTransits(lc, interpmodel, tc, dur, Rs=Rstar[0],Ms=rhostar[0]*Rstar[0]**3)

    #Asses whether any dips are significant enough:
    if planet_dic_1['01']['SNR']>mono_SNRthresh:
        #Check if the Depth/Rp suggests we have a very likely EB, we search for a secondary
        if planet_dic_1['01']['rp_rs']<PL_ror_thresh:
            #Searching for other (periodic) planets in the system
            planet_dic_2=search.SearchForOtherPlanets(lc, planet_dic_1['01'], SNRthresh=other_planet_SNRthresh)
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

def get_savename(ID, mission, how='load', suffix='mcmc.pickle', overwrite=False, savefileloc=None):
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
    if savefileloc is None:
        savefileloc=os.path.join(MonoData_savepath,id_dic[mission]+str(ID).zfill(11))
    if not os.path.isdir(savefileloc):
        os.mkdir(savefileloc)
    pickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"*"+suffix))
    if how == 'load' and len(pickles)>1:
        #finding most recent pickle:
        date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
        datepickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_*_"+suffix))
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
        datepickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_*_"+suffix))
        if len(datepickles)==0:
            nsim=0
        elif overwrite:
            nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
        else:
            #Finding next unused number with this date:
            nsim=1+np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])

    return [os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_"+str(int(nsim))+"_"+suffix), os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+'_'+suffix)]


def load_pickle(ID, mission,loadname=None,savefileloc=None):
    #Pickle file style: folder/TIC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
    if loadname is None:
        loadname=get_savename(ID, mission, how='load', suffix='mcmc.pickle', savefileloc=savefileloc)[0]
    if os.path.exists(loadname):
        n_bytes = 2**31
        max_bytes = 2**31 - 1

        ## read
        bytes_in = bytearray(0)
        input_size = os.path.getsize(loadname)
        with open(loadname, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        trace = pickle.loads(bytes_in)
        return trace
    else:
        return None

def save_pickle(trace,ID,mission,savename=None,overwrite=False,savefileloc=None):
    if savename is None:
        savename=get_savename(ID, mission, how='save', suffix='mcmc.pickle', overwrite=overwrite, savefileloc=savefileloc)[0]

    n_bytes = 2**31
    max_bytes = 2**31 - 1

    ## write
    bytes_out = pickle.dumps(trace)
    with open(savename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def get_lds(Ts,logg=4.43812,FeH=0.0,mission="TESS"):
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d

    if mission[0]=="T" or mission[0]=="t":
        import pandas as pd
        from astropy.io import ascii
        #
        from astroquery.vizier import Vizier
        #v = Vizier(catalog=[])
        Vizier.ROW_LIMIT = -1
        TessLDs=Vizier.get_catalogs('J/A+A/600/A30/tableab')[0].to_pandas()
        TessLDs=TessLDs.loc[(TessLDs['Type']=='r')&((TessLDs['Mod']=="PD")^(TessLDs['Teff']>3000))]
        #TessLDs=ascii.read(os.path.join(MonoData_tablepath,'tessLDs.txt')).to_pandas()
        a_interp=ct2d(np.column_stack((TessLDs['Teff'].values.astype(float),TessLDs['logg'].values.astype(float))),TessLDs['aLSM'].values.astype(float))
        b_interp=ct2d(np.column_stack((TessLDs['Teff'].values.astype(float),TessLDs['logg'].values.astype(float))),TessLDs['bLSM'].values.astype(float))

        if type(Ts) in [float,int,np.float64,np.int64]:
            Ts=np.array([Ts])
        if type(logg) is float:
            outarr=np.column_stack((np.array([a_interp(T,logg) for T in np.clip(Ts,2000,12000)]),
                                    np.array([b_interp(T,logg) for T in np.clip(Ts,2000,12000)])))
        else:
            outarr=np.column_stack((a_interp(np.clip(Ts,2000,12000),logg),b_interp(np.clip(Ts,2000,12000),logg)))
        return outarr
    elif mission[0]=="k" or mission[0]=="K":
        #Get Kepler Limb darkening coefficients.
        #print(label)
        types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
        how='2'#quadratic
        if how in types:
            checkint = types[how]
            #print(checkint)
        else:
            print("no key...")

        arr = np.genfromtxt(os.path.join(MonoData_tablepath,"KeplerLDlaws.txt"),skip_header=2)
        FeHarr=np.unique(arr[:, 2])
        FeH=find_nearest_2D(FeH,FeHarr)

        outarr=np.zeros((len(FeH),len(checkint)))
        for met in np.unique(FeH):
            #Selecting FeH manually:
            arr2=arr[arr[:,2]==met]
            for n,i in enumerate(checkint):
                ix_to_take=(FeH==met)*(Ts<50000.)*(Ts>=2000.)
                u_interp=ct2d(np.column_stack((arr2[:,0],arr2[:,1])),arr2[:,i])
                outarr[ix_to_take,n]=u_interp(np.clip(Ts[ix_to_take],3500,50000),np.clip(logg[ix_to_take],0,5))
        return outarr


def PlotCorner(trace, ID, mission='TESS', varnames=["b", "ecc", "period", "r_pl","u_star","vrel"],
               savename=None, overwrite=False,savefileloc=None,returnfig=False,tracemask=None):
    #Plotting corner of the parameters to see correlations
    import corner
    import matplotlib.pyplot as plt
    print("varnames = ",varnames)

    if savename is None:
        savename=get_savename(ID, mission, how='save', suffix='_corner.png',
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

def vals_to_latex(vals):
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

def to_latex_table(trace, ID, mission='TESS', varnames='all',order='columns',
               savename=None, overwrite=False, savefileloc=None, tracemask=None):
    #Plotting corner of the parameters to see correlations
    print("MakingLatexTable")
    if savename is None:
        savename=get_savename(ID, mission, how='save', suffix='_table.txt',overwrite=False, savefileloc=savefileloc)[0]
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    if varnames is None or varnames == 'all':
        varnames=[var for var in trace.varnames if var[-2:]!='__' and var not in ['gp_pred','light_curves']]

    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    samples = samples.loc[tracemask]
    facts={'r_pl':109.07637,'Ms':1.0,'rho':1.0,"t0":1.0,"period":1.0,"vrel":1.0,"tdur":24}
    units={'r_pl':"$ R_\\oplus $",'Ms':"$ M_\\odot $",'rho':"$ \\rho_\\odot $",
           "t0":"BJD-2458433","period":'d',"vrel":"$R_s/d$","tdur":"hours"}
    if order=="rows":
        #Table has header as a single row and data as a single row
        rowstring=str("ID")
        valstring=str(ID)
        for row in samples.columns:
            fact=[fact for fact in list(facts.keys()) if fact in row]
            if fact is not []:
                rowstring+=' & '+str(row)+' ['+units[fact[0]]+']'
                valstring+=' & '+vals_to_latex(np.percentile(facts[fact[0]]*samples[row],[16,50,84]))
            else:
                rowstring+=' & '+str(row)
                valstring+=' & '+vals_to_latex(np.percentile(samples[row],[16,50,84]))
        outstring=rowstring+"\n"+valstring
    else:
        #Table has header as a single column and data as a single column
        outstring="ID & "+str(ID)
        for row in samples.columns:
            fact=[fact for fact in list(facts.keys()) if fact in row]
            if len(fact)>0:
                outstring+="\n"+row+' ['+units[fact[0]]+']'+" & "+vals_to_latex(np.percentile(facts[fact[0]]*samples[row],[16,50,84]))
            else:
                outstring+="\n"+row+" & "+vals_to_latex(np.percentile(samples[row],[16,50,84]))
    with open(savename,'w') as file_to_write:
        file_to_write.write(outstring)
    #print("appending to file,",savename,"not yet supported")
    return outstring

def partition_list(a, k):
    """AI is creating summary for partition_list

    Args:
        a (list): Ordered list of lengths that we wish to evenly split into k pieces
        k (int): Number of parts along which to split a

    Returns:
        list: Ordered index of which of `k` bins the value in `a` belongs
    """
    if k <= 1: return np.tile(0,len(a))
    if k == len(a): return np.arange(k)
    assert k<len(a) #Cannot have more plot rows that data sectors...
    partition_between = [(i+1)*len(a) // k for i in range(k-1)]
    average_height = float(sum(a))/k
    best_score = None
    best_partitions = None
    count = 0

    while True:
        starts = [0] + partition_between
        ends = partition_between + [len(a)]
        partitions = [a[starts[i]:ends[i]] for i in range(k)]
        heights = list(map(sum, partitions))
        abs_height_diffs = list(map(lambda x: abs(average_height - x), heights))
        worst_partition_index = abs_height_diffs.index(max(abs_height_diffs))
        worst_height_diff = average_height - heights[worst_partition_index]

        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1

        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return np.hstack([np.tile(i,ends[i]-starts[i]) for i in range(len(starts))])
            #best_partitions
        count += 1

        move = -1 if worst_height_diff < 0 else 1
        bound_to_move = 0 if worst_partition_index == 0\
                        else k-2 if worst_partition_index == k-1\
                        else worst_partition_index-1 if (worst_height_diff < 0) ^ (heights[worst_partition_index-1] > heights[worst_partition_index+1])\
                        else worst_partition_index
        direction = -1 if bound_to_move < worst_partition_index else 1
        partition_between[bound_to_move] += move * direction

def kepler_spline(time, flux, flux_mask = None, transit_mask = None, bk_space=1.25, maxiter=5, outlier_cut=3, polydegree=3, reflect=False):
    """Computes a best-fit spline curve for a light curve segment.
    The spline is fit using an iterative process to remove outliers that may cause
    the spline to be "pulled" by discrepent points. In each iteration the spline
    is fit, and if there are any points where the absolute deviation from the
    median residual is at least 3*sigma (where sigma is a robust estimate of the
    standard deviation of the residuals), those points are removed and the spline
    is re-fit.
    Args:
        time: Numpy array; the time values of the light curve.
        flux: Numpy array; the flux (brightness) values of the light curve.
        flux_mask (np.ndarray of booleans, optional): Numpy array where False values refer to anomalies. Defaults to None
        transit_mask (np.ndarray of booleans, optional): Numpy array where False values refer to in-transit points
        bk_space: Spline break point spacing in time units.
        maxiter: Maximum number of attempts to fit the spline after removing badly
                fit points.
        outlier_cut: The maximum number of standard deviations from the median
                spline residual before a point is considered an outlier.
        polydegree: Polynomial degre. Defaults to 3
        reflect: Whether to perform spline fit using reflection of final time...
    Returns:
        spline: The values of the fitted spline corresponding to the input time
                values.
        mask: Boolean mask indicating the points used to fit the final spline.
    Raises:
        InsufficientPointsError: If there were insufficient points (after removing
                outliers) for spline fitting.
        SplineError: If the spline could not be fit, for example if the breakpoint
                spacing is too small.
    """
    region_starts=np.sort(time)[1+np.hstack((-1,np.where(np.diff(np.sort(time))>bk_space)[0]))]
    region_ends  =np.sort(time)[np.hstack((np.where(np.diff(np.sort(time))>bk_space)[0],len(time)-1))]
    spline = []
    mask = []
    for n in range(len(region_starts)):
        ix=(time>=region_starts[n])*(time<=region_ends[n])
        assert len(time[ix])>4

        # Rescale time into [0, 1].
        #t_min = np.min(time[ix])
        #t_max = np.max(time[ix])
        #n_interior_knots = int(np.round((t_max-t_min)/bk_space))
        #qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
        qs = np.arange(np.min(time[ix]), np.max(time[ix]), bk_space)[1:-1]
        qs = (qs-np.min(time[ix]))/(np.max(time[ix])-np.min(time[ix]))

        # Values of the best fitting spline evaluated at the time points.
        ispline = None

        # Mask indicating the points used to fit the spline.
        imask = np.ones_like(time[ix]).astype(bool)
        imask = imask*flux_mask[ix] if flux_mask is not None else imask
        imask = imask*transit_mask[ix] if transit_mask is not None else imask

        if reflect and (region_ends[n]-region_starts[n])>1.8*bk_space:
            incad=np.nanmedian(np.diff(time[ix]))
            xx=[np.arange(region_starts[n]-bk_space*0.9,region_starts[n]-incad,incad),
                np.arange(region_ends[n]+incad,region_ends[n]+bk_space*0.9,incad)]
            # Adding the lc, plus a reflected region either side of each part.
            # Also adding a boolean array to show where the reflected parts are
            # Also including zeros to make sure flux spline does not wander far from lightcurve
            itime=np.hstack((time[ix][0]-1.35*bk_space, time[ix][0]-1.3*bk_space, xx[0], time[ix], xx[1], time[ix][-1]+1.3*bk_space,time[ix][-1]+1.35*bk_space))
            imask=np.hstack((True, True, imask[:len(xx[0])][::-1], imask, imask[-1*len(xx[1]):][::-1], True, True))
            iflux=np.hstack((0.0,0.0,flux[ix][:len(xx[0])][::-1], flux[ix], flux[ix][-1*len(xx[1]):][::-1], 0.0, 0.0 ))
            ibool=np.hstack((np.zeros(len(xx[0])+2),np.ones(np.sum(ix)),np.zeros(len(xx[1])+2))).astype(bool)

        else:
            itime=time[ix]
            iflux=flux[ix]
            ibool=np.tile(True,len(itime))

        for ni in range(maxiter):
            if ispline is not None:
                # Choose points where the absolute deviation from the median residual is
                # less than outlier_cut*sigma, where sigma is a robust estimate of the
                # standard deviation of the residuals from the previous spline.
                residuals = iflux - ispline
                
                new_imask = robust_mean(residuals[imask], cut=outlier_cut)[2]
                # in ",ni,"th run")
                if np.all(new_imask):
                    break    # Spline converged.
                #Otherwise we're adding the updated mask to the mask
                imask[imask] = new_imask

            if np.sum(imask) > 4:
                # Fewer than 4 points after removing outliers. We could plausibly return
                # the spline from the previous iteration because it was fit with at least
                # 4 points. However, since the outliers were such a significant fraction
                # of the curve, the spline from the previous iteration is probably junk,
                # and we consider this a fatal error.
                try:
                    with warnings.catch_warnings():
                        # Suppress warning messages printed by pydlutils.bspline. Instead we
                        # catch any exception and raise a more informative error.
                        warnings.simplefilter("ignore")

                        # Fit the spline on non-outlier points.
                        #curve = BSpline.iterfit(time[mask], flux[mask], bkspace=bk_space)[0]
                        knots = np.quantile(itime[imask], qs)
                        #print(np.all(np.isfinite(flux[mask])),np.average(flux[mask]))
                        tck = splrep(itime[imask], iflux[imask], t=knots, k=polydegree)
                    ispline = splev(itime, tck)

                    # Evaluate spline at the time points.
                    #spline = curve.value(time)[0]

                    #spline = np.copy(flux)
                except (IndexError, TypeError, ValueError) as e:
                    raise ValueError(
                            "Fitting spline failed with error: '%s'. This might be caused by the "
                            "breakpoint spacing being too small, and/or there being insufficient "
                            "points to fit the spline in one of the intervals." % e)
            else:
                ispline=np.tile(np.nanmedian(iflux[imask]),len(iflux))
                break
        spline+=[ispline[ibool]]
        mask+=[imask[ibool]]


    return np.hstack(spline), np.hstack(mask)

def robust_mean(y, cut):
    """Computes a robust mean estimate in the presence of outliers.
    Args:
        y: 1D numpy array. Assumed to be normally distributed with outliers.
        cut: Points more than this number of standard deviations from the median are
                ignored.
    Returns:
        mean: A robust estimate of the mean of y.
        mean_stddev: The standard deviation of the mean.
        mask: Boolean array with the same length as y. Values corresponding to
                outliers in y are False. All other values are True.
    """
    # First, make a robust estimate of the standard deviation of y, assuming y is
    # normally distributed. The conversion factor of 1.4826 takes the median
    # absolute deviation to the standard deviation of a normal distribution.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    absdev = np.abs(y - np.median(y))
    sigma = 1.4826 * np.median(absdev)

    # If the previous estimate of the standard deviation using the median absolute
    # deviation is zero, fall back to a robust estimate using the mean absolute
    # deviation. This estimator has a different conversion factor of 1.253.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    if sigma < 1.0e-24:
        sigma = 1.253 * np.mean(absdev)

    # Identify outliers using our estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # of non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The
    # following formula is an approximation, see
    # http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Identify outliers using our second estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # with non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Final estimate is the sample mean with outliers removed.
    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask

def err_string_parse(s):
    for estr in ['e_','eneg_','epos_']:
        if len(s)>len(estr) and s[:len(estr)]==estr:
            return True, s[len(estr):]
    
    for estr in [' err','_err1','err1','err2','_err2',' Error']:
        if len(s)>len(estr) and s[-1*len(estr):]==estr:
            return True, s[:-1*len(estr)]
    return False, None

def make_bokeh_table(df, dftype='toi', cols2use=None, cols2avoid=None, errtype=' err', width=300, height=350):
    """Form Bokeh table from an input pandas dataframe

    Args:
        df ([type]): [description]
        dftype (str, optional): [description]. Defaults to 'toi'.
        cols2use ([type], optional): [description]. Defaults to None.
        cols2avoid ([type], optional): [description]. Defaults to None.
        width (int, optional): [description]. Defaults to 300.
        height (int, optional): [description]. Defaults to 350.

    Returns:
        [type]: [description]
    """
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

    if type(df)==pd.Series:
        df=pd.DataFrame(df).T
    if cols2use is None:
        if dftype is None or dftype=='':
            cols2use=df.columns
        elif dftype=='toi':
            cols2use=['TIC ID','TESS Disposition', 'TFOPWG Disposition', 'TESS Mag','RA', 'Dec',
                'Epoch (BJD)','Period (days)','Duration (hours)', 'Depth (mmag)','Planet Radius (R_Earth)','SNR_per_transit',
                'Stellar Eff Temp (K)', 'Stellar Radius (R_Sun)','Comments','Cheops_Observability','Cheops_Max_Efficiency',
                'Cheops_Obs_dates','Year2_obs_times', 'Year3_obs_times','Year4_obs_times', 'TESS_data', 'TESS_dvr']
        elif dftype=='tic':
            cols2use='ID, ra, dec, Tmag, plx, eclong, eclat, Bmag, Vmag, Jmag,  Kmag, GAIAmag, Teff, logg, MH, rad, mass, rho, d'.split(', ')
    if cols2avoid is None:
        if dftype is None or dftype=='':
            cols2avoid=[]
        elif dftype=='toi':
            cols2avoid=['SG1A','SG1B','SG2','SG3','SG4','SG5','ACWG ESM','ACWG TSM',
                                'Time Series Observations','Spectroscopy Observations','Imaging Observations',
                                'TESS Disposition','Master','Planet Insolation (Earth Flux)','Depth (mmag)',
                                'Planet Equil Temp (K)','Previous CTOI','PM RA (mas/yr)','PM Dec (mas/yr)']
        elif dftype=='tic':
            cols2avoid='pmRA, pmDEC, objType, typeSrc, version, HIP, TYC, UCAC, TWOMASS, SDSS, ALLWISE, GAIA, APASS, KIC, POSflag, PMflag, lumclass, lum, ebv, numcont, contratio, disposition, duplicate_id, priority, EBVflagTeffFlag, gaiabp, gaiarp, gaiaqflag, starchareFlag, VmagFlag, BmagFlag, splists, RA_orig, Dec_orig, raddflag, wdflag, dstArcSec'.split(', ')
    #Making Datatable inset:
    err_cols=[]
    nonerr_cols=[]
    #errless_cols=[]

    cols2use=[c for c in cols2use if not err_string_parse(c)[0]]
   
    df=df.rename(columns={col:col[:-1] for col in df.columns if col[-1]==' '}) #Removing trailing spaces

    #Creating error arrays
    for col in df.columns:
        if col in cols2use and col not in cols2avoid and 'e_'+col in df.columns:
            nonerr_cols+=[col]
            err_cols+=['e_'+col]
            #If we have multiple errors, we'll do a median to make sure we only end up with one:
            df['e_'+col]=np.nanmedian(np.vstack([abs(df[ecol].values.astype(float)) for ecol in ['e_'+col,'epos_'+col,'eneg_'+col,col+' err',col+'_err1',col+'_err2',col+' Error'] if ecol in df.columns]),axis=0)

        elif col in cols2use and col not in cols2avoid and type(df[col].values[0]) in [int,float,np.float64,np.int64,str] and 'e_'+col not in df.columns:
            nonerr_cols+=[col]
            err_cols+=['e_'+col]
            df['e_'+col]=np.tile(np.nan,df.shape[0])

    nonerr_cols=np.nan_to_num(np.array(nonerr_cols),0.0)
    err_cols=np.nan_to_num(np.array(err_cols),0.0)
    #errless_cols=np.nan_to_num(np.array(errless_cols),0.0)
    newdf=pd.DataFrame()
    newdf['col']=nonerr_cols
    columns=[TableColumn(field='col', title='Column')]
    for pl in range(df.shape[0]):
        if 'TOI' in df.columns:
            name = str(df.iloc[pl]['TOI'])+' '
        elif 'CTOI' in df.columns:
            name = str(df.iloc[pl]['CTOI'])+' '
        elif 'id' in df.columns:
            name = str(df.iloc[pl]['id'])+' '
        else:
            name = str(df.iloc[pl].name)+' '
        newdf[name+'Value']=[0.0 if df.iloc[pl][val] in [None,np.nan,-np.inf,np.inf,''] else df.iloc[pl][val] for val in nonerr_cols]
        newdf[name+'Errs']=[0.0 if df.iloc[pl][val] in [None,np.nan,-np.inf,np.inf,''] else df.iloc[pl][val] for val in err_cols]
        #newdf=newdf.fillna(0.0)
        columns+=[TableColumn(field=name+'Value', title=name+'Value')]
        columns+=[TableColumn(field=name+'Errs', title=name+'Errs')]
    #print(newdf)
    data_table = DataTable(source=ColumnDataSource(newdf), columns=columns, width=width, height=height)    
    return data_table

def gap_cull(t0,t,dat,std_thresh=10,boolean=None,time_jump_thresh=0.4):
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

def update_period_w_tls(time,flux,per,N_search_pers=200,per_range_search=0.01,oversampling=5):
    """
    Args:
        time (ndarray) - numpy array of times
        flux (ndarray) - numpy array of fluxes in ppm (normalised to zero)
        per (float) - period to test/update
        N_search_pers (int,optional) - Minimum number of periods to search. Must be greater than 50 or TLS fails
        per_range_search (float,optional) - Range, in days, to search for true period."""
    from transitleastsquares import transitleastsquares as tls
    tlsmodel=tls(t=np.sort(time), 
                 y=1+flux[np.argsort(time)]*1e-3)
    #Backed out the period timestep from TLS to ensure >100 points (otherwise it defaults to searching the whole period range)
    span=np.max(time)-np.min(time)
    deltaP=per**(4/3)/(13.23*oversampling*span)
    #print("TLS period update. period_min=",per-0.5*N_search_pers*deltaP,"period_max=",per+0.5*N_search_pers*deltaP)
    outtls=tlsmodel.power(period_min=per-0.5*np.max([N_search_pers*deltaP,per_range_search]),period_max=per+0.5*np.max([N_search_pers*deltaP,per_range_search]),
                          use_threads=4, duration_grid_step=1.2,
                           oversampling_factor=oversampling,transit_depth_min=100e-6)
    return outtls.period

def iteratively_determine_GP_params(pmmodel,time,flux,flux_err,tdurs,debug=False):
    """Iteratively determining best start parameter arrays for SHO GP kernel w0 and power."""
    lcrange=27
    av_dur = np.average(tdurs)
    exps=np.array([np.log((2*np.pi)/(av_dur)), np.log((2*np.pi)/(0.1*lcrange))])
    #Max power as half the 1->99th percentile in flux
    print(flux)
    maxpowers=0.5*np.ptp(np.nanpercentile(flux,[2,98]))
    logmaxpowers=np.log(0.5*np.ptp(np.nanpercentile(flux,[1,99])))
    #([np.nanstd(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values) for scope in self.lcs]))
    
    #Min power as 2x the average point-to-point displacement
    logminpowers=np.log(2*np.nanmedian(abs(np.diff(flux))))
    minpowers=0.5*np.nanmedian(abs(np.diff(flux)))
    span=abs(np.min(logmaxpowers)-np.max(logminpowers))

    import pymc_ext as pmx
    target=0.01
    success=np.array([False,False]);target=0.01
    try:
        while np.any(~success) and target<0.2:
            if not success[0]:
                try:
                    low=(2*np.pi)/(abs(np.random.normal(3*(0.03/target)**0.25,1)))#Targetting ~3 days at max (start 3.5d, end 2.0d)
                    up=np.clip((2*np.pi)/(av_dur*(0.1/target)),2*low,10000) #And ~2.5x average transit duration at min (start 8x, end 0.5x)
                    #itarg=abs(np.random.normal(target,0.5*target))
                    print(low,up,(2*np.pi)/av_dur,(2*np.pi)/(av_dur*(0.1/target)),2*low)
                    w0vals = pmx.utils.estimate_inverse_gamma_parameters(lower=low, upper=up, target=target)
                    success[0]=True
                except:
                    success[0]=False
                    
            if not success[1]:
                try:
                    sigmavals = pmx.utils.estimate_inverse_gamma_parameters(lower=abs(np.random.normal(np.min(minpowers),0.4*np.min(minpowers))), upper=np.min(maxpowers)*np.sqrt(target/0.1),
                                                                            target=target)
                    success[1]=True
                except:
                    success[1]=False
            target*=1.15
        assert np.all(success), "InverseGamma estimation of "+"&".join(list(np.array(["w0","sigma"])[~success]))+" failed"
        with pmmodel:
            w0=pm.InverseGamma("w0",**w0vals)
            sigma=pm.InverseGamma("sigma",**sigmavals)
        return w0, sigma
    except:
        with pmmodel:
            log_w0 = pm.TruncatedNormal("log_w0", mu=exps[1]+0.2*np.ptp(exps), sigma=0.01*np.ptp(exps), initval=exps[1]+0.15*np.ptp(exps),lower=exps[1],upper=exps[0])
            w0 = pm.Deterministic("w0", pm.math.exp(log_w0))
            log_sigma = pm.Normal("log_sigma", mu=(np.min(logmaxpowers)+np.max(logminpowers))/2, sigma=0.2*abs(np.min(logmaxpowers)-np.max(logminpowers)),initval=np.min(logmaxpowers)-0.1)
            sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        return w0, sigma

        # av_dur = np.nanmean(tdurs)
        # #freqs bounded from 2pi/10 to 2pi/2, but seauentially increasing these bounds until the estimate_inverse_gamma_parameters works
        # success=False
        # minw0=(2*np.pi)/10
        # maxw0=(2*np.pi)/1.5
        # while not success and maxw0<((2*np.pi)/(av_dur*0.33)):
        #     try:
        #         w0_alpha_beta = pmx.estimate_inverse_gamma_parameters(lower=minw0,upper=maxw0)
        #         w0_mode = (w0_alpha_beta['beta']/(1+w0_alpha_beta['alpha']))
        #         if debug: print(maxw0,(2*np.pi)/(av_dur*0.33),minw0,w0_alpha_beta,w0_mode)
        #         phot_w0 = pm.InverseGamma("phot_w0",testval=(2*np.pi)/10, **w0_alpha_beta)
        #         success=True
        #     except:
        #         minw0/=1.2
        #         maxw0*=1.1
        #         success=False
        # if debug: print(success, w0_mode,minw0,maxw0)

        # #We want the power to be somewhere between the point-to-point MAD, and the STD of the lightcurve binned at the w0 timescales:
        # power_est = np.nanstd(bin_lc_segment(np.column_stack((time,flux,flux_err)),binsize=2*np.pi/w0_mode)[:,1])
        # maxpower = power_est
        # minpower = np.nanmedian(abs(np.diff(flux)))
        # success=False
        # while not success and maxpower<5*power_est:
        #     try:
        #         if debug: print(maxpower,power_est,minpower)
        #         phot_sigma = pm.InverseGamma("phot_sigma",testval=minpower*5,
        #                                 **pmx.estimate_inverse_gamma_parameters(lower=minpower, upper=maxpower))
        #         success=True
        #     except:
        #         maxpower*=1.1
        #         minpower/=1.2
        #         success=False
        # if debug: print(success, minpower,maxpower)
    
