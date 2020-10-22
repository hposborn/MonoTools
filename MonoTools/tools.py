import numpy as np
import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.io import ascii
from scipy.signal import savgol_filter

import h5py

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

MonoData_tablepath = os.path.join(os.path.dirname(os.path.dirname( __file__ )),'data','tables')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname(os.path.dirname( __file__ )),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')

from . import starpars

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}

#goto='/Users/hosborn' if 'Users' in os.path.dirname(os.path.realpath(__file__)).split('/') else '/home/hosborn'


def openFits(f,fname,mission,cut_all_anom_lim=4.0,use_ppt=True,force_raw_flux=False,**kwargs):
    '''
    # opens and processes all lightcurve files (especially, but not only, fits files).
    # Processing involvesd iteratively masking anomlaous flux values
    '''
    #print(type(f),"opening ",fname,fname.find('everest')!=-1,f[1].data,f[0].header['TELESCOP']=='Kepler')
    mask=None
    end_of_orbit=False #Boolean as to whether we need to cut/fix the end-of-orbit flux
    
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
                    'raw_flux':f[1].data['fraw'],'bg_flux':f[1].data['BKG'],'qual':f[1].data['QUALITY']}
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
            print("TESS file")
            time = f[1].data['TIME']
            sap = f[1].data['SAP_FLUX']/np.nanmedian(f[1].data['SAP_FLUX'])
            pdcsap = f[1].data['PDCSAP_FLUX']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
            pdcsap_err = f[1].data['PDCSAP_FLUX_ERR']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
            lc={'time':time,'flux':pdcsap,'flux_err':pdcsap_err,'raw_flux':f[1].data['SAP_FLUX'],
                'bg_flux':f[1].data['SAP_BKG']}
            if ~np.isnan(np.nanmedian(f[1].data['PSF_CENTR2'])):
                lc['cent_1']=f[1].data['PSF_CENTR1'];lc['cent_2']=f[1].data['PSF_CENTR2']
            else:
                lc['cent_1']=f[1].data['MOM_CENTR1'];lc['cent_2']=f[1].data['MOM_CENTR2']
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
    elif type(f)==eleanor.targetdata.TargetData:
        #Eleanor TESS object
        lc={'time':f.time,'flux':f.corr_flux,'flux_err':f.flux_err,'raw_flux':f.raw_flux,
            'cent_1':f.centroid_xs,'cent_2':f.centroid_ys,'quality':f.quality,
            'flux_sm_ap':f.all_corr_flux[1],
            'flux_xl_ap':f.all_corr_flux[16]}
        #Fixing negative quality values as 2^15
        lc['quality'][lc['quality']<0.0]=np.power(2,15)
        lc['quality']=lc['quality'].astype(int)
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
        print('cannot identify fits type to identify with')
        #logging.debug('Found fits file but cannot identify fits type to identify with')
        return None
    
    if force_raw_flux and 'raw_flux' in lc:
        #Here we'll force ourselves to use raw flux, and not the detrended flux, if it exists:
        lc['detrended_flux']=lc.pop('flux')
        lc['flux']=lc['raw_flux'][:]
    
    lc['mask']=maskLc(lc,fname,cut_all_anom_lim=cut_all_anom_lim,use_ppt=use_ppt,end_of_orbit=end_of_orbit,input_mask=mask)
    
    #Including the cadence in the lightcurve as ["t2","t30","k1","k30"] mission letter + cadence
    lc['cadence']=np.tile(mission[0]+str(np.round(np.nanmedian(np.diff(lc['time']))*1440).astype(int)),len(lc['time']))
    
    # Only discard positive outliers
    
    print(np.sum(~lc['mask']),"points masked in lc of",len(lc['mask']))
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

    
def maskLc(lc,fhead,cut_all_anom_lim=5.0,use_ppt=False,end_of_orbit=True,
           use_binned=False,use_flat=False,mask_islands=True,input_mask=None):
    # Mask bad data (nans, infs and negatives) 
    
    prefix= 'bin_' if use_binned else ''
    suffix='_flat' if use_flat else ''
    
    mask = np.isfinite(lc[prefix+'flux'+suffix]) & np.isfinite(lc[prefix+'time']) & np.isfinite(lc[prefix+'flux_err'])
    if np.sum(mask)>0:
        # & (lc[prefix+'flux'+suffix]>0.0)
        print(np.sum(mask))
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
            #print(np.sum(~lc['mask']),"points before CutAnomDiff")
            mask[mask]=CutAnomDiff(lc[prefix+'flux'+suffix][mask],cut_all_anom_lim)
            #Doing this a second time with more stringent limits to cut two-point outliers:
            mask[mask]=CutAnomDiff(lc[prefix+'flux'+suffix][mask],cut_all_anom_lim+3.5)
            #print(np.sum(~lc['mask']),"after before CutAnomDiff")
        mu = np.median(lc[prefix+'flux'+suffix][mask])
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

def openPDC(epic,camp,use_ppt=True,**kwargs):
    if camp == '10':
    #https://archive.stsci.edu/missions/k2/lightcurves/c1/201500000/69000/ktwo201569901-c01_llc.fits
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c102_llc.fits'
    else:
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp).zfill(2)+'_llc.fits'
    if requests.get(urlfilename1, timeout=600).status_code==200:
        with fits.open(urlfilename1,show_progress=False) as hdus:
            lc=openFits(hdus,urlfilename1,mission='kepler',use_ppt=use_ppt,**kwargs)
            lc['src']['K2']='K2_pdc'
        return lc
    else:
        return None

def openVand(epic,camp,v=1,use_ppt=True,**kwargs):
    lcvand=[]
    #camp=camp.split(',')[0] if len(camp)>3
    if camp=='10' or camp==10 or camp=='10.0':
        camp='102'
    elif camp=='et' or camp=='E' or camp=='e':
        camp='e'
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
    else:
        camp=str(int(float(camp))).zfill(2)
    if camp in ['09','11']:
        #C91: https://archive.stsci.edu/missions/hlsp/k2sff/c91/226200000/35777/hlsp_k2sff_k2_lightcurve_226235777-c91_kepler_v1_llc.fits
        url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
        print("Vanderburg LC at ",url1)
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1,show_progress=False) as hdus:
                lcvand+=[openFits(hdus,url1,mission='k2',use_ppt=use_ppt,**kwargs)]
        url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1,show_progress=False) as hdus:
                lcvand+=[openFits(hdus,url2,mission='k2',use_ppt=use_ppt)]
    elif camp=='e':
        print("Engineering data")
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
        url='https://www.cfa.harvard.edu/~avanderb/k2/ep'+str(epic)+'alldiagnostics.csv'
        print("Vanderburg LC at ",url)
        df=pd.read_csv(url,index_col=False)
        lc={'time':df['BJD - 2454833'].values,
            'flux':df[' Corrected Flux'].values,
            'flux_err':np.tile(np.median(abs(np.diff(df[' Corrected Flux'].values))),df.shape[0])}
        lcvand+=[openFits(lc,url,mission='k2',use_ppt=use_ppt,**kwargs)]
    else:
        urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
        if requests.get(urlfitsname, timeout=600).status_code==200:
            with fits.open(urlfitsname,show_progress=False) as hdus:
                lcvand+=[openFits(hdus,urlfitsname,mission='k2',use_ppt=use_ppt,**kwargs)]
            print("Extracted vanderburg LC from ",urlfitsname)
        else:
            print("Cannot find vanderburg LC at ",urlfitsname)
    lc=lcStack(lcvand)
    lc['src']='K2_vand'
    return lc
 
def openEverest(epic,camp,pers=None,durs=None,t0s=None,use_ppt=True,**kwargs):
    import everest
    if camp in [10,11,10.0,11.0,'10','11','10.0','11.0']:
        camp=[int(str(int(float(camp)))+'1'),int(str(int(float(camp)))+'2')]
    else:
        camp=[int(float(camp))]
    lcs=[]
    lcev={}
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
                lcev={'time':np.vstack((lcev['time'],st1.time)),
                      'flux':np.vstack((lcev['flux'],st1.flux,st1.flux)),
                      'flux_err':np.vstack((lcev['flux_err'],st1.fraw_err,st1.fraw_err)),
                      'raw_flux':np.vstack((lcev['raw_flux'],st1.fraw,st1.fraw)),
                      'raw_flux_err':np.vstack((lcev['raw_flux_err'],st1.fraw_err,st1.fraw_err)),
                      'quality':np.vstack((lcev['quality'],st1.quality))}
            hdr={'cdpp':st1.cdpp,'ID':st1.ID,'Tmag':st1.mag,'mission':'K2','name':st1.name,'campaign':camp,'lcsource':'everest'}
            lcs+=[openFits(lcev,hdr,mission='k2',use_ppt=use_ppt,**kwargs)]
        except:
            print(c,"not possible to load")
            return None
        #elif int(camp)>=14:
        #    lcloc='https://archive.stsci.edu/hlsps/everest/v2/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_everest_k2_llc_'+str(epic)+'-c'+str(int(camp))+'_kepler_v2.0_lc.fits'
        #    lcev=openFits(fits.open(lcloc),lcloc)
    #print(np.unique(lcev['quality']))
    lc=lcStack(lcs)
    lc['src']='K2_ev'
    return lc
   

def getK2lc(epic,camp,saveloc=None,pers=None,durs=None,t0s=None,use_ppt=True):
    '''
    Gets (or tries to get) all LCs from K2 sources. Order is Everest > Vanderburg > PDC.
    '''
    from urllib.request import urlopen
    import everest
    lcs=[]
    lcs+=[openEverest(int(epic), camp, pers=pers, durs=durs, t0s=t0s, use_ppt=use_ppt)]
    lcs+=[openVand(int(epic), camp, use_ppt=use_ppt)]
    lcs=[lc for lc in lcs if lc is not None]
    if len(lcs)==0:
        try:
            return [openPDC(int(epic),int(camp),use_ppt=use_ppt)]
        except:
            print("No LCs for "+str(epic)+" campaign "+str(camp)+" at all")
            return None
    elif len(lcs)==1:
        return lcs[0]
    elif len(lcs)>1:
        stds = np.array([np.nanmedian(abs(np.diff(l['flux'][l['mask']]))) for l in lcs])
        
        if len(lcs[0]['time'])>1.5*len(lcs[1]['time']) or ((len(lcs[0]['time'])>0.66*len(lcs[1]['time']))&(stds[0]<stds[1])):
            return lcs[0]
        else:
            return lcs[1]

def K2_lc(epic,pers=None,durs=None,t0s=None, use_ppt=True):
    '''
    # Opens K2 lc
    '''
    df,_=starpars.GetExoFop(epic,"k2")
    lcs=[]
    print("K2 campaigns to search:",str(df['campaign']).split(','))
    for camp in str(df['campaign']).split(','):
        lcs+=[getK2lc(epic,camp,pers=pers,durs=durs,t0s=t0s, use_ppt=use_ppt)]
    lcs=lcStack(lcs)
    return lcs,df


def getKeplerLC(kic,cadence='long',use_ppt=True,**kwargs):
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
                    ilc=openFits(hdu,lcloc,mission='kepler',use_ppt=use_ppt,**kwargs)
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
                    ilc=openFits(hdu,lcloc,mission='kepler',use_ppt=use_ppt,**kwargs)
                    if ilc is not None:
                        lcs+=[ilc]
                    hdr=hdu[1].header
    if len(lcs)>0:
        lc=lcStack(lcs)
        return lc,hdr
    else:
        return None,None

def lcStack(lcs):
    #Stacks multiple lcs together
    outlc={}
    allkeys=np.unique(np.hstack([list(lcs[nlc].keys()) for nlc in range(len(lcs)) if lcs[nlc] is not None]))
    allkeys=allkeys[allkeys!='flux_format'] #This is the only non-timeseries keyword
    #Stacking each timeseries on top of each other
    for nlc in range(len(lcs)):
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

def observed(tic):
    # Using "webtess" page to check if TESS object was observed:
    # Returns dictionary of each sector and whether it was observed or not
    if type(tic)==int or type(tic)==float:
        page = requests.get('https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry='+str(int(tic)))
    elif type(tic)==SkyCoord:
        page = requests.get('https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry='+str(tic.ra.deg)+"%2C+"+str(tic.dec.deg))
    else:
        print(type(tic),"- unrecognised")
    print('https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry='+str(tic))
    tree = html.fromstring(page.content)
    Lamp = tree.xpath('//pre/text()') #stores class of //pre html element in list Lamp
    tab=tree.xpath('//pre/text()')[0].split('\n')[2:-1]
    out_dic={int(t[7:9]): True if t.split(':')[1][1]=='o' else False for t in tab}
    #print(out_dic)
    return out_dic

def getCorotLC(corid,use_ppt=True,**kwargs):
    #These are pre-computed CoRoT LCs I have lying around. There is no easy API as far as I can tell.
    lc=openFits(np.load('/'.join(MonoData_tablepath.split('/')[:-1]+["CorotLCs","CoRoT"+str(corid)+".npy"])),
                        "CorotLCs/CoRoT"+str(corid)+".npy",mission="Corot",use_ppt=use_ppt,**kwargs)
    
    #Some of these arrays have flux errors of zoer, so we need to nip that in the bud:
    lc['flux_err'][lc['flux_err']==0]=np.nanmedian(abs(np.diff(lc['flux'][lc['flux_err']==0])))
    lc['jd_base']=2451545
    return lc

def TESS_lc(tic, sectors='all',use_ppt=True, coords=None, use_eleanor=True, data_loc=None,**kwargs):
    #Downloading TESS lc     
    if data_loc is None:
        data_loc=MonoData_savepath+"/TIC"+str(int(tic)).zfill(11)
    
    epoch={1:'2018206045859_0120',2:'2018234235059_0121',3:'2018263035959_0123',4:'2018292075959_0124',
           5:'2018319095959_0125',6:'2018349182459_0126',7:'2019006130736_0131',8:'2019032160000_0136',
           9:'2019058134432_0139',10:'2019085135100_0140',11:'2019112060037_0143',12:'2019140104343_0144',
           13:'2019169103026_0146',14:'2019198215352_0150',15:'2019226182529_0151',16:'2019253231442_0152',
           17:'2019279210107_0161',18:'2019306063752_0162',19:'2019331140908_0164',20:'2019357164649_0165',
           21:'2020020091053_0167',22:'2020049080258_0174',23:'2020078014623_0177',24:'2020106103520_0180',
           25:'2020133194932_0182',26:'2020160202036_0188',27:'2020186164531_0189',28:'2020212050318_0190'}
    lcs=[];lchdrs=[]
    if sectors == 'all':
        if coords is not None and type(coords)==SkyCoord:
            sect_obs=observed(coords)
        else:
            sect_obs=observed(tic)
        epochs=[key for key in epoch if sect_obs[key]]
        
        if epochs==[]:
            #NO EPOCHS OBSERVABLE APPARENTLY. USING THE EPOCHS ON EXOFOP/TIC8
            toi_df=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
            if tic in toi_df['TIC ID'].values:
                print("FOUND TIC IN TOI LIST")
                epochs=list(np.array(toi_df.loc[toi_df['TIC ID']==tic,'Sectors'].values[0].split(',')).astype(int))
    elif type(sectors)==list or type(sectors)==np.ndarray:
        epochs=sectors
    else:
        epochs=[sectors]
        #observed_sectors=observed(tic)
        #observed_sectors=np.array([os for os in observed_sectors if observed_sectors[os]])
        #if observed_sectors!=[-1] and len(observed_sectors)>0:
        #    observed_sectors=observed_sectors[np.in1d(observed_sectors,np.array(list(epoch.keys())))]
        #else:
        #    observed_sectors=sector
        #print(observed_sectors)
    get_qlp=0
    print(epochs,type(epochs))
    for key in epochs:
        try:
            #2=minute cadence data from tess website
            fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(key).zfill(4)+"/"+str(tic).zfill(16)[:4]+"/"+str(tic).zfill(16)[4:8]+"/"+str(tic).zfill(16)[-8:-4]+"/"+str(tic).zfill(16)[-4:]+"/tess"+epoch[key].split('_')[0]+"-s"+str(key).zfill(4)+"-"+str(tic).zfill(16)+"-"+epoch[key].split('_')[1]+"-s_lc.fits"
            h = httplib2.Http()
            resp = h.request(fitsloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(fitsloc,show_progress=False) as hdus:
                    lcs+=[openFits(hdus,fitsloc,mission='tess',use_ppt=use_ppt,**kwargs)]
                    lchdrs+=[hdus[0].header]
            else:
                raise Exception('No TESS lightcurve')
        except:
            if os.path.isdir(data_loc) and len(glob.glob(data_loc+"/*.h5"))>0:
                get_qlp+=1
            elif use_eleanor:
                print("No QLP files at",data_loc,"Loading Eleanor Lightcurve")
                try:
                    #Getting eleanor lightcurve:
                    try:
                        star = eleanor.eleanor.Source(tic=tic, sector=key)
                    except:
                        star = eleanor.eleanor.Source(coords=coords, sector=key)
                    try:
                        elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True,save_postcard=False)
                    except:
                        try:
                            elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=False,save_postcard=False)
                        except:
                            elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False,save_postcard=False)
                    elen_hdr={'ID':star.tic,'GaiaID':star.gaia,'Tmag':star.tess_mag,
                              'RA':star.coords[0],'dec':star.coords[1],'mission':'TESS','campaign':key,'source':'eleanor',
                              'ap_masks':elen_obj.all_apertures,'ap_image':np.nanmedian(elen_obj.tpf[50:80],axis=0)}
                    elen_lc=openFits(elen_obj,elen_hdr,mission='tess',use_ppt=use_ppt,**kwargs)
                    lcs+=[elen_lc]
                    lchdrs+=[elen_hdr]
                except Exception as e:
                    print(e, tic,"not observed by TESS in sector",key)

    #Acessing QLP data from local files - only happens if there's .h5 lightcurves in a TICXXXXXXX folder in the folder where this is run
    if get_qlp>0:
        print("# Loading QLP lightcurves")
        print(tic,type(tic))
        for orbit in glob.glob(data_loc+"/*.h5"):
            f=h5py.File(orbit)
            if len(lcs)==0 or np.nanmin(abs(np.nanmedian(f['LightCurve']['BJD'])-np.hstack([l['time'] for l in lcs])))>5:
                # This specific QLP orbit does not have a SPOC lightcurve attached (i.e. no other obs within 5days)
                lcs+=[openFits(f,orbit,mission='tess',use_ppt=use_ppt,**kwargs)]
                lchdrs+=[{'source':'qlp'}]
    if len(lcs)>1:
        lc=lcStack(lcs)
        return lc,lchdrs[0]
    elif len(lcs)==1:
        #print(lcs,lchdrs)
        return lcs[0],lchdrs[0]
    else:
        return None,None

def openLightCurve(ID,mission,coor=None,use_ppt=True,other_data=True,
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
            tess_id=tess_id.iloc[np.argmin(tess_id['Tmag'])] if type(tess_id)==pd.DataFrame else tess_id
            if tess_id is not None:
                IDs['tess']=tess_id['ID']
            else:
                IDs['tess']=None
        #else:
        #    tess_id, tess_dat, sects = tess_stars2px_mod.SectFromCoords(coor,tic=ID)
        if mission.lower()!='kepler':
            v = Vizier(catalog=['V/133/kic'])
            res=v.query_region(coor, radius=5*units.arcsec, catalog=['V/133/kic'])
            if 'V/133/kic' in res and len(res['V/133/kic'])>0:
                IDs['kepler'] = res['V/133/kic']['KIC'][0]
            else:
                IDs['kepler'] = None
    
    #Opening using url search:
    lcs={};hdrs={}
    if IDs['tess'] is not None:
        lcs['tess'],hdrs['tess'] = TESS_lc(IDs['tess'], use_ppt=use_ppt, coords=coor, **kwargs)
        if lcs['tess'] is not None:
            lcs['tess']['time']-=(jd_base-2457000)
    if IDs['k2'] is not None:
        lcs['k2'],hdrs['k2'] = K2_lc(IDs['k2'],pers=kwargs.get('periods',None),
                                     durs=kwargs.get('initdurs',None),
                                     t0s=kwargs.get('initt0',None),
                                     use_ppt=use_ppt)
        if lcs['k2'] is not None:
            lcs['k2']['time']-=(jd_base-2454833)
    if IDs['kepler'] is not None:
        lcs['kepler'],hdrs['kepler'] = getKeplerLC(IDs['kepler'],use_ppt=use_ppt)
        if lcs['kepler'] is not None:
            lcs['kepler']['time']-=(jd_base-2454833)
    if mission.lower() == 'corot':
        lcs['corot'] = getCorotLC(ID,use_ppt=use_ppt)
        lcs['corot']['time']-=(jd_base-lcs['corot']['jd_base'])
        lcs['corot']['jd_base']=jd_base
        hdrs['corot'] = None
    #print(IDs,lcs)
    if len(lcs.keys())>1:
        lc=lcStack([lcs[lc] for lc in lcs if lcs[lc] is not None])
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
        pickle.dump(lc,open(MonoData_savepath+'/'+ID_string+'/'+ID_string+'_lc.pickle','wb'))
    
    return lc,hdrs[mission.lower()]

def LoadLc(lcid,mission='tess',file_loc=None):
    # Quick tool to load pickled lightcurve dict.
    # lcid = ID
    # mission = 'tess'. mission string (TESS, K2, Kepler, etc)
    # file_loc = None ; loction of pickle. If None defaults to $MONOTOOLSPATH
    ID_string=id_dic[mission]+str(lcid).zfill(11)
    file_loc=MonoData_savepath+'/'+ID_string if file_loc is not None else file_loc
    return pickle.load(open(MonoData_savepath+'/'+ID_string+'/'+ID_string+'_lc.pickle','rb'))

def cutLc(lctimes,max_len=10000,return_bool=True,transit_mask=None):
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

def weighted_avg_and_std(values, errs, axis=None): 
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
    else:
        return [values[0], errs[0]]

def lcBin(lc,binsize=1/48,split_gap_size=0.8,use_flat=True,use_masked=True, extramask=None,modify_lc=True):
    #Binning lightcurve to e.g. 30-min cadence for planet search
    # Can optionally use the flatted lightcurve
    binlc={}
        
    #Using flattened lightcurve as well as normal one:
    if use_flat and 'flux_flat' not in lc:
        lc=lcFlatten(lc)
    if use_flat:
        flux_dic=['flux_flat','flux'] 
        binlc['flux_flat']=[]
        binlc['flux']=[]
    else:
        flux_dic=['flux']
        binlc['flux']=[]
    binlc['bin_cadence']=[]
        
    if np.nanmax(np.diff(lc['time']))>split_gap_size:
        loop_blocks=np.array_split(np.arange(len(lc['time'])),np.where(np.diff(lc['time'])>2.0)[0])
    else:
        loop_blocks=[np.arange(len(lc['time']))]
    if extramask is not None and type(extramask)==np.ndarray and (type(extramask[0])==bool)|(type(extramask[0])==np.bool_):
        mask=lc['mask']&extramask
        print("extramask with n=",np.sum(extramask))
    else:
        mask=lc['mask']
    for sh_time in loop_blocks:
        nodata=False
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
        if binsize>(1.66*np.nanmedian(np.diff(lc['time'][sh_time]))):
            binlc['bin_cadence']+=[np.array([cads[digi==d] if type(cads[digi==d])==str else cads[digi==d][0] for d in np.unique(digi)])[:,np.newaxis]]
        else:
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

def bin_lc_segment(lc_segment, binsize,return_digi=False):
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

def lcFlatten(lc, winsize = 3.5, stepsize = 0.15, polydegree = 2, 
              niter = 10, sigmaclip = 3., gapthreshold = 1.0,
              use_binned=False, use_mask=True, reflect=True, transit_mask=None):
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
            print("transit mask:",type(initmask),len(initmask),
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
            #print(partlc.shape,len(xx[0]),len(xx[1]),refl_t.shape,refl_flux.shape,refl_bool.shape)
            uselc_w_reflect+=[np.column_stack((refl_t,refl_flux,refl_bool))]
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
        win,box = formwindow(uselc,stepcent,winsize,stepsize,gapthreshold)  #should return window around box not including box
        newbox=box[uselc[:,4].astype(bool)] # Excluding from our box any points which are actually part of the "reflection"
        #Checking that we have points in the box where the window is not entirely junk/masked
        if np.sum(newbox)>0 and np.sum(win&uselc[:,3].astype(bool))>0:
            #Forming the polynomial fit from the window around the box:
            baseline = dopolyfit(uselc[win,:3],mask=uselc[win,3].astype(bool),
                                 stepcent=stepcent,d=polydegree,ni=niter,sigclip=sigmaclip)
            lc[prefix+'flux_flat'][newbox] = lc[prefix+'flux'][newbox] - np.polyval(baseline,lc[prefix+'time'][newbox]-stepcent)*lc['flux_unit']
            #Here we have 
        
    return lc
    


def init_model(lc, initdepth, initt0, Rstar, rhostar, Teff, logg=np.array([4.3,1.0,1.0]),initdur=None, 
               periods=None,assume_circ=False,
               use_GP=True,constrain_LD=True,ld_mult=3,useL2=True,
               mission='TESS',FeH=0.0,LoadFromFile=False,cutDistance=0.0,
               debug=True):
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
    
    lcmask = np.tile(True,len(x)) if 'mask' not in lc else lc['mask']
    
    x,y,yerr = lc['time'],lc['flux'],lc['flux_err']
    
    n_pl=len(initt0)
    
    print("Teff:",Teff)
    start=None
    with pm.Model() as model:
        
        # We're gonna need a bounded normal:
        #BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)

        #Stellar parameters (although these aren't really fitted.)
        #Using log rho because otherwise the distribution is not normal:
        if len(rhostar)==3:
            logrho_S = pm.Normal("logrho_S", mu=np.log(rhostar[0]), sd=np.average(abs(rhostar[1:]/rhostar[0])),testval=np.log(rhostar[0]))
        else:
            logrho_S = pm.Normal("logrho_S", mu=np.log(rhostar[0]), sd=rhostar[1]/rhostar[0],testval=np.log(rhostar[0]))

        rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S))
        if len(Rstar)==3:
            Rs = pm.Normal("Rs", mu=Rstar[0], sd=np.average(Rstar[1:]),testval=Rstar[0],shape=1)
        else:
            Rs = pm.Normal("Rs", mu=Rstar[0], sd=Rstar[1],testval=Rstar[0],shape=1)
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
        
        # The time of a reference transit for each planet
        # If multiple times are given that means multiple transits *without* a set period - a "complex" t0
        is_complex_t0=np.array([type(it0)==list for it0 in initt0])
        print("Checking if there's complex t0s suggesting two transits with gap:",initt0,is_complex_t0)
        if np.all(~is_complex_t0):
            #Normal transit fit - no complex gaps
            t0 = pm.Normal("t0", mu=initt0, sd=1.0, shape=n_pl, testval=initt0)
        else:
            # In cases where we have two transit with a gap between, we will have two t0s for a single planet (and N periods)
           
            mus=np.array([initt0[nt0][1] for nt0 in range(len(initt0)) if is_complex_t0[nt0]])
            sds=np.tile(1.0,np.sum(is_complex_t0))
            #print(mus,mus.shape,type(mus),sds,sds.shape,type(sds))
            t0_second_trans = pm.Normal("t0_second_trans",
                                        mu=mus, 
                                        sd=sds,
                                        shape=np.sum(is_complex_t0),
                                        testval=mus)

            if np.all(is_complex_t0)>1:
                t0 = pm.Normal("t0", mu=[t0s[0] for t0s in initt0], sd=1.0,
                               shape=n_pl,testval=[t0s[0] for t0s in initt0])
            else:
                mus=np.array([initt0[nt0] for nt0 in range(len(initt0)) if not is_complex_t0[nt0]]+[initt0[nt0][0] for nt0 in range(len(initt0)) if is_complex_t0[nt0]])
                sds=np.tile(1.0,len(is_complex_t0))
                #print(mus,mus.shape,type(mus),sds,sds.shape,type(sds))
                t0 = pm.Normal("t0",
                               mu=mus,
                               sd=sds,
                               shape=n_pl,
                               testval=mus)
            for nt in range(len(initt0)):
                if is_complex_t0[nt] and (periods[nt]==0.0)|(periods[nt] is None)|(np.isnan(periods[nt])):
                    #need to make periods[nt] into a list of possible periods.
                    inputdur=0.5 if initdur is None or np.isnan(initdur[nt]) or initdur[nt]==0.0 else initdur[nt]
                    max_P=initt0[nt][1]-initt0[nt][0]
                    folded_x=np.sort(abs((x[lcmask]-initt0[nt][0]-max_P*0.5)%max_P-max_P*0.5))
                    min_P=np.where(np.diff(folded_x)>inputdur)[0]
                    min_P=min_P[0] if len(min_P)>1 else min_P
                    min_P=np.max(folded_x) if len(min_P)==0 else min_P
                    #print(min_P,max_P)
                    periods[nt]=list(max_P/np.arange(1,max_P/min_P,1))
                    #Periods is then from max_P down to the limit where max_P/N<min_P, i.e. floor(max_P/min_P)
                    
        #Calculating minimum period:
        P_gap_cuts=[];pertestval=[]
        #print(initt0)
        for n,nt0 in enumerate(initt0):
            if not is_complex_t0[n]:
                #Looping over all t0s - i.e. all planets
                if periods is None or np.isnan(periods[n]) or periods[n]==0.0:
                    #computing minimum periods and gaps for each planet, given the LC
                    dist_from_t0=np.sort(abs(nt0-x[lcmask]))
                    inputdur=0.5 if initdur is None or np.isnan(initdur[n]) or initdur[n]==0.0 else initdur[n]
                    #print(x[lcmask],nt0,inputdur)
                    P_gap_cuts+=[PeriodGaps(x[lcmask],nt0,inputdur)]
                    #Estimating init P using duration:
                    initvrel=(2*(1+np.sqrt(initdepth[n]))*np.sqrt(1-(0.41/(1+np.sqrt(initdepth[n])))**2))/inputdur
                    initper=18226*(rhostar[0]/1.408)/(initvrel**3)
                    #print(initper,P_gap_cuts[n])
                    if initper>P_gap_cuts[n][0]:
                        pertestval+=[initper/P_gap_cuts[n][0]]
                    else:
                        pertestval+=[0.5]
                else:
                    P_gap_cuts+=[[0.75*periods[n]]]
                    pertestval+=[periods[n]/P_gap_cuts[n][0]]
        
        #Cutting points for speed of computation:
        if not np.any(is_complex_t0):
            speedmask=np.tile(False, len(x))
            for n,it0 in enumerate(initt0):
                if not is_complex_t0[n]:
                    if periods is not None and not np.isnan(periods[n]) and not periods[n]==0.0:
                        #For known periodic planets, need to keep many transits, so masking in the period space:
                        speedmask[(((x-it0)%periods[n])<cutDistance)|(((x-it0)%periods[n])>(periods[n]-cutDistance))]=True
                    elif cutDistance>0.0:
                        speedmask[abs(x-it0)<cutDistance]=True
                    else:
                        #No parts of the lc to cut
                        speedmask=np.tile(True,len(x))
            print(np.sum(~speedmask),"points cut from lightcurve leaving",np.sum(speedmask),"to process")
            totalmask=speedmask*lcmask
        else:
            #Using all points in the 
            totalmask=lcmask[:]
        
        P_min=np.array([P_gap_cuts[n][0] for n in range(len(P_gap_cuts)) if not is_complex_t0[n]])
        pertestval=np.array(pertestval)
        print("Using minimum period(s) of:",P_min)
        
        #From Dan Foreman-Mackey's thing:
        period = pm.Pareto("period", m=min_period, alpha=2./3)

        #P_index = pm.Bound("P_index", upper=1.0, lower=0.0)("P_index", mu=0.5, sd=0.33, shape=n_pl)
        
        if np.any(is_complex_t0):
            #Let's do the complex period using "Categorical" discrete distribution (we give an array where all possible periods must sum to 1)
            #Computing priors as n**-(8/3) and making input array from 0 to 25 populated by the derived periods
            #print([periods[npl] for npl in range(len(initt0)) if is_complex_t0[npl]])
            complex_pers=pm.Deterministic("complex_pers", tt.as_tensor_variable([periods[npl] for npl in range(len(initt0)) if is_complex_t0[npl]]))
        logp = pm.Deterministic("logp", tt.log(period))

        # The Espinoza (2018) parameterization for the joint radius ratio and
        # impact parameter distribution
        if useL2:
            #EB case as second light needed:
            RpRs, b = xo.distributions.get_joint_radius_impact(
                min_radius=0.001, max_radius=1.25,
                testval_r=np.array(initdepth)**0.5,
                testval_b=np.random.rand(n_pl)
            )
        else:
            RpRs, b = xo.distributions.get_joint_radius_impact(
                min_radius=0.001, max_radius=0.25,
                testval_r=np.array(initdepth)**0.5,
                testval_b=np.random.rand(n_pl)
            )

        r_pl = pm.Deterministic("r_pl", RpRs * Rs)
        
        #Initialising Limb Darkening:
        if constrain_LD:
            n_samples=1200
            # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.
            ld_dists=getLDs(np.random.normal(Teff[0],Teff[1],n_samples),
                            np.random.normal(logg[0],logg[1],n_samples),FeH,mission=mission)
            print("contrain LDs - ",Teff[0],Teff[1],logg[0],logg[1],FeH,n_samples,
                  np.clip(np.nanmedian(ld_dists,axis=0),0,1),np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0))
            u_star = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star", 
                                        mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                        sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
        else:
            # The Kipping (2013) parameterization for quadratic limb darkening paramters
            u_star = xo.distributions.QuadLimbDark("u_star", testval=np.array([0.3, 0.2]))
        
        #Initialising GP kernel (i.e. without initialising Potential)
        if use_GP:
            # Transit jitter & GP parameters
            #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
            cads=np.unique(lc['cadence'])
            #print(cads)
            #Transit jitter will change if observed by Kepler/K2/TESS - need multiple logs^2
            if len(cads)==1:
                logs2 = pm.Uniform("logs2", upper=np.log(np.std(y[totalmask]))+4,
                                   lower=np.log(np.std(y[totalmask]))-4)
                min_cad=np.nanmedian(np.diff(x))#Limiting to <1 cadence
            else:
                logs2 = pm.Uniform("logs2", upper=[np.log(np.std(y[totalmask&(lc['cads']==c)]))+4 for c in cads],
                                   lower=[np.log(np.std(y[totalmask&(lc['cads']==c)]))-4 for c in cads], shape=len(cads))
                min_cad=np.min([np.nanmedian(np.diff(x[lc['cads']==c])) for c in cads])
            
            logw0_guess = np.log(2*np.pi/10)
            
            lcrange=x[totalmask][-1]-x[totalmask][0]
            
            #freqs bounded from 2pi/minimum_cadence to to 2pi/(4x lc length)
            logw0 = pm.Uniform("logw0",lower=np.log((2*np.pi)/(4*lcrange)), 
                               upper=np.log((2*np.pi)/min_cad))

            # S_0 directly because this removes some of the degeneracies between
            # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
            logpower = pm.Uniform("logpower",lower=-20,upper=np.log(np.nanmedian(abs(np.diff(y[totalmask])))))
            logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
        
        if not assume_circ:
            # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
            BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
            ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, shape=n_pl,
                              testval=np.tile(0.1,n_pl))
            omega = xo.distributions.Angle("omega", shape=n_pl, testval=np.tile(0.1,n_pl))

        # Complex T requires special treatment of orbit (i.e. period) and potential:
        if np.any(is_complex_t0):
            #Marginalising over each possible period
            if np.sum(is_complex_t0)==1:
                #Single planet with two transits and a gap
                logprobs = []
                all_lcs = []
                for i, index in enumerate([periods[np] for np in range(len(periods)) if is_complex_t0[np]]):
                    with pm.Model(name="per_{0}".format(i), model=model) as submodel:
                        if not np.all(is_complex_t0):
                            #Have other planets:
                            p2use=tt.concatenate(period,complex_pers[0][i])
                        else:
                            p2use=complex_pers[0][i]
                        
                        # Set up a Keplerian orbit for the planets
                        if assume_circ:
                            orbit = xo.orbits.KeplerianOrbit(
                                r_star=Rs, rho_star=rho_S,
                                period=p2use, t0=t0, b=b)
                        else:
                            orbit = xo.orbits.KeplerianOrbit(
                                r_star=Rs, rho_star=rho_S,
                                ecc=ecc, omega=omega,
                                period=p2use, t0=t0, b=b)
                        print("orbit set up")
                        vx, vy, vz = orbit.get_relative_velocity(t0)
                        #vsky = 
                        if n_pl>1:
                            vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
                        else:
                            vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)
                        tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)
                        
                        #print(x[totalmask])
                        if debug:
                            tt.printing.Print('u_star')(u_star)
                            tt.printing.Print('r_pl')(r_pl)
                            tt.printing.Print('mult')(mult)
                            tt.printing.Print('tdur')(tdur)
                            tt.printing.Print('t0')(t0)
                            tt.printing.Print('b')(b)
                            tt.printing.Print('p2use')(p2use)
                            tt.printing.Print('rho_S')(rho_S)
                            tt.printing.Print('Rs')(Rs)
                        # Compute the model light curve using starry
                        light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,
                                                                                     t=x[totalmask])*1e3/mult
                        
                        light_curve = pm.math.sum(light_curves, axis=-1) + mean     
                        all_lcs.append(light_curve)

                        if use_GP:
                            if len(cads)==1:
                                #Observed by a single telescope
                                gp = xo.gp.GP(kernel, x[totalmask], tt.exp(logs2) + tt.zeros(np.sum(totalmask)), J=2)

                                loglike = tt.sum(gp.log_likelihood(y[totalmask] - light_curve))
                                gp_pred = pm.Deterministic("gp_pred", gp.predict())
                            else:
                                #We have multiple logs2 terms due to multiple telescopes:
                                for n in range(len(cads)):
                                    gp_i += [xo.gp.GP(kernel, x[totalmask&(lc['cads']==cads[n])], tt.exp(logs2[n]) + tt.zeros(np.sum(totalmask&(lc['cads']==cads[n]))), J=2)]
                                    llk_gp_i += [gp.log_likelihood(y[totalmask&(lc['cads']==cads[n])] - light_curve)]
                                    gp_pred_i += [gp.predict()]

                                loglike = tt.sum(tt.stack(llk_gp_i))
                                gp_pred = pm.Deterministic("gp_pred", tt.stack(gp_pred_i))

                            #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                            #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                            #llk = pm.Deterministic("llk", model.logpt)
                        else:
                            loglike = tt.sum(pm.Normal.dist(mu=light_curve, sd=yerr[totalmask]).logp(y[totalmask]))
                        
                        logprior = tt.log(orbit.dcosidb) - 2 * tt.log(complex_pers[0][i])
                        logprobs.append(loglike + logprior)
                        
            elif np.sum(is_complex_t0)==2:
                #Two planets with two transits...
                for i_1, index in enumerate(periods[is_complex_t0][0]):
                    for i_2, index in enumerate(periods[is_complex_t0][1]):
                        with pm.Model(name="per_{0}_{1}".format(i_1,i_2), model=model) as submodel:
                            if not np.all(is_complex_t0):
                                #Have other planets:
                                p2use=tt.concatenate(period,tt.as_tensor_variable([complex_pers[0][i_1],complex_pers[1][i_2]]))
                            else:
                                p2use=tt.as_tensor_variable([complex_pers[0][i_1],complex_pers[1][i_2]])
                            if debug:
                                tt.printing.Print('p2use')(p2use)
                            # Set up a Keplerian orbit for the planets
                            if assume_circ:
                                orbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    period=p2use, t0=t0, b=b)
                            else:
                                orbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    ecc=ecc, omega=omega,
                                    period=p2use, t0=t0, b=b)
                            vx, vy, vz = orbit.get_relative_velocity(t0)
                            #vsky = 
                            if n_pl>1:
                                vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
                            else:
                                vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)
                            tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)

                            # Compute the model light curve using starry
                            light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,
                                                                                         t=x[totalmask])*1e3/mult
                            light_curve = pm.math.sum(light_curves, axis=-1) + mean     
                            all_lcs.append(light_curve)

                            if use_GP:
                                if len(cads)==1:
                                    #Observed by a single telescope
                                    gp = xo.gp.GP(kernel, x[totalmask], tt.exp(logs2) + tt.zeros(np.sum(totalmask)), J=2)

                                    loglike = tt.sum(gp.log_likelihood(y[totalmask] - light_curve))
                                    gp_pred = pm.Deterministic("gp_pred", gp.predict())
                                else:
                                    #We have multiple logs2 terms due to multiple telescopes:
                                    for n in range(len(cads)):
                                        gp_i += [xo.gp.GP(kernel, x[totalmask&(lc['cads']==cads[n])], tt.exp(logs2[n]) + tt.zeros(np.sum(totalmask&(lc['cads']==cads[n]))), J=2)]
                                        llk_gp_i += [gp.log_likelihood(y[totalmask&(lc['cads']==cads[n])] - light_curve)]
                                        gp_pred_i += [gp.predict()]

                                    loglike = tt.sum(tt.stack(llk_gp_i))
                                    gp_pred = pm.Deterministic("gp_pred", tt.stack(gp_pred_i))

                                #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                                #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                                #llk = pm.Deterministic("llk", model.logpt)
                            else:
                                loglike = tt.sum(pm.Normal.dist(mu=light_curve, sd=yerr[totalmask]).logp(y[totalmask]))

                            logprior = tt.sum(tt.log(orbit.dcosidb)) -\
                                       2 * tt.log(periods[is_complex_t0][i_1]) -\
                                       2 * tt.log(periods[is_complex_t0][i_1])
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
            if assume_circ:
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=Rs, rho_star=rho_S,
                    period=period, t0=t0, b=b)
            else:
                # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, shape=n_pl,
                                  testval=np.tile(0.1,n_pl))
                omega = xo.distributions.Angle("omega", shape=n_pl, testval=np.tile(0.1,n_pl))
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
            
            light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,t=x[totalmask])*1e3/mult
            light_curve = pm.math.sum(light_curves, axis=-1)
            pm.Deterministic("light_curves", light_curves)
            # Compute the model light curve using starry
            if use_GP:
                if len(cads)==1:
                    gp = xo.gp.GP(kernel, x[totalmask], tt.exp(logs2) + tt.zeros(np.sum(totalmask)), J=2)

                    llk_gp = pm.Potential("transit_obs", gp.log_likelihood(y[totalmask] - light_curve))
                    gp_pred = pm.Deterministic("gp_pred", gp.predict())
                else:
                    #We have multiple logs2 terms due to multiple telescopes:
                    gp_i = []
                    llk_gp_i = []
                    gp_pred_i = []
                    for n in range(len(cads)):
                        gp_i += [xo.gp.GP(kernel, x[totalmask&(lc['cads']==cads[n])], tt.exp(logs2[n]) + tt.zeros(np.sum(totalmask&(lc['cads']==cads[n]))), J=2)]
                        llk_gp_i += [gp.log_likelihood(y[totalmask&(lc['cads']==cads[n])] - light_curve)]
                        gp_pred_i += [gp.predict()]

                    llk_gp = pm.Potential("transit_obs", tt.stack(llk_gp_i))
                    gp_pred = pm.Deterministic("gp_pred", tt.stack(gp_pred_i))

                #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
                #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
                #llk = pm.Deterministic("llk", model.logpt)
            else:
                pm.Normal("obs", mu=light_curve, sd=yerr[totalmask], observed=y[totalmask])

        if debug:
            tt.printing.Print('Rs')(Rs)
            tt.printing.Print('RpRs')(RpRs)
            tt.printing.Print('u_star')(u_star)
            tt.printing.Print('r_pl')(r_pl)
            tt.printing.Print('t0')(t0)
        
        #print(P_min,t0,type(x[totalmask]),x[totalmask][:10],np.nanmedian(np.diff(x[totalmask])))
        
        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        #print(model.test_point)
        if not LoadFromFile:
            if debug:
                print("before",model.check_test_point())
            map_soln = xo.optimize(start=start, vars=[RpRs, b])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[period, t0])
            map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower])
            map_soln = xo.optimize(start=map_soln, vars=[logw0])
            map_soln = xo.optimize(start=map_soln)
            if debug:
                print("after",model.check_test_point())
            
            return model, map_soln, totalmask, P_gap_cuts
        else:
            return model, None, totalmask, P_gap_cuts

        # This shouldn't make a huge difference, but I like to put a uniform
        # prior on the *log* of the radius ratio instead of the value. This
        # can be implemented by adding a custom "potential" (log probability).
        #pm.Potential("r_prior", -pm.math.log(r))


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
    #print(savenames)
    
    if os.path.exists(savenames[1].replace('_mcmc.pickle','.lc')) and os.path.exists(savenames[1].replace('_mcmc.pickle','_hdr.pickle')) and not overwrite:
        print("loading from",savenames[1].replace('_mcmc.pickle','.lc'))
        #Loading lc from file
        df=pd.read_csv(savenames[1].replace('_mcmc.pickle','.lc'))
        lc={col.replace('# ',''):df[col].values for col in df.columns}
        hdr=pickle.load(open(savenames[1].replace('_mcmc.pickle','_hdr.pickle'),'rb'))
    else:
        lc,hdr = openLightCurve(ID,mission,**kwargs)
        #print([len(lc[key]) for key in list(lc.keys())])
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
    #print(lc['time'],type(lc['time']),type(lc['time'][0]))
    model, soln, lcmask, P_gap_cuts = init_model(lc,initdepth, initt0, Rstar, rhostar, Teff,
                                                 logg=logg, **kwargs)
    #initdur=None,n_pl=1,periods=None,per_index=-8/3,
    #assume_circ=False,use_GP=True,constrain_LD=True,ld_mult=1.5,
    #mission='TESS',LoadFromFile=LoadFromFile,cutDistance=cutDistance)
    print("Model loaded")


    #try:
    if LoadFromFile and not overwrite:
        trace = LoadPickle(ID, mission, savenames[0])
    else:
        trace=None
   
    if trace is None:
        #Running sampler:
        np.random.seed(int(ID))
        with model:
            #print(type(soln))
            #print(soln.keys())
            trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=soln, chains=4,
                                  step=xo.get_dense_nuts_step(target_accept=0.9),compute_convergence_checks=False)

        SavePickle(trace, ID, mission, savenames[0])
            
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

def GetSavename(ID, mission, how='load', suffix='mcmc.pickle', overwrite=False, savefileloc=None):
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
                                
    
def LoadPickle(ID, mission,loadname=None,savefileloc=None):
    #Pickle file style: folder/TIC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
    if loadname is None:
        loadname=GetSavename(ID, mission, how='load', suffix='mcmc.pickle', savefileloc=savefileloc)[0]
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

def SavePickle(trace,ID,mission,savename=None,overwrite=False,savefileloc=None):
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='mcmc.pickle', overwrite=overwrite, savefileloc=savefileloc)[0]
        
    n_bytes = 2**31
    max_bytes = 2**31 - 1

    ## write
    bytes_out = pickle.dumps(trace)
    with open(savename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def getLDs(Ts,logg=4.43812,FeH=0.0,mission="TESS"):
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d

    if mission[0]=="T" or mission[0]=="t":
        import pandas as pd
        from astropy.io import ascii
        TessLDs=ascii.read(os.path.join(MonoData_tablepath,'tessLDs.txt')).to_pandas()
        TessLDs=TessLDs.rename(columns={'col1':'logg','col2':'Teff','col3':'FeH','col4':'L/HP','col5':'a',
                                           'col6':'b','col7':'mu','col8':'chi2','col9':'Mod','col10':'scope'})
        a_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.a.values.astype(float))
        b_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.b.values.astype(float))

        if (type(Ts)==float) or (type(Ts)==int):
            Ts=np.array([Ts])
        if type(logg) is float:
            outarr=np.column_stack((np.array([a_interp(T,logg) for T in np.clip(Ts,2300,12000)]),
                                    np.array([b_interp(T,logg) for T in np.clip(Ts,2300,12000)])))
        else:
            outarr=np.column_stack((a_interp(np.clip(Ts,2300,12000),logg),b_interp(np.clip(Ts,2300,12000),logg)))
        return outarr
    elif mission[0]=="k" or mission[0]=="K": 
        #Get Kepler Limb darkening coefficients.
        #print(label)
        types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
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
    
def ToLatexTable(trace, ID, mission='TESS', varnames='all',order='columns',
               savename=None, overwrite=False, savefileloc=None, tracemask=None):
    #Plotting corner of the parameters to see correlations
    print("MakingLatexTable")
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_table.txt',overwrite=False, savefileloc=savefileloc)[0]
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
