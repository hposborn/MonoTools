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

MonoTools_path = os.path.dirname(os.path.abspath( __file__ ))
from .stellar import starpars

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}

#goto='/Users/hosborn' if 'Users' in os.path.dirname(os.path.realpath(__file__)).split('/') else '/home/hosborn'

def K2_lc(epic,pers=None,durs=None,t0s=None, use_ppt=True):
    '''
    # Opens K2 lc
    '''
    df,_=starpars.GetExoFop(epic,"k2")
    lcs=[]
    print("K2 campaigns to search:",df['campaign'])
    for camp in str(df['campaign']).split(','):
        lcs+=[getK2lc(epic,int(float(camp)),pers=pers,durs=durs,t0s=t0s, use_ppt=use_ppt)]
        
    lcs=lcStack(lcs)
    return lcs,df

def getK2lc(epic,camp,saveloc=None,pers=None,durs=None,t0s=None,use_ppt=True):
    '''
    Gets (or tries to get) all LCs from K2 sources. Order is Everest > Vanderburg > PDC.
    '''
    from urllib.request import urlopen
    import everest
    try:
        lc=openEverest(epic,camp,pers=pers,durs=durs,t0s=t0s,use_ppt=use_ppt)
    except:
        print("No everest")
        try:
            lc=openVand(epic,camp,use_ppt=use_ppt)
        except:
            print("No vand")
            try:
                lc=openPDC(epic,camp,use_ppt=use_ppt)
            except:
                print("No LCs at all")
    return lc


def openFits(f,fname,mission,cut_all_anom_lim=4.0,use_ppt=True):
    '''
    # opens and processes all lightcurve files (especially, but not only, fits files).
    # Processing involvesd iteratively masking anomlaous flux values
    '''
    #print(type(f),"opening ",fname,fname.find('everest')!=-1,f[1].data,f[0].header['TELESCOP']=='Kepler')
    
    end_of_orbit=False #Boolean as to whether we need to cut/fix the end-of-orbit flux
    
    if type(f)==fits.hdu.hdulist.HDUList or type(f)==fits.fitsrec.FITS_rec:
        if f[0].header['TELESCOP']=='Kepler' or fname.find('kepler')!=-1:
            if fname.find('k2sff')!=-1:
                lc={'time':f[1].data['T'],'flux':f[1].data['FCOR'],
                    'flux_err':np.tile(np.median(abs(np.diff(f[1].data['FCOR']))),len(f[1].data['T'])),
                    'flux_raw':f[1].data['FRAW'],
                    'bg_flux':f[1+np.argmax([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['flux_raw']}
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
    elif type(f)==dict:
        lc=f
    else:
        print('cannot identify fits type to identify with')
        #logging.debug('Found fits file but cannot identify fits type to identify with')
        return None
    
    # Mask bad data (nans, infs and negatives) 
    lc['mask'] = np.isfinite(lc['flux']) & np.isfinite(lc['time']) & (lc['flux']>0.0) 
    # Mask data if it's 4.2-sigma from its points either side (repeating at 7-sigma to get any points missed)
    #print(np.sum(~lc['mask']),"points before quality flags")
    if 'quality' in lc:
        qs=[1,2,3,4,6,7,8,9,13,15,16,17]#worst flags to cut - for the moment just using those in the archive_manual
        if type(fname)==dict and 'lcsource' in fname.keys() and fname['lcsource']=='everest':
            qs+=[23]
            print("EVEREST file with ",np.log(np.max(lc['quality']))/np.log(2)," max quality")
        lc['mask']=lc['mask']&(np.sum(np.vstack([lc['quality'] & 2 ** (q - 1) for q in qs]),axis=0)==0)
    #print(np.sum(~lc['mask']),"points after quality flags")
    if cut_all_anom_lim>0:
        #print(np.sum(~lc['mask']),"points before CutAnomDiff")
        lc['mask'][lc['mask']]=CutAnomDiff(lc['flux'][lc['mask']],cut_all_anom_lim)
        #print(np.sum(~lc['mask']),"after before CutAnomDiff")
    mu = np.median(lc['flux'][lc['mask']])
    if use_ppt:
        # Convert to parts per thousand
        lc['flux'] = (lc['flux'] / mu - 1) * 1e3
        lc['flux_err'] *= 1e3/mu
    else:
        lc['flux'] = (lc['flux'] / mu - 1)
        lc['flux_err'] /= mu
    
    #End-of-orbit cut
    # Arbritrarily looking at the first/last 15 points and calculating STD of first/last 300 pts.
    # We will cut the first/last points if the lightcurve STD is drastically better without them
    if end_of_orbit:
        stds=np.array([np.nanstd(lc['flux'][lc['mask']][n:(300+n)]) for n in np.arange(0,17)])
        stds/=np.min(stds)
        newmask=np.tile(True,np.sum(lc['mask']))
        for n in np.arange(15):
            if stds[n]>1.05*stds[-1]:
                newmask[n]=False
                newmask[n+1]=False
        stds=np.array([np.nanstd(lc['flux'][lc['mask']][(-300+n):n]) for n in np.arange(-17,0)])
        stds/=np.min(stds)
        for n in np.arange(-15,0):
            if stds[n]>1.05*stds[0]:
                newmask[n]=False
                newmask[n-1]=False
        lc['mask'][lc['mask']]=newmask
    
    # Identify outliers
    m2 = lc['mask']
    
    for i in range(10):
        try:
            y_prime = np.interp(lc['time'], lc['time'][m2], lc['flux'][m2])
            smooth = savgol_filter(y_prime, 101, polyorder=3)
            resid = lc['flux'] - smooth
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
            resid = np.zeros(len(lc['flux']))
            sigma = 1.0
            pass
    
    #Including the cadence in the lightcurve as ["t2","t30","k1","k30"] mission letter + cadence
    lc['cadence']=np.tile(mission[0]+str(np.round(np.nanmedian(np.diff(lc['time']))*1440).astype(int)),len(lc['time']))
    
    # Only discard positive outliers
    lc['mask']*=m2
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
    lc['flux_unit']=0.001 if use_ppt else 1.0
    return lc

def openPDC(epic,camp,use_ppt=True):
    if camp == '10':
    #https://archive.stsci.edu/missions/k2/lightcurves/c1/201500000/69000/ktwo201569901-c01_llc.fits
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c102_llc.fits'
    else:
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp).zfill(2)+'_llc.fits'
    if requests.get(urlfilename1, timeout=600).status_code==200:
        with fits.open(urlfilename1) as hdus:
            lc=openFits(hdus,urlfilename1,mission='kepler',use_ppt=use_ppt)
        return lc
    else:
        return None

def openVand(epic,camp,v=1,use_ppt=True):
    lcvand=[]
    #camp=camp.split(',')[0] if len(camp)>3
    if camp=='10':
        camp='102'
    elif camp=='et' or camp=='E':
        camp='e'
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
    else:
        camp=str(int(camp)).zfill(2)
    if camp in ['09','11']:
        #C91: https://archive.stsci.edu/missions/hlsp/k2sff/c91/226200000/35777/hlsp_k2sff_k2_lightcurve_226235777-c91_kepler_v1_llc.fits
        url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
        print("Vanderburg LC at ",url1)
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1) as hdus:
                lcvand+=[openFits(hdus,url1,mission='k2',use_ppt=use_ppt)]
        url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1) as hdus:
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
        lcvand+=[openFits(lc,url,mission='k2',use_ppt=use_ppt)]
    else:
        urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
        print("Vanderburg LC at ",urlfitsname)
        if requests.get(urlfitsname, timeout=600).status_code==200:
            with fits.open(urlfitsname) as hdus:
                lcvand+=[openFits(hdus,urlfitsname,mission='k2',use_ppt=use_ppt)]
    return lcStack(lcvand)
 
def openEverest(epic,camp,pers=None,durs=None,t0s=None,use_ppt=True):
    import everest
    if camp in ['10','11']:
        camp=[camp+'1',camp+'2']
    else:
        camp=[int(camp)]
    
    lcev={}
    for c in camp:
        st1=everest.Everest(int(epic),season=c)
        
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
    
        #elif int(camp)>=14:
        #    lcloc='https://archive.stsci.edu/hlsps/everest/v2/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_everest_k2_llc_'+str(epic)+'-c'+str(int(camp))+'_kepler_v2.0_lc.fits'
        #    lcev=openFits(fits.open(lcloc),lcloc)
    #print(np.unique(lcev['quality']))
    hdr={'cdpp':st1.cdpp,'ID':st1.ID,'Tmag':st1.mag,'mission':'K2','name':st1.name,'campaign':camp,'lcsource':'everest'}
    return openFits(lcev,hdr,mission='k2',use_ppt=use_ppt)
   
def getKeplerLC(kic,cadence='long',use_ppt=True):
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
    if cadence=='long' and 'llc' in q:
        for q in [qc for qc in qcodes if qc[-4]=='_llc']:
            lcloc='http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/kplr'+str(int(kic)).zfill(9)+'-'+str(q)+'.fits'
            h = httplib2.Http()
            resp = h.request(lcloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(lcloc) as hdu:
                    ilc=openFits(hdu,lcloc,mission='kepler',use_ppt=use_ppt)
                    if ilc is not None:
                        lcs+=[ilc]
                    hdr=hdu[1].header
    elif cadence is 'short' and 'slc' in q:
        for q in [qc for qc in qcodes if qc[-4]=='_slc']:
            lcloc='http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/kplr'+str(int(kic)).zfill(9)+'-'+str(q)+'.fits'
            h = httplib2.Http()
            resp = h.request(lcloc, 'HEAD')
            if int(resp[0]['status']) < 400:
                with fits.open(lcloc) as hdu:
                    ilc=openFits(hdu,lcloc,mission='kepler',use_ppt=use_ppt)
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
    allkeys=np.unique(np.hstack([list(lcs[nlc].keys()) for nlc in range(len(lcs))]))
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
    page = requests.get('https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry='+str(tic))
    print('https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry='+str(tic))
    tree = html.fromstring(page.content)
    Lamp = tree.xpath('//pre/text()') #stores class of //pre html element in list Lamp
    tab=tree.xpath('//pre/text()')[0].split('\n')[2:-1]
    out_dic={int(t[7:9]): True if t.split(':')[1][1]=='o' else False for t in tab}
    #print(out_dic)
    return out_dic

def getCorotLC(corid,use_ppt=True):
    lc=openFits(np.load("data/CorotLCs/CoRoT"+str(corid)+".npy"),"CorotLCs/CoRoT"+str(corid)+".npy",mission="Corot")
    return lc

def TESS_lc(tic,sector='all',use_ppt=True, coords=None, use_eleanor=True):
    #Downloading TESS lc 
    if sector is 'all':
        sect_obs=observed(tic)
    else:
        sect_obs={sector:True}
    
    epoch={1:'2018206045859_0120',2:'2018234235059_0121',3:'2018263035959_0123',4:'2018292075959_0124',
           5:'2018319095959_0125',6:'2018349182459_0126',7:'2019006130736_0131',8:'2019032160000_0136',
           9:'2019058134432_0139',10:'2019085135100_0140',11:'2019112060037_0143',12:'2019140104343_0144',
           13:'2019169103026_0146',14:'2019198215352_0150',15:'2019226182529_0151',16:'2019253231442_0152',
           17:'2019279210107_0161',18:'2019306063752_0162',19:'2019331140908_0164',20:'2019357164649_0165',
           21:'2020020091053_0167',22:'2020049080258_0174'}
    lcs=[];lchdrs=[]
    if type(sector)==str and sector=='all':
        epochs=list(epoch.keys())
    else:
        epochs=[sector]
        #observed_sectors=observed(tic)
        #observed_sectors=np.array([os for os in observed_sectors if observed_sectors[os]])
        #if observed_sectors!=[-1] and len(observed_sectors)>0:
        #    observed_sectors=observed_sectors[np.in1d(observed_sectors,np.array(list(epoch.keys())))]
        #else:
        #    observed_sectors=sector
        #print(observed_sectors)
    for key in epochs:
        if sect_obs[key]:
            try:
                #2=minute cadence data from tess website
                fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(key).zfill(4)+"/"+str(tic).zfill(16)[:4]+"/"+str(tic).zfill(16)[4:8]+"/"+str(tic).zfill(16)[-8:-4]+"/"+str(tic).zfill(16)[-4:]+"/tess"+epoch[key].split('_')[0]+"-s"+str(key).zfill(4)+"-"+str(tic).zfill(16)+"-"+epoch[key].split('_')[1]+"-s_lc.fits"
                h = httplib2.Http()
                resp = h.request(fitsloc, 'HEAD')
                if int(resp[0]['status']) < 400:
                    with fits.open(fitsloc) as hdus:
                        lcs+=[openFits(hdus,fitsloc,mission='tess',use_ppt=use_ppt)]
                        lchdrs+=[hdus[0].header]
                else:
                    raise Exception('No TESS lightcurve')
            except:
                if use_eleanor:
                    try:
                        #Getting eleanor lightcurve:
                        try:
                            star = eleanor.eleanor.Source(tic=tic, sector=key)
                        except:
                            star = eleanor.eleanor.Source(coords=coords, sector=key)
                        try:
                            elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True)
                        except:
                            try:
                                elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=False)
                            except:
                                elen_obj=eleanor.eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False)
                        elen_hdr={'ID':star.tic,'GaiaID':star.gaia,'Tmag':star.tess_mag,
                                  'RA':star.coords[0],'dec':star.coords[1],'mission':'TESS','campaign':key,
                                  'ap_masks':elen_obj.all_apertures,'ap_image':np.nanmedian(elen_obj.tpf[50:80],axis=0)}
                        elen_lc=openFits(elen_obj,elen_hdr,mission='tess',use_ppt=use_ppt)
                        lcs+=[elen_lc]
                        lchdrs+=[elen_hdr]
                    except Exception as e:
                        print(e, tic,"not observed by TESS in sector",key)
    if len(lcs)>1:
        lc=lcStack(lcs)
        return lc,lchdrs[0]
    elif len(lcs)==1:
        #print(lcs,lchdrs)
        return lcs[0],lchdrs[0]
    else:
        return None,None

def openLightCurve(ID,mission,use_ppt=True,other_data=True,jd_base=2457000,**kwargs):
    #Doing this to get coordinates:
    df,_=starpars.GetExoFop(ID,mission)
    print(df)
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
        if mission.lower() is not 'k2':
            v = Vizier(catalog=['J/ApJS/224/2'])
            res = v.query_region(coor, radius=5*units.arcsec, catalog=['J/ApJS/224/2'])
            if len(res)>0 and len(res[0])>0:
                IDs['k2']=res[0]['EPIC'][0]
            else:
                IDs['k2']=None
        if mission.lower() is not 'tess':
            # Let's search for associated TESS lightcurve:    
            tess_id = Catalogs.query_criteria("TIC",coordinates=coor,radius=12*units.arcsec,
                                              objType="STAR",columns=['ID','KIC','Tmag']).to_pandas()
            #print(tess_id)
            if tess_id.shape[1]>1:
                IDs['tess']=tess_id.iloc[np.argmin(tess_id['Tmag'])]['ID']
            elif tess_id.shape[1]==1:
                IDs['tess']=tess_id['ID'].values[0]
            else:
                IDs['tess']=None
        if mission.lower() is not 'kepler':
            v = Vizier(catalog=['V/133/kic'])
            res=v.query_region(coor, radius=5*units.arcsec, catalog=['V/133/kic'])
            if 'V/133/kic' in res and len(res['V/133/kic'])>0:
                IDs['kepler'] = res['V/133/kic']['KIC'][0]
            else:
                IDs['kepler'] = None
    
    #Opening using url search:
    lcs={};hdrs={}
    if IDs['tess'] is not None:
        lcs['tess'],hdrs['tess'] = TESS_lc(IDs['tess'],use_ppt=use_ppt,coords=coor)
        lcs['tess']['time']-=(jd_base-2457000)
    if IDs['k2'] is not None:
        lcs['k2'],hdrs['k2'] = K2_lc(IDs['k2'],pers=kwargs.get('periods',None),
                                     durs=kwargs.get('initdurs',None),
                                     t0s=kwargs.get('initt0',None),
                                     use_ppt=use_ppt)
        lcs['k2']['time']-=(jd_base-2454833)
    if IDs['kepler'] is not None:
        lcs['kepler'],hdrs['kepler'] = getKeplerLC(IDs['kepler'],use_ppt=use_ppt)
        lcs['kepler']['time']-=(jd_base-2454833)
    if mission.lower() is 'corot':
        lcs['corot'] = getCorotLC(ID,use_ppt=use_ppt)
        lcs['corot']['time']-=(jd_base-2454833)
        hdrs['corot'] = None
    #print(IDs,lcs)
    if len(lcs.keys())>1:
        lc=lcStack([lcs[lc] for lc in lcs if lcs[lc] is not None])
    elif not other_data:
        lc=lcs[mission.lower()]
    else:
        lc=lcs[list(lcs.keys())[0]]
    lc['jd_base']=jd_base
    
    #Maing sure lightcurve is sorted by time, and that there are no nans in the time array:
    for key in lc:
        if key!='time' and type(lc[key])==np.ndarray and len(lc[key])==len(lc['time']):
            lc[key]=lc[key][~np.isnan(lc['time'])]
            lc[key]=lc[key][:][np.argsort(lc['time'][~np.isnan(lc['time'])])]
    lc['time']=np.sort(lc['time'][~np.isnan(lc['time'])])
    
    return lc,hdrs[mission.lower()]

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
        savefileloc=os.path.join(NamastePymc3_path,'data',id_dic[mission]+str(ID).zfill(11))
    if not os.path.isdir(savefileloc):
        os.mkdir(savefileloc)
    pickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"*"+suffix))
    if how is 'load' and len(pickles)>1:
        #finding most recent pickle:
        date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
        datepickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_*_"+suffix))
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
        TessLDs=ascii.read(os.path.join(NamastePymc3_path,'data','tables','tessLDs.txt')).to_pandas()
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

        arr = np.genfromtxt(os.path.join(NamastePymc3_path,"data","KeplerLDlaws.txt"),skip_header=2)
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
    if varnames is None or varnames is 'all':
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

def PlotLC_interactive(lc, trace, ID):
    from bokeh.plotting import figure, output_file, save
    
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_TransitFit.png', 
                             overwrite=overwrite, savefileloc=savefileloc)[0]
        print(savename)
    
    output_file(savename)
    
    #Initialising figure:
    p = figure(plot_width=1000, plot_height=600,title=str(ID)+" Transit Fit")
    
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
        
    if lcmask is None:
        assert len(lc['time'])==len(trace['gp_pred'][0,:])
        lcmask=np.tile(True,len(lc['time']))
    else:
        assert len(lc['time'][lcmask])==len(trace['gp_pred'][0,:])
    
    #Finding if there's a single enormous gap in the lightcurve:
    x_gap=np.max(np.diff(lc['time'][lcmask]))>10
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
    gp_mod = np.median(trace["gp_pred"][tracemask,:] + trace["mean"][tracemask, None], axis=0)
    
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
        gap_pos=np.average(lc['time'][np.argmax(np.diff(lc['time'])):(1+np.argmax(np.diff(lc['time'])))])
        before_gap_lc,before_gap_gp=(lc['time']<gap_pos)&lcmask,(lc['time'][lcmask]<gap_pos)
        after_gap_lc,after_gap_gp=(lc['time']>gap_pos)&lcmask,(lc['time'][lcmask]>gap_pos)
        
        print(np.sum(before_gap_lc),len(lc['time'][before_gap_lc]),np.sum(before_gap_gp),len(gp_mod[before_gap_gp]))
        
        f_all_1.circle(lc['time'][before_gap_lc], lc['flux'][before_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.circle(lc['time'][after_gap_lc], lc['flux'][after_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        f_all_1.line(lc['time'][before_gap_lc], gp_mod[before_gap_gp]+2.5*min_trans, color="C3", label="GP fit")
        f_all_2.line(lc['time'][after_gap_lc], gp_mod[after_gap_gp]+2.5*min_trans, color="C3", label="GP fit")

        f_all_1.circle(lc['time'][before_gap_lc], lc['flux'][before_gap_lc] - gp_mod[before_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.circle(lc['time'][after_gap_lc], lc['flux'][after_gap_lc] - gp_mod[after_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid_1.circle(lc['time'][before_gap_lc], 
                         lc['flux'][before_gap_lc] - gp_mod[before_gap_gp] - np.sum(pred[1,before_gap_gp,:],axis=1), 
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_1.set_xlabel('Time (BJD-245700)')
        f_all_resid_1.set_xlim(lc['time'][before_gap_lc][0],lc['time'][before_gap_lc][-1])

        f_all_resid_2.circle(lc['time'][after_gap_lc], 
                         lc['flux'][after_gap_lc] - gp_mod[after_gap_gp] - np.sum(pred[1,after_gap_gp,:],axis=1),
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_2.set_xlabel('Time (BJD-245700)')
        f_all_resid_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
        #print(len(lc[:,0]),len(lc[lcmask,0]),len(gp_mod))
        f_all_resid_1.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid_2.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        for n_pl in range(len(pred[1,0,:])):
            f_all_1.plot(lc['time'][before_gap_lc], pred[1,before_gap_gp,n_pl], color="C1", label="model")
            
            art = f_all_1.patch(np.append(lc['time'][before_gap_lc], lc['time'][before_gap_lc][::-1]),
                                np.append(pred[0,before_gap_gp,n_pl], pred[2,before_gap_gp,n_pl][::-1]),
                                       color="C1", alpha=0.5, zorder=1000)
            f_all_1.set_xlim(lc['time'][before_gap_lc][0],lc['time'][before_gap_lc][-1])

            f_all_2.line(lc['time'][after_gap_lc], pred[1,after_gap_gp,n_pl], color="C1", label="model")
            art = f_all_2.patch(np.append(lc['time'][after_gap_lc], lc['time'][after_gap_lc][::-1]),
                                np.append(pred[0,after_gap_gp,n_pl], pred[2,after_gap_gp,n_pl][::-1]),
                                color="C1", alpha=0.5, zorder=1000)
            f_all_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
            
            f_all_1.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
            f_all_2.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
        
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
        print(len(lc['time']),len(lc['flux']),len(lc['time'][lcmask]),len(lc['flux'][lcmask]),len(gp_mod),len(np.sum(pred[1,:,:],axis=1)))
        f_all.circle(lc['time'][lcmask], lc['flux'][lcmask]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all.line(lc['time'][lcmask], gp_mod+2.5*min_trans, color="C3", label="GP fit")

        # Plot the data
        f_all.plot(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid.circle(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid.set_xlabel('Time (BJD-245700)')
        f_all_resid.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        for n_pl in range(len(pred[1,0,:])):
            f_all.line(lc['time'][lcmask], pred[1,:,n_pl], color="C1", label="model")
            art = f_all.patch(np.append(lc['time'][lcmask],lc['time'][lcmask][::-1]),
                              np.append(pred[0,:,n_pl], pred[2,:,n_pl][::-1]),
                              color="C1", alpha=0.5, zorder=1000)
            f_all.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        f_all.set_xticks([])
        f_zoom=figure(width=250, plot_height=400, title=None)
        f_zoom_resid=figure(width=250, plot_height=150, title=None)
    
    min_trans=0;min_resid=0
    for n_pl in range(len(pred[1,0,:])):
        # Get the posterior median orbital parameters
        p = np.median(trace["period"][tracemask,n_pl])
        t0 = np.median(trace["t0"][tracemask,n_pl])
        tdur = np.nanmedian(trace['tdur'][tracemask,n_pl])
        #tdurs+=[(2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2))/np.nanmedian(trace['vrel'][tracemask,n_pl])]
        #print(min_trans,tdurs[n_pl],2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2),np.nanmedian(trace['vrel'][tracemask, n_pl]))
        
        phase=(lc['time'][lcmask]-t0+p*0.5)%p-p*0.5
        zoom_ind=abs(phase)<tdur
        
        resids=lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind] - np.sum(pred[1,zoom_ind,:],axis=1)

        if zoom_plot_time:
            #Plotting time:
            f_zoom.plot(phase[zoom_ind], min_trans+lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
            f_zoom.plot(phase[zoom_ind], min_trans+pred[1,zoom_ind,n_pl], color="C"+str(n_pl+1), label="model")
            art = f_zoom.patch(phase[zoom_ind], min_trans+pred[0,zoom_ind,n_pl], min_trans+pred[2,zoom_ind,n_pl],
                                      color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
            f_zoom_resid.plot(phase[zoom_ind],min_resid+resids,
                              ".k", label="data", zorder=-1000,alpha=0.5)
            f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                              "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

        else:
            print("#Normalising to transit duration",min_trans)
            f_zoom.plot(phase[zoom_ind]/tdur, min_trans+lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
            
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


def PlotLC(lc, trace, ID, mission='TESS', savename=None,overwrite=False, savefileloc=None, 
           returnfig=False, lcmask=None,tracemask=None, zoom_plot_time=False):
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
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,6))
    
    if lcmask is None:
        assert len(lc['time'])==len(trace['gp_pred'][0,:])
        lcmask=np.tile(True,len(lc['time']))
    else:
        assert len(lc['time'][lcmask])==len(trace['gp_pred'][0,:])
    
    #Finding if there's a single enormous gap in the lightcurve:
    x_gap=np.max(np.diff(lc['time'][lcmask]))>10
    if x_gap:
        print(" GAP IN X OF ",np.argmax(np.diff(lc['time'])))
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
    gp_mod = np.median(trace["gp_pred"][tracemask,:] + trace["mean"][tracemask, None], axis=0)
    
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
        gap_pos=np.average(lc['time'][np.argmax(np.diff(lc['time'])):(1+np.argmax(np.diff(lc['time'])))])
        before_gap_lc,before_gap_gp=(lc['time']<gap_pos)&lcmask,(lc['time'][lcmask]<gap_pos)
        after_gap_lc,after_gap_gp=(lc['time']>gap_pos)&lcmask,(lc['time'][lcmask]>gap_pos)
        
        print(np.sum(before_gap_lc),len(lc['time'][before_gap_lc]),np.sum(before_gap_gp),len(gp_mod[before_gap_gp]))
        
        f_all_1.plot(lc['time'][before_gap_lc], lc['flux'][before_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.plot(lc['time'][after_gap_lc], lc['flux'][after_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        f_all_1.plot(lc['time'][before_gap_lc], gp_mod[before_gap_gp]+2.5*min_trans, color="C3", label="GP fit")
        f_all_2.plot(lc['time'][after_gap_lc], gp_mod[after_gap_gp]+2.5*min_trans, color="C3", label="GP fit")

        f_all_1.plot(lc['time'][before_gap_lc], lc['flux'][before_gap_lc] - gp_mod[before_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.plot(lc['time'][after_gap_lc], lc['flux'][after_gap_lc] - gp_mod[after_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid_1.plot(lc['time'][before_gap_lc], 
                         lc['flux'][before_gap_lc] - gp_mod[before_gap_gp] - np.sum(pred[1,before_gap_gp,:],axis=1), 
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_1.set_xlabel('Time (BJD-245700)')
        f_all_resid_1.set_xlim(lc['time'][before_gap_lc][0],lc['time'][before_gap_lc][-1])

        f_all_resid_2.plot(lc['time'][after_gap_lc], 
                         lc['flux'][after_gap_lc] - gp_mod[after_gap_gp] - np.sum(pred[1,after_gap_gp,:],axis=1),
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_2.set_xlabel('Time (BJD-245700)')
        f_all_resid_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
        #print(len(lc[:,0]),len(lc[lcmask,0]),len(gp_mod))
        f_all_resid_1.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid_2.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        for n_pl in range(len(pred[1,0,:])):
            f_all_1.plot(lc['time'][before_gap_lc], pred[1,before_gap_gp,n_pl], color="C1", label="model")
            art = f_all_1.fill_between(lc['time'][before_gap_lc], pred[0,before_gap_gp,n_pl], pred[2,before_gap_gp,n_pl],
                                       color="C1", alpha=0.5, zorder=1000)
            f_all_1.set_xlim(lc['time'][before_gap_lc][0],lc['time'][before_gap_lc][-1])

            f_all_2.plot(lc['time'][after_gap_lc], pred[1,after_gap_gp,n_pl], color="C1", label="model")
            art = f_all_2.fill_between(lc['time'][after_gap_lc], pred[0,after_gap_gp,n_pl], pred[2,after_gap_gp,n_pl],
                                       color="C1", alpha=0.5, zorder=1000)
            f_all_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
            
            f_all_1.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
            f_all_2.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
        
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
        print(len(lc['time']),len(lc['flux']),len(lc['time'][lcmask]),len(lc['flux'][lcmask]),len(gp_mod),len(np.sum(pred[1,:,:],axis=1)))
        f_all.plot(lc['time'][lcmask], lc['flux'][lcmask]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all.plot(lc['time'][lcmask], gp_mod+2.5*min_trans, color="C3", label="GP fit")

        # Plot the data
        f_all.plot(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid.plot(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid.set_xlabel('Time (BJD-245700)')
        f_all_resid.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        for n_pl in range(len(pred[1,0,:])):
            f_all.plot(lc['time'][lcmask], pred[1,:,n_pl], color="C1", label="model")
            art = f_all.fill_between(lc['time'][lcmask], pred[0,:,n_pl], pred[2,:,n_pl], color="C1", alpha=0.5, zorder=1000)
            f_all.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        f_all.set_xticks([])
        f_zoom=fig.add_subplot(gs[:3, 3])
        f_zoom_resid=fig.add_subplot(gs[3, 3])
    
    min_trans=0;min_resid=0
    for n_pl in range(len(pred[1,0,:])):
        # Get the posterior median orbital parameters
        p = np.median(trace["period"][tracemask,n_pl])
        t0 = np.median(trace["t0"][tracemask,n_pl])
        tdur = np.nanmedian(trace['tdur'][tracemask,n_pl])
        #tdurs+=[(2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2))/np.nanmedian(trace['vrel'][tracemask,n_pl])]
        #print(min_trans,tdurs[n_pl],2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2),np.nanmedian(trace['vrel'][tracemask, n_pl]))
        
        phase=(lc['time'][lcmask]-t0+p*0.5)%p-p*0.5
        zoom_ind=abs(phase)<tdur
        
        resids=lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind] - np.sum(pred[1,zoom_ind,:],axis=1)

        if zoom_plot_time:
            #Plotting time:
            f_zoom.plot(phase[zoom_ind], min_trans+lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
            f_zoom.plot(phase[zoom_ind], min_trans+pred[1,zoom_ind,n_pl], color="C"+str(n_pl+1), label="model")
            art = f_zoom.fill_between(phase[zoom_ind], min_trans+pred[0,zoom_ind,n_pl], min_trans+pred[2,zoom_ind,n_pl],
                                      color="C"+str(n_pl+1), alpha=0.5,zorder=1000)
            f_zoom_resid.plot(phase[zoom_ind],min_resid+resids,
                              ".k", label="data", zorder=-1000,alpha=0.5)
            f_zoom_resid.plot([-1,1],[min_resid,min_resid],
                              "-",color="C"+str(n_pl+1), label="data", zorder=-1000,alpha=0.75,linewidth=2.0)

        else:
            print("#Normalising to transit duration",min_trans)
            f_zoom.plot(phase[zoom_ind]/tdur, min_trans+lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
            
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
