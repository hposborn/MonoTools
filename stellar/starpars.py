import numpy as np
import scipy
import pandas as pd

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

import requests
import re
import os
import sys

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import json
import pickle

from astropy.io import fits
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

from astropy.table import Table

import requests
from lxml import html

def mastQuery(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content


def TICdata(tics,sect=None,getImageData=False):
    """
    Download TESS stellar data

    Arguments:
        tics -- list of TESS IDs

    Returns:
        pandas DataFrame of stellar info from TIC.
    """
    if type(tics)==int or type(tics)==str:
        tics=np.array([tics])
    ticStringList=tics.astype(str)
    allData=[]

    #cols=['objID','objType','MH','logg','Teff','rho','rad','mass','ra','dec','pmRA','pmDEC','Tmag','contratio','d','gallat','gallong']
    tess_df=pd.DataFrame()

    request= {'service':'Mast.Catalogs.Filtered.Tic',
         'params':{'columns':'*', 'filters':[{'paramName':'ID', 'values':list(ticStringList)}]},
         'format':'json', 'removenullcolumns':True}
    while True:
        headers, outString = mastQuery(request)
        outObject = json.loads(outString)
        #allData.append(outObject)
        if outObject['status'] != 'EXECUTING':
            print("Breaking because status = "+outObject['status'])
            break
        if time.time() - startTime >30:
            print("Working...")
            startTime = time.time()
        time.sleep(5)
    for ni in range(len(outObject['data'])):
        tess_df=tess_df.append(pd.Series({col:outObject['data'][ni][col] for col in outObject['data'][ni]},name=int(outObject['data'][ni]['ID'])))

    if getImageData and sect is not None:
        tess_df['sector']=np.tile(np.nan,len(tess_df))
        tess_df['camera']=np.tile(np.nan,len(tess_df))
        tess_df['CCD']=np.tile(np.nan,len(tess_df))
        tess_df['colPix']=np.tile(np.nan,len(tess_df))
        tess_df['rowPix']=np.tile(np.nan,len(tess_df))
        for n,row in tess_df.iterrows():
            out = tess_stars2px_function_entry(int(row['ID']), tess_df['ra'], tess_df['dec'])
            outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo = out
            try:
                sectloc=list(outSec).index(sect)
                tess_df.loc[int(row['ID']),'sector']=outSec[sectloc]
                tess_df.loc[int(row['ID']),'camera']=outCam[sectloc]
                tess_df.loc[int(row['ID']),'CCD']=outCcd[sectloc]
                tess_df.loc[int(row['ID']),'colPix']=outColPix[sectloc]
                tess_df.loc[int(row['ID']),'rowPix']=outRowPix[sectloc]
            except:
                print(sect," not in observed sectors, ",outSec)
    return tess_df


def QueryGaiaAndSurveys(sc,CONESIZE=15*u.arcsec,savefile=None,mission='tess'):
    #Getting Gaia DR2 RVs:
    job = Gaia.launch_job_async("SELECT * \
    FROM gaiadr2.gaia_source \
    WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),\
    CIRCLE('ICRS',"+str(float(sc.ra.deg))+","+str(float(sc.dec.deg))+","+str(CONESIZE.to(u.deg).value)+"))=1;",verbose=False)

    gaia_res=job.get_results().to_pandas()
    #"closeness" function combines proximity of source to RA and DEC (as function of CONESIZE) 
    #                             *AND* brightness (scaled 1 arcsec ~ 9th mag, or 2*sep ~ deltamag=2.3)
    closeness=(np.hypot(sc.ra.deg-gaia_res.ra,sc.dec.deg-gaia_res.dec)/CONESIZE.to(u.deg).value)*np.exp(0.3*(gaia_res.phot_g_mean_mag-18))
    
    #Getting APASS results:
    apasshtml="https://www.aavso.org/cgi-bin/apass_dr10_download.pl?ra={0:.4f}&dec={1:.4f}&radius={2:.4f}&output=csv".format(sc.ra.deg,sc.dec.deg,CONESIZE.to(u.deg).value)
    out=requests.get(apasshtml)
    if out.status_code==200:
        #Modifying the output HTML to include a <tr> before the first row of header labels:
        APASS=pd.read_html(out.text.replace('<table border=1>\n\t\t<td','<table border=1>\n<tr><td'),header=0)
        if len(APASS)>0:
            APASS=APASS[0]
        else:
            APASS=None
    else:
        APASS=None
    #pd.DataFrame.from_csv("https://www.aavso.org/cgi-bin/apass_dr10_download.pl?ra="+str(SC.ra.deg)+"&dec="+str(SC.dec.deg)+"&radius="+str(CONESIZE/3600)+"&output=csv")
    
    alldat=pd.DataFrame()
    
    #Looping through Gaia results to find best match
    for n,row in enumerate(gaia_res.iterrows()):
        #Name of series becomes 00_gaiaid (for target) and then 01_gaiaid (for subsequent blends)
        sername=str(list(np.sort(closeness)).index(closeness[n]))+'_'+str(row[1]['source_id'])
        alldattemp=pd.Series({'mission':mission},name=sername)
        #print(alldattemp)
        alldattemp=alldattemp.append(gaia_res.iloc[n])
        print("Querying catalogues. "+str(n)+" of "+str(len(gaia_res)))
        #multiple rows, let's search using the Gaia RA/DECs
        #newra=row[1]['ra']
        #newdec=row[1]['dec']
        #print(row[1]['designation'],"<desig, id>",int(row[1]['source_id']))
        
        if APASS is not None:
            
            #"closeness algorithm = [dist in arcsec]*exp(0.3*[delta mag])
            closeness_apass=3600*np.hypot(row[1]['ra']-APASS['RA (deg)'],row[1]['dec']-APASS['Dec (deg)'])*np.exp(0.3*(row[1]['phot_g_mean_mag']-APASS['Sloan g\' (SG)']))
            if np.min(closeness_apass)<2.5:
                #Takes best APASS source if there is a source: (within 1 arcsec and deltamag<3) or (<2.5arcsec and deltamag=0.0)
                #Appending APASS info:
                nrby_apas=APASS.iloc[np.argmin(closeness_apass)]
                nrby_apas=nrby_apas.rename(index={col:'ap_'+col for col in nrby_apas.index if col not in gaia_res.columns})

                alldattemp=alldattemp.append(nrby_apas.drop([col for col in gaia_res.columns if col in nrby_apas.index]))
        
        dr=int(row[1]['designation'].decode("utf-8")[7])
        gid=row[1]['source_id']
        #Now searching the cross-matched cats with the GAIA ID
        jobs={}
        jobs['2m'] = Gaia.launch_job_async("SELECT * \
            FROM gaiadr"+str(dr)+".gaia_source AS g, gaiadr"+str(dr)+".tmass_best_neighbour AS tbest, gaiadr1.tmass_original_valid AS tmass \
            WHERE g.source_id = tbest.source_id AND tbest.tmass_oid = tmass.tmass_oid \
            AND g.source_id = "+str(gid), dump_to_file=False,verbose=False)
        jobs['sd'] = Gaia.launch_job_async("SELECT * \
            FROM gaiadr"+str(dr)+".gaia_source AS g, gaiadr"+str(dr)+".sdss"+"_"[:(2-dr)]+"dr9_best_neighbour AS sdbest, gaiadr1.sdssdr9_original_valid AS sdss \
            WHERE g.source_id = sdbest.source_id AND sdbest.sdssdr9_oid = sdss.sdssdr9_oid \
            AND g.source_id = "+str(gid), dump_to_file=False,verbose=False)
        jobs['ur'] = Gaia.launch_job_async("SELECT * \
            FROM gaiadr"+str(dr)+".gaia_source AS g, gaiadr"+str(dr)+".urat1_best_neighbour AS uratbest, gaiadr1.urat1_original_valid AS urat1 \
            WHERE g.source_id = uratbest.source_id AND uratbest.urat1_oid = urat1.urat1_oid \
            AND g.source_id = "+str(gid), dump_to_file=False,verbose=False)
        jobs['wise'] = Gaia.launch_job_async("SELECT * \
            FROM gaiadr"+str(dr)+".gaia_source AS g, gaiadr"+str(dr)+".allwise_best_neighbour AS wisest, gaiadr1.allwise_original_valid AS wise \
            WHERE g.source_id = wisest.source_id AND wisest.allwise_oid = wise.allwise_oid \
            AND g.source_id = "+str(gid), dump_to_file=False,verbose=False)
        for job in jobs:
            res=jobs[job].get_results().to_pandas()
            if res.shape[0]>0:
                #Making 
                res=res.rename(columns={col:job+'_'+col for col in res.columns if col not in gaia_res.columns})
                alldattemp=alldattemp.append(res.iloc[0].drop([col for col in gaia_res.columns if col in res.columns]))
        alldattemp=alldattemp.drop_duplicates()
        #print(alldattemp,,job_sd.get_results().to_pandas(),
        #                      job_ur.get_results().to_pandas(),job_wise.get_results().to_pandas())
        #alldattemp=pd.concat([alldattemp,job_2m.get_results().to_pandas(),job_sd.get_results().to_pandas(),
        #                      job_ur.get_results().to_pandas(),job_wise.get_results().to_pandas()],
        #                     axis=1)
        alldat=alldat.append(alldattemp.rename(sername))
        
    
    alldat['dilution_ap']=np.tile(CONESIZE.to(u.arcsec).value,len(alldat))
    alldat['prop_all_flux']=alldat['phot_g_mean_flux'].values/np.nansum(alldat['phot_g_mean_flux'].values)
    alldat['diluted_by']=1.0-alldat['prop_all_flux']
    if type(alldat)==pd.DataFrame and len(alldat)>1:
        targ=alldat.iloc[np.argmin(closeness)]
    elif type(alldat)==pd.DataFrame:
        targ=alldat.iloc[0]
    elif type(alldat)==pd.Series:
        targ=alldat
    if savefile is not None:
        alldat.iloc[np.argsort(closeness)].to_csv(savefile.replace('.csv','_all_contams.csv'))
        targ.to_csv(savefile)
    #print(str(alldat.shape)," dic created with data from ",','.join([cats[i] if len(jobs[i])>0 else "" for i in range(5)]))
    
    return targ


def GetKICinfo(kic):
    #Getting Kepler stellar info from end-of-Kepler Q1-Q17 data table:
    kicdat=pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_stellar&where=kepid=%27"+str(int(kic))+"%27")
    if len(kicdat.shape)>1:
        kicdat=kicdat.iloc[0]
    for row in kicdat.index:
        newname=row[:]
        if 'dens' in row:
            newname=newname.replace('dens','rho_gcm3')
        elif 'radius' in row:
            newname=newname.replace('radius','rad')
        if '_err1' in row:
            newname=newname.replace('_err1','ep')
        elif '_err2' in row:
            newname=newname.replace('_err2','em')
        kicdat=kicdat.rename(index={row:newname})
        try:
            kicdat[newname]=float(kicdat[newname])
        except:
            continue
    for row in kicdat.index:
        #Adding simple average errors:
        if 'em' in row and row[:-1] not in kicdat.index:
            kicdat[row[:-1]]=0.5*(abs(kicdat[row])+abs(kicdat[row[:-1]+'p']))
    #Adding rho in terms of solar density:
    kicdat['rho']=kicdat['rho_gcm3']/1.411
    kicdat['rhoep']=kicdat['rho_gcm3ep']/1.411
    kicdat['rhoem']=kicdat['rho_gcm3em']/1.411
    kicdat['rhoe']=0.5*(abs(kicdat['rhoep'])+abs(kicdat['rhoem']))
    kicdat['rho_gcm3e']=1.411*kicdat['rhoe']

    #kicdat=pd.DataFrame.from_csv("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=keplerstellar&where=epic_number=%27"+str(int(kic))+"%27")
    kicdat['mission']='kepler'
    kicdat['id']=kicdat['kepid']
    kicdat['spec']=None
    kicdat['source']='kic'
    return kicdat

def GetExoFop(icid,mission='tess',file=''):
    cols={'Telescope':'telescope','Instrument':'instrument','Teff (K)':'teff','Teff (K) Error':'teffe',
          'Teff':'teff','Teff Error':'teffe','log(g)':'logg',
          'log(g) Error':'logge','Radius (R_Sun)':'rad','Radius':'rad','Radius Error':'rade',
          'Radius (R_Sun) Error':'rade','logR\'HK':'logrhk',
          'logR\'HK Error':'logrhke','S-index':'sindex','S-index Error':'sindexe','H-alpha':'haplha','H-alpha Error':'halphae',
          'Vsini':'vsini','Vsini Error':'vsinie','Rot Per':'rot_per','Rot Per Error':'rot_pere','Metallicity':'feh',
          'Metallicity Error':'fehe','Mass (M_Sun)':'mass','Mass':'mass','Mass Error':'masse',
          'Mass (M_Sun) Error':'masse','Density (g/cm^3)':'rho_gcm3',
          'Density':'rho_gcm3',
          'Density (g/cm^3) Error':'rho_gcm3e','Luminosity':'lum','Luminosity Error':'lume',
          'Observation Time (BJD)':'obs_time_bjd','Distance':'dis','Distance Error':'dise',
          'RV (m/s)':'rv_ms','RV Error':'rv_mse','Distance (pc)':'dis','Distance (pc) Error':'dise',
          '# of Contamination sources':'n_contams', 'B':'bmag', 'B Error':'bmage', 'Dec':'dec', 'Ecliptic Lat':'lat_ecl',
          'Ecliptic Long':'long_ecl', 'Gaia':'gmag', 'Gaia Error':'gmage', 'Galactic Lat':'lat_gal', 'Galactic Long':'long_gal',
          'H':'hmag', 'H Error':'hmage', 'In CTL':'in_ctl', 'J':'jmag', 'J Error':'jmage', 'K':'kmag', 'K Error':'kmage',
          'Planet Name(s)':'planet_names', 'Proper Motion Dec (mas/yr)':'pm_dec',
          'Proper Motion RA (mas/yr)':'pm_ra', 'RA':'ra','RA (J2015.5)':'ra', 'Dec (J2015.5)':'dec',
          'Star Name & Aliases':'star_name', 'TESS':'tmag','Kep':'kepmag',
          'TESS Error':'tmage', 'TIC Contamination Ratio':'ratio_contams', 'TOI':'toi', 'V':'vmag', 'V Error':'vmage',
          'WISE 12 micron':'w3mag', 'WISE 12 micron Error':'w3mage', 'WISE 22 micron':'w4mag',
          'WISE 22 micron Error':'w4mage', 'WISE 3.4 micron':'w1mag', 'WISE 3.4 micron Error':'w1mage',
          'WISE 4.6 micron':'w2mag', 'WISE 4.6 micron Error':'w2mag', 'n_TOIs':'n_tois','spec':'spec',
          'Campaign':'campaign','Object Type':'objtype'}
    '''
    Index(['mission', 'ra', 'dec', 'GalLong', 'GalLat', 'Aliases', 'campaign',
           'Proposals', 'objtype', 'bmag', 'bmag_err', 'g', 'g_err', 'vmag',
           'vmag_err', 'r', 'r_err', 'kepmag', 'kepmag_err', 'i', 'i_err', 'jmag',
           'jmag_err', 'hmag', 'hmag_err', 'kmag', 'kmag_err', 'w1mag',
           'w1mag_err', 'w2mag', 'w2mag_err', 'w3mag', 'w3mag_err', 'w4mag',
           'w4mag_err', 'Teff', 'Teff_err', 'logg', 'logg_err', 'Radius',
           'Radius_err', 'FeH', 'FeH_err', 'Distance', 'Distance_err', 'Mass',
           'Mass_err', 'Density', 'Density_err', 'spec', 'bmagem', 'bmagep', 'gem',
           'gep', 'vmagem', 'vmagep', 'rem', 'rep', 'kepmagem', 'kepmagep'],
          dtype='object')
    Index(['iem', 'iep', 'jmagem', 'jmagep', 'hmagem', 'hmagep', 'kmagem',
           'kmagep', 'w1magem', 'w1magep', 'w2magem', 'w2magep', 'w3magem',
           'w3magep', 'w4magem', 'w4magep', 'Teffem', 'Teffep', 'loggem', 'loggep',
           'Radiusem', 'Radiusep', 'FeHem', 'FeHep', 'Distanceem', 'Distanceep',
           'Massem', 'Massep', 'Densityem', 'Densityep', 'bmage', 'ge', 'vmage',
           're', 'ie', 'jmage', 'hmage', 'kmage', 'w1mage', 'w2mage', 'w3mage',
           'Teffe', 'logge', 'Radiuse', 'FeHe', 'Distancee', 'Masse', 'Densitye'],
          dtype='object')
    '''
    
    #Strips online file for a given epic/tic
    if mission.lower() in ['kep','kepler']:
        kicinfo=GetKICinfo(icid)
        #Checking if the object is also in the TIC:
        ticout=Catalogs.query_criteria(catalog="Tic",coordinates=str(kicinfo['ra'])+','+str(kicinfo['dec']),
                                       radius=20*u.arcsecond,objType="STAR",columns=['ID','KIC','Tmag','Vmag']).to_pandas()
        if len(ticout.shape)>1:
            ticout=ticout.loc[np.argmin(ticout['Tmag'])]
            icid=ticout['ID']
            mission='tess'
        elif ticout.shape[0]>0:
            #Not in TIC
            return kicinfo
    else:
        kicinfo = None
    assert mission.lower() in ['tess','k2']
    outdat={}
    outdat['mission']=mission.lower()
    #Searching TESS and K2 ExoFop for info (and TIC-8 info):
    req=requests.get("https://exofop.ipac.caltech.edu/"+mission.lower()+"/download_target.php?id="+str(icid), timeout=120)
    if req.status_code==200:
        #Splitting into each 'paragraph'
        sections=req.text.split('\n\n')
        for sect in sections:
            #Processing each section:
            if sect[:2]=='RA':
                #This is just general info - saving
                for line in sect.split('\n'):
                    if mission.lower()=='tess':
                        if line[:28].strip() in cols:
                            outdat[cols[line[:28].strip()]]=line[28:45].split('  ')[0].strip()
                        else:
                            outdat[re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',line[:28])]=line[28:45].split('  ')[0].strip()
                    elif mission.lower()=='k2':
                        if line[:13].strip() in cols:
                            outdat[cols[line[:13].strip()]]=line[13:].strip()
                        else:
                            outdat[re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',line[:13])]=line[13:].strip()
            elif sect[:24]=='TESS Objects of Interest':
                #Only taking number of TOIs and TOI number:
                outdat['n_TOIs']=len(sect.split('\n'))-2
                outdat['TOI']=sect.split('\n')[2][:15].strip()
            elif sect[:7]=='STELLAR':
                #Stellar parameters
                labrow=sect.split('\n')[1]
                boolarr=np.array([s==' ' for s in labrow])
                splits=[0]+list(2+np.where(boolarr[:-3]*boolarr[1:-2]*~boolarr[2:-1]*~boolarr[3:])[0])+[len(labrow)]
                labs = [re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',labrow[splits[i]:splits[i+1]]) for i in range(len(splits)-1)]
                spec=[]
                if mission.lower()=='tess':
                    #Going through all sources of Stellar params:
                    for row in sect.split('\n')[2:]:
                        stpars=np.array([row[splits[i]:splits[i+1]].strip() for i in range(len(splits)-1)])
                        for nl in range(len(labs)):
                            if labs[nl].strip() not in cols:
                                label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', labs[nl]).replace('Error','_err') 
                            else:
                                label=cols[labs[nl].strip()]
                            if not label in outdat.keys() and stpars[1]=='' and stpars[nl].strip()!='':
                                #Stellar info just comes from TIC, so saving simply:
                                outdat[label] = stpars[nl]
                            elif stpars[1]!='' and stpars[nl].strip()!='':
                                #Stellar info comes from follow-up, so saving with _INSTRUMENT:
                                spec+=['_'+row[splits[3]:splits[4]].strip()]
                                outdat[labs[nl]+'_'+stpars[1]] = stpars[nl]
                elif mission.lower()=='k2':
                    for row in sect.split('\n')[1:]:
                        if row[splits[0]:splits[1]].strip() not in cols:
                            label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', row[splits[0]:splits[1]]).replace('Error','_err') 
                        else:
                            label=cols[row[splits[0]:splits[1]].strip()]

                        if not label in outdat.keys() and row[splits[3]:splits[4]].strip()=='huber':
                            outdat[label] = row[splits[1]:splits[2]].strip()
                            outdat[label+'_err'] = row[splits[2]:splits[3]].strip()
                        elif label in outdat.keys() and row[splits[3]:splits[4]].strip()!='huber':
                            if row[splits[3]:splits[4]].strip()!='macdougall':
                                spec+=['_'+row[splits[3]:splits[4]].strip()]
                                #Adding extra stellar params with _user (no way to tell the source, e.g. spectra)
                                outdat[label+'_'+row[splits[3]:splits[4]].strip()] = row[splits[1]:splits[2]].strip()
                                outdat[label+'_err'+'_'+row[splits[3]:splits[4]].strip()] = row[splits[2]:splits[3]].strip()
                outdat['spec']=None if len(spec)==0 else ','.join(list(np.unique(spec)))
            elif sect[:9]=='MAGNITUDE':
                labrow=sect.split('\n')[1]
                boolarr=np.array([s==' ' for s in labrow])
                splits=[0]+list(2+np.where(boolarr[:-3]*boolarr[1:-2]*~boolarr[2:-1]*~boolarr[3:])[0])+[len(labrow)]
                for row in sect.split('\n')[2:]:
                    if row[splits[0]:splits[1]].strip() not in cols:
                        label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', row[splits[0]:splits[1]]).replace('Error','_err') 
                    else:
                        label=cols[row[splits[0]:splits[1]].strip()]
                    outdat[label] = row[splits[1]:splits[2]].strip()
                    outdat[label+'_err'] = row[splits[2]:splits[3]].strip()
                    
        outdat=pd.Series(outdat,name=icid)
        
        #Replacing err and err1/2 with em and ep
        for col in outdat.index:
            try:
                outdat[col]=float(outdat[col])
            except:
                pass
            if col.find('_err1')!=-1:
                outdat=outdat.rename(index={col:col.replace('_err1','ep')})
            elif col.find('_err2')!=-1:
                outdat=outdat.rename(index={col:col.replace('_err2','em')})
            elif col.find('_err')!='1':
                outdat[col.replace('_err','em')]=outdat[col]
                outdat[col.replace('_err','ep')]=outdat[col]
                outdat=outdat.rename(index={col:col.replace('_err','e')})
        for col in outdat.index:
            if 'radius' in col:
                outdat=outdat.rename(index={col:col.replace('radius','rad')})
            if col[-2:]=='em' and col[:-1] not in outdat.index and type(outdat[col])!=str:
                #average of em and ep -> e
                outdat[col[:-1]]=0.5*(abs(outdat[col])+abs(outdat[col[:-1]+'p']))
        return outdat, kicinfo
    elif kicinfo is not None:
        return None, kicinfo
    else:
        return None, None

def LoadModel():
    #Loading isoclassify "mesa" model from file:
    from .isoclassify import classify, pipeline
    mist_loc='/'.join(classify.__file__.split('/')[:-3])+'/mesa.h5'
    import h5py
    file = h5py.File(mist_loc,'r+', driver='core', backing_store=False)
    model = {'age':np.array(file['age']),\
             'mass':np.array(file['mass']),\
             'feh':np.array(file['feh']),\
             'teff':np.array(file['teff']),\
             'logg':np.array(file['logg']),\
             'rad':np.array(file['rad']),\
             'lum':np.array(file['rad']),\
             'rho':np.array(file['rho']),\
             'dage':np.array(file['dage']),\
             'dmass':np.array(file['dmass']),\
             'dfeh':np.array(file['dfeh']),\
             'eep':np.array(file['eep']),\
             'bmag':np.array(file['bmag']),\
             'vmag':np.array(file['vmag']),\
             'btmag':np.array(file['btmag']),\
             'vtmag':np.array(file['vtmag']),\
             'gmag':np.array(file['gmag']),\
             'rmag':np.array(file['rmag']),\
             'imag':np.array(file['imag']),\
             'zmag':np.array(file['zmag']),\
             'jmag':np.array(file['jmag']),\
             'hmag':np.array(file['hmag']),\
             'kmag':np.array(file['kmag']),\
             'd51mag':np.array(file['d51mag']),\
             'gamag':np.array(file['gamag']),\
             'fdnu':np.array(file['fdnu']),\
             'avs':np.zeros(len(np.array(file['gamag']))),\
             'dis':np.zeros(len(np.array(file['gamag'])))}
    model['rho'] = np.log10(model['rho'])
    model['lum'] = model['rad']**2*(model['teff']/5777.)**4
    # next line turns off Dnu scaling relation corrections
    model['fdnu'][:]=1.
    model['avs']=np.zeros(len(model['teff']))
    model['dis']=np.zeros(len(model['teff']))
    '''
    # load MIST models
    homedir=os.path.expanduser('~/')
    import pickle
    model=pickle.load(open(mist_loc,'rb'),encoding='latin')
    # prelims to manipulate some model variables (to be automated soon ...)
    model['rho']=np.log10(model['rho'])
    model['fdnu'][:]=1.

    model['avs']=np.zeros(len(model['teff']))
    model['dis']=np.zeros(len(model['teff']))
    '''
    return model

def LoadDust(sc,plx,dust='allsky'):
    import mwdust
    av=mwdust.SFD()(sc.galactic.l.deg,sc.galactic.b.deg,1000.0/plx)
    #sfdmap(sc.ra.deg.to_string(),sc.dec.deg.to_string())
    if dust == 'allsky':
        dustmodel = pipeline.query_dustmodel_coords_allsky(sc.ra.deg,sc.dec.deg)
        ext = pipeline.extinction('cardelli')
    if dust == 'green18':
        dustmodel = pipeline.query_dustmodel_coords(sc.ra.deg,sc.dec.deg)
        ext = pipeline.extinction('schlafly16')
    if dust == 'none':
        dustmodel = 0
        ext = pipeline.extinction('cardelli')
    return dustmodel,ext

def dens2(logg,loggerr1,loggerr2,rad,raderr1,raderr2,mass,masserr1,masserr2,nd=6000,returnpost=False):
    #Returns a density as the weighted average of that from logg and mass
    dens1 = lambda logg,rad: (np.power(10,logg-2)/(1.33333*np.pi*6.67e-11*rad*695500000))/1410.0
    dens2 = lambda mass,rad: ((mass*1.96e30)/(4/3.0*np.pi*(rad*695500000)**3))/1410.0

    loggs= np.random.normal(logg,0.5*(loggerr1+loggerr2),nd)
    rads= np.random.normal(rad,0.5*(raderr1+raderr2),nd)
    rads[rads<0]=abs(np.random.normal(rad,0.25*(raderr1+raderr2),np.sum(rads<0)))
    masses= np.random.normal(mass,0.5*(masserr1+masserr2),nd)
    masses[masses<0]=abs(np.random.normal(mass,0.25*(masserr1+masserr2),np.sum(masses<0)))
    d1=np.array([dens1(loggs[l],rads[l]) for l in range(nd)])
    d2=np.array([dens2(masses[m],rads[m]) for m in range(nd)])
    #Combining up/down dists alone for up.down uncertainties. COmbining all dists for median.
    #Gives almost identical values as a weighted average of resulting medians/errors.
    #print("logg/rad: "+str(np.median(d1))+"+/-"+str(np.std(d1))+", mass/rad:"+str(np.median(d2))+"+/-"+str(np.std(d2)))
    post=d1 if np.std(d1)<np.std(d2) else d2
    if returnpost:
        #Returning combined posterier...
        return post
    else:
        dens=np.percentile(post,[16,50,84])
        return np.array([dens[1],np.diff(dens)[0],np.diff(dens)[1]])

def QueryNearbyGaia(sc,CONESIZE,file=None):
    
    job = Gaia.launch_job_async("SELECT * \
    FROM gaiadr2.gaia_source \
    WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),\
    CIRCLE('ICRS',"+str(sc.ra.deg)+","+str(sc.dec.deg)+","+str(CONESIZE/3600.0)+"))=1;" \
    , dump_to_file=True,output_file=file)
    
    df=job.get_results().to_pandas()
    '''
    if df:
        job = Gaia.launch_job_async("SELECT * \
        FROM gaiadr1.gaia_source \
        WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),\
        CIRCLE('ICRS',"+str(sc.ra.deg)+","+str(sc.dec.deg)+","+str(CONESIZE/3600.0)+"))=1;" \
        , dump_to_file=True,output_file=file)
    '''
    print(np.shape(df))
    if np.shape(df)[0]>1:
        print(df.shape[0],"stars with mags:",df.phot_g_mean_mag.values,'and teffs:',df.teff_val.values)
        #Taking brightest star as target
        df=df.loc[np.argmin(df.phot_g_mean_mag)]
    if len(np.shape(df))>1:
        df=df.iloc[0]
    if np.shape(df)[0]!=0 or np.isnan(float(df['teff_val'])):
        outdf={}
        #print(df[['teff_val','teff_percentile_upper','radius_val','radius_percentile_upper','lum_val','lum_percentile_upper']])
        outdf['Teff']=float(df['teff_val'])
        outdf['e_Teff']=0.5*(float(df['teff_percentile_upper'])-float(df['teff_percentile_lower']))
        #print(np.shape(df))
        #print(df['lum_val'])
        if not np.isnan(df['lum_val']):
            outdf['lum']=float(df['lum_val'])
            outdf['e_lum']=0.5*(float(df['lum_percentile_upper'])-float(df['lum_percentile_lower']))
        else:
            if outdf['Teff']<9000:
                outdf['lum']=np.power(10,5.6*np.log10(outdf['Teff']/5880))
                outdf['e_lum']=1.0
            else:
                outdf['lum']=np.power(10,8.9*np.log10(outdf['Teff']/5880))
                outdf['e_lum']=0.3*outdf['lum']
        if not np.isnan(df['radius_val']):
            outdf['rad']=float(df['radius_val'])
            outdf['e_rad']=0.5*(float(df['radius_percentile_upper'])-float(df['radius_percentile_lower']))
        else:
            mass=outdf['lum']**(1/3.5)
            if outdf['Teff']<9000:
                outdf['rad']=mass**(3/7.)
                outdf['e_rad']=0.5*outdf['rad']
            else:
                outdf['rad']=mass**(19/23.)
                outdf['e_rad']=0.5*outdf['rad']
        outdf['GAIAmag_api']=df['phot_g_mean_mag']
    else:
        print("NO GAIA TARGET FOUND")
        outdf={}
    return outdf



def CheckSpecCsv(radec,icid,thresh=20*u.arcsec):
    specs=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath( __file__ )),"spectra_all.csv"))
    spec_coords=SkyCoord(specs['ra']*u.deg,specs['dec']*u.deg)
    seps=radec.separation(spec_coords)
    
    #Searching by ID
    if icid in specs.input_id:
        out=specs.loc[specs.input_id.values==icid,['teff','teff_err','logg','logg_err','feh','feh_err']]
    elif np.min(seps)<thresh:
        #And searching by RA/DEC
        out=specs.iloc[np.argmin(seps),['teff','teff_err','logg','logg_err','feh','feh_err']]
    else:
        return None
    
    #Converting from df to Series:
    out=out.iloc[0] if type(out)==pd.DataFrame else out
    return out
    
    
def Assemble_and_run_isoclassify(icid,sc,mission,survey_dat,exofop_dat,errboost=0.2,spec_dat=None,
                                 useGaiaLum=True,useGaiaBR=True,useBV=True,useGaiaSpec=True,
                                 use2mass=True,useGriz=True,useGaiaAg=True):
    from .isoclassify import classify, pipeline
    ############################################
    #    Building isoclassify input data:      #
    ############################################
    
    x=classify.obsdata()
    mag=False
    x.addcoords(sc.ra.deg,sc.dec.deg)
    
    #Luminosity from Gaia:
    if useGaiaLum and 'lum_val' in survey_dat.index:
        if not np.isnan((survey_dat.lum_val+survey_dat.lum_percentile_upper+survey_dat.lum_percentile_lower)):
            x.addlum([survey_dat.lum_val],[0.5*(survey_dat.lum_percentile_upper-survey_dat.lum_percentile_lower)])
    #BR from Gaia:
    if useGaiaBR and 'phot_g_mean_mag' in survey_dat.index and survey_dat.phot_g_mean_mag is not None:
        if not np.isnan(np.sum(survey_dat[['phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
                                           'phot_g_mean_flux_over_error','phot_bp_mean_flux_over_error',
                                           'phot_rp_mean_flux_over_error']].values.astype(np.float64))):
            x.addgaia(survey_dat[['phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag']].values.astype(np.float64),
                  errboost+np.log(1.0+1.0/survey_dat[['phot_g_mean_flux_over_error',
                                              'phot_bp_mean_flux_over_error',
                                              'phot_rp_mean_flux_over_error']].values.astype(np.float64))*2.5)
            mag+=True
    else:
        print("No Gaia mag for",icid)
        mag+=False
    #BV from either APASS or Exofop:
    if useBV and 'ap_Johnson B (B)' in survey_dat.index and not np.isnan(np.sum(survey_dat[['ap_Johnson B (B)','ap_Johnson V (V)','ap_Berr','ap_Verr']].values)):
        #BV photometry (eg apass)
        x.addbv([survey_dat['ap_Johnson B (B)'],survey_dat['ap_Johnson V (V)']],
                [errboost+survey_dat['ap_Berr'],errboost+survey_dat['ap_Verr']])
        mag+=True
    elif useBV and 'B' in exofop_dat.index and not np.isnan(np.sum(exofop_dat[['B','V','Be','Ve']].values.astype(float))):
        x.addbv([float(exofop_dat['B']),float(exofop_dat['V'])],
                [errboost+float(exofop_dat['Be']),errboost+float(exofop_dat['Ve'])])
        mag+=True
    else:
        print("No BV for",icid)
        mag+=False

    #Spectra either from APASS, or from user-uploaded file, or from Gaia spectrum:
    if exofop_dat['spec'] is not None:
        #From ExoFop - either has _user (K2) or _INSTRUMENT (TESS)
        
        #If there's multiple spectra, we'll take the one with lowest Teff err:
        if len(exofop_dat['spec'].split(','))>1:
            src=exofop_dat['spec'].split(',')[np.min([spec_dat['teff_err'+spec_src] for spec_src in exofop_dat['spec'].split(',')])]
        else:
            src=exofop_dat['spec'].split(',')[0]
        if 'logg'+src in spec_dat.index:
            #Correcting possible problems (e.g. missing columns):
            exofop_dat['feh'+src]=0.0 if 'feh'+src not in exofop_dat.index else exofop_dat['feh'+src]
            if 'fehe'+src not in exofop_dat.index:
                if 'fehep'+src in spec_dat.index:
                    exofop_dat['fehe'+src]=0.5*(abs(exofop_dat['fehep'+src])+abs(exofop_dat['fehem'+src]))
                else:
                    exofop_dat['fehe'+src]=2.0
            if 'logge'+src not in exofop_dat.index:
                if 'loggep'+src in exofop_dat.index:
                    exofop_dat['logge'+src]=0.5*(abs(exofop_dat['loggep'+src])+abs(exofop_dat['loggem'+src]))
                else:
                    exofop_dat['logge'+src]=2.5
            if 'teffe'+src not in exofop_dat.index:
                if 'teffep'+src in exofop_dat.index:
                    exofop_dat['teffe'+src]=0.5*(abs(exofop_dat['teffem'+src])+abs(exofop_dat['teffem'+src]))
                else:
                    exofop_dat['teffe'+src]=250
            x.addspec([exofop_dat['teff'+src], exofop_dat['logg'+src], exofop_dat['feh'+src]],
                      [exofop_dat['teffe'+src], exofop_dat['logge'+src], exofop_dat['fehe'+src]])
    elif spec_dat is not None:
        #From LAMOST or AAT or Coralie (cross-matched list in stellar folder)
        x.addspec([spec_dat.teff, spec_dat.logg, spec_dat.feh],
                  [spec_dat.teff_err, spec_dat.logg_err, spec_dat.feh_err])
    elif useGaiaSpec and 'teff_val' in survey_dat.index and not np.isnan(survey_dat.teff_val):
        #From Gaia:
        x.addspec([survey_dat.teff_val, survey_dat.rv_template_logg, 0.0],
                  [0.5*(survey_dat.teff_percentile_upper-survey_dat.teff_percentile_lower), 0.4, 1.0])
    #2MASS JHK from Gaia-xmatched catalogue or from ExoFop:
    if use2mass and '2m_ks_m' in survey_dat.index and not np.isnan(np.sum(survey_dat[['2m_j_m','2m_h_m','2m_ks_m',
                                                                                     '2m_j_msigcom','2m_h_msigcom','2m_ks_msigcom']].values)):
        # 2MASS photometry
        x.addjhk([survey_dat['2m_j_m'],survey_dat['2m_h_m'],survey_dat['2m_ks_m']],
                 [errboost+survey_dat['2m_j_msigcom'],errboost+survey_dat['2m_h_msigcom'],errboost+survey_dat['2m_ks_msigcom']])
        mag+=True
    elif use2mass and 'K' in exofop_dat.index  and not np.isnan(np.sum(exofop_dat[['J','H','K',
                                                                                   'Je','He','Ke']].values.astype(float))):
        x.addjhk([float(exofop_dat['J']),float(exofop_dat['H']),float(exofop_dat['K'])],
                 [errboost+float(exofop_dat['Je']),errboost+float(exofop_dat['He']),errboost+float(exofop_dat['Ke'])])
        mag+=True
    else:
        print("No 2MASS for",icid)
        mag+=False
    #GRIZ photometry from APASS or Gaia-xmatched SDSS catalogue or from ExoFop:
    if useGriz and "ap_Sloan z' (SZ)" in survey_dat.index and not np.isnan(np.sum(survey_dat[["ap_Sloan g' (SG)","ap_Sloan r' (SR)","ap_Sloan i' (SI)","ap_Sloan z' (SZ)","ap_SGerr","ap_SRerr","ap_SIerr","ap_SZerr"]].values)):
        # 2MASS photometry
        x.addgriz([survey_dat["ap_Sloan g' (SG)"],survey_dat["ap_Sloan r' (SR)"],
                   survey_dat["ap_Sloan i' (SI)"],survey_dat["ap_Sloan z' (SZ)"]],
                 [errboost+survey_dat["ap_SGerr"],errboost+survey_dat["ap_SRerr"],
                  errboost+survey_dat["ap_SIerr"],errboost+survey_dat["ap_SZerr"]])
        mag+=True
    elif useGriz and 'sd_z_mag' in survey_dat.index and not np.isnan(np.sum(survey_dat[["sd_g_mag","sd_r_mag","sd_i_mag","sd_z_mag","sd_g_mag_error","sd_r_mag_error","sd_i_mag_error","sd_z_mag_error"]].values)):
        x.addgriz([survey_dat['sd_g_mag'],survey_dat['sd_r_mag'],survey_dat['sd_i_mag'],survey_dat['sd_z_mag']],
                  [errboost+survey_dat['sd_g_mag_error'],errboost+survey_dat['sd_r_mag_error'],
                   errboost+survey_dat['sd_i_mag_error'],errboost+survey_dat['sd_z_mag_error']])
        mag+=True
    elif useGriz and 'z' in exofop_dat.index and not np.isnan(np.sum(exofop_dat[["g","r","i","z","ge","re","ie","ze"]].values)):
        x.addgriz([float(exofop_dat['g']),float(exofop_dat['r']),float(exofop_dat['i']),float(exofop_dat['z'])],
                  [errboost+float(exofop_dat['ge']),errboost+float(exofop_dat['re']),
                   errboost+float(exofop_dat['ie']),errboost+float(exofop_dat['ze'])])
        mag+=True
    else:
        mag+=False
        print("No griz for",icid)
    #Gaia Ag
    if useGaiaAg and 'a_g_val' in survey_dat.index and survey_dat['a_g_val'] is not None:
        av=survey_dat['a_g_val']
    else:
        av=-99
    #Gaia Parallax:
    if 'parallax' in survey_dat.index and survey_dat.parallax is not None:
        x.addplx(survey_dat.parallax/1000,survey_dat.parallax_error/1000)
    #In a case where no magnitude is set, we assume V~kepmag/V~Tmag:
    if not mag:
        if 'tmag' in exofop_dat.index and ~np.isnan(exofop_dat['tmag']):
            print("No archival photometry! Adding Tmag from input catalogue magnitude as V:",exofop_dat['tmag'])
            x.addbv([-99,exofop_dat['tmag']],[-99,0.2])
        elif 'kepmag' in exofop_dat.index and ~np.isnan(exofop_dat['kepmag']):
            print("No archival photometry! Adding Kepmaf from input catalogue magnitude as V:",exofop_dat['kepmag'])
            x.addbv([-99,exofop_dat['kepmag']],[-99,0.2])
    
    ############################################
    #           Running isoclassify:           #
    ############################################
    print("Isoclassifying")
    mod=LoadModel()
    try:
        dustmodel,ext = LoadDust(sc,survey_dat.parallax/1000.,dust='allsky')
        paras = classify.classify(input=x, model=mod, dustmodel=dustmodel, useav=av, ext=ext, plot=0)
    except:
        paras = classify.classify(input=x, model=mod)

    ############################################
    #       Assembling all output data:        #
    ############################################
    
    #Extracting parameters from isoclassify output class into pandas df:
    col_names=['teff','teffep','teffem','logg','loggep','loggem','feh','fehep','fehem',
               'rad','radep','radem','mass','massep','massem','rho','rhoep','rhoem',
               'lum','lumep','lumem','avs','avsep','avsem','dis','disep','disem']#,'plx','plxep','plxem','mabs']
    isoclass_df=pd.Series()
    for c in col_names:
        exec('isoclass_df[\"'+c+'\"]=paras.'+c)
    #isoclass_df=isoclass_df.rename(index={'rho':'rho_gcm3','rhoep':'rho_gcm3ep','rhoem':'rho_gcm3em'})
    #After much posturing, I have determined that these output "rho"s are in rho_S and not gcm3, so adding gcm3 values here:
    isoclass_df['rho_gcm3']=isoclass_df['rho']*1.411
    isoclass_df['rho_gcm3ep']=isoclass_df['rhoep']*1.411
    isoclass_df['rho_gcm3em']=isoclass_df['rhoem']*1.411
    return isoclass_df, paras
    
def starpars(icid,mission,errboost=0.1,return_best=True,
             useGaiaLum=True,useGaiaBR=True,useGaiaSpec=True,
             useBV=True,use2mass=True,useGriz=True,useGaiaAg=True,use_surveys=False):
    # Estimating stellar parameters given survey data, input catalogues, and possibly follow-up data
    #INPUTS:
    # - icid         (Mission ID in input catalogue)
    # - mission      ('tess','k2','kepler')
    # - errboost     (amount to artificially boost photometry errors due to possible systematics)
    # - return_best  (boolean. Only return best info, or return all data objects?)
    # - useGaiaLum   (Use luminosity as determined by Gaia)
    # - useGaiaBR    (Use Gaia B and R filter photometry)
    # - useGaiaSpec  (Use the Gaia estimates of logg and Teff as input spectra)
    # - useBV        (Use BV from survey data - e.g. APASS)
    # - use2mass     (Use 2MASS JHK from survey)
    # - useGriz      (Use BV from survey data - e.g. APASS or SDSS)
    # - useGaiaAg    (Use Reddening as determined by Gaia)
    
    
    ############################################
    #    Getting stellar data from Exofop:     #
    ############################################
    if mission is not 'CoRoT':
        exofop_dat, kicinfo = GetExoFop(icid,mission)
        if exofop_dat is not None:
            tic_info=Catalogs.query_criteria(catalog="Tic",
                                            coordinates=str(exofop_dat['ra'])+','+str(exofop_dat['dec']),
                                            radius=20*u.arcsecond,objType="STAR").to_pandas()
        else:
            tic_info=Catalogs.query_criteria(catalog="Tic",
                                            coordinates=str(kicinfo['ra'])+','+str(kicinfo['dec']),
                                            radius=20*u.arcsecond,objType="STAR").to_pandas()
        #In the case where we only get KIC info, we just call this "ExoFop" too:
        if exofop_dat is None and kicinfo is not None:
            exofop_dat=kicinfo
            exofop_dat['mission']='kep_or_k2'
    else:
        corot_dat=pd.read_csv("CorotLCs/CorotSinglesFixed.csv")
        corot_dat=corot_dat.loc[corot_dat['ID']==icid].iloc[0]
        exofop_dat=Catalogs.query_criteria(catalog="Tic",coordinates=str(corot_dat['ra'])+','+str(corot_dat['dec']),
                                       radius=20*u.arcsecond,objType="STAR").to_pandas()
        exofop_dat=exofop_dat.append(corot_dat)
        kicinfo=None
    ############################################
    #   Getting survey data from [Various]:    #
    ############################################
    
    if use_surveys:
        #Loading RA and Dec:
        if type(exofop_dat['ra'])==str and (exofop_dat['ra'].find(':')!=-1)|(exofop_dat['ra'].find('h')!=-1):
            coor=SkyCoord(exofop_dat['ra'],exofop_dat['dec'],unit=(u.hourangle,u.deg))
        elif (type(exofop_dat['ra'])==float)|(type(exofop_dat['ra'])==np.float64) or (type(exofop_dat['ra'])==str)&(exofop_dat['ra'].find(',')!=-1):
            coor=SkyCoord(exofop_dat['ra'],exofop_dat['dec'],unit=u.deg)

        #Getting TIC, Spec and survey data:
        #tic_dat = Catalogs.query_criteria("TIC",coordinates=coor,radius=20/3600,objType="STAR")#This is not used, as TIC is on ExoFop
        spec_dat = CheckSpecCsv(coor,icid)
        survey_dat=QueryGaiaAndSurveys(coor,mission=mission)

        order_of_kw_to_remove=['useGaiaAg','useGriz','useBV','useGaiaBR','use2mass','useGaiaSpec','useGaiaLum']
        n_kw_to_remove=0
        isoclass_dens_is_NaN=True
        #Isoclass often fails, e.g. due to photometry. So let's loop through the kwargs and gradually remove contraints:
        while isoclass_dens_is_NaN and n_kw_to_remove<=len(order_of_kw_to_remove):
            kwars={order_of_kw_to_remove[nkw]:(True if nkw>=n_kw_to_remove else False) for nkw in range(len(order_of_kw_to_remove))}
            #print(n_kw_to_remove,'/',len(order_of_kw_to_remove),kwars)
            #try:
            if 1==1:
                isoclass_df, paras = Assemble_and_run_isoclassify(icid,coor,mission,survey_dat,exofop_dat,
                                                   errboost=errboost*(1+0.33*n_kw_to_remove),spec_dat=spec_dat,**kwars)
                #print(isoclass_df[['rho_gcm3','rho_gcm3ep','rho_gcm3em']])
                isoclass_dens_is_NaN=(np.isnan(isoclass_df[['rho_gcm3','rho_gcm3ep','rho_gcm3em']]).any())|(isoclass_df[['rho_gcm3','rho_gcm3ep','rho_gcm3em']]==0.0).all()
            #except Exception as e:
            #    exc_type, exc_obj, exc_tb = sys.exc_info()
            #    print(exc_type, exc_tb.tb_lineno)
            #    isoclass_df,paras=None,None
                #print(n_kw_to_remove,'|',isoclass_df)
            n_kw_to_remove+=1
        #Assessing which available data source is the *best* using lowest density error
    else:
        survey_dat=None
        isoclass_df=None
        paras=None
    if isoclass_df is not None:
        isoclass_err_rho=(0.5*(abs(isoclass_df['rho_gcm3ep'])+abs(isoclass_df['rho_gcm3em'])))/isoclass_df['rho_gcm3']
    else:
        isoclass_err_rho=100
    #Calculating 
    if 'rho' not in exofop_dat.index and 'rho_gcm3' not in exofop_dat.index and 'rad' in exofop_dat.index and (('mass' in exofop_dat.index)|('logg' in exofop_dat.index)):
        #Getting Density from R, M and logg:
        rhos=[];rhoems=[];rhoeps=[];rhoes=[]
        if 'mass' in exofop_dat.index:
            rhos+=[1.411*exofop_dat['mass']/exofop_dat['rad']**3]
            rhoeps+=[1.411*(exofop_dat['mass']+exofop_dat['massep'])/((exofop_dat['rad']-exofop_dat['radem'])**3)-rhos[-1]]
            rhoems+=[rhos[-1] - 1.411*(exofop_dat['mass']-exofop_dat['massem'])/((exofop_dat['rad']+exofop_dat['radep'])**3)]
            rhoes+=[0.5*(abs(rhoeps[-1])+abs(rhoems[-1]))]
        if 'logg' in exofop_dat.index:
            rhos+=[np.power(10,exofop_dat['logg']-4.4377)/exofop_dat['rad']]
            rhoeps+=[np.power(10,(exofop_dat['logg']+exofop_dat['loggep'])-4.4377)/(exofop_dat['rad']-exofop_dat['radem'])-rhos[-1]]
            rhoems+=[rhos[-1]-np.power(10,(exofop_dat['logg']-exofop_dat['loggem'])-4.4377)/(exofop_dat['rad']+exofop_dat['rad'])]
            rhoes+=[0.5*(abs(rhoeps[-1])+abs(rhpems[-1]))]
        rhos=np.array(rhos)
        rhoes=np.array(rhoes)
        exofop_dat['rho_gcm3']=rhos[np.argmin(rhoes)]
        exofop_dat['rho_gcm3e']=np.min(rhoes)
        exofop_dat['rho_gcm3em']=rhoems[np.argmin(rhoes)]
        exofop_dat['rho_gcm3ep']=rhoeps[np.min(rhoes)]
        exofop_dat['rho']=exofop_dat['rho_gcm3']/1.411
        exofop_dat['rhoe']=exofop_dat['rho_gcm3e']/1.411
        exofop_dat['rhoem']=exofop_dat['rho_gcm3em']/1.411
        exofop_dat['rhoep']=exofop_dat['rho_gcm3ep']/1.411
    elif 'rho' in exofop_dat.index and 'rho_gcm3' not in exofop_dat.index:
        exofop_dat['rho_gcm3']=exofop_dat['rho']*1.411
        exofop_dat['rho_gcm3e']=0.5*(abs(exofop_dat['rhoep'])+abs(exofop_dat['rhoem']))*1.411
    elif 'rho_gcm3' in exofop_dat.index and 'rho' not in exofop_dat.index:
        exofop_dat['rho']=exofop_dat['rho_gcm3']/1.411
        exofop_dat['rhoe']=0.5*(abs(exofop_dat['rho_gcm3ep'])+abs(exofop_dat['rho_gcm3em']))/1.411
        exofop_dat['rhoep']=exofop_dat['rho_gcm3ep']/1.411
        exofop_dat['rhoem']=exofop_dat['rho_gcm3em']/1.411
    elif 'rho_gcm3' not in exofop_dat.index:
        exofop_dat['rho_gcm3e']=100
        exofop_dat['rho_gcm3']=1
    if 'rho_gcm3em' in exofop_dat.index and 'rho_gcm3e' not in exofop_dat.index:
        exofop_dat['rho_gcm3e']=0.5*(abs(exofop_dat['rho_gcm3ep'])+abs(exofop_dat['rho_gcm3em']))
        #elif 
        #exofop_dat['rho_gcm3e']=0.5*(exofop_dat['rho_gcm3em']+exofop_dat['rho_gcm3ep'])
    #Calculating percentage error on density from exofop/input catalogues:
    if 'rho_gcm3' in exofop_dat.index and not np.isnan(float(exofop_dat['rho_gcm3'])):
        #Checking if there is also a Kepler Input catalogue file, and whether the quoted density error is lower:
        if kicinfo is not None and 'rho' in kicinfo.index:
            if (kicinfo['rho_gcm3e']/kicinfo['rho_gcm3'])<(exofop_dat['rho_gcm3e']/exofop_dat['rho_gcm3']):
                #Replacing data in exofop_dat with that from kicdat
                for col in kicinfo.index:
                    exofop_dat[col]=kicinfo[col]
                exofop_dat['source']='KIC'
        inputcat_err_rho=(exofop_dat['rho_gcm3e'])/exofop_dat['rho_gcm3']
    else:
        inputcat_err_rho=100
    print(inputcat_err_rho,exofop_dat['rho_gcm3e'],'<err | rho>',exofop_dat['rho_gcm3'])
    print('Density errors.  isoclassify:',isoclass_err_rho,', input cat:',inputcat_err_rho)
    if isoclass_df is not None and abs(exofop_dat['rho_gcm3']-isoclass_df['rho_gcm3']) > abs(0.5*(abs(isoclass_df['rho_gcm3ep'])+abs(isoclass_df['rho_gcm3em']))+exofop_dat['rho_gcm3e']):
        print('Densities disagree at >1-sigma | isoclassify:',isoclass_df['rho_gcm3'],0.5*(abs(isoclass_df['rho_gcm3ep'])+abs(isoclass_df['rho_gcm3em'])),'| input cat:',exofop_dat['rho_gcm3'],exofop_dat['rho_gcm3e'])
    
    #Now we know which is best, we put that best info into "best_df"
    best_df=pd.Series()
    if mission[0] in ['T','t']:
        best_df['tmag']=exofop_dat['tmag']
    elif mission[0] in ['K','k']:
        best_df['kepmag']=exofop_dat['kepmag']
    best_df['ra']=exofop_dat['ra']
    best_df['dec']=exofop_dat['dec']
    
    #selecting the best stellar parameter source from input cat vs isoclassify
    if isoclass_err_rho<inputcat_err_rho or np.isnan(inputcat_err_rho):
        #Generated Density better constrained by isoclassify:
        col_names=['teff','teffep','teffem','logg','loggep','loggem','lum','lumep','lumem',
                   'rad','radep','radem','mass','massep','massem','rho_gcm3','rho_gcm3ep','rho_gcm3em',
                   'dis','disep','disem']
        for col in col_names:
            best_df[col]=isoclass_df[col]
        best_df['source']='isoclassify'
    elif inputcat_err_rho<=isoclass_err_rho or np.isnan(isoclass_err_rho):
        #input catalogue info better constrained
        col_names=['teff','teffep','teffem','teffe','teff_prov','logg','loggep','loggem','logge','logg_prov',
                   'rad','radep','radem','rade','mass','massep','massem','masse'
                   'rho_gcm3','rho_gcm3e','rho_gcm3ep','rho_gcm3em','rho','rhoe','rhoep','rhoem']
        if 'av' in exofop_dat.index:
            col_names+=['avs','avsem','avsep']
        if 'feh' in exofop_dat.index:
            col_names+=['feh','fehem','fehep']
        for col in col_names:
            if col in exofop_dat.index:
                best_df[col]=exofop_dat[col]
        best_df['source']='input_catalogue'
    
    #Converting rho in gcm3 to rho in rho_s
    if 'rho_gcm3' in best_df.index:
        coldic={'rho_gcm3':'rho','rho_gcm3em':'rhoem','rho_gcm3ep':'rhoep'}
        for key in coldic:
            best_df[coldic[key]]=best_df[key]/1.411
            
    if return_best:
        return best_df
    else:
        return exofop_dat,survey_dat,isoclass_df,paras,best_df

    
def getStellarDensity(ID,mission,errboost=0.1):
    #Compiling dfs (which may have spectra)
    exofop_dat,_,isoclass_df,_,_=starpars(ID,mission,errboost=0.1,return_best=False)
    
    #Sorting out missing data and getting important info - Mass, Radius, density and logg:
    if pd.isnull(exofop_dat[['logg','mass']]).all() and ~np.isnan(exofop_dat['lum']):
        if 'lume' not in exofop_dat.index:
            exofop_dat['lume']=0.5*(abs(exofop_dat['lumem'])+exofop_dat['lumep'])
        #Mass-Luminosity relation for objects with no Mass but have Luminosity
        if ((~np.isnan(exofop_dat['rad']))&(exofop_dat['rad']<0.55))|((exofop_dat['teff']<5000)&(exofop_dat['lum']<0.3)):
            #M < 0.43
            exofop_dat['mass']=np.power(exofop_dat['lum']/0.23,1/2.3)
            exofop_dat['massep']=(1/2.3)*np.power(exofop_dat['lum']/0.23,1/2.3-1.0)*exofop_dat['lume']
            exofop_dat['massem']=-1*exofop_dat['massep']
            exofop_dat['logg']=np.power(10,exofop_dat['mass']/exofop_dat['rad']**2)+4.43
        elif best_stardf['teff']>8550:
            #2Ms < M < 55Ms
            exofop_dat['mass']=np.power(exofop_dat['lum']/1.4,1/3.5)
            exofop_dat['massep']=(1/3.5)*np.power(exofop_dat['lum']/1.4,1/3.5-1.0)*exofop_dat['lume']
            exofop_dat['massem']=-1*exofop_dat['massep']
        else:
            #0.43 < M < 2
            exofop_dat['mass']=np.power(exofop_dat['lum'],1/4)
            exofop_dat['massep']=(1/4)*np.power(exofop_dat['lum']/0.23,(1/4-1.0))*exofop_dat['lume']
            exofop_dat['massem']=-1*exofop_dat['massep']

    #compiling a logg array:
    if pd.isnull(exofop_dat['logg']) and ~pd.isnull(exofop_dat[['mass','rad']]).any():
        exofop_dat['logg']=np.array([np.log10(exofop_dat['mass']/exofop_dat['rad']**2)+4.438,0.5,0.5])
        exofop_dat['loggep']=np.array([np.log10((exofop_dat['mass']+exofop_dat['massep'])/(exofop_dat['rad']-exofop_dat['radem'])**2)+4.438,0.5,0.5])-exofop_dat['logg']
        exofop_dat['loggem']=exofop_dat['logg']-np.array([np.log10((exofop_dat['mass']-exofop_dat['massem'])/(exofop_dat['rad']+exofop_dat['radep'])**2)+4.438,0.5,0.5])
    
    #compiling a rho array:
    if not pd.isnull(exofop_dat['rho']) and exofop_dat['rho']!=0.0:
        rhos=np.array([exofop_dat['rho'],exofop_dat['rhoem'],exofop_dat['rhoep']])
    elif not np.isnan(exofop_dat[['logg','rad','mass']]).all():
        rhos=namaste.dens2(*exofop_dat[['logg','loggem','loggep','rad','radem','radep','mass','massem','massep']])
    else:
        rhos=None
        
    if isoclass_df is not None and not np.isnan(isoclass_df['rho']):
        rhos_iso=np.array([isoclass_df['rho'],isoclass_df['rhoem'],isoclass_df['rhoep']])
    else:
        rhos_iso=None
        
    return rhos,rhos_iso

def make_numeric(df):
    outcol=pd.DataFrame()
    for col in df.columns:
        try:
            outcol[col]=df[col].values.astype(float)
        except:
            outcol[col]=df[col].values
    return outcol

def MainSequenceFit(dist,V):
    #In the case where we only have distance and an optical magnitude, let's just make a guess with the bolometric magnitude
    
    #Loading fits of absolute V mag versus
    fits=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath( __file__ )),"BolMag_interpolations.models"),"rb"))
    
    M_V = V - 5*np.log(dist/10)
    info={'rad':fits[0](M_V),'mass':fits[1](M_V),'teff':fits[2](M_V)}
    info['eneg_rad']=0.5*info['rad']
    info['epos_rad']=0.5*info['rad']
    info['eneg_mass']=0.5*info['mass']
    info['epos_mass']=0.5*info['mass']
    info['eneg_teff']=0.2*info['teff']
    info['epos_teff']=0.2*info['teff']
    return info

def getStellarInfoFromCsv(ID,mission,k2tab=None):
    # Kepler on Vizier J/ApJ/866/99/table1
    # New K2: "http://kevinkhu.com/table1.txt"
    # TESS on Vizier (TICv8)
    if mission.lower()=='tess':
        info = TICdata(ID).iloc[0]
        print("TESS object",info)
        k2tab = None
    else:
        if mission.lower()=='kepler':
            info = Vizier(catalog='J/ApJ/866/99/table1').query_constraints(KIC=int(ID))[0].to_pandas().iloc[0]
            info['mission']='Kepler'

            info=info.rename(index={'KIC':'ID','Gaia':'GAIA',
                                     'Dist':'dist','E_Dist':'epos_dist','e_Dist':'eneg_dist',
                                     'R_':'rad','E_R_':'epos_Rad', 'e_R_':'eneg_Rad',
                                     '_RA':'ra','_DE':'dec'})
            info['eneg_Teff']=info['e_Teff']
            info['epos_Teff']=info['e_Teff']
            info['e_d']=0.5*(abs(info['epos_dist'])+abs(info['eneg_dist']))
            info['e_rad']=0.5*(abs(info['eneg_Rad'])+abs(info['epos_Rad']))

            radec = SkyCoord(info['ra']*u.deg,info['dec']*u.deg)
            print("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_supp_stellar&select=kepid,teff,teff_err1,teff_err2,teff_prov,logg,logg_err1,logg_err2,feh,radius,radius_err1,radius_err2,mass,mass_err1,mass_err2&order=kepid&format=csv&where=kepid=%27"+str(int(ID))+"%27")
            xtra=pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_supp_stellar&select=kepid,teff,teff_err1,teff_err2,teff_prov,logg,logg_err1,logg_err2,feh,radius,radius_err1,radius_err2,mass,mass_err1,mass_err2&format=csv&where=kepid=%27"+str(int(ID))+"%27").iloc[0]
            print(xtra['radius'],info['rad'],xtra['radius_err1'])
            if (xtra['radius']-info['rad'])<(0.5*xtra['radius_err1']):
                info['mass']=xtra['mass']
                info['eneg_Mass']=xtra['mass_err1']
                info['epos_Mass']=xtra['mass_err2']
                info['e_mass']=0.5*(abs(info['eneg_Mass'])+abs(info['epos_Mass']))
                info['logg']=xtra['logg']
                info['eneg_logg']=xtra['logg_err1']
                info['epos_logg']=xtra['logg_err2']
                info['e_logg']=0.5*(abs(info['eneg_logg'])+abs(info['epos_logg']))

                info['feh']=xtra['feh']
                
            k2tab=None
        elif mission.lower()=='k2':
            if k2tab is None:
                k2tab=ascii.read('http://kevinkhu.com/table1.txt').to_pandas()
            info = k2tab.loc[k2tab['EPIC']==int(ID)].iloc[0]
            info['mission']='K2'
            info=info.rename(index={'EPIC':'ID','Gaia':'GAIA',
                                      'Dist':'dist','E_Dist':'epos_dist','e_Dist':'eneg_dist',
                                      'Mstar':'mass','E_Mstar':'eposmMass','e_Mstar':'eneg_mass',
                                      'Rstar':'rad','E_Rstar':'epos_rad', 'e_Rstar':'eneg_rad',
                                      'E_logg':'epos_logg', 'e_logg':'eneg_logg',
                                      '[Fe/H]':'feh','E_[Fe/H]':'epos_feh','e_[Fe/H]':'eneg_feh'})
            info['eneg_Teff']=info['e_Teff']
            info['epos_Teff']=info['e_Teff']
            info['e_d']=0.5*(abs(info['epos_dist'])+abs(info['eneg_dist']))
            info['e_rad']=0.5*(abs(info['eneg_Rad'])+abs(info['epos_Rad']))
            info['e_mass']=0.5*(abs(info['eneg_Mass'])+abs(info['epos_Mass']))
            info['e_logg']=0.5*(abs(info['eneg_logg'])+abs(info['epos_logg']))
            print(info['GAIA'])
            if not np.isnan(info['GAIA']):
                gaiainfo=[]
                #DR2:
                gaiainfo=Gaia.launch_job_async("SELECT * \
                                                  FROM gaiadr2.gaia_source \
                                                  WHERE gaiadr2.gaia_source.source_id="+str(info['GAIA'])
                                                 ).results
                if len(gaiainfo)==0:
                    #Try DR1
                    gaiainfo=Gaia.launch_job_async("SELECT * \
                                                      FROM gaiadr1.gaia_source \
                                                      WHERE gaiadr1.gaia_source.source_id="+str(info['GAIA'])
                                                     ).results
                gaiainfo=gaiainfo.to_pandas().iloc[0]
                #EPIC has no RA/Dec so we do a Gaia query
                radec = SkyCoord(gaiainfo['ra']*u.deg,gaiainfo['dec']*u.deg)
            else:
                kicdat=pd.DataFrame.from_csv("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2targets&where=epic_number=%27"+str(int(ID))+"%27")
                if len(kicdat.shape)>0:
                    if len(kicdat.shape)>1:
                        kicdat=kicdat.iloc[0]
                    radec=SkyCoord(kicdat['rastr'],kicdat['decstr'],unit=(u.hourangle,u.deg))
                    info['ra']=radec.ra.deg
                    info['dec']=radec.dec.deg

        #Now getting TIC, which may be present for the above too
        try:
            tic_dat = Catalogs.query_criteria("TIC",coordinates=radec,radius=15/3600,objType="STAR")#This is not used, as TIC is on ExoFop
            tic_dat=tic_dat.iloc[np.argmin(tic_dat['Tmag'])]
        except:
            print(ID," not in TIC")
            tic_dat=None
        if tic_dat is not None and (tic_dat['e_rad']/tic_dat['rad'])<(info['e_rad']/info['rad']):
            #tic dat is better, we should use ticdat:
            info=tic_dat
        elif tic_dat is not None:
            for key in tic_dat:
                if key not in info:
                    info[key]=tic_dat[key]
    
    #Switching out capitals:
    if 'd' in info.index:
        info=info.rename(index={'d':'dist'})
    #Switching all e_ and E_ to eneg and epos:
    for col in ['teff','rad','mass','dist','logg','rho']:
        captd=col[0].upper()+col[1:]
        if captd in info.index:
            info=info.rename(index={captd:col})
        if 'eneg_'+captd in info.index and 'eneg_'+col not in info.index:
            info=info.rename(index={'eneg_'+captd:'eneg_'+col})
        elif 'e_'+captd in info.index and 'eneg_'+col not in info.index:
            info['eneg_'+col]=info['e_'+captd]
        elif 'e_'+col in info.index and 'eneg_'+col not in info.index:
            info['eneg_'+col]=info['e_'+col]
        if 'epos_'+captd in info.index:
            info=info.rename(index={'epos_'+captd:'epos_'+col})
        elif 'E_'+captd in info.index and 'epos_'+col not in info.index:
            info=info.rename(index={'E_'+captd:'epos_'+col})
        elif 'E_'+col in info.index and 'epos_'+col not in info.index:
            info=info.rename(index={'E_'+col:'epos_'+col})
        elif 'e_'+captd in info.index and 'epos_'+col not in info.index:
            info['epos_'+col]=info['e_'+captd]
        elif 'e_'+col in info.index and 'epos_'+col not in info.index:
            info['epos_'+col]=info['e_'+col]
        
        if 'eneg_'+col not in info.index and col in info:
            #NO ERRORS PRESENT???
            info['eneg_'+col]=info[col]*0.5
        if 'epos_'+col not in info.index and col in info:
            info['epos_'+col]=info[col]*0.5
        
    if 'rad' not in info:
        try:
            print("#Using Isoclassify to attempt to get star parameters:")
            Rstar, rhos, Teff, logg, src = getStellarInfo(ID,info,mission)
            info['rad']=Rstar[0]
            info['eneg_rad']=Rstar[1]
            info['epos_rad']=Rstar[2]
            info['teff']=Teff[0]
            info['eneg_teff']=Teff[1]
            info['epos_teff']=Teff[2]
            info['logg']=logg[0]
            info['eneg_logg']=logg[1]
            info['epos_logg']=logg[2]
            info['rhos']=rhos[0]
            info['eneg_rhos']=rhos[1]
            info['epos_rhos']=rhos[2]
        except:
            print("getStellarInfo fails")
            
    if 'rad' not in info or np.isnan(info['rad']) or info['rad'] is None:
        if 'dist' in info.index:
            nm=0;nomag=True
            mags=['Vmag','V','GAIAmag','kepmag','Tmag']
            while nomag:
                if mags[nm] in info.index:
                    msinfo=MainSequenceFit(info[mags[nm]],info['dist'])
                    for col in msinfo:
                        info[col]=msinfo[col]
                    nomag=False

    if 'mass' not in info:
        #Problem here - need mass. Derive from Teff and Rs, plus Dist vs. mag?
        info['mass']=np.nan
        info['eneg_mass']=np.nan
        info['epos_mass']=np.nan
        
    if 'logg' not in info:
        #Problem here - need logg. Derive from Teff and Rs, plus Dist vs. mag?
        if not np.isnan(info['mass']):
            info['logg']=np.power(10,info['mass']/info['rad']**2)-4.43
            info['eneg_logg']=info['logg']-(np.power(10,(info['mass']-info['eneg_mass'])/(info['rad']+info['eneg_rad'])**2)-4.43)
            info['epos_logg']=(np.power(10,(info['mass']+info['eneg_mass'])/(info['rad']-info['eneg_rad'])**2)-4.43)-info['logg']
        else:
            info['logg']=np.nan
            info['eneg_logg']=np.nan
            info['epos_logg']=np.nan

    if 'rho' not in info:
        #Problem here - need rho. Derive from Teff and Rs, plus Dist vs. mag?
        if 'mass' in info and 'rad' in info:
            info['rho']=1.411*(info['mass']/info['rad']**3)
            info['eneg_rho']=info['rho']-1.411*((info['mass']-abs(info['eneg_mass']))/(info['rad']+info['epos_rad'])**3)
            info['epos_rho']=1.411*((info['mass']+info['epos_mass'])/(info['rad']-abs(info['eneg_rad']))**3)-info['rho']

    return info, k2tab
        

def getStellarInfo(ID,hdr,mission,overwrite=False,fileloc=None,savedf=True,use_surveys=True):
    #Compiling dfs (which may have spectra)
    if not overwrite and fileloc is not None and os.path.exists(fileloc.replace('.csv','_best.csv')):
        print("Loading stellar params from file")
        exofop_dat=make_numeric(pd.read_csv(fileloc.replace('.csv','_exofop.csv'), index_col=0,header=None).T)
        survey_dat=make_numeric(pd.read_csv(fileloc.replace('.csv','_survey.csv'), index_col=0,header=None).T)
        isoclass_df=make_numeric(pd.read_csv(fileloc.replace('.csv','_isoclass.csv'), index_col=0,header=None).T)
        best_stardf=make_numeric(pd.read_csv(fileloc.replace(".csv","_best.csv"), index_col=0,header=None).T)
    else:
        exofop_dat,survey_dat,isoclass_df,paras,best_stardf=starpars(ID,mission,errboost=0.1,
                                                                     return_best=False,useGaiaLum=True,
                                                                     useGaiaBR=True,useGaiaSpec=True,
                                                                     useBV=True,use2mass=True,
                                                                     useGriz=True,useGaiaAg=True,
                                                                     use_surveys=use_surveys)
        if savedf and fileloc is not None:
            if exofop_dat is not None:
                exofop_dat.to_csv(fileloc.replace('.csv','_exofop.csv'))
            if survey_dat is not None:
                survey_dat.to_csv(fileloc.replace('.csv','_survey.csv'))
            if isoclass_df is not None:
                isoclass_df.to_csv(fileloc.replace('.csv','_isoclass.csv'))
            best_stardf.to_csv(fileloc.replace('.csv','_best.csv'))
            #tic_df.to_csv(fileloc.replace('.csv','_tic.csv'))
    #Taking brightest star if multiple:
    if type(best_stardf)==pd.DataFrame and best_stardf.shape[0]>1:
        print("stardf has shape:", best_stardf.shape)
        #Taking brightest star:
        if 'Tmag' in best_stardf.columns:
            print(best_stardf.Tmag)
            best_stardf=best_stardf.iloc[np.argmin(best_stardf.Tmag)]
        elif 'kepmag' in best_stardf.columns:
            print(best_stardf['kepmag'])
            best_stardf=best_stardf.iloc[np.argmin(best_stardf['kepmag'])]
        else:
            print(best_stardf.columns)
    elif type(best_stardf)==pd.DataFrame and best_stardf.shape[0]==1:
        #Pandas df -> Series
        best_stardf=best_stardf.iloc[0]

    #Sorting out missing data and getting important info - Mass, Radius, density and logg:
    if pd.isnull(best_stardf[['logg','mass']]).all() and ~np.isnan(best_stardf['lum']):
        if 'lume' not in best_stardf.index:
            best_stardf['lume']=0.5*(abs(best_stardf['lumem'])+best_stardf['lumep'])
        
        #Mass-Luminosity relation for objects with no Mass but have Luminosity
        if ((~np.isnan(best_stardf['rad']))&(best_stardf['rad']<0.55))|((best_stardf['teff']<5000)&(best_stardf['lum']<0.3)):
            #M < 0.43
            best_stardf['mass']=np.power(best_stardf['lum']/0.23,1/2.3)
            best_stardf['massep']=(1/2.3)*np.power(best_stardf['lum']/0.23,1/2.3-1.0)*best_stardf['lume']
            best_stardf['massem']=-1*best_stardf['massep']
            best_stardf['logg']=np.power(10,best_stardf['mass']/best_stardf['rad']**2)+4.43
        elif best_stardf['teff']>8550:
            #2Ms < M < 55Ms
            best_stardf['mass']=np.power(best_stardf['lum']/1.4,1/3.5)
            best_stardf['massep']=(1/3.5)*np.power(best_stardf['lum']/1.4,1/3.5-1.0)*best_stardf['lume']
            best_stardf['massem']=-1*best_stardf['massep']
        else:
            #0.43 < M < 2
            best_stardf['mass']=np.power(best_stardf['lum'],1/4)
            best_stardf['massep']=(1/4)*np.power(best_stardf['lum']/0.23,(1/4-1.0))*best_stardf['lume']
            best_stardf['massem']=-1*best_stardf['massep']
    if ~pd.isnull(best_stardf['rad']):
        Rstar=np.array([best_stardf['rad'], best_stardf['radem'],abs(best_stardf['radep'])])
    else:
        print("No Rs",best_stardf['rad'])
    
    if ~pd.isnull(best_stardf['teff']):
        Teff=np.array([best_stardf['teff'], best_stardf['teffem'],abs(best_stardf['teffep'])])
    else:
        print("No Teff",best_stardf['teff'])

    #compiling a logg array:
    if ~pd.isnull(best_stardf['logg']):
        logg=np.array([best_stardf['logg'], best_stardf['loggem'],best_stardf['loggep']])
    elif ~pd.isnull(best_stardf[['mass','rad']]).any():
        logg=np.array([np.log10(best_stardf['mass']/best_stardf['rad']**2)+4.438,0.5,0.5])
    else:
        logg=np.array([4,1,1])
    
    #compiling a rho array:
    if not pd.isnull(best_stardf['rho']) and best_stardf['rho']!=0.0:
        rhos=np.array([best_stardf['rho'],best_stardf['rhoem'],best_stardf['rhoep']])
    else:
        rhos=namaste.dens2(*best_stardf[['logg','loggem','loggep','rad','radem','radep','mass','massem','massep']])
    
    return Rstar, rhos, Teff, logg, best_stardf['source']