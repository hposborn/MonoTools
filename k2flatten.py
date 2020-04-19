import numpy as np
import hpo.planetlib as pl

def dopolyfit(win,d,ni,sigclip):
    base = np.polyfit(win[:,0],win[:,1],w=1.0/np.power(win[:,2],2),deg=d)
    #for n iterations, clip 3(?) sigma, redo polyfit
    for iter in range(ni):
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        offset = np.abs(win[:,1]-np.polyval(base,win[:,0]))/win[:,2]
        clippedregion = win[offset<sigclip,:]
        if (offset<sigclip).sum()>int(0.8*len(win[:, 0])):
            clippedregion = win[offset<sigclip,:]
        else:
            clippedregion=win[offset<np.average(offset)]
        base = np.polyfit(clippedregion[:,0],clippedregion[:,1],w=1.0/np.power(clippedregion[:,2],2),deg=d)
    return base

def formwindow(dat,cent,size,boxsize,gapthresh):
    winlowbound = np.searchsorted(dat[:,0],cent-size/2.)
    winhighbound = np.searchsorted(dat[:,0],cent+size/2.)
    boxlowbound = np.searchsorted(dat[:,0],cent-boxsize/2.)
    boxhighbound = np.searchsorted(dat[:,0],cent+boxsize/2.)
    if winhighbound == len(dat[:,0]):
        winhighbound -= 1
    highgap = dat[winhighbound,0] < (cent+size/2.)-gapthresh #uses a 1d threshold for gaps
    lowgap = dat[winlowbound,0] > (cent-size/2.)+gapthresh
    flag = 0
    if highgap:
        if lowgap:
            flag = 1
        else:
            winlowbound = np.searchsorted(dat[:,0],dat[winhighbound,0]-size)
    else:
        if lowgap:
            winhighbound =  np.searchsorted(dat[:,0],dat[winlowbound,0]+size)
    window = np.concatenate((dat[winlowbound:boxlowbound,:],dat[boxhighbound:winhighbound,:]))
    box = dat[boxlowbound:boxhighbound,:]
    return window,boxlowbound,boxhighbound


if __name__ == '__main__':
    RedFromFile(sys.argv[1], sys.argv[2])

def ReadFromFile(fname):
    outlc=ReduceNoise(pl.ReadLC(fname))
    np.savetxt(fnameout, outlc)

def ReduceNoise(lc2, winsize = 2, stepsize = 0.2, polydegree = 3, niter = 20, sigmaclip = 3., gapthreshold = 1.0 ):
    '''set up flattening parameters
    winsize = 2   #days, size of polynomial fitting region
    stepsize = 0.2  #days, size of region within polynomial region to detrend
    polydegree = 3  #degree of polynomial to fit to local curve
    niter = 20      #number of iterations to fit polynomial, clipping points significantly deviant from curve each time.
    sigmaclip = 3.   #significance at which points are clipped (as niter)
    gapthreshold = 1.0  #days, threshold at which a gap in the time series is detected and the local curve is adjusted to not run over it
    '''
    lc=np.copy(lc2)
    
    lc[:,1]+=(1.0-np.nanmedian(lc[:,1]))
    lcdetrend=np.column_stack((lc[:, 0], np.zeros(len(lc[:, 0])), lc[:, 2]))
    #general setup
    JD0 = lc[0,0]
    lc[:,0] = lc[:,0] - JD0
    #lcdet[:,0] = lcdet[:,0] - JD0
    lenlc = lc[-1,0]
    lcbase = np.median(lc[:,1])
    lc[:,1] /= lcbase
    lc[:,2] = lc[:,2]/lcbase
    nsteps = np.ceil(lenlc/stepsize).astype('int')
    stepcentres = np.arange(nsteps)/float(nsteps) * lenlc + stepsize/2.

    #print nsteps
    #for each step centre:
    #actual flattening
    for s in range(nsteps):
        stepcent = stepcentres[s]
        winregion,boxlowbound,boxhighbound = formwindow(lc,stepcent,winsize,stepsize,gapthreshold)  #should return window around box not including box
        baseline = dopolyfit(winregion,polydegree,niter,sigmaclip)
        lcdetrend[boxlowbound:boxhighbound, 1] = lc[boxlowbound:boxhighbound,1] - np.polyval(baseline,lc[boxlowbound:boxhighbound,0]) + 1

     #   winregiondet,boxlowbounddet,boxhighbounddet = formwindow(lcdet,stepcent,winsize,stepsize,gapthreshold)  #should return window around box not including box
     #   baselinedet = dopolyfit(winregiondet,polydegree,niter,sigmaclip)
        
    #    lcdetdetrend[boxlowbounddet:boxhighbounddet] = lcdet[boxlowbounddet:boxhighbounddet,1] - np.polyval(baselinedet,lcdet[boxlowbounddet:boxhighbounddet,0]) + 1
    return lcdetrend
