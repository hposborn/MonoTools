
def CheckMonoPairs_old(lc_time, all_pls):
    #Loop through each pair of monos without a good period, and check:
    # - if they correspond in terms of depth/duration
    # - and whether they could be periodic given other data
    all_monos=[pl for pl in all_pls if (all_pls[pl]['orbit_flag']=='mono')&(all_pls[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability','FP - confusion'])]
    all_others=[pl for pl in all_pls if (all_pls[pl]['orbit_flag'] in ['periodic', 'duo'])&(all_pls[pl]['flag'] not in ['asteroid','EB','instrumental','lowSNR','variability','FP - confusion'])]
    
    print("monos:",all_monos,"others",{m:all_pls[m]['orbit_flag'] for m in all_others})
    
    import itertools
    if len(all_monos)>1:
        pairs=list(itertools.combinations(range(len(all_monos)), 2))
        pair_match_proxs=[]
        pair_match_Pmins=[]
        for p in pairs:
            # Compute statistic of how closely the two transits match:
            pl0=all_monos[p[0]]
            pl1=all_monos[p[1]]
            
            prox = (np.log(all_pls[pl0]['depth'])-np.log(all_pls[pl1]['depth']))**2/0.25**2+\
                    (np.log(all_pls[pl0]['tdur'])-np.log(all_pls[pl1]['tdur']))**2/0.2**2
            pair_match_proxs+=[prox]
            if prox<2:
                #match. Check if periodicity is possible, or is observed, and compute minimum P
                P_guess=abs(all_pls[pl0]['tcen']-all_pls[pl1]['tcen'])
                print(pl0,pl1,P_guess,prox)
                lc_cut = (abs(lc_time-all_pls[pl0]['tcen'])>(0.5*all_pls[pl0]['tdur']))&(abs(lc_time-all_pls[pl1]['tcen'])>(0.5*all_pls[pl1]['tdur']))
                n_trans_pnts_observed=0;niter=0
                #Loop through each possible period alias until we find one which produces >2 in-transit points
                while n_trans_pnts_observed <= 2 and niter<50:
                    niter+=1
                    newP=P_guess/niter
                    data_phases=(lc_time[lc_cut]-all_pls[pl0]['tcen'])%newP
                    n_trans_pnts_observed = np.sum((data_phases<0.4*all_pls[pl0]['tdur'])|(data_phases>(newP-0.4*all_pls[pl0]['tdur'])))
                    #lc[(>ix_in_trans[0])&(lc[:,0]<ix_in_trans[1]),0])
                
                if niter-1==0:
                    # In this case, even the transit-to-transit period seems to be ruled out by other data...
                    # This could mean:
                    # - these are two seperate mono transits...
                    # - This is an EB and one is a secondary... We will check this with Rp/Rs later, but assume they are monos for now
                    # - this is periodic at P=P_guess, and has a third transit is in the LC!
                    pair_match_Pmins+=[-99]
                else:
                    pair_match_Pmins+=[P_guess/(niter-1)]
            else:
                pair_match_Pmins+=[-100]
        pair_match_proxs=np.array(pair_match_proxs)
        pair_match_Pmins=np.array(pair_match_Pmins)
        #In this case we have multiple matches...
        match_pairs=np.array([pairs[i] for i in np.where((pair_match_proxs*pair_match_Pmins)>0)[0]])
        if len(match_pairs)>0:
            print("we have matching >1 monotransit pair - ",match_pairs)
            if len(match_pairs.ravel())==len(np.unique(match_pairs.ravel())):
                #The planets which match are seperate - no confusion here!
                npairs_to_match=range(len(match_pairs))
            else:
                #Confusion - some planets are matched twice! Take the best match
                doubles=[i for i, x in enumerate(match_pairs.ravel()) if list(match_pairs.ravel()).count(x) > 1]
                sort_on_prox=pair_match_Pmins[np.where((pair_match_proxs*pair_match_Pmins)>0)[0]].argsort()
                npairs_to_match=[]
                for n in sort_on_prox:
                    if len(npairs_to_match)==0:
                        npairs_to_match+=[n]
                    else:
                        already_in_ptm = np.array([match_pairs[npm] for npm in npairs_to_match]).ravel()
                        if not np.in1d(match_pairs[n],already_in_ptm).any():
                            #the planets in this matching pair have not already been added to the list of those planets to conjoin,
                            # ... so we're free to add them
                            npairs_to_match+=[n]
            print(npairs_to_match)
            for n in npairs_to_match:
                #For each identified monotransit pair, let's combine them into a single planet in the dict:
                n_pair=np.where((pair_match_proxs*pair_match_Pmins)>0)[0][n]
                keys=[all_monos[match_pairs[n][0]],all_monos[match_pairs[n][1]]]
                all_pls[keys[0]]=deepcopy(all_pls[keys[0]])
                for key in all_pls[keys[0]]:
                    if key in all_pls[keys[1]]:
                        if key is 'period':
                            print("tcens = ",all_pls[keys[0]]['tcen'],all_pls[keys[1]]['tcen'])
                            all_pls[keys[0]]['period']=abs(all_pls[keys[0]]['tcen']-all_pls[keys[1]]['tcen'])
                        elif key is 'snr':
                            all_pls[keys[0]]['snr']=np.hypot(all_pls[keys[0]]['snr'],all_pls[keys[1]]['snr'])
                        elif type(all_pls[keys[1]][key])==float:
                            #Average of two:
                            all_pls[keys[0]][key]=0.5*(all_pls[keys[0]][key]+all_pls[keys[1]][key])
                all_pls[keys[0]]['tcen_2']=all_pls[keys[1]]['tcen']
                all_pls[keys[0]]['orbit_flag']='duo'
                all_pls[keys[0]]['P_min']=pair_match_Pmins[n_pair]
                all_pls[keys[1]]['orbit_flag']='FP - Confusion with '+keys[0]
        # The returns planets have either been conjoined into "duo" planet pairs (ie no longer monos) with period and minimum period computed
        # Or we have tested each pair and found that either the do not seem closely matched, or when combined they suggest some data already excludes their affinity
    return all_pls
    
    
def MonoTransitSearch_old(lc,ID,sigma_threshold=4,mod_per_tdur=50,use_flat=True,use_binned=True,binsize=1/96.,
                          Rs=None,Ms=None,Teff=None,n_oversamp=75,plot=False,plot_loc=None):
    #Searches LC for monotransits
    print("Searching "+str(ID)+" for monotransits")
    if use_flat and 'flux_flat' not in lc:
        #Flattening but protecting the typical durations:
        if Rs is not None:
            typ_dur=1.5*((np.nanmax(lc['time'])-np.nanmin(lc['time']))/(18226*Rs**-2))**(1/3)
        else:
            typ_dur=0.4
        lc=lcFlatten(lc,winsize=typ_dur*7.5,stepsize=typ_dur*0.15)
    if use_binned:
        lc=lcBin(lc,binsize=binsize,use_flat=use_flat)
    flux_key = 'flux_flat' if use_flat else 'flux'
        
    #Computing a fine x-range to search:
    search_xrange=[]

    cad=np.nanmedian(np.diff(lc['time']))
    
    Rs=1.0 if Rs is None else float(Rs)
    Ms=1.0 if Ms is None else float(Ms)
    Teff=5800.0 if Teff is None else float(Teff)

    interpmodels,tdurs=get_interpmodels(Rs,Ms,Teff,lc['time'],lc['flux_unit'])

    #Removing gaps bigger than 2d (with no data)
    for arr in np.array_split(lc['time'][lc['mask']],1+np.where(np.diff(lc['time'][lc['mask']])>tdurs[2])[0]):
        search_xrange+=[np.arange(arr[0]+0.66*tdurs[0],arr[-1]-0.66*tdurs[0],tdurs[1]/n_oversamp)]
    search_xrange=np.hstack((search_xrange))

    #Making depth vary from 0.1 to 1.0
    search_dep_shifts=np.exp(np.random.normal(0.0,n_oversamp*0.005,len(search_xrange)))

    #Making duration vary around primary duration with sd=0.3
    search_dur_shifts=np.exp(np.random.normal(0.0,n_oversamp*0.01,len(search_xrange)))

    #Looping through search and computing chi-sq at each position:
    outparams=[]

    def trans_model_neglnlik(pars,x,y,sigma2,tcen,initdur,initdep,interpmodel):
        # pars = depth, duration, poly1, poly2
        #Returns chi-squared for transit model:
        model=(pars[0]/initdep)*interpmodel(np.clip((x-tcen)*(initdur/pars[1]),-9999,9999))
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    def trans_model_wtrend_neglnlik(pars,x,y,sigma2,tcen,initdur,initdep,interpmodel):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = depth, duration, poly1, poly2
        model=np.polyval(pars[-2:],x-tcen)+(np.clip(pars[0],0.0,1000)/(initdep))*interpmodel((x-tcen)*(initdur/pars[1]))
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
    
    def sin_model_wtrend_neglnlik(pars,x,y,sigma2,tcen):
        #Returns chi-squared for transit model, plus linear background flux trend
        # pars = depth, duration, poly1, poly2
        model=np.polyval(pars[-2:],x-tcen)
        newt=(x-tcen)/pars[1]*(np.pi*2)
        amp=np.exp(-1*np.power(newt, 2.) / (2 * np.power(np.pi, 2.)))
        model=np.polyval(pars[-2:],x-tcen)+pars[0]*amp*np.sin(newt-np.pi*0.5)
        return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    #What is the probability of transit given duration (used in the prior calculation)
    p_transit=0.5*tdurs[1]/(lc['time'][-1]-lc['time'][0])
    methods=['SLSQP','Nelder-Mead','Powell']
    deps=[abs(interpmodels[n%3](0.0)) for n in range(3)]
    for n_mod,x2s in enumerate(search_xrange):
        randn=np.random.randint(2,size=2)
        #minimise two params
        round_tr=lc['mask']&(abs(lc['time']-x2s)<(5*tdurs[n_mod%3]))
        in_tr=abs(lc['time'][round_tr]-x2s)<(0.4*tdurs[n_mod%3])
        if not np.isnan(lc[flux_key][round_tr][in_tr]).all():
            init_poly=np.polyfit(lc['time'][round_tr][~in_tr]-x2s,lc[flux_key][round_tr][~in_tr],2)
            init_dep=np.nanmedian(lc[flux_key][round_tr][~in_tr])-np.nanmedian(lc[flux_key][round_tr][in_tr])
            sigma2=lc['flux_err'][round_tr]**2
            '''
            res1=optim.minimize(trans_model_neglnlik, 
                                (init_dep*search_dep_shifts[n_mod],tdurs[n_mod%3]*search_dur_shifts[n_mod]),
                                args=(lc['time'][round_tr],
                                      lc[flux_key][round_tr],sigma2,
                                      x2s,tdurs[n_mod%3],interpmodels[n_mod%3](0.0),interpmodels[n_mod%3])
                               )
            '''
            res_trans=optim.minimize(trans_model_wtrend_neglnlik, 
                                (init_dep*search_dep_shifts[n_mod],
                                 tdurs[n_mod%3]*search_dur_shifts[n_mod],
                                 init_poly[0],init_poly[1]),
                                args=(lc['time'][round_tr],lc[flux_key][round_tr],sigma2,x2s,
                                      tdurs[n_mod%3],deps[n_mod%3],interpmodels[n_mod%3]),
                                method = methods[randn[0]])
            res_sin=optim.minimize(sin_model_wtrend_neglnlik, 
                                (init_dep*search_dep_shifts[n_mod], tdurs[n_mod%3]*search_dur_shifts[n_mod],
                                 init_poly[0],init_poly[1]),
                                args=(lc['time'][round_tr],lc[flux_key][round_tr],sigma2,x2s),
                                method = methods[randn[1]])

            #print(np.shape(lc[flux_key][round_tr]),np.shape(lc['time'][round_tr]),np.shape(lc[flux_key][round_tr]),np.shape(np.polyfit(lc['time'][round_tr],lc[flux_key][round_tr],1)),np.shape(lc['flux_err'][round_tr]**2))
            flat_model=np.polyval(np.polyfit(lc['time'][round_tr],lc[flux_key][round_tr],1),lc['time'][round_tr])
            flat_neg_llik=0.5 * np.sum((lc[flux_key][round_tr] - flat_model)**2 / sigma2 + np.log(sigma2))
            #(2*best_dip_res.fun + np.log(np.sum(roundTransit*lc['mask']))*2*order+2+2)
            log_len=np.log(np.sum(round_tr))

            #BIC = log(n_points)*n_params - 2*(log_likelihood + log_prior)
            #BIC = log(n_points)*n_params + 2*(neg_log_likelihood-log_prior). 
            # Setting log prior of transit as sum of:
            #   - 0.5*tdur/len(x)  - the rough probability that, given some point in time, it's in the centre of a transit
            #   X - normal prior on log(duration) to be within 50% (0.5 in logspace)
            #   X   +(np.log(res_trans.x[1]) - np.log(tdurs[n_mod%3]))**2/0.5
            BICs=[log_len*len(res_trans.x)+2*(res_trans.fun-np.log(p_transit)),
                  log_len*len(res_sin.x)+2*(res_sin.fun-np.log(p_transit)),
                  log_len*len(init_poly)+2*(flat_neg_llik-np.log(1-2*p_transit))]
            outparams+=[[x2s,BICs[0],BICs[1],BICs[2],
                         init_dep,tdurs[n_mod%3],init_poly[0],init_poly[1],
                         np.clip(res_trans.x[0],0.0,1000),res_trans.x[1],res_trans.x[2],res_trans.x[3],
                         res_sin.x[0],res_sin.x[1],res_sin.x[2],res_sin.x[3],
                         n_mod%3,int(res_trans.success),int(res_sin.success),int(res_trans.success&res_sin.success)]]
        else:
            outparams+=[[np.nan,np.nan,np.nan,
                         init_dep,tdurs[n_mod%3],init_poly[0],init_poly[1],
                         np.nan,np.nan,np.nan,np.nan,
                         np.nan,np.nan,np.nan,np.nan,
                         n_mod%3,0.0,0.0,0.0]]
        #if n_mod%1000==945:
        #    print(n_mod,BICs[-1],outparams[-1])      
    outparams=pd.DataFrame(np.array(outparams),columns=['tcen','BIC_trans','BIC_sin','BIC_poly',
                                                        'init_dep','init_dur','init_poly_a','init_poly_b',
                                                        'trans_dep','trans_dur','trans_poly_a','trans_poly_b',
                                                        'sin_dep','sin_dur','sin_poly_a','sin_poly_b',
                                                        'n_mod','tran_success','sin_success','all_success'])
    
    #Negative depth & even the "worst" DeltaBIC for the transit model over both sinusoid & polynomial is significant
    outparams['worstBIC']=np.max([outparams['BIC_trans'].values-outparams['BIC_sin'].values,
                                  outparams['BIC_trans'].values-outparams['BIC_poly'].values],axis=0)
    signfct=np.where(((outparams['worstBIC'])<-0)&(outparams['trans_dep']>0.0))[0]
    
    if plot:
        fig, ax1 = plt.subplots(figsize=(12,5))
        ax1.set_title(str(ID).zfill(11)+" - Monotransit search")
        ax1.plot(lc['time'][lc['mask']],(lc[flux_key][lc['mask']]-np.nanmedian(lc[flux_key]))*lc['flux_unit'],',k')
        if 'bin_time' not in lc:
            if flux_key=='flux_flat':
                lc=lcBin(lc,30/1440,use_masked=True,use_flat=True)
            else:
                lc=lcBin(lc,30/1440,use_masked=True)
        ax1.plot(lc['bin_time'],(lc['bin_flux']-np.nanmedian(lc['bin_flux']))*lc['flux_unit'],'.k',alpha=0.6)
        ax2 = ax1.twinx()
        ax2.scatter(outparams.loc[outparams['tran_success']==0.0,'tcen'],outparams.loc[outparams['tran_success'].values==0.0,'worstBIC'],
                    c=outparams.loc[outparams['tran_success']==0.0,'trans_dur'],
                    s=3+np.sqrt(2000*abs(outparams.loc[outparams['tran_success']==0.0,'trans_dep'])),alpha=0.15)

        ax2.scatter(outparams.loc[outparams['tran_success']==1.0,'tcen'],outparams.loc[outparams['tran_success'].values==1.0,'worstBIC'],
                    c=outparams.loc[outparams['tran_success']==1.0,'trans_dur'],
                    s=3+np.sqrt(2000*abs(outparams.loc[outparams['tran_success']==1.0,'trans_dep'])),alpha=0.85)
        ax2.plot([search_xrange[0],search_xrange[-1]],[-10,-10],'--',linewidth=3.0,alpha=0.3)
        ax2.set_ylim(np.nanmedian(outparams['worstBIC'])*2,np.nanmin(outparams['worstBIC']))
        minflux=np.nanmin(lc['bin_flux'])
        ax1.set_ylim(minflux*2.5,np.nanmax(lc['bin_flux'])*1.5)
        ax1.set_xlim(np.nanmin(lc['bin_time']),np.nanmax(lc['bin_time']))
        ax2.set_ylabel("DeltaBIC")
        ax1.set_ylabel("Flux")
                 
    if len(signfct)>0:
        jumps=np.hstack((0,1+np.where(np.diff(signfct)>0.66*n_oversamp)[0],len(signfct)))
        min_ixs=[]
        #Looping through clustered regions of "detection" space and finding the maximum value within
        for n_jump in range(len(jumps)-1):
            ix=signfct[jumps[n_jump]:jumps[n_jump+1]]
            min_ix=ix[np.argmin(outparams.iloc[ix]['worstBIC'])]
            min_ixs+=[[min_ix,outparams.iloc[min_ix]['worstBIC']]]
            '''print('nearbys:',outparams[min_ix-1,6],outparams[min_ix-1,0]-outparams[min_ix-1,1],
                             outparams[min_ix,6],outparams[min_ix,0]-outparams[min_ix,1],
                             outparams[min_ix+1,6],outparams[min_ix+1,0]-outparams[min_ix+1,1])'''
        min_ixs=np.array(min_ixs)
        min_ixs=min_ixs[min_ixs[:,1].argsort()]
        detns = {}
        lc_std=np.nanstd(lc['flux'][lc['mask']])
        cad=np.nanmedian(np.diff(lc['time']))
        for nix,ix in enumerate(min_ixs):
            detn_row=outparams.iloc[int(ix[0])]
            detns[str(nix).zfill(2)]={}
            detns[str(nix).zfill(2)]['BIC_trans']=detn_row['BIC_trans']
            detns[str(nix).zfill(2)]['BIC_sin']=detn_row['BIC_sin']
            detns[str(nix).zfill(2)]['BIC_poly']=detn_row['BIC_poly']
            detns[str(nix).zfill(2)]['tcen']=detn_row['tcen']
            detns[str(nix).zfill(2)]['period']=np.nan
            detns[str(nix).zfill(2)]['period_err']=np.nan
            detns[str(nix).zfill(2)]['DeltaBIC']=ix[1],
            detns[str(nix).zfill(2)]['tdur']=detn_row['trans_dur']
            detns[str(nix).zfill(2)]['depth']=detn_row['trans_dep']
            detns[str(nix).zfill(2)]['oot_polyfit']=[detn_row['trans_poly_a'],detn_row['trans_poly_b']]
            detns[str(nix).zfill(2)]['orbit_flag']='mono'
            detns[str(nix).zfill(2)]['snr']=detns[str(nix).zfill(2)]['depth']/(lc_std/np.sqrt(detns[str(nix).zfill(2)]['tdur']/cad))
        if plot:
            for det in detns:
                ax1.text(detns[det]['tcen'],minflux,det)
            if plot_loc is None:
                plot_loc = str(ID).zfill(11)+"_mono_search_2.png"
            fig.savefig(plot_loc)
    else:
        detns={}
    return detns, outparams
