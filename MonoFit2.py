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
        
        print(len(self.planets),self.planets,'monos:',self.monos,'multis:',self.multis,'duos:',self.duos)
        
        n_pl=len(self.planets)
        self.cads=np.unique(self.lc['cadence'])
        #In the case of different cadence/missions, we need to separate their respective errors to fit two logs2
        self.lc['flux_err_index']=np.column_stack([np.where(self.lc['cadence']==cad,1.0,0.0) for cad in self.cads])

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
            #   Masking out-of-transit flux:
            ######################################
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

            ######################################
            #     Initialising Periods & tcens
            ######################################

            if len(self.monos)>0:
                # The period distributions of monotransits are tricky as we often have gaps to contend with
                # We cannot sample the full period distribution while some regions have p=0.
                # Therefore, we need to find each region and marginalise over each
                
                min_Ps=np.array([self.planets[pls]['P_min'] for pls in self.monos])
                print(min_Ps)
                #From Dan Foreman-Mackey's thing:
                #log_soft_per = pm.Uniform("log_soft_per", lower=np.log(min_Ps), upper=np.log(100*min_Ps),shape=len(min_Ps))
                #soft_period = pm.Deterministic("soft_period", tt.exp(log_soft_per))
                #pm.Potential("mono_per_prior",-2*log_soft_per) # prior from window function and occurrence rates
                test_ps=np.array([self.planets[pls]['period'] if self.planets[pls]['period']>self.planets[pls]['P_min'] else 1.25*self.planets[pls]['P_min'] for pls in self.monos])
                mono_periods={}
                for pl in self.monos:
                    mono_periods[pl]=pm.Bound(pm.Pareto,
                                         upper=self.planets[pl]['per_gaps'][:,0],
                                         lower=self.planets[pl]['per_gaps'][:,1])("mono_period_"+pl, 
                                                                                m=self.planets[pl]['per_gaps'][0,0],
                                                                                alpha=1.0, 
                                                                                shape=len(self.planets[pl]['per_gaps'][:,0]),
                                                                                testval=0.5+self.planets[pl]['per_gaps'][:,0])
                tcens=np.array([self.planets[pls]['tcen'] for pls in self.monos])
                tdurs=np.array([self.planets[pls]['tdur'] for pls in self.monos])

            if len(self.duos)>0:
                #Creating deterministic array of duo periods.
                for npl,pl in enumerate(self.duos):
                    duo_periods[pl]=pm.Deterministic("duo_period_"+pl,
                                                                  init_t0[-1*(len(self.duos)+npl)]-tcen2[npl],
                                                                  self.planets[pl]['period_int_aliases'])
                tcens2=np.array([self.planets[pls]['tcen_2'] for pls in self.duos])
                tdurs=np.array([self.planets[pls]['tdur'] for pls in self.duos])
                t0_second_trans = pm.Bound(pm.Normal, 
                                           upper=tcens2+tdurs*0.5, 
                                           lower=tcens2-tdurs*0.5)("t0_second_trans",mu=tcens2,
                                                                  sd=np.tile(0.2,len(self.duos)),
                                                                  shape=len(self.duos),testval=tcens2)

            if len(self.multis)>0:
                inipers=np.array([self.planets[pls]['period'] for pls in self.multis])
                inipererrs=np.array([self.planets[pls]['period_err'] for pls in self.multis])
                multi_periods = pm.Normal("multi_periods", 
                                          mu=inipers,
                                          sd=np.clip(inipererrs*0.25,np.tile(0.005,len(inipers)),0.02*inipers),
                                          shape=len(self.multis),
                                          testval=inipers)
            tcens=np.array([self.planets[pls]['tcen'] for pls in self.multis+self.monos+self.duos])
            tdurs=np.array([self.planets[pls]['tdur'] for pls in self.multis+self.monos+self.duos])
            print(tcens,tdurs)
            t0 = pm.Bound(pm.Normal, upper=tcens+tdurs*0.5, lower=tcens-tdurs*0.5)("t0",mu=tcens, sd=tdurs*0.05,
                                        shape=len(self.multis),testval=tcens)

                
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
                                   J=2, mean=mean)
            
            ################################################
            #     Creating function to generate transits
            ################################################
            def gen_lc(i_orbit,i_r,mask=None,prefix=''):
                # Short method to create stacked lightcurves, given some input time array and some input cadences:
                # This function is needed because we may have 
                #   -  1) multiple cadences and 
                #   -  2) multiple telescopes (and therefore limb darkening coefficients)
                lc_c=[]
                lc_cad_xs=[]
                mask = ~np.isnan(self.lc['time']) if mask is None else mask
                for cad in self.cads:
                    t_cad=self.lc['time'][mask][self.lc['cadence'][mask]==cad].astype(np.float64)
                    t_exp=np.nanmedian(np.diff(t_cad))
                    lc_cad_xs+=[t_cad]
                    if cad[0]=='t':
                        lc_c +=[xo.LimbDarkLightCurve(u_star_tess).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=t_cad,
                                                                 texp=t_exp
                                                             )/(self.lc['flux_unit']*mult)]
                    elif cad[0]=='k':
                        lc_c += [xo.LimbDarkLightCurve(u_star_kep).get_light_curve(
                                                                 orbit=i_orbit, r=i_r,
                                                                 t=t_cad,
                                                                 texp=t_exp
                                                             )/(self.lc['flux_unit']*mult)]
                #Sorting by time so that it's in the correct order here:
                lc_cad_xs=np.hstack(lc_cad_xs)
                #print(len(lc_cad_xs),np.min(lc_cad_xs),np.max(lc_cad_xs),len(self.lc['time'][mask]),self.lc['time'][mask][0],self.lc['time'][mask][-1])
                assert (np.sort(lc_cad_xs)==self.lc['time'][mask]).all()
                return pm.Deterministic(prefix+"light_curves", tt.concatenate(lc_c,axis=0)[np.argsort(lc_cad_xs)])
            
            ################################################
            #     Analysing Multiplanets
            ################################################
            if len(self.multis)>0:
                multi_inds=[pl in self.multis for pl in self.multis+self.monos+self.duos]
                if assume_circ:
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        period=multi_period, t0=t0[multi_inds], b=b[multi_inds])
                else:
                    # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
                    multi_orbit = xo.orbits.KeplerianOrbit(
                        r_star=Rs, rho_star=rho_S,
                        ecc=ecc[multi_inds], omega=omega[multi_inds],
                        period=multi_period, t0=t0[multi_inds], b=b[multi_inds])
                
                #Generating lightcurves using pre-defined gen_lc function:
                multi_mask_light_curves = gen_lc(multi_orbit,r[multi_inds],mask=self.lc['oot_mask'],prefix='mask_')
                multi_mask_light_curve = pm.math.sum(multi_mask_light_curves, axis=-1) #Summing lightcurve over n planets
            else:
                multi_mask_light_curve = tt.zeros_like(self.lc['flux'][self.lc['oot_mask']])
            ################################################
            #     Marginalising over Duo periods
            ################################################
            if len(self.duos)>0:
                duo_per_info={}
                for nduo,duo in enumerate(self.duos):
                    #Marginalising over each possible period
                    #Single planet with two transits and a gap
                    duo_gap_info[duo]={'logpriors':[],
                                        'logliks':[],
                                        'lcs':[]}
                    
                    duo_ind=np.where([pl==duo for pl in self.multis+self.monos+self.duos])[0][0]

                    for i,p_int in enumerate(self.planets[duo]['period_int_aliases']):
                        with pm.Model(name="duo_per_{0}".format(i), model=model) as submodel:
                            # Set up a Keplerian orbit for the planets
                            if assume_circ:
                                duoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    period=duo_periods[nduo], t0=t0[duo_ind], b=b[duo_ind])
                            else:
                                duoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    ecc=ecc[duo_ind], omega=omega[duo_ind],
                                    period=duo_periods[nduo], t0=t0[duo_ind], b=b[duo_ind])
                            print(self.lc['time'][self.lc['oot_mask']])
                            
                            # Compute the model light curve using starry
                            duo_mask_light_curve = gen_lc(duoorbit,r[duo_ind],mask=self.lc['oot_mask'],prefix='duo_mask_')

                            duo_gap_info[duo] = pm.math.sum(duo_mask_light_curve, axis=-1) #Summing lightcurve over n planets
                            
                            duo_gap_info[duo]['logpriors'] +=[tt.log(duoorbit.dcosidb[-1]) - 2 * tt.log(period[-1])]
                            if use_GP:
                                duo_gap_info[duo]['logliks']+=[self.gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - duo_mask_light_curve - multi_mask_light_curve - mean)]
                            else:
                                new_yerr = self.lc['flux_err'][self.lc['oot_mask']]**2 + \
                                           tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)),
                                duo_gap_info[duo]['logliks']+=[tt.sum(pm.Normal.dist(mu=light_curve, sd=new_yerr
                                                                                    ).logp(self.lc['flux'][self.lc['oot_mask']]-\
                                                                                           multi_mask_light_curve - mean))]
                    # Compute the marginalized probability and the posterior probability for each period
                    logprobs = tt.stack(duo_gap_info[duo]['logpriors']+duo_gap_info[duo]['logliks'])
                    logprob_marg = pm.math.logsumexp(logprobs)
                    duo_per_info[duo]['logprob_class'] = pm.Deterministic("logprob_class_"+duo, logprobs - logprob_marg)
                    pm.Potential("logprob_"+duo, logprob_marg)

                    # Compute the marginalized light curve
                    duo_per_info[duo]['marg_lc']=pm.Deterministic("light_curve_"+duo,
                                                                  tt.sum(tt.stack(duo_per_info[duo]['lcs']) *\
                                                                         tt.exp(duo_per_info[duo]['logprob_class'])[:, None],
                                                                         axis=0
                                                                        ))
                duo_light_curves=pm.Deterministic("duo_light_curves",
                                                  tt.stack([duo_per_info[duo]['marg_lc'] for duo in self.duos]))
                duo_light_curve=pm.Deterministic("duo_light_curve",tt.sum(duo_light_curves,axis=0))
            else:
                duo_light_curve = tt.zeros_like(self.lc['flux'][self.lc['oot_mask']])

            ################################################
            #     Marginalising over Mono gaps
            ################################################
            if len(self.monos)>0:
                mono_gap_info={}
                #Two planets with two transits... This has to be the max.
                for nmono,mono in enumerate(self.monos):
                    #Marginalising over each possible period
                    #Single planet with two transits and a gap
                    mono_gap_info[mono]={'logpriors':[],
                                         'logliks':[],
                                         'lcs':[]}
                    mono_ind=np.where([pl==mono for pl in self.multis+self.monos+self.duos])[0][0]

                    for i,gaps in enumerate(self.planets['per_gaps']):
                        with pm.Model(name="mono_per_{0}".format(i), model=model) as submodel:
                            # Set up a Keplerian orbit for the planets
                            if assume_circ:
                                monoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    period=mono_periods[mono], t0=mono_t0s[nmono], b=b)
                            else:
                                monoorbit = xo.orbits.KeplerianOrbit(
                                    r_star=Rs, rho_star=rho_S,
                                    ecc=ecc[mono_ind], omega=omega[mono_ind],
                                    period=period, t0=t0, b=b)

                            # Compute the model light curve using starry
                            mono_mask_light_curves = gen_lc(monoorbit,r[mono_ind],
                                                                     mask=self.lc['oot_mask'],prefix='mono_mask_')

                            mono_gap_info[mono]['lcs']+=[pm.math.sum(mask_light_curves, axis=-1)]
                            mono_gap_info[mono]['logpriors'] +=[tt.log(monoorbit.dcosidb[-1])]
                            if use_GP:
                                mono_gap_info[mono]['logliks']+=[self.gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - duo_mask_light_curve - multi_mask_light_curve - mean)]
                            else:
                                new_yerr = self.lc['flux_err'][self.lc['oot_mask']]**2 + \
                                           tt.dot(self.lc['flux_err_index'][self.lc['oot_mask']],tt.exp(logs2)),
                                mono_gap_info[mono]['logliks']+=[tt.sum(pm.Normal.dist(mu=light_curve, sd=new_yerr
                                                                                    ).logp(self.lc['flux'][self.lc['oot_mask']]- \
                                                                                           duo_mask_light_curve - \
                                                                                           multi_mask_light_curve - mean))]
                    # Compute the marginalized probability and the posterior probability for each period
                    logprobs = tt.stack(mono_gap_info[mono]['logpriors']+mono_gap_info[mono]['logliks'])
                    logprob_marg = pm.math.logsumexp(logprobs)
                    mono_per_info[mono]['logprob_class'] = pm.Deterministic("logprob_class_"+mono, logprobs - logprob_marg)
                    pm.Potential("logprob_"+mono, logprob_marg)

                    # Compute the marginalized light curve
                    mono_per_info[mono]['marg_lc']=pm.Deterministic("light_curve_"+mono,
                                                                  tt.sum(tt.stack(mono_per_info[mono]['lcs']) *\
                                                                         tt.exp(mono_per_info[mono]['logprob_class'])[:, None],
                                                                         axis=0
                                                                        ))
                #Creating a marginalised lightcurve from all monotransits:
                mono_light_curves=pm.Deterministic("mono_light_curves",
                                                   tt.stack([mono_per_info[mono]['marg_lc'] for mono in self.monos]))
                mono_light_curve=pm.Deterministic("mono_light_curve",tt.sum(mono_light_curves,axis=0))

            else:
                mono_light_curve = tt.zeros_like(self.lc['flux'][self.lc['oot_mask']])

            ################################################
            #            Compute predicted LCs:
            ################################################
            #Now we have lightcurves for each of the possible parameters we want to marginalise, we need to sum them
            mask_light_curve = pm.Deterministic("mask_light_curve" tt.sum(tt.stack((duo_mask_light_curve,
                                                                                    multi_mask_light_curve,
                                                                                    mono_mask_light_curve)),axis=-1))
            if use_GP:
                total_llc = pm.Deterministic("total_llk",self.gp.log_likelihood(self.lc['flux'][self.lc['oot_mask']] - \
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
                start = self.model.test_point
            print(model.test_point)
            if not LoadFromFile:
                ################################################
                #               Optimizing:
                ################################################
                
                #Setting up optimization depending on what planet models we have:
                initvars0=[r, b]
                initvars1=[logs2]
                initvars2=[r, b, period, t0, rho_S]
                initvars3=[]
                initvars4=[r, b, period]
                if len(self.monos)>1:
                    initvars1+=['mono_period_'+pl for pl in self.monos]
                    initvars3+=['mono_period_'+pl for pl in self.monos]
                if len(self.multis)>1:
                    initvars1+=[multi_periods]
                    initvars4+=[multi_periods]
                if len(self.duos)>1:
                    #for pl in self.duos:
                    #    eval("initvars1+=[duo_period_"+pl+"]")
                    initvars1+=['duo_period_'+pl for pl in self.duos]
                    initvars4+=['duo_period_'+pl for pl in self.duos]
                if not assume_circ:
                    initvars2+=[ecc, omega]
                if use_GP:
                    initvars3+=[logs2, logpower, logw0]
                else:
                    initvars3+=[mean]
                    
                print("before",model.check_test_point())
                map_soln = xo.optimize(start=start, vars=initvars0,verbose=True)
                map_soln = xo.optimize(start=start, vars=initvars1,verbose=True)
                map_soln = xo.optimize(start=start, vars=initvars2,verbose=True)
                map_soln = xo.optimize(start=start, vars=initvars3,verbose=True)
                map_soln = xo.optimize(start=start, vars=initvars4,verbose=True)
                map_soln = xo.optimize(start=map_soln)
                '''
                tt.printing.Print('logs2')(logs2)
                map_soln = xo.optimize(start=start)
                tt.printing.Print('logs2')(logs2)
                
                #map_soln = xo.optimize(start=map_soln, vars=[period, t0])
                map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower])
                map_soln = xo.optimize(start=map_soln, vars=[logw0])
                #if not assume_circ:
                #    map_soln = xo.optimize(start=map_soln, vars=[ecc, omega, period, t0])
                map_soln = xo.optimize(start=map_soln, vars=[r, b],verbose=True)
                map_soln = xo.optimize(start=map_soln)
                map_soln = xo.optimize(start=map_soln, vars=[mean, r])
                map_soln = xo.optimize(start=map_soln, vars=[mean, b])
                map_soln = xo.optimize(start=map_soln, vars=[mean, period])
                map_soln = xo.optimize(start=map_soln, vars=[mean, logs2, logpower, logw0])
                map_soln = xo.optimize(start=map_soln, vars=[mean, ecc, omega])
                map_soln = xo.optimize(start=map_soln, vars=[mean, logr, b, period])
                map_soln = xo.optimize(start=map_soln)
                
                print("after",model.check_test_point())
                '''
                self.model = model
                self.init_soln = map_soln