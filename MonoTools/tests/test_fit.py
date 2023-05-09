from MonoTools import fit
import matplotlib.pyplot as plt
import exoplanet as xo

import numpy as np
np.random.seed(1812)

import unittest

class TestModelFit(unittest.TestCase):

    def setUp(self):
        #Creating 5 varied models
        #1 Model with only duotransit in photometry. 
        self.simple_duo_lc_model=self.setUpData(test_rv=False, test_duo=True, test_mono=False, test_multi=False)
        #2 Model with both monotransit and multi-transiting planets in 2 sources of photometry
        self.mid_mono_lc_model=self.setUpData(test_rv=False, test_duo=False, test_mono=True, test_multi=True, multi_phot=True)
        #3 Model with duo, mono and multi-transiting planets in 2 sources of photometry
        self.complex_lc_model=self.setUpData(test_rv=False, test_duo=True, test_mono=True, test_multi=True)
        #4 Model with duo and RVs, with only a single RV scope but multiple photometry sources
        self.simple_duo_rv_model=self.setUpData(test_rv=True, test_duo=True, test_mono=False, 
                                           test_multi=False, test_nontr=False, use_multiple_scopes=False, multi_phot=True)
        #5 Model with mono, multi a non-transiting planet, with RVs from 2 scopes.
        self.complex_mono_rv_model=self.setUpData(test_rv=True, test_duo=False, test_mono=True, test_multi=True, 
                                             test_nontr=True, use_multiple_scopes=True)
              
    def setUpData(self, test_rv=True, test_duo=True, test_mono=True, test_multi=True, 
                  test_nontr=True, use_multiple_scopes=True, multi_phot=True):
        ###### INITIALISING LC MODELS ######
        x = np.hstack((np.arange(100.5,109.5,30/1440), np.arange(110.5,119.5,30/1440)))

        multi_phot=True

        if multi_phot:
            scopeind = np.tile('t30',len(x))
            newx = np.hstack((np.arange(300.5,309.5,10/1440), np.arange(310.5,319.5,10/1440)))
            x = np.hstack((x,newx))
            scopeind = np.hstack((scopeind,np.tile('k20',len(newx))))
        else:
            x = np.hstack((x,np.arange(300.5,309.5,30/1440), np.arange(310.5,319.5,30/1440)))
            scopeind = np.tile('t30',len(x))
        y_sv=abs(np.random.normal(0.0002,0.0005))*np.sin((x+np.random.random()*2*np.pi)/(np.random.normal(8,1))) + \
             abs(np.random.normal(0.0001,0.0005))*np.sin((x+np.random.random()*2*np.pi)/(np.random.normal(3,0.8)))
        noise=0.0008
        y = np.random.normal(1.0,noise,len(x))+y_sv
        yerr = noise*1.1+np.zeros_like(y)

        inputs={'mono_period':44.1237,'mono_t0':107.127,'mono_depth':0.0045,
                    'mono_ecc':0.001,'mono_omega':0.4,'mono_tdur':4/24,'mono_b':0.3,'mono_K':12.5,
                    'duo_period':33.61,'duo_t0':105.57,'duo_depth':0.003,'duo_tdur':0.2,
                    'duo_ecc':0.001,'duo_omega':0.4,'duo_b':0.6,'duo_K':8.25,
                    'multi_period':8.4789,'multi_t0':101.7892,'multi_depth':0.0014,'multi_tdur':0.13,
                    'multi_ecc':0.05,'multi_omega':0.4,'multi_b':0.12,'multi_K':3.25,
                    'nontr_period':4.822,'nontr_t0':106.7892,
                    'nontr_ecc':0.05,'nontr_omega':0.4,'nontr_K':2.5,
                    't_u1':0.4,'t_u2':0.3,'k_u1':0.3,'k_u2':0.1}
        duo_orbit   = xo.orbits.KeplerianOrbit(period=inputs['duo_period'], t0=inputs['duo_t0'], b=inputs['duo_b'],
                                               omega=inputs['duo_omega'], ecc=inputs['duo_ecc'])
        mono_orbit  = xo.orbits.KeplerianOrbit(period=inputs['mono_period'], t0=inputs['mono_t0'], b=inputs['mono_b'],
                                               omega=inputs['mono_omega'], ecc=inputs['mono_ecc'])
        multi_orbit = xo.orbits.KeplerianOrbit(period=inputs['multi_period'], t0=inputs['multi_t0'], b=inputs['multi_b'],
                                               omega=inputs['multi_omega'], ecc=inputs['multi_ecc'])
        nontr_orbit = xo.orbits.KeplerianOrbit(period=inputs['nontr_period'], t0=inputs['nontr_t0'], 
                                               omega=inputs['nontr_omega'], ecc=inputs['nontr_ecc'])
        self.inputs=inputs
        if test_rv:
            ###### INITIALISING RV MODELS ######
            x_rv=np.sort(330+80*np.random.random(25))
            scopeindex=np.tile('s1',25)
            if use_multiple_scopes:
                x_rv=np.hstack((x_rv, np.sort(410+50*np.random.random(12))))
                scopeindex=np.hstack((scopeindex,np.tile('s2',12)))

            #Adding 8.1m/s scatter, and jitter using a couple of sin functions
            # - a) amplitude 6.5 and period of ~11d, b) 9m/s and period ~250
            jitps=np.random.normal([0.95,250],[0.1,40])
            jitt0s=np.random.random(2)*jitps
            y_jitter= 6.5*np.sin(np.pi*2*(x_rv-jitt0s[0])/(jitps[0])) + 19*np.sin(np.pi*2*(x_rv-jitt0s[1])/(jitps[1]))
            yerr_rv = np.random.normal(7.13,0.5,len(x_rv))
            offsets=np.array([-23.2378 if scopeindex[i]=='s1' else 79.2348 for i in range(len(x_rv))])
            y_rv = np.random.normal(0.0,[6.1 if scopeindex[i]=='s1' else 9.8 for i in range(len(x_rv))])+y_jitter+offsets
            
        
        lc = {'time':x,'flux_err':np.tile(noise,len(x)),'flux':y,'jd_base':2459000,'cadence':scopeind,
                   'flux_unit':0.001,'mask':np.random.random(len(x))<0.996}
        if test_duo:
            if multi_phot:
                lc['flux'][scopeind=='t30']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                              orbit=duo_orbit, t=x[scopeind=='t30'], r=np.sqrt(inputs['duo_depth'])).eval()[:, 0]
                lc['flux'][scopeind=='k20']+=xo.LimbDarkLightCurve([inputs['k_u1'], inputs['k_u2']]).get_light_curve(
                                              orbit=duo_orbit, t=x[scopeind=='k20'], r=np.sqrt(inputs['duo_depth'])).eval()[:, 0]
            else:
                lc['flux']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                                orbit=duo_orbit, t=x, r=np.sqrt(inputs['duo_depth'])).eval()[:, 0]
        if test_mono:
            if multi_phot:
                lc['flux'][scopeind=='t30']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                            orbit=mono_orbit, t=x[scopeind=='t30'], r=np.sqrt(inputs['mono_depth'])).eval()[:, 0]
                lc['flux'][scopeind=='k20']+=xo.LimbDarkLightCurve([inputs['k_u1'], inputs['k_u2']]).get_light_curve(
                                            orbit=mono_orbit, t=x[scopeind=='k20'], r=np.sqrt(inputs['mono_depth'])).eval()[:, 0]
            else:
                lc['flux']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                            orbit=mono_orbit, t=x, r=np.sqrt(inputs['mono_depth'])).eval()[:, 0]
        if test_multi:
            if multi_phot:
                lc['flux'][scopeind=='t30']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                            orbit=multi_orbit,t=x[scopeind=='t30'], r=np.sqrt(inputs['multi_depth'])).eval()[:, 0]
                lc['flux'][scopeind=='k20']+=xo.LimbDarkLightCurve([inputs['k_u1'], inputs['k_u2']]).get_light_curve(
                                            orbit=multi_orbit,t=x[scopeind=='k20'], r=np.sqrt(inputs['multi_depth'])).eval()[:, 0]
            else:
                lc['flux']+=xo.LimbDarkLightCurve([inputs['t_u1'], inputs['t_u2']]).get_light_curve(
                                            orbit=multi_orbit,t=x, r=np.sqrt(inputs['multi_depth'])).eval()[:, 0]
        lc['flux']-=np.nanmedian(lc['flux'])
        lc['flux']*=1000
        if test_rv:
            rvs = {'time':x_rv,'rv':y_rv,'rv_err':yerr_rv,'jd_base':2459000,'derive_K':False,'rv_unit':'m/s'}
            if test_duo:
                rvs['rv']+=duo_orbit.get_radial_velocity(x_rv, K=inputs['duo_omega']).eval()
            if test_mono:
                rvs['rv']+=mono_orbit.get_radial_velocity(x_rv, K=inputs['mono_omega']).eval()
            if test_multi:
                rvs['rv']+=multi_orbit.get_radial_velocity(x_rv, K=inputs['multi_omega']).eval()
            if test_nontr:
                rvs['rv']+=nontr_orbit.get_radial_velocity(x_rv, K=inputs['nontr_K']).eval()
        else:
            rvs=None
        return lc, rvs

    def test_simple_duo_lc_model(self):
        mod = fit.monoModel('test_simple_duo_lc_model','tess',self.simple_duo_lc_model[0])
        mod.init_starpars(Rstar=[1.0,0.05,0.05], rhostar=[1.0,0.1,0.1], Teff=[5700,120,120])
        mod.add_duo({'tcen':np.random.normal(self.inputs['duo_t0'],0.005),
                     'tcen_2':np.random.normal(self.inputs['duo_t0']+6*self.inputs['duo_period'],0.005),
                     'tdur':self.inputs['duo_tdur'], 'depth':np.random.normal(1.0,0.25)*self.inputs['duo_depth']}, 'duo')
        mod.init_model(use_GP=False)
        fig = mod.Plot(n_samp=1,return_fig=True,overwrite=True)
        assert fig is not None
        print("simple_duo_lc_model passes")

    def test_mid_mono_lc_model(self):
        mod2 = fit.monoModel('test_mid_mono_lc_model','tess',self.mid_mono_lc_model[0])
        mod2.init_starpars(Rstar=[1.0,0.05,0.05], rhostar=[1.0,0.1,0.1], Teff=[5700,120,120])
        mod2.add_mono({'tcen':np.random.normal(self.inputs['mono_t0'],0.01),
                      'tdur':self.inputs['mono_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['mono_depth']},'b')
        mod2.add_multi({'tcen':np.random.normal(self.inputs['multi_t0'],0.01),
                       'period':np.random.normal(self.inputs['multi_period'],0.0001),'period_err':0.0002,
                       'tdur':self.inputs['multi_tdur'],'depth':np.random.normal(1.0,0.25)*self.inputs['multi_depth']},'c')
        mod2.init_model(use_GP=True)
        fig = mod2.Plot(n_samp=1,return_fig=True,overwrite=True)
        assert fig is not None
        print("mid_mono_lc_model passes")

    def test_complex_lc_model(self):
        mod3 = fit.monoModel('test_complex_lc_model','tess',self.complex_lc_model[0])
        mod3.init_starpars(Rstar=[1.0,0.05,0.05], rhostar=[1.0,0.1,0.1], Teff=[5700,120,120])
        mod3.add_duo({'tcen':np.random.normal(self.inputs['duo_t0'],0.01),
                     'tcen_2':np.random.normal(self.inputs['duo_t0']+6*self.inputs['duo_period'],0.01),
                     'tdur':self.inputs['duo_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['duo_depth']},'00')
        mod3.add_mono({'tcen':np.random.normal(self.inputs['mono_t0'],0.01),
                      'tdur':self.inputs['mono_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['mono_depth']},'01')
        mod3.add_multi({'tcen':np.random.normal(self.inputs['multi_t0'],0.01),
                       'period':np.random.normal(self.inputs['multi_period'],0.0001),'period_err':0.0002,
                       'tdur':self.inputs['multi_tdur'],'depth':np.random.normal(1.0,0.25)*self.inputs['multi_depth']},'02')
        mod3.init_model(use_GP=False)
        fig = mod3.Plot(n_samp=1,return_fig=True,overwrite=True)
        assert fig is not None
        print("complex_lc_model passes")

    def test_simple_duo_rv_model(self):
        mod4 = fit.monoModel('test_simple_duo_rv_model','tess',self.simple_duo_rv_model[0])
        mod4.init_starpars(Rstar=[1.0,0.05,0.05], rhostar=[1.0,0.1,0.1], Teff=[5700,120,120])
        mod4.add_rvs(self.simple_duo_rv_model[1],n_poly_trend=1)
        mod4.add_duo({'tcen':np.random.normal(self.inputs['duo_t0'],0.01),
                     'tcen_2':np.random.normal(self.inputs['duo_t0']+6*self.inputs['duo_period'],0.01),
                     'tdur':self.inputs['duo_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['duo_depth']},'00')
        mod4.init_model(use_GP=False)
        fig = mod4.Plot(n_samp=1,return_fig=True,overwrite=True)
        assert fig is not None
        fig2 = mod4.PlotRVs(n_samp=1,return_fig=True,overwrite=True)
        assert fig2 is not None
        print("simple_duo_rv_model passes")

    def test_complex_mono_rv_model(self):
        mod5 = fit.monoModel('test_complex_mono_rv_model','tess',self.complex_mono_rv_model[0])
        mod5.init_starpars(Rstar=[1.0,0.05,0.05], rhostar=[1.0,0.1,0.1], Teff=[5700,120,120])
        
        mod5.add_rvs(self.complex_mono_rv_model[1],n_poly_trend=3)
        mod5.add_mono({'tcen':np.random.normal(self.inputs['mono_t0'],0.01),
                      'tdur':self.inputs['mono_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['mono_depth']},'00')
        mod5.add_multi({'tcen':np.random.normal(self.inputs['multi_t0'],0.01),
                       'period':np.random.normal(self.inputs['multi_period'],0.0001),'period_err':0.0002,
                       'tdur':self.inputs['multi_tdur'],'depth':np.random.normal(1.0,0.01)*self.inputs['multi_depth'],
                       'K':np.random.normal(1.0,0.25)*self.inputs['multi_K']},'01')
        mod5.add_rvplanet({'tcen':np.random.normal(self.inputs['nontr_t0'],0.2),
                          'period':np.random.normal(self.inputs['nontr_t0'],0.1),'period_err':0.1,
                          'K':np.random.normal(1.0,0.25)*self.inputs['nontr_K']},'02')
        mod5.init_model(use_GP=True)
        fig = mod5.Plot(n_samp=1,return_fig=True,overwrite=True)
        assert fig is not None
        fig2 = mod5.PlotRVs(n_samp=1,return_fig=True,overwrite=True)
        assert fig2 is not None
        print("complex_mono_rv_model passes")
