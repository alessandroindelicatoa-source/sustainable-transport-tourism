"""Advanced interactive exercises. Demo data are synthetic, not empirical."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm

RNG = np.random.default_rng(202611)

def demo_gravity():
    origins=np.arange(1,81)
    population=RNG.uniform(.2,9,80)*1e6
    attraction=RNG.uniform(20,130,80)
    distance=RNG.uniform(100,3100,80)
    mean=np.exp(3.2+0.85*np.log(population)+.65*np.log(attraction)-1.05*np.log(distance))
    flow=RNG.poisson(np.maximum(mean,1))
    return pd.DataFrame({'origin_id':origins,'population':population.astype(int),'attractiveness':attraction.round(2),'distance_km':distance.round(2),'passengers':flow})

def download(df,name):
    st.download_button('Download results (CSV)',df.to_csv(index=False).encode('utf-8'),file_name=name,mime='text/csv',key='dl_'+name)

def section_gravity():
    st.subheader('Advanced lab · estimate a gravity model')
    st.markdown('Estimate a log-linear gravity equation by OLS, compare elasticities and examine residuals. Synthetic example; not observed passenger traffic.')
    demo=demo_gravity(); download(demo,'synthetic_gravity_data.csv')
    upload=st.file_uploader('Optional CSV: population, attractiveness, distance_km, passengers',type='csv',key='gravity_file')
    try:
        df=pd.read_csv(upload) if upload is not None else demo
        cols=['population','attractiveness','distance_km','passengers']; d=df[cols].dropna()
        d=d[(d[cols]>0).all(axis=1)].copy()
        if len(d)<6: st.warning('At least six positive complete observations are required.');return
        x=np.column_stack([np.ones(len(d))]+[np.log(d[c]) for c in cols[:3]])
        y=np.log(d.passengers.to_numpy())
        b=np.linalg.lstsq(x,y,rcond=None)[0]; residual=y-x@b
        se=np.sqrt(np.diag(np.linalg.pinv(x.T@x))*np.sum(residual**2)/(len(y)-x.shape[1]))
        fit=np.exp(x@b)
        coef=pd.DataFrame({'Parameter':['Intercept','alpha · population','gamma · attractiveness','beta · distance (signed)'],'Estimate':b,'OLS standard error':se})
        st.dataframe(coef,hide_index=True,use_container_width=True)
        st.latex(r'\ln T_{ij}=\ln k+\alpha\ln P_i+\gamma\ln A_j-\beta\ln D_{ij}+\epsilon_{ij}')
        st.metric('R² (log scale)',f'{1-np.sum(residual**2)/np.sum((y-y.mean())**2):.3f}')
        st.scatter_chart(pd.DataFrame({'Fitted log flow':x@b,'Residual':residual}),x='Fitted log flow',y='Residual')
        download(pd.DataFrame({'observed':d.passengers,'predicted_retransformed':fit,'log_residual':residual}),'gravity_fitted.csv')
        st.caption('OLS standard errors assume homoskedastic independent errors. Exponentiating fitted log outcomes does not automatically recover conditional mean flows; Poisson pseudo-maximum likelihood may be preferable with zeros.')
    except (ValueError,KeyError) as err: st.error(f'Invalid dataset: {err}')

def section_ghg():
    st.subheader('Advanced lab · Monte Carlo uncertainty')
    st.markdown('Uncertain activity and emission factors. Report the simulated interval, not a certified emissions inventory.')
    a=st.number_input('Activity (passenger-km)',1000,10000000,200000,10000,key='mc_a')
    f=st.number_input('Assumed emission factor (kg CO₂e/pkm)',.001,2.,.09,.005,key='mc_f')
    sd=st.slider('Coefficient of variation of activity and emission factor',0.,.6,.20,.025,key='mc_sd')
    rng=np.random.default_rng(54)
    sigma=np.sqrt(np.log(1+sd**2)); mu=-sigma*sigma/2
    v=(a*f/1000)*rng.lognormal(mu,sigma,10000)*rng.lognormal(mu,sigma,10000)
    q=np.quantile(v,[.025,.5,.975]);st.metric('Median simulated emissions',f'{q[1]:,.1f} t CO₂e')
    st.write(f'95% simulation interval: {q[0]:,.1f}–{q[2]:,.1f} t CO₂e')
    st.line_chart(pd.DataFrame({'Percentile':np.arange(0,101),'t CO2e':np.percentile(v,np.arange(0,101))}).set_index('Percentile'))
    download(pd.DataFrame({'draw':np.arange(1,10001),'t_CO2e':v}),'ghg_uncertainty_draws.csv')
    st.caption('Independent lognormal distributions are illustrative assumptions, not estimated uncertainty.')

def section_choice():
    st.subheader('Advanced lab · estimate binary mode choice')
    st.markdown('Estimate a binary logit from synthetic choices (rail = 1, air = 0). Both the data-generating process and model are simplified.')
    rng=np.random.default_rng(614)
    n=800; dcost=rng.normal(5,38,n); dtime=rng.normal(-.4,1.4,n)
    true=.35-.027*dcost-.85*dtime
    prob=1/(1+np.exp(-np.clip(true,-500,500)))
    choice=rng.binomial(1,prob)
    d=pd.DataFrame({'rail_minus_air_cost_eur':dcost,'rail_minus_air_time_h':dtime,'rail_chosen':choice})
    download(d,'synthetic_mode_choice.csv')
    upl=st.file_uploader('Optional CSV with these three columns',type='csv',key='choice_data')
    if upl is not None:
        try: d=pd.read_csv(upl)
        except Exception as exc: st.error(str(exc));return
    try:
        x=np.column_stack([np.ones(len(d)),d.rail_minus_air_cost_eur,d.rail_minus_air_time_h]); y=d.rail_chosen.to_numpy()
        if len(np.unique(y))!=2: st.warning('Both binary outcomes required.');return
        def obj(b): return np.sum(np.logaddexp(0,x@b)-y*(x@b))
        result=minimize(obj,np.array([0.,-.01,-.2]),method='BFGS')
        coef=result.x
        df=pd.DataFrame({'Term':['Intercept','Cost difference (EUR)','Time difference (h)'],'Estimate':coef})
        st.dataframe(df,hide_index=True,use_container_width=True)
        if coef[1]!=0:st.metric('Implied value of time',f'€{60*coef[2]/coef[1]/60:,.2f} per hour')
        st.caption('VOT = time coefficient / cost coefficient in EUR/hour, provided both variables use hours and EUR. Results can be unstable with poor data or separation.')
        delta=st.slider('Counterfactual rail price reduction (EUR)',0,50,15,1,key='choice_delta')
        def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-50,50)))
        base=sigmoid(x@coef).mean();alt=sigmoid(x@coef-coef[1]*delta).mean()
        st.metric('Predicted rail-choice share under scenario',f'{alt*100:.1f}%',delta=f'{(alt-base)*100:+.1f} pp')
        download(pd.DataFrame({'observed':y,'baseline_probability':sigmoid(x@coef),'policy_probability':sigmoid(x@coef-coef[1]*delta)}),'mode_choice_predictions.csv')
    except (KeyError,ValueError) as exc:st.error(f'Please check the dataset columns: {exc}')

def section_access():
    st.subheader('Advanced lab · Hansen accessibility and decay parameter')
    d=pd.DataFrame({'Area':['Centre','Suburbs','Coast','Rural'],'Opportunities':[200,125,90,60],'Airport_minutes':[25,45,60,95]})
    lam=st.slider('Exponential decay λ (per minute)',.001,.15,.035,.001,key='acc_lam')
    d['Contribution']=d.Opportunities*np.exp(-lam*d.Airport_minutes)
    st.latex(r'A_i=\sum_j O_j\exp(-\lambda c_{ij})')
    st.dataframe(d,hide_index=True,use_container_width=True)
    st.metric('Total Hansen accessibility index',f'{d.Contribution.sum():.1f}')
    val=np.linspace(.001,.15,80)
    s=[np.sum(d.Opportunities*np.exp(-ll*d.Airport_minutes)) for ll in val]
    st.line_chart(pd.DataFrame({'lambda':val,'accessibility':s}).set_index('lambda'))
    download(d,'accessibility_areas.csv')

def section_carbon():
    st.subheader('Advanced lab · carbon price, demand elasticity and emissions')
    st.latex(r'Q_1=Q_0\left(\frac{p_1}{p_0}\right)^\varepsilon')
    q=st.number_input('Baseline trips',1000,100000000,100000,1000,key='cp_q')
    fare=st.number_input('Baseline fare (EUR)',1.,2000.,100.,5.,key='cp_fare')
    e=st.slider('Own-price demand elasticity (negative)',-2.,0.,-.6,.05,key='cp_e')
    intensity=st.number_input('Emissions per trip (t CO₂)',.001,10.,.12,.01,key='cp_intensity')
    share=st.slider('Carbon cost passed through to fares',0.,1.,1.,.05,key='cp_share')
    prices=np.arange(0,301,10)
    q1=q*((fare+share*prices*intensity)/fare)**e
    z=pd.DataFrame({'carbon_price_eur_t':prices,'trips':q1,'emissions_t_co2':q1*intensity,'tax_revenue_eur':q1*intensity*prices})
    st.line_chart(z.set_index('carbon_price_eur_t')[['emissions_t_co2']]);st.dataframe(z,hide_index=True,use_container_width=True);download(z,'carbon_policy_elasticity.csv')
    st.caption('Constant-elasticity demand with full matching of carbon cost to intensity. No induced operational response, intermodal spillover, ETS cap equilibrium or welfare effects.')

def section_aviation():
    st.subheader('Advanced lab · decompose emissions changes')
    st.latex(r'\Delta \ln E = \Delta\ln A+\Delta\ln I')
    annual_growth=st.slider('Annual passenger-km growth (%)',-5.,10.,2.5,.25,key='av_growth')/100
    annual_efficiency=st.slider('Annual intensity improvement (%)',0.,10.,3.,.25,key='av_eff')/100
    baseline=st.number_input('2026 reference emissions (t CO₂)',1000,100000000,1000000,10000,key='av_base')
    yr=np.arange(2026,2051); k=yr-2026
    q=(1+annual_growth)**k; ii=(1-annual_efficiency)**k
    d=pd.DataFrame({'year':yr,'activity_index':q,'intensity_index':ii,'emissions_t_co2':baseline*q*ii})
    st.line_chart(d.set_index('year')[['emissions_t_co2']]);st.metric('Change in 2050',f'{(q[-1]*ii[-1]-1)*100:+.1f}%');download(d,'aviation_decomposition.csv')
    st.caption('Intensity is defined here per passenger-km. Non-CO₂ effects and aircraft lifecycle are not included.')

def section_airport():
    st.subheader('Advanced lab · marginal abatement cost ranking')
    d=pd.DataFrame({'Measure':['Terminal lighting retrofit','Ground fleet electrification','On-site photovoltaic','Heat-pump replacement'],'Abatement_t_CO2e':[160,310,220,190],'Annualized_net_cost_EUR':[-18000,22000,9000,13000]})
    d['MAC_eur_per_t']=d.Annualized_net_cost_EUR/d.Abatement_t_CO2e
    d=d.sort_values('MAC_eur_per_t')
    budget=st.slider('Annual investment envelope (EUR)',0,120000,50000,5000,key='airport_budget')
    d['selected']=False;spent=0
    for i,r in d.iterrows():
        cost=max(0,r.Annualized_net_cost_EUR)
        if spent+cost<=budget:d.loc[i,'selected']=True;spent+=cost
    st.dataframe(d,hide_index=True,use_container_width=True)
    st.metric('Annual abatement selected',f'{d.loc[d.selected,"Abatement_t_CO2e"].sum():.0f} t CO₂e')
    download(d,'airport_mac.csv')
    st.caption('Illustrative greedy ranking by average abatement cost. Budget constraint treats negative costs as zero and does not account for interactions; it is not a globally optimal investment model.')

def section_rail():
    st.subheader('Advanced lab · rail demand sensitivity surface')
    bc=st.slider('Fare coefficient (per EUR)',.005,.08,.025,.005,key='rail_bc')
    bt=st.slider('Time coefficient (per hour)',.05,1.5,.7,.05,key='rail_bt')
    prices=np.arange(35,156,10); times=np.arange(2,6.1,.5)
    airutility=-bc*90-bt*4.5
    shares=np.array([[np.exp(-bc*p-bt*t)/(np.exp(-bc*p-bt*t)+np.exp(airutility)) for p in prices] for t in times])
    d=pd.DataFrame(shares,index=np.round(times,2),columns=prices)
    st.write('Modelled rail share as a function of rail door-to-door time (rows, h) and fare (columns, EUR).')
    st.dataframe(d.style.format('{:.1%}'),use_container_width=True)
    out=d.stack().reset_index();out.columns=['rail_time_h','rail_fare_eur','rail_probability'];download(out,'rail_probability_surface.csv')
    st.caption('Binary logit with fixed aviation alternative, no capacity constraints or induced demand.')

def section_maas():
    st.subheader('Advanced lab · break-even trip frequency')
    subscription=st.slider('Subscription per visitor (€)',10,150,60,5,key='maas_sub')
    fare=st.slider('Average pay-as-you-go trip (€)',1.,25.,4.,.5,key='maas_fare')
    fixed=st.slider('Non-covered monthly trips (€)',0,80,15,5,key='maas_fixed')
    trips=np.arange(0,61)
    df=pd.DataFrame({'trips':trips,'pay_as_you_go_eur':trips*fare,'subscription_total_eur':np.repeat(subscription+fixed,len(trips))})
    st.line_chart(df.set_index('trips'))
    breakeven=(subscription+fixed)/fare
    st.metric('Break-even included trips',f'{breakeven:.1f}')
    download(df,'maas_break_even.csv')
    st.caption('A private monetary break-even calculation, not a measure of public-sector cost, emissions or social welfare.')

SECTIONS=[section_ghg,section_gravity,section_choice,section_access,section_carbon,section_aviation,section_airport,section_rail,section_maas]
def render_advanced(lesson):
    n=int(lesson.split(' · ')[0])-1
    st.divider()
    with st.expander('Advanced practice · estimation, sensitivity and downloadable results',expanded=True):
        SECTIONS[n]()
