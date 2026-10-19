"""Interactive teaching companion for Sustainable Transport Systems and Mobility in Tourism.
All starting inputs are illustrative, not observed statistics or official emission factors.
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sustainable Transport Lab', page_icon='🚆', layout='wide')
st.title('Sustainable Transport Systems and Mobility in Tourism')
st.caption('Interactive teaching lab · Demonstration parameters only · November 2026')
LESSONS = ['1 · GHG accounting','2 · Tourism gravity model','3 · Why fly? Mode choice','4 · Airport access and egress','5 · Carbon pricing','6 · Sustainable aviation scenarios','7 · Green airport inventory','8 · High-speed rail competition','9 · MaaS bundle designer']
lesson=st.sidebar.radio('Select lesson',LESSONS)
st.sidebar.info('All preset figures are hypothetical teaching inputs. They must not be interpreted as measured transport performance, official emission factors, or estimated behaviour.')

def num(label,value,min_value=0.0,step=1.0):
    return st.number_input(label,min_value=float(min_value),value=float(value),step=float(step))
def show_table(df):
    st.dataframe(df, hide_index=True, use_container_width=True)
def task(question):
    st.info('Class task: '+question)

def softmax(v):
    x=np.array(v,dtype=float); ex=np.exp(x-x.max()); return ex/ex.sum()

if lesson.startswith('1'):
    st.header('Activity-based GHG emissions inventory')
    st.latex(r'E_s = A_s \times EF_s')
    st.write('Enter activity data and hypothetical conversion factors for three emission scopes. Units are shown in every input.')
    col1,col2,col3=st.columns(3)
    with col1:
        a1=num('Scope 1: fuel (litres)',10000,step=100)
        f1=num('Assumed factor (kg CO₂e/litre)',2.6,step=.1)
    with col2:
        a2=num('Scope 2: electricity (kWh)',50000,step=1000)
        f2=num('Assumed factor (kg CO₂e/kWh)',.2,step=.01)
    with col3:
        a3=num('Scope 3: passenger-km',300000,step=10000)
        f3=num('Assumed factor (kg CO₂e/pkm)',.08,step=.01)
    vals=np.array([a1*f1,a2*f2,a3*f3])/1000
    df=pd.DataFrame({'Scope':['Scope 1','Scope 2','Scope 3'],'t CO₂e':vals})
    show_table(df);st.bar_chart(df.set_index('Scope'));st.metric('Total emissions',f'{vals.sum():,.1f} t CO₂e')
    task('Reduce one activity level by 20%. Which scope contributes most to the baseline? What sources would you need for defensible emission factors?')
elif lesson.startswith('2'):
    st.header('Calibrating a gravity model')
    st.latex(r'T_{ij}= k \frac{P_i^{\alpha} A_j^{\gamma}}{D_{ij}^{\beta}}')
    c1,c2,c3=st.columns(3)
    with c1:
        pop=num('Reference origin population (persons)',6700000,step=100000)
        appeal=num('Reference destination attractiveness (index)',100,step=5)
    with c2:
        dist=num('Reference distance (km)',1793,step=10)
        observed=num('Reference observed flow (passengers/year; illustrative)',1500000,step=10000)
    with c3:
        alpha=st.slider('Population elasticity α',0.0,2.0,1.0,.05)
        gamma=st.slider('Attractiveness elasticity γ',0.0,2.0,.8,.05)
        beta=st.slider('Distance elasticity β',0.0,3.0,1.2,.05)
    if min(pop,appeal,dist)>0:
        k=observed*dist**beta/(pop**alpha*appeal**gamma)
        st.metric('Calibrated scale k',f'{k:.5g}')
        st.subheader('Scenario to predict')
        q1,q2,q3=st.columns(3)
        with q1: pp=num('Scenario origin population',5500000,step=100000)
        with q2: aa=num('Scenario attractiveness index',100,step=5)
        with q3: dd=num('Scenario distance (km)',2000,step=10)
        prediction=k*pp**alpha*aa**gamma/dd**beta if dd>0 else float('nan')
        st.metric('Predicted annual passenger flow',f'{prediction:,.0f}')
    task('Change distance by 10% while holding everything else constant. Which parameter governs the resulting percentage change?')
elif lesson.startswith('3'):
    st.header('Illustrative mode choice model')
    st.latex(r'P_m = \frac{\exp(V_m)}{\sum_j\exp(V_j)},\qquad V_m=-\beta_C C_m - \beta_T t_m')
    st.write('Prices are EUR per trip and time is total door-to-door hours. Coefficients are assumed, not econometrically estimated.')
    beta_c=st.slider('Cost sensitivity (per EUR)',0.001,.1,.025,.001)
    beta_t=st.slider('Travel-time sensitivity (per hour)',.01,2.0,.65,.01)
    data=[]
    cols=st.columns(3)
    for c,mode,p,t in zip(cols,['Air','Rail','Car'],[95,70,80],[4.0,5.0,7.0]):
        with c:
            st.subheader(mode)
            price=num(f'{mode}: total cost (EUR)',p,step=5)
            time=st.slider(f'{mode}: time (hours)',.5,15.0,t,.25)
            data.append((mode,price,time))
    probs=softmax([-beta_c*p-beta_t*t for _,p,t in data])
    show_table(pd.DataFrame({'Mode':[x[0] for x in data],'Probability':probs,'Share (%)':np.round(probs*100,1)}))
    st.bar_chart(pd.DataFrame({'Mode':[x[0] for x in data],'Predicted share (%)':probs*100}).set_index('Mode'))
    task('Raise the air fare by EUR 30. Describe the modelled substitution and why these are not observed market shares.')
elif lesson.startswith('4'):
    st.header('Airport access: generalised cost')
    st.latex(r'GC_m = C_m + VOT\cdot t_m + R_m')
    vot=st.slider('Value of travel time (EUR/hour)',0,60,15)
    rows=[]
    for mode,c,t,r in [('Bus',5,1.0,5),('Taxi',35,.45,2),('Train',12,.55,3)]:
        with st.expander(mode,expanded=True):
            col1,col2,col3=st.columns(3)
            with col1: cost=num(f'{mode}: fare (EUR)',c,step=1)
            with col2: tm=st.slider(f'{mode}: total time (hours)',.1,3.0,t,.05)
            with col3: risk=num(f'{mode}: reliability / inconvenience penalty (EUR)',r,step=1)
            rows.append((mode,cost,tm,risk,cost+vot*tm+risk))
    df=pd.DataFrame(rows,columns=['Mode','Cost EUR','Time h','Penalty EUR','Generalised cost EUR'])
    show_table(df);st.bar_chart(df.set_index('Mode')[['Generalised cost EUR']]); st.success('Lowest generalised cost: '+df.loc[df['Generalised cost EUR'].idxmin(),'Mode'])
    task('Which access mode would you recommend to travellers with high and low values of time?')
elif lesson.startswith('5'):
    st.header('Carbon tax and allowance cost simulator')
    st.latex(r'\text{Carbon payment} = E_{\mathrm{tCO_2}}\times p_{\mathrm{EUR/tCO_2}}')
    emissions=num('Emissions covered (t CO₂)',12000,step=500)
    price=st.slider('Carbon price (EUR / t CO₂)',0,300,85,5)
    allowance=st.slider('Free allocation (%)',0,100,20,5)
    riders=num('Passengers across which the payment is recovered',300000,step=10000)
    pay=emissions*price*(1-allowance/100)
    st.metric('Payment after assumed free allocation',f'€{pay:,.0f}')
    st.metric('Full pass-through per passenger',f'€{pay/riders:,.2f}' if riders else 'Not defined')
    task('Compare full free allocation and zero free allocation. Does this arithmetic predict a change in passengers or emissions?')
elif lesson.startswith('6'):
    st.header('Aviation: activity growth vs emissions-intensity reduction')
    st.latex(r'E_1=E_0(1+g)(1-r)')
    baseline=num('Baseline annual emissions (t CO₂)',1000000,step=50000)
    growth=st.slider('Activity growth to scenario year (%)',-50,200,40,5)/100
    reduction=st.slider('Emission intensity reduction (%)',0,95,30,5)/100
    after=baseline*(1+growth)*(1-reduction)
    st.metric('Scenario annual emissions (t CO₂)',f'{after:,.0f}',delta=f'{(after/baseline-1)*100:+.1f}% vs baseline' if baseline else None,delta_color='inverse')
    yrs=np.arange(2026,2051)
    paths=pd.DataFrame({'Year':yrs,'Constant emissions':[baseline]*len(yrs),'Scenario':[baseline*(1+growth*(y-2026)/24)*(1-reduction*(y-2026)/24) for y in yrs]})
    st.line_chart(paths.set_index('Year'))
    task('Find the intensity reduction needed to offset 40% activity growth. What does this simplified model omit?')
elif lesson.startswith('7'):
    st.header('Green airport: illustrative inventory and interventions')
    energy=num('Terminal electricity (MWh/year)',20000,step=1000)
    energyef=num('Electricity factor (t CO₂e/MWh; hypothetical)',.20,step=.05)
    vehicles=num('Owned vehicles / equipment emissions (t CO₂e/year)',3500,step=100)
    improvement=st.slider('Electricity consumption reduction (%)',0,70,20,5)/100
    electrify=st.slider('Vehicle/equipment emission reduction (%)',0,100,30,5)/100
    base=energy*energyef+vehicles
    future=energy*energyef*(1-improvement)+vehicles*(1-electrify)
    df=pd.DataFrame({'Scenario':['Baseline','After interventions'],'t CO₂e':[base,future]})
    show_table(df); st.bar_chart(df.set_index('Scenario'))
    st.metric('Illustrative reduction',f'{base-future:,.0f} t CO₂e')
    task('How would you distinguish operational Scope 1–2 emissions from emissions attributable to aircraft?')
elif lesson.startswith('8'):
    st.header('Air, rail and car: door-to-door competition')
    beta_c=st.slider('Cost coefficient',.001,.10,.02,.001,key='bc8')
    beta_t=st.slider('Time coefficient',.01,2.0,.6,.01,key='bt8')
    options=[]
    for mode,p,t in [('High-speed rail',75,3.0),('Air',95,4.5),('Car',100,5.5)]:
        with st.expander(mode,expanded=True):
            c1,c2=st.columns(2)
            with c1: money=num(f'{mode} cost (EUR)',p,step=5)
            with c2: hours=st.slider(f'{mode} door-to-door time (h)',.5,12.0,t,.25)
            options.append((mode,money,hours))
    probs=softmax([-beta_c*p-beta_t*t for _,p,t in options])
    df=pd.DataFrame({'Mode':[a for a,_,_ in options],'Illustrative predicted share (%)':probs*100})
    show_table(df);st.bar_chart(df.set_index('Mode'))
    task('Reduce rail station access time by 30 minutes. Compare its effect with a EUR 20 rail fare reduction.')
else:
    st.header('MaaS: build and test a mobility bundle')
    sub=num('MaaS bundle monthly subscription (EUR)',65,step=5)
    bus=n_bus=st.slider('Bus trips during stay',0,60,18)
    bike=n_bike=st.slider('Shared bicycle uses',0,50,8)
    cab=n_taxi=st.slider('Taxi rides',0,30,3)
    c1,c2,c3=st.columns(3)
    with c1: busprice=num('Standalone bus fare (EUR)',2.5,step=.5)
    with c2: bikeprice=num('Standalone bicycle use (EUR)',3,step=.5)
    with c3: taxiprice=num('Standalone taxi fare (EUR)',16,step=1)
    includes_bus=st.checkbox('Bundle includes bus trips',value=True)
    includes_bike=st.checkbox('Bundle includes bicycle uses',value=True)
    includes_taxi=st.checkbox('Bundle includes taxi rides',value=False)
    standalone=bus*busprice+bike*bikeprice+cab*taxiprice
    bundle=sub+(0 if includes_bus else bus*busprice)+(0 if includes_bike else bike*bikeprice)+(0 if includes_taxi else cab*taxiprice)
    st.metric('Standalone trip spending',f'€{standalone:,.2f}')
    st.metric('Subscription plus excluded travel',f'€{bundle:,.2f}')
    st.metric('Difference (standalone minus bundle)',f'€{standalone-bundle:,.2f}')
    task('Change included services and travel frequency. Is a cheaper MaaS bundle necessarily more sustainable?')

st.divider()
st.caption('Methodological note: results are deterministic demonstrations. No external datasets, location tracking, or individual passenger data are used.')

from advanced import render_advanced
render_advanced(lesson)
