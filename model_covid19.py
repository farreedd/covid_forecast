#!/usr/bin/env python
# coding: utf-8

# In[283]:


# importing required libraries
import math 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime


# In[284]:


# importing files and assigning variables

cases_ft = pd.read_csv('D:/HBKU/Spring_2020/AI&ML in Healthcare/Dataset/confirmed.csv')

cases_table = cases_ft.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"],
	var_name="Date", value_name="Confirmed").fillna('').drop(['Lat', 'Long'], axis=1)
cases_table ['Date'] = pd.to_datetime(cases_table['Date'])

# print(cases_table)

# Data Clensing

def get_time(country):
    if cases_table[cases_table['Country/Region'] == country]['Province/State'].nunique() >= 1:
        country_table = cases_table[cases_table['Country/Region'] == country]
        country_ft = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed'], 
                                                 index='Date', aggfunc=sum).to_records()) 
        return country_ft.set_index('Date')[['Confirmed']]

    df = cases_table(cases_table['Country/Region'] == country)&(
        cases_table['Province/State'].isin(['', country]))
    return df.set_index('Date')[['Confirmed']]

country = 'Qatar'
df = get_time(country)
if len(df)>1 and df.iloc[-2,0] >= df.iloc[-1,0]:
    df.drop(d.tail(1).index, inplace=True)
# (df.tail(10))


# In[285]:


# *********individual country until datapoints************


def model_lag(n, a, alpha, lag, t):
    lag = min(max(lag, -100), 100)
    return max(n, 0) * (1 - math.e ** (min(-a, 0) * (t - lag))) ** max(alpha, 0)
def model(n, a, alpha, t):
    return max(n, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)


def model_loss(var):
    n, a, alpha = var
    model_x = []
    r = 0
    for t in range(len(df)):
        r += (model(n, a, alpha, t) - df.iloc[t, model_index]) ** 2
    return math.sqrt(r) 

use_lag_model = False
if use_lag_model:
    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15, 0]), method='Nelder-Mead', tol=1e-5).x
else:
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed']

plot_color = ['#99990077', '#FF000055']
#               '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']

# pd.concat([model_sim, df], axis=1).plot(color = plot_color)
# plt.show()


# In[286]:


# ***********Extended model for next 30 days*************

start_date = df.index[0]
pred_days = len(df) + 30
extended_model_x = []
last_row = []

isValid = True
for t in range(pred_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t)])
    if (t > len(df)):
        last_row = extended_model_x[-1]
        if (isValid):
            last_row2 = extended_model_x[-2]
            isValid = False
extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed']
plot_color = ['#99990077', '#FF000055']
pd.concat([extended_model_sim, df], axis=1).plot(color = plot_color)
print(country + ' COVID-19 Forecast')
plt.show()


# In[287]:


# pd.options.display.float_format = '{:20,.0f}'.format
# concat_df = pd.concat([df, extended_model_sim], axis=1)
# concat_df[concat_df.index.day % 3 == 0]


# In[288]:


# **********Forecasting model for the world datapoints for next 30 days************

def display_fit(df, opt_confirmed, ax):
    model_x = []
    
    isValid = True
    
    for t in range(len(df)):
        model_x.append([df.index[t], model(*opt_confirmed, t)])
        if (t > len(df)):
            last_row = model_x[-1]
            if (isValid):
                    last_row2 = model_x[-2]
                    isValid = False
    model_sim = pd.DataFrame(model_x, dtype=int)
    model_sim.set_index(0, inplace=True)
    model_sim.columns = ['Model-Confirmed']
    plot_color = ['#99990077', '#FF000055']
    
    return pd.concat([model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)

def display_extended_curve(df, opt_confirmed, ax):
    start_date = df.index[0]
    n_days = len(df) + 40
    extended_model_x = []
    
    isValid = True
    
    for t in range(n_days):
        extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t)])
        
        if (t > len(df)):
            last_row = extended_model_x[-1]
            if (isValid):
                last_row2 = extended_model_x[-2]
                isValid = False
                    
    extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
    extended_model_sim.set_index(0, inplace=True)
    extended_model_sim.columns = ['Model-Confirmed']
    
    plot_color = ['#99990077', '#FF000055']
    
    return pd.concat([extended_model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)

def opt_display_model(df, stats):
    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
        df.drop(df.tail(1).index,inplace=True)
    global model_index
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 1
    if min(opt_confirmed) > 0:
        stats.append([country, *opt_confirmed])
        n_plot = len(stats)
        plt.figure(1)
#         ax1 = plt.subplot(221)
#         display_fit(df, opt_confirmed, ax1)
        ax2 = plt.subplot(222)
        display_extended_curve(df, opt_confirmed, ax2)
        plt.show()


# In[289]:


# ********Calling the model for world datapoints********

# stats = []

# df = cases_table[['Country/Region', 'Date', 'Confirmed']].groupby('Date').sum()
# print('World COVID-19 Prediction (World data)')
# opt_display_model(df, stats)


# In[290]:


stats = []
for country in ['Qatar', 'Bahrain', 'Oman', 'Kuwait', 'United Arab Emirates', 'China', 
               'Italy', 'Spain']:
    df = get_time(country)
    print('{} COVID-19 Prediction'.format(country))
    opt_display_model(df, stats)


# In[291]:


# *********Qatar and Neighboring Countries*********

cc = (get_time('Qatar')+get_time('Bahrain')+get_time('Oman')+get_time('Saudi Arabia')+
            get_time('Iran')+get_time('Oman'))

# print(ctr)

def model_lag(n, a, alpha, lag, t):
    lag = min(max(lag, -100), 100)
    return max(n, 0) * (1 - math.e ** (min(-a, 0) * (t - lag))) ** max(alpha, 0)
def model(n, a, alpha, t):
    return max(n, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)


def model_loss(var):
    n, a, alpha = var
    model_x = []
    r = 0
    for t in range(len(cc)):
        r += (model(n, a, alpha, t) - cc.iloc[t, model_index]) ** 2
    return math.sqrt(r) 

use_lag_model = False
if use_lag_model:
    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15, 0]), method='Nelder-Mead', tol=1e-5).x
else:
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_x = []
for t in range(len(cc)):
    model_x.append([cc.index[t], model(*opt_confirmed, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed']

plot_color = ['#99990077', '#FF000055']


start_date = cc.index[0]
pred_days = len(cc) + 30
extended_model_x = []
last_row = []

isValid = True
for t in range(pred_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t)])
    if (t > len(cc)):
        last_row = extended_model_x[-1]
        if (isValid):
            last_row2 = extended_model_x[-2]
            isValid = False
extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed']
plot_color = ['#99990077', '#FF000055']
pd.concat([extended_model_sim, cc], axis=1).plot(color = plot_color)
print('Middle East' + ' COVID-19 Forecast')
plt.show()


# In[282]:


# pd.options.display.float_format = '{:20,.0f}'.format
# concat_cc = pd.concat([cc, extended_model_sim], axis=1)
# concat_cc[concat_cc.index.day % 3 == 0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




