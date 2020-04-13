# importing required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import datetime
import model

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

country = 'Spain'
df = get_time(country)
# if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
#     df.drop(df.tail(1).index,inplace=True)
print(df.tail(10))