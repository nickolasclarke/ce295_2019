# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # HW 4: Forecasting Residential Electricity Power Consumption
# You have been hired as a Building Energy Data Scientist. Congratulations! 
# What an exciting new opportunity! In your first job assignment, you must 
# construct algorithms to forecast residential electricity consumption with data
# and machine learning (ML). The assignment is organized in a tutorial fashion, 
# thereby allowing you to practice ML on a relevant real-world energy system 
# example.
# ## Reading
# [Optional] Read “Gated Ensemble Learning Method for Demand-Side Electricity 
# Load Forecasting” first authored by Eric Burger, available at this 
# [URL](https://ecal.berkeley.edu/pubs/MultiModelForecasterBurger.pdf).
#
# ## Background
# Several years ago we collected 
# [Green Button Data](http://www.greenbuttondata.org/) from the CE 295 students.
# This data includes hourly electricity consumption from over 20 residences in 
# the East Bay. The data has been (mostly) cleansed for your convenience 
# including (i) aligning time-stamps, (ii) filling-in missing data, and (iii) 
# organizing it into an easily readable format.

# ## Problem 1: Exploratory Data Analysis
# Download the data file HW4_Train_Data.csv. Load the CSV data into Matlab or 
# Python

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('fivethirtyeight')

test_df = pd.read_excel('HW4_Test_Data.xlsx')
train_df = pd.read_csv('HW4_Train_Data.csv', 
                       parse_dates=['Start Time (GMT-0800,PST)',
                                    'End Time (GMT-0800,PST)'])

#rename columns
train_cols = {'UNIX Timestamp (seconds)':'unix_time',
              'Start Time (GMT-0800,PST)':'start_time',
              'End Time (GMT-0800,PST)':'end_time'
              }
train_df = train_df.rename(columns=train_cols)
train_df.columns = train_df.columns.str.replace('\(kWh\)','')

# %%  [markdown]
# (a) Create a bar plot of the average hourly energy consumption [kWh] for each 
# building. Make the x-axis the building index number. Make the y-axis the 
# average hourly energy consumption. In red, super  impose error bars on your 
# bars that indicate the standard deviation of hourly energy consumption. 
# Provide this plot in your report.

# %%
#take the average of each building and plot
train_avg = train_df.iloc[0:,3:].mean()
errors = train_df.iloc[0:,3:].std()

ax = train_avg.plot(kind='bar',
               title='Building Hourly Average Power [kWh]',
               label ='Building kWh',
               yerr=errors,
               ecolor='r',
               capsize=3,
               legend=True,)
ax.set(xlabel='Building', ylabel='Avg kWh')
plt.show()

#drop Building 6, since it has negative values
train_df = train_df.drop(columns = ['Bldg6 '])
# rerun analysis
train_avg = train_df.iloc[0:,3:].mean()
errors = train_df.iloc[0:,3:].std()

ax = train_avg.plot(kind='bar',
               title='Building Hourly Average Power [kWh]',
               label ='Building kWh',
               yerr=errors,
               ecolor='r',
               capsize=3,
               legend=True,)
ax.set(xlabel='Building', ylabel='Avg kWh')
plt.show()




# %% [markdown]
# (b) Which building has abnormally high variance? Do any buildings have moments
# of NEGATIVE power consumption? If so, then we will remove this building from 
# our analysis. This building likely installed solar during the year and the
# smart meter data aggregates power consumption and solar power generation, 
# so it’s un-usable for this homework.
#
# ## Answer:
# *yes, building 6 has negative values, suggesting a solar installation.*
#
# (c) Re-organize your energy data set into a 4-D array. In order, the dimensions 
# correspond to (i) building index, (ii) week-of-year, (iii) day-of-week, and 
# (iv) hour-of-day. For each element in the 4-D vector,store the normalized 
# energy. That is, divide kWh by the maximum hourly energy consumption for that 
# building. Normalize the energy for each building. That is, divide kWh by the 
# maximum hourly energy consumption for that building. In industry and academia,
# we sometimes refer to these normalized building electricity consumption 
# trajectories as “load shapes”. Now, generate the following plots:
#
#  - In seven separate figures (one for each day-of-week), plot the hourly 
# energy consumption load shapes vs. hour (x-axis) for each building - all 
# super-imposed.
# - In each of the seven figures, plot the average hourly energy consumption in 
# a thick black line.


# %%
train_df = train_df.set_index(train_df['start_time'])
train_df = train_df.drop(columns=['unix_time', 'start_time', 'end_time'])

train_max = train_df.max()
train_norm = train_df / train_max

#group into
train_week = train_norm.groupby(train_norm.index.week)
train_dow = train_norm.groupby(train_norm.index.dayofweek)
train_hr = train_norm.groupby(train_norm.index.hour)
 
# %%
