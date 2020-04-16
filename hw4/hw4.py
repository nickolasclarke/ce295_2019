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
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

#group
train_norm['week']      = train_norm.index.week
train_norm['dayofweek'] = train_norm.index.dayofweek
train_norm['hour']      = train_norm.index.hour
#shift new columns to far left of df
cols = list(train_norm.columns)
cols = cols[-3:] + cols[:-3]
train_norm = train_norm[cols]

grouped = train_norm.groupby(['dayofweek','week','hour']).mean()
group_keys = grouped.index.get_level_values('dayofweek').unique().tolist()

#plot
def plot_dow(df, grp_keys, avg=True):
    '''
    Returns seven plots of building energy data. Each solid line represents 
    hourly data for one building for each week in the dataset. i.e. 52 lines per
    building, for a year's worth of data. The dashed line represents the hourly 
    avg for all buildings and weeks.
    
    '''
    #change color cycle to uniform color
    
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color",
    #     plt.cm.Reds(np.linspace(2,1,len(grp_keys))))

    fig, axes = plt.subplots(3,3,figsize=(15,10),sharey=True)
    labels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

    for (key, ax) in zip(grp_keys,axes.flatten()):
        df.xs(key).transpose().plot(ax=ax,
                            title=labels[key],
                            legend=False
                            )
        if avg == True:
            df.xs(key).groupby('hour').mean().transpose().mean() \
                .plot(ax=ax,c='k',ls='--',lw=len(grp_keys)) #wow, this is hack. what is a cleaner way? Pivot?
            #df.xs(key).transpose().mean().plot(ax=ax,c='k',ls='--',lw=len(grp_keys))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Normalized Power [kW]')
    plt.tight_layout()

plot_dow(grouped, group_keys)

# for (key, ax) in zip(group_keys,axes.flatten()):
#     grouped.xs(key).plot(ax=ax,
#                          title=labels[key],
#                          legend=False
#                          )
#     grouped.xs(key).transpose().mean().plot(ax=ax, c='k', ls='--', lw=6)
#     ax.set_xlabel('Hour')
#     ax.set_ylabel('Normalized Power [kW]')

# %% [markdown]
# ## Problem 2: Average Model
# In this problem, we design an extremely simple forecasting model that is often
# effective. We call it the“Average Model”. The average model forecasts the
# building power to be the average value from historical data at the HoD and 
# DoW. See PDF for mathematical representation.
# Answer the following questions:
# - (a) Download test data file HW4_Test_Data.csv. Load the CSV data into Matlab or Python. The test
# data includes one week of normalized hourly load for Sunday Sept 7, 2014 00:00 – Saturday Sept 13,
# 2014 23:00. Generate seven plots (one for each DoW) which visualize HoD (x-axis) and the normalized
# hourly energy for the Test Data and Average Model.

# %%
#use datetime index, create grouped version of test_data
test_df = test_df.set_index('TestTime')
grouped_test = test_df.groupby([test_df.index.dayofweek,
                                test_df.index.hour
                                ]).mean()

train_davg = pd.DataFrame() #empty df to store hourly avg per day
#take a cross section, comprised of the 24 hours in a day, build an hourly avg
for key in group_keys:
    day_avg = grouped.xs(key).groupby('hour').mean().transpose().mean()
    # day_avg = new_df.xs(key).mean()
    day_avg = day_avg.rename(key)
    train_davg = pd.concat([train_davg, day_avg,], axis=1)

fig, ax = plt.subplots(sharey=True)
train_davg.plot(ax=ax)

fig, axes = plt.subplots(3,3,figsize=(15,10),sharey=True)
labels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

for (key, ax) in zip(group_keys,axes.flatten()):
    grouped.xs(key).groupby('hour').mean().transpose().mean().plot(ax=ax,
        label='Predicted Value [DoW Avg]')
    grouped_test.xs(key).plot(ax=ax,label='Test Building',title=labels[key])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Normalized Power [kW]')
    plt.tight_layout()

# %% [markdown]
# (b) Compute the mean absolute error (MAE)
# %%
#calculate the MAE, aggregate up to DoW and overall
def mae_dow(test, train):
    mae_raw = np.abs(test['TestBldg'].to_numpy() - train.melt()['value'])
    mae_raw = pd.Series(mae_raw.values,index=test.index)
    mae_dow = mae_raw.groupby(mae_raw.index.dayofweek).mean()
    mae_week = mae_dow.mean()
    return (mae_raw, mae_dow, mae_week)

avg_mae,avg_mae_dow,avg_mae_week = mae_dow(test_df, train_davg)

print(f'MAE Day of Week:{avg_mae_dow.values}\nMAE Weekly: {avg_mae_week:.5f}')
# %% [markdown]
# ## Problem 3: Autoregressive with eXogeneous Inputs Model (ARX)
# In this problem, we will design a forecasting model based on the Autoregressive with eXogenous inputs model
# (ARX). We consider the ARX model given by: *see pdf*
#
# - (a) Write the ARX model in linear-in-the-parameters form: Y = Φθ as described in the video lectures.
# Define the vector Y and matrix Φ.
#
# - (b) Formulate a least squares optimization problem to find the optimal 
# parameters α_l`, l` = 1, 2,...L which fit your data. Write down the objective 
# function. Define your notation. Is this a convex program? Why?
#
# - (c) Solve your optimization problem with L = 3. In your report, give the 
# optimal values of a^star_1, a^star_2, a^star_3
#
# - (d) Test your ARX model on the test data set. Generate seven plots (one for 
# each DoW) which visualize HoD (x-axis) and the normalized hourly energy for 
# the Test Data, the Average Model, and the ARX model. Report the MAE for each 
# DoW, and the entire week.

# %%
# import pandas_datareader as pdr
# import statsmodels.api as sm
# from statsmodels.tsa.ar_model import AutoReg
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

# model = AutoReg(train_df.iloc[:,:1], 3)
# results = model.fit()
# # print(f'Lag: {results.k_ar}')
# # print(f'Coefficients: {results.params}')
# predictions = results.predict()
# arx_mse = mean_squared_error(test_df, predictions[test_df.index])
# arx_mae = mean_absolute_error(test_df, predictions[test_df.index])
# print(f'ARX MSE: {arx_mse}\n ARX MAE: {arx_mae}')

# fix, ax = plt.subplots()
# predictions[test_df.index].plot(ax=ax)
# test_df.plot(ax=ax)


# pred_grouped = predictions[test_df.index]
# pred_grouped = pred_grouped.groupby([pred_grouped.index.dayofweek, pred_grouped.index.hour]).mean()
# #preds = results.predict(test_df['TestBldg'])
# # %%
# from statsmodels.tsa.api import VAR
# var_mod = VAR(train_norm.iloc[:,3:].transpose())
# var_res = var_mod.fit()
# var_res.summary()

# %%

#create a lagmat for each building for every hour.
def create_lag(df,maxlag):
    '''create a lag matrix of df
        maxlag: the desired lag
    '''
    #TODO use_pandas=True on lagmat doesnt seem to work, this is a hacky way of forcing it.
    cols = ['L.0','L.1','L.2','L.3']
    df_lag = df.stack().groupby(level=1).apply(lambda g: pd.DataFrame( \
        lagmat(g,maxlag,trim='both',original='in'), \
        index=g.droplevel(1).index[3:],columns = cols)).sort_index()

    hourly_avg = df.groupby([df.index.dayofweek,
                             df.index.hour]).mean().mean(axis=1)
    hourly_avg.name = 'hourly_avg'
    df_lag['day'] = df_lag.index.get_level_values(1).dayofweek
    df_lag['hour'] = df_lag.index.get_level_values(1).hour
    df_lag = df_lag.join(hourly_avg, on=['day','hour'])
    # df_lag = df_lag.drop(columns=['day','hour'])
    return df_lag

#create one for both the training and testing data so they are the same shape for OLS
train_lag = create_lag(train_df,3)
test_lag = create_lag(test_df,3)

# test_lag = test_df.stack().groupby(level=1).apply(
#     lambda g: pd.DataFrame(lagmat(g, 3, trim="both", original="in"), index=g.droplevel(1).index[3:], columns = ["L.0", "L.1", "L.2", "L.3"])
# ).sort_index()
# test_lag["day"] = test_lag.index.get_level_values("TestTime").dayofweek
# test_lag["hour"] = test_lag.index.get_level_values("TestTime").hour
# test_lag = test_lag.join(hourly_avg, on=["day","hour"])

#Fit an OLS model
ols_cols = ['L.1','L.2','L.3','hourly_avg']
y = train_lag['L.0']
X = train_lag[ols_cols]
ols_mod = sm.OLS(y, X)
ols_fit = ols_mod.fit()
#predict for the hours covered by test_df
ols_pred = ols_fit.predict(test_lag[ols_cols])
ols_pred = ols_pred.droplevel(0)
ols_pred.name = 'ARX(3)'
#plot the full week results
fig, ax = plt.subplots()
test_df.plot(ax=ax, label='Test Building')
ols_pred.plot(ax=ax, label='ARX(3)')
ax.legend()

pred_subset = ols_pred[test_df.index]

#print the MAE and MSE results. Drop the first 3 rows since they are NaN in ols_pred
arx_mse = mean_squared_error(test_df[3:], pred_subset[3:])
arx_mae = mean_absolute_error(test_df[3:],pred_subset[3:])
print(f'ARX MSE: {arx_mse}\n ARX MAE: {arx_mae}')
# %%
#group the subset by DoW for plotting
pred_subset = pred_subset.groupby(pred_subset.index.dayofweek)
#plot DoW with Avg, ARX, and Test data
fig, axes = plt.subplots(3,3,figsize=(15,10),sharey=True)
labels = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
for (key, axes) in zip(group_keys,axes.flatten()):
    grouped.xs(key).groupby('hour').mean().transpose().mean().plot(ax=axes,
         label='DoW Avg')
    grouped_test.xs(key).plot(ax=axes,label='Test Building',title=labels[key])
    axes.plot(pred_subset.get_group(key).values, label='ARX')
    # pred_subset.get_group(key).plot(ax=ax, )
    axes.legend()
    axes.set_xlabel('Hour')
    axes.set_ylabel('Normalized Power [kW]')
    plt.tight_layout()
# %%
