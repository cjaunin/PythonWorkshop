#!/usr/bin/env python3
########################################################################################################################
# EMF-QARM - Python Workshop - Main
# Authors: Maxime Borel, Coralie Jaunin
# Creation Date: February 27, 2023
# Revised on: February 27, 2023
########################################################################################################################
# load package
import pickle
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, DateOffset
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
########################################################################################################################
# load modules
from Code.project_parameters import path, start_date, end_date, firm_sample_ls
from Code.project_functions import winsorize
########################################################################################################################
# open data
path_to_file = path.get('Inputs') + '/SP500.csv'
sp500_df = pd.read_csv(path_to_file, delimiter=',')
########################################################################################################################
# select sample data
# format date variables
sp500_df['clean_date'] = pd.to_datetime(sp500_df['date'], format='%Y-%m-%d')
# time period
mask = (sp500_df.clean_date >= pd.to_datetime(start_date)) & (sp500_df.clean_date <= pd.to_datetime(end_date))
sp500_subset_df = sp500_df[mask].copy()
# monthly
sp500_subset_df.sort_values(['Name', 'clean_date'], inplace=True)
sp500_subset_df['clean_yearmonth'] = sp500_subset_df.clean_date.dt.to_period('M')
sp500_subset_df = sp500_subset_df.groupby(['clean_yearmonth', 'Name']).last().reset_index()
# firm sample
mask = sp500_subset_df['Name'].astype('category').isin(firm_sample_ls)
sp500_subset_df = sp500_subset_df[mask]
########################################################################################################################
# slices
sp500_subset_close_df = sp500_subset_df.pivot(index=['clean_date'], columns=['Name'], values=['close'])
sp500_subset_close_df.columns = sp500_subset_close_df.columns.get_level_values(1)
sp500_subset_close_df.iloc[:, 0]
sp500_subset_close_df.iloc[4:12, 0]
sp500_subset_close_df.loc[:, 'AAL']
sp500_subset_close_df.loc[pd.to_datetime('2016-06-30'), 'AAL']
sp500_subset_close_df.loc[sp500_subset_close_df.index > pd.to_datetime('2016-06-30'), 'AAL']
sp500_subset_close_df.loc[sp500_subset_close_df.index > pd.to_datetime('2016-06-30'), ['AAL', 'T']]
########################################################################################################################
# monthly returns
# a very bad idea
sp500_subset_df_copy = sp500_subset_df.copy()
sp500_subset_df['ret'] = sp500_subset_df.groupby('Name', group_keys=False)['close'].apply(
    lambda s: np.log(s) - np.log(s.shift(1)))
# what can go wrong???? --> missing observation <-- and --> weekends and bank holidays <--
mask = sp500_subset_df.clean_date != pd.to_datetime('2016-06-30')
sp500_subset_df = sp500_subset_df[mask]
sp500_subset_df['ret_missing'] = sp500_subset_df.groupby('Name', group_keys=False)['close'].apply(
    lambda s: np.log(s) - np.log(s.shift(1)))
# one of many solution using the original dataframe
sp500_subset_df = sp500_subset_df_copy.copy()
sp500_subset_df['monthend_date'] = sp500_subset_df.clean_date + MonthEnd(0)
lagged_dates = sp500_subset_df.monthend_date + DateOffset(months=-1) + MonthEnd(0)
lagged_index = pd.MultiIndex.from_arrays([sp500_subset_df['Name'], lagged_dates])
sp500_subset_df = sp500_subset_df.set_index(['Name', 'monthend_date'])
sp500_subset_df['ret'] = np.log(sp500_subset_df.close) - np.log(sp500_subset_df.close.reindex(lagged_index)).values
# what if some observations are missing  --> it still works
sp500_subset_df_copy = sp500_subset_df.copy()
mask = sp500_subset_df.clean_date != pd.to_datetime('2016-06-30')
sp500_subset_df = sp500_subset_df[mask].reset_index()
lagged_dates = sp500_subset_df.monthend_date + DateOffset(months=-1) + MonthEnd(0)
lagged_index = pd.MultiIndex.from_arrays([sp500_subset_df['Name'], lagged_dates])
sp500_subset_df = sp500_subset_df.set_index(['Name', 'monthend_date'])
sp500_subset_df['ret_missing'] = np.log(sp500_subset_df.close) - np.log(sp500_subset_df.close.reindex(lagged_index)).values
print(sp500_subset_df.loc[('AAL', pd.to_datetime('2016-07-31')), ['ret', 'ret_missing']])
sp500_subset_df = sp500_subset_df_copy.copy()
# one of many solution using the pivoted dataframe
lagged_index = (sp500_subset_close_df.index + DateOffset(months=-1)).to_period('M')
sp500_subset_close_df.index = sp500_subset_close_df.index.to_period('M')
sp500_subset_ret_df = np.log(sp500_subset_close_df).values - np.log(sp500_subset_close_df.reindex(lagged_index)).values
sp500_subset_ret_df = pd.DataFrame(sp500_subset_ret_df, index=sp500_subset_close_df.index, columns=sp500_subset_close_df.columns)
########################################################################################################################
# summary statistics and correlations
sumstat_df = sp500_subset_df.describe()
print(sumstat_df.to_latex())
sumstat_tab = sumstat_df.to_latex(float_format="{:,.4f}".format)
print(sumstat_tab)
# returns by ticker
ret_sumstat_firm_df = sp500_subset_df.groupby('Name')['ret'].describe()
# correlation
corr_df = sp500_subset_ret_df.corr()
print(corr_df.to_latex(float_format="{:,.4f}".format))
########################################################################################################################
# functions
sp500_df[['close', 'volume']].describe([0.025, 0.975])
sp500_df['volume_win'] = winsorize(sp500_df[['volume']], 0.025)
sp500_df.sort_values('volume', inplace=True)
sp500_df.head(10)
########################################################################################################################
# save output
# as pickle 1
filename = '/workshop_output1.pkl'
sp500_subset_df.to_pickle(path.get('Outputs') + filename)
# as pickle 2 - when the structure you want to save is not a dataframe
filename = '/sumstat_table.txt'
with open(path.get('Outputs') + filename, 'wb') as f:
    pickle.dump(sumstat_tab, f)
# as csv
filename = '/workshop_output2.csv'
sp500_subset_df.to_csv(path.get('Outputs') + filename)
########################################################################################################################
# a nice figure
sns.set()
sns.set_context('paper', font_scale=1.2)
sns.set_style('ticks')
major_xticks = mdates.YearLocator(1, month=1, day=1)
major_xticks_fmt = mdates.DateFormatter('%Y')
minor_xticks = mdates.MonthLocator()
palette = sns.cubehelix_palette(n_colors=4, start=1, rot=1, hue=1, light=0.5, dark=0)
gs = plt.GridSpec(3, 2, hspace=0.2,
                  bottom=0.025, top=0.95, left=0.075, right=0.95,
                  height_ratios=[1, 1, 1], width_ratios=[1, 1])
fig = plt.figure(figsize=(12, 15))
# plot data
to_plot = sp500_subset_df.loc[:, ['ret', 'close']].reset_index()
to_plot['ret'] = to_plot.ret * 100
# ax1 - all prices
ax1 = fig.add_subplot(gs[0, :])
sns.lineplot(x='monthend_date', y='close', hue='Name', data=to_plot, palette=palette, ax=ax1)
# ax1 - title
ax1.title.set_text(r'Panel A: Evolution of Stock Prices')
# ax1 - axes
ax1.set_ylabel(r'Stock Price')
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax1.set_ylim(0, 75)
ax1.set_xlabel('')
ax1.xaxis.set_major_locator(major_xticks)
ax1.xaxis.set_minor_locator(minor_xticks)
ax1.xaxis.set_major_formatter(major_xticks_fmt)
ax1.set_xlim(pd.to_datetime(start_date) + MonthEnd(0), pd.to_datetime(end_date))
# ax1 - legend
ax1.legend(title='', loc=3, framealpha=1)
# ax2 - AAL returns
ax2 = fig.add_subplot(gs[1, 0])
sns.lineplot(x='monthend_date', y='ret', data=to_plot[to_plot['Name'] == 'AAL'], color=palette[0], ax=ax2)
ax2.axhline(y=0, c='0', ls='--')
# ax2 - title
ax2.title.set_text(r'Panel B: American Airline Returns')
# ax2 - axes
ax2.set_ylabel(r'Monthly Stock Returns')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax2.set_ylim(-25, 25)
ax2.set_xlabel('')
ax2.xaxis.set_major_locator(major_xticks)
ax2.xaxis.set_minor_locator(minor_xticks)
ax2.xaxis.set_major_formatter(major_xticks_fmt)
ax2.set_xlim(pd.to_datetime(start_date) + MonthEnd(0), pd.to_datetime(end_date))
# ax3 - NKE returns
ax3 = fig.add_subplot(gs[1, 1])
sns.lineplot(x='monthend_date', y='ret', data=to_plot[to_plot['Name'] == 'NKE'], color=palette[1], ax=ax3)
ax3.axhline(y=0, c='0', ls='--')
# ax3 - title
ax3.title.set_text(r'Panel C: Nike Returns')
# ax3 - axes
ax3.set_ylabel(r'Monthly Stock Returns')
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax3.set_ylim(-25, 25)
ax3.set_xlabel('')
ax3.xaxis.set_major_locator(major_xticks)
ax3.xaxis.set_minor_locator(minor_xticks)
ax3.xaxis.set_major_formatter(major_xticks_fmt)
ax3.set_xlim(pd.to_datetime(start_date) + MonthEnd(0), pd.to_datetime(end_date))
# ax4 - T returns
ax4 = fig.add_subplot(gs[2, 0])
sns.lineplot(x='monthend_date', y='ret', data=to_plot[to_plot['Name'] == 'T'], color=palette[2], ax=ax4)
ax4.axhline(y=0, c='0', ls='--')
# ax4 - title
ax4.title.set_text(r'Panel D: AT&T Returns')
# ax4 - axes
ax4.set_ylabel(r'Monthly Stock Returns')
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax4.set_ylim(-25, 25)
ax4.set_xlabel('')
ax4.xaxis.set_major_locator(major_xticks)
ax4.xaxis.set_minor_locator(minor_xticks)
ax4.xaxis.set_major_formatter(major_xticks_fmt)
ax4.set_xlim(pd.to_datetime(start_date) + MonthEnd(0), pd.to_datetime(end_date))
# ax5 - VZ returns
ax5 = fig.add_subplot(gs[2, 1])
sns.lineplot(x='monthend_date', y='ret', data=to_plot[to_plot['Name'] == 'VZ'], color=palette[3], ax=ax5)
ax5.axhline(y=0, c='0', ls='--')
# ax5 - title
ax5.title.set_text(r'Panel D: Verizon Returns')
# ax5 - axes
ax5.set_ylabel(r'Monthly Stock Returns')
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax5.set_ylim(-25, 25)
ax5.set_xlabel('')
ax5.xaxis.set_major_locator(major_xticks)
ax5.xaxis.set_minor_locator(minor_xticks)
ax5.xaxis.set_major_formatter(major_xticks_fmt)
ax5.set_xlim(pd.to_datetime(start_date) + MonthEnd(0), pd.to_datetime(end_date))
plt.show()
fig.savefig(path.get('Outputs') + '/workshop_figure.png')
