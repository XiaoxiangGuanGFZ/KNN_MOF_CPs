# -*- coding: utf-8 -*-
"""
MAIN file for disaggregation model: k nearest neighbour resampling +
method of fragments + circulation patterns + multisite

** AIR TEMPERATURE **

Created on 28.02.2022
Last updated on 02.03.2022
@author: Xiaoxiang Guan (guan.xiaoxiang@gfz-potsdam.de)
"""

import pandas as pd
import numpy as np
import os
import fnmatch
# import necessary function sources (customized)
import Funcs_Metrics
import Funcs_kNN_MOF_CP

ws = 'D:/KNN_MOF_CPs/'
"""
 Import/input the data into memory including daily synoptic variables to be disaggregated: 
    rainfall and (or) air temperature, and sub-daily (hourly) historical observations providing fragments
 Rename the columns of each dataframe 
"""

Daily_Prec_tar = pd.read_csv(
    ws + 'data/Daily_Tem.txt',
    sep='\t', header=None
)
Site_index = list(range(0, Daily_Prec_tar.shape[1] - 3))
Site_index_name = ['site' + str(i) for i in Site_index]
Daily_Prec_tar.columns = ['year', 'month', 'day'] + Site_index_name

Hourly_Prec = pd.read_csv(
    ws + 'data/Hourly_Tem.txt',
    sep='\t', header=None
)
Hourly_Prec.columns = ['year', 'month', 'day', 'hour'] + Site_index_name

# ---------------Simulation experiment setup --------------

sim_range = range(2002, 2021)  # for the period of 2002-2020
Scenarios = [4, 5, 6, 7, 8, 0]
for s in range(0, len(Scenarios)):
    if Scenarios[s] > 0:
        CLA = Scenarios[s]
        # CLA = 4
        CPs_obs = pd.read_csv(
            ws + 'data/CLA_ERA5_' + str(CLA) + '_psl-anom_1979-2021.cla',
            # data sequence in 2021 is nor complete!
            header=None, sep='\t'
        )
        CPs_obs.columns = ['year', 'month', 'day', 'Ignore', 'CP']
        CPs_obs = CPs_obs[(CPs_obs['year'] >= sim_range[0]) & (CPs_obs['year'] <= sim_range[-1])]
        CPs_obs = CPs_obs.reset_index().drop(['index'], axis=1)
        CP_toggle = True
    else:
        # Scenario is 0, condition the disaggregation at monthly scale, excluding the CPs
        CP_toggle = False
        CPs_obs = None

    for iyear in range(0, len(sim_range)):
        # historical daily rainfall dataset aggregated from hourly observations: as reference in matching
        Hourly_Prec_his = Hourly_Prec[(Hourly_Prec['year'] != sim_range[iyear])]
        Hourly_Prec_his = Hourly_Prec_his.reset_index().drop(['index'], axis=1)

        Daily_Prec = Daily_Prec_tar[(Daily_Prec_tar['year'] == sim_range[iyear])]  # to be disaggregated
        Daily_Prec = Daily_Prec.reset_index().drop(['index'], axis=1)

        DIS_results = Funcs_kNN_MOF_CP.MOF_kNN_mul(
            # the parameters are different from those in rainfall disaggregation; continuity=0 (must)
            Daily_Prec, Hourly_Prec_his, CPs_obs, variable='tem', CP=CP_toggle, season=False, continuity=0
        )
        DIS_results['Hourly_results'].to_csv(
            ws + 'results/Tem_CLA_' + str(Scenarios[s]) + '.csv',
            index=False, mode='a', header=False
        )

Scenarios = [4, 5, 6, 7, 8]  # considering the seasonality in kNN_MOF_CP
for s in range(0, len(Scenarios)):
    if Scenarios[s] > 0:
        CLA = Scenarios[s]
        # CLA = 4
        CPs_obs = pd.read_csv(
            ws + 'data/CLA_ERA5_' + str(CLA) + '_psl-anom_1979-2021.cla',
            # data sequence in 2021 is nor complete!
            header=None, sep='\t'
        )
        CPs_obs.columns = ['year', 'month', 'day', 'Ignore', 'CP']
        CPs_obs = CPs_obs[(CPs_obs['year'] >= sim_range[0]) & (CPs_obs['year'] <= sim_range[-1])]
        CPs_obs = CPs_obs.reset_index().drop(['index'], axis=1)
        CP_toggle = True
    else:
        # Scenario is 0, condition the disaggregation at monthly scale, excluding the CPs
        CP_toggle = False
        CPs_obs = None

    for iyear in range(0, len(sim_range)):
        # historical daily rainfall dataset aggregated from hourly observations: as reference in matching
        Hourly_Prec_his = Hourly_Prec[(Hourly_Prec['year'] != sim_range[iyear])]
        Hourly_Prec_his = Hourly_Prec_his.reset_index().drop(['index'], axis=1)

        Daily_Prec = Daily_Prec_tar[(Daily_Prec_tar['year'] == sim_range[iyear])]  # to be disaggregated
        Daily_Prec = Daily_Prec.reset_index().drop(['index'], axis=1)

        DIS_results = Funcs_kNN_MOF_CP.MOF_kNN_mul(
            Daily_Prec, Hourly_Prec_his, CPs_obs, variable='tem', CP=CP_toggle, season=True, continuity=0
        )
        DIS_results['Hourly_results'].to_csv(
            ws + 'results/Tem_SW_CLA_' + str(Scenarios[s]) + '.csv',
            index=False, mode='a', header=False
        )
# ---- temperature disaggregation DONE!

# --------- metrics calculation for air temperature ----------------
files = list()
for file in os.listdir(ws + 'results/'):
    if fnmatch.fnmatch(file, 'Tem_CLA_*.csv') | fnmatch.fnmatch(file, 'Tem_SW_CLA_*.csv'):
        files.append(file)
experiments = [i.replace('.csv', '') for i in files]

for e in range(0, len(experiments)):
    # e = 0
    # -- import the disaggregated hourly rainfall series --
    df = pd.read_csv(
        ws + 'results/' + files[e],
        sep=',', header=None
    )
    Site_index = list(range(0, df.shape[1] - 4))
    Site_colname = ['site' + str(i) for i in Site_index]
    df.columns = ['year', 'month', 'day', 'hour'] + Site_colname  # rename the columns
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]  # exclude the NAs

    # -- calculate the rainfall statistics --
    DF_mean = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Mean_Tem)
    Out_DF_mean = Funcs_Metrics.Metric_short_long(DF_mean, i=['year'], metric='mean', experiment=experiments[e])
    out = Out_DF_mean.copy()

    DF_std = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Std_Tem)
    Out_DF_std = Funcs_Metrics.Metric_short_long(DF_std, i=['year'], metric='std', experiment=experiments[e])
    out = pd.concat([out, Out_DF_std])

    qtls = [0.75, 0.9, 0.95, 0.97, 0.99, 0.995]
    for qtl in qtls:
        DF_ext = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Extremes_rainfall, qtl)
        Out_DF_ext = Funcs_Metrics.Metric_short_long(
            DF_ext, i=['year'], metric='qtl' + str(qtl), experiment=experiments[e]
        )
        out = pd.concat([out, Out_DF_ext])

    DF_lag1auto = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Lag_auto_correlation)
    Out_DF_lag1auto = Funcs_Metrics.Metric_short_long(
        DF_lag1auto, i=['year'], metric='lag1auto', experiment=experiments[e]
    )
    out = pd.concat([out, Out_DF_lag1auto])
    out.to_csv(ws + 'results/metrics_Tem.csv', index=False, mode='a', header=False)

    DF_dual_cor = Funcs_Metrics.Inter_site_dual_cor(df, Site_colname)
    DF_dual_cor['Experiment'] = experiments[e]
    DF_dual_cor.to_csv(ws + 'results/Inter_Cors_Tem.csv', index=False, mode='a', header=False)

# --- metrics computation for obs
df = pd.read_csv(
    ws + 'data/Hourly_Tem.txt',
    sep='\t', header=None
)
Site_index = list(range(0, df.shape[1] - 4))
Site_colname = ['site' + str(i) for i in Site_index]
df.columns = ['year', 'month', 'day', 'hour'] + Site_colname  # rename the columns
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]  # exclude the NAs

# -- calculate the statistics --
DF_mean = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Mean_Tem)
Out_DF_mean = Funcs_Metrics.Metric_short_long(DF_mean, i=['year'], metric='mean', experiment='obs')
out = Out_DF_mean.copy()

DF_std = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Std_Tem)
Out_DF_std = Funcs_Metrics.Metric_short_long(DF_std, i=['year'], metric='std', experiment='obs')
out = pd.concat([out, Out_DF_std])

qtls = [0.75, 0.9, 0.95, 0.97, 0.99, 0.995]
for qtl in qtls:
    DF_ext = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Extremes_rainfall, qtl)
    Out_DF_ext = Funcs_Metrics.Metric_short_long(
        DF_ext, i=['year'], metric='qtl' + str(qtl), experiment='obs'
    )
    out = pd.concat([out, Out_DF_ext])

DF_lag1auto = df.groupby(['year'])[Site_colname].aggregate(Funcs_Metrics.Lag_auto_correlation)
Out_DF_lag1auto = Funcs_Metrics.Metric_short_long(
    DF_lag1auto, i=['year'], metric='lag1auto', experiment='obs'
)
out = pd.concat([out, Out_DF_lag1auto])
out.to_csv(ws + 'results/metrics_Tem.csv', index=False, mode='a', header=False)

DF_dual_cor = Funcs_Metrics.Inter_site_dual_cor(df, Site_colname)
DF_dual_cor['Experiment'] = 'obs'
DF_dual_cor.to_csv(ws + 'results/Inter_Cors_Tem.csv', index=False, mode='a', header=False)

# Evaluation metrics DONE!
