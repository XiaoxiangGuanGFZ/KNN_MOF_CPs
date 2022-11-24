# -*- coding: utf-8 -*-
"""
FUNCTIONS for performance evaluation of disaggregation model
Main metrics (evaluation statistics) including common ones
and those specially designed for the feature of non-stationarity

Created on 10.01.2022
Last updated on 17.02.2022
@author: Xiaoxiang Guan (guan.xiaoxiang@gfz-potsdam.de)
"""

import numpy as np
import pandas as pd
import scipy.stats as scistat
from itertools import combinations

# ---
def Extremes_rainfall(x, qtl=0.5):
    """
    Extreme property of rainfall (50th, 75th, 90th, 95th percentiles)
    Parameter:
    ----------
    x: a list of numeric values
    y: outputted quantiles
    """
    x = np.array(x)
    x = x[x > 0]
    y = np.quantile(x, qtl)
    return round(y, 3)


def Mean_rainfall(x):
    """
    return the arithmetic average value of x (excluding zero values non-rainy)
    """
    x = np.array(x)
    x = x[x > 0]
    y = np.mean(x)
    return round(y, 3)


def Std_rainfall(x):
    """
    return the standard deviation of x
    """
    x = np.array(x)
    x = x[x > 0]
    y = np.std(x)
    return round(y, 3)


def Mean_Tem(x):
    """
    return the arithmetic average value of x
    """
    x = np.array(x)
    # x = x[x > 0]
    y = np.mean(x)
    return round(y, 3)


def Std_Tem(x):
    """
    return the standard deviation of x
    """
    x = np.array(x)
    # x = x[x > 0]
    y = np.std(x)
    return round(y, 3)


def Skewness_rainfall(x):
    """
    return the skewness (one of the moments order 3) of x
    """
    x = np.array(x)
    x = x[x > 0]
    y = scistat.skew(x)
    return round(y, 3)  # return a scalar value


def Lag_auto_correlation(x, lag=1):
    """
    Compute the partial auto-correlation in one sequence (x)
    lag: time lag, time period part; can be 1 or 2
    """
    x1 = x[lag:]
    x2 = x[:-lag]
    y = scistat.pearsonr(np.array(x1), np.array(x2))
    return np.round(y[0], 3)  # return a scalar value

# ---- correlation coefficients between sites -----
def Inter_site_cor(x, y):
    """
    A spatial correlation structure metric;
    Calculate the Pearson correlation coefficient between x any y
    parameter:
    ----------------
    x,y: a vector of the hourly rainfall at two sites
    """
    cor = scistat.pearsonr(x, y)
    return np.round(cor[0], 3)

def combine(pop, set=2):
    """
    derive the combinations (pairs), with the 'set' as size of the pairs
    :param pop: population to sample
    :param set: size of the sampled pairs
    :return: a tuple with tuples of all the combinations
    """
    return tuple(combinations(pop, set))

def Inter_site_dual_cor(df, sitenames):
    """
    frequency: Pr(Ri,24 > 0 | Ri+1,1 > 0) + Pr(Ri,24 = 0 | Ri+1,1 == 0)
    :param df: DataFrame for the multisite hourly rainfall series
    :param sitenames: the column names of the df should be ['year', 'month', 'day', 'hour'] + sitenames
    :return: return the inter-day wet-dry status continuity (frequency)
    """
    coms = combine(sitenames, 2)
    cors = list()
    for i in range(0, len(coms)):
        value = Inter_site_cor(df[coms[i][0]], df[coms[i][1]])
        cors.append(value)
    out = pd.DataFrame({
        'pair': coms,
        'coref': cors
    })
    return out


def Continuity_ratio(x, y):
    """
    a spatial metric;
    The continuity ratio defines the ratio of the mean of the precipitation at site x
    depending on whether site y is wet or dry.
    If the correlation between sites is high the continuity ratio will be relatively small,
    and if the correlation is low, it will be relatively large.

    :param x: hourly rainfall series at site x
    :param y: hourly rainfall series at site y
    :return: continuity ratio
    """
    x = np.array(x)
    y = np.array(y)
    cr = np.mean(x[(y <= 0) & (x > 0)]) / np.mean(x[(y > 0) & (x > 0)])
    return np.round(cr)


def Mean_wd_spell_len(x, dw='w'):
    """
    intra-day wet and dry spell characteristics;
    * A within-day wet spell is defined as consecutive hours of precipitation within a wet day
    * A within-day dry spell is thereafter simply defined as the consecutive dry hours that intersperse rainfall events.
    ----Parameter:
    x: a vector of hourly rainfall series in one rainy day
        (with total rainfall greater than 0.5)
    dw: wet-dry status, 'w' indicates outputting the mean spell length of wet hours
    ----Return:
    out: a vector with the length of 2, indicating the average wet and dry spell length respectively
    ----Do a test:
    x = [0, 0, 5, 0, 0, 1.2, 2.3, 4, 12, 15, 14.3, 2.3, 1.2, \
     0, 0, 0, 0, 1.2, 1.3, 5, 0, 0, 0, 0]
    """
    x = list(x)
    # n = len(x)
    wd = [float(i) > 0 for i in x]
    wns = list()
    dns = list()
    wd_enu = wd.pop()
    while len(wd) > 0:
        if wd_enu:
            # wet spell
            enu = 0 if len(wd) > 0 else 1
            while wd_enu:
                enu = enu + 1
                if len(wd) > 0:
                    wd_enu = wd.pop()
                else:
                    break
            wns.append(enu)
        else:
            # dry spell
            enu = 0 if len(wd) > 0 else 1
            while not wd_enu:
                enu = enu + 1
                if len(wd) > 0:
                    wd_enu = wd.pop()
                else:
                    break
            dns.append(enu)
    out = round(np.mean(np.array(wns)), 0) if wd == 'w' else round(np.mean(np.array(dns)), 0)
    return out

# ----- Continuity_frequency computation ----
def Continuity_frequency(x):
    """
    calculate the continuity frequency one sequence of hourly rainfall data in a year
    :param x: a vector of hourly rainfall in a year
    :return: return a scalar, indicating the continuity frequency
    """
    # remove the first and the last hour of the series, because they have no matching.
    x = list(x)
    x.pop(0)
    x.pop(-1)
    n = len(x)
    x_index = range(0, n, 2)
    cf = 0
    for i in range(0, len(x_index)):
        if (x[x_index[i]] > 0) & (x[x_index[i] + 1] > 0):
            cf += 1
        elif (x[x_index[i]] <= 0.) & (x[x_index[i] + 1] <= 0.):
            cf += 1
    return round(cf / len(x_index), 3)

def Continuity_inter_day(df, sitenames):
    """
    frequency: Pr(Ri,24 > 0 | Ri+1,1 > 0) + Pr(Ri,24 = 0 | Ri+1,1 == 0)
    :param df: DataFrame for the multisite hourly rainfall series
    :param sitenames: the column names of the df should be ['year', 'month', 'day', 'hour'] + sitenames
    :return: return the inter-day wet-dry status continuity (frequency)
    """
    df_023 = df[(df['hour'] == 0) | (df['hour'] == 23)]
    DF_cid = df_023.groupby(['year'])[sitenames].aggregate(Continuity_frequency)
    return DF_cid


def Metric_short_long(df, i=['year'], metric='qtl', experiment='CLA4'):
    df = pd.wide_to_long(df.reset_index(), stubnames='site', i=i, j='Sites')
    df.rename(columns={'site': 'value'}, inplace=True)
    df['metric'] = metric
    df['Experiment'] = experiment
    df.reset_index(inplace=True)
    return df
