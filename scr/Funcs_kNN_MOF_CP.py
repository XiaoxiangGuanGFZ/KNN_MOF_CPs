# -*- coding: utf-8 -*-
"""
FUNCTIONs for disaggregation model: k nearest neighbour resampling +
method of fragments + circulation patterns + multisite
multivariable: precipitation and air temperature (disaggregated separately)

Created on 10.01.2022
Last updated on 19.01.2022
@author: Xiaoxiang Guan (guan.xiaoxiang@gfz-potsdam.de)
"""

import datetime
import numpy as np
import math
import pandas as pd

def FILTER_season(Daily_obs, season='summer'):
    """
    seasons: summer or winter
    summer: from May to October (5-10); Winter: from November to next April (10-4)
    Parameters:
    ----------
        Daily_obs, dataframe, 2D
        season: indicate the seasons, value range ['summer', 'winter']
    :return
    output: filtered dataframe 'out'
    """
    month_values = Daily_obs['month']
    if season.lower() not in ['summer', 'winter']:
        print('parameter season value error: "summer" or "winter", case insensitive')
        # return None
    else:
        if season.lower() == 'summer':
            out = Daily_obs[(month_values >= 5) & (month_values <= 10)]
        else:
            # winter
            out = Daily_obs[(month_values < 5) | (month_values > 10)]
        return out  # filtered Daily_obs to be returned


def FILTER_CP(Daily_obs, CPs_obs, cp=1):
    """
    filter the Daily_obs dataframe by circulation pattern class
    Parameters:
    ----------
        Daily_obs: historical multisite variable series
        CPs_obs: dataframe of circulation pattern of each day
            *****
            CPs_obs should cover the date range (or the same row-index) of Daily_obs.
        cp: input parameter; an integer; indicating the class of circulation pattern on the target day
    :return
        the filtered(updated) Daily_obs DataFrame with one certain CP type
    """
    Daily_obs.reset_index(inplace=True)
    Daily_obs.drop(['index'], axis=1, inplace=True)
    CPs_obs.reset_index(inplace=True)
    CPs_obs.drop(['index'], axis=1, inplace=True)
    Daily_obs = Daily_obs.merge(CPs_obs, left_on=['year', 'month', 'day'], right_on=['year', 'month', 'day'])
    Daily_obs = Daily_obs[Daily_obs['CP'] == cp]
    Daily_obs.reset_index(inplace=True)
    Daily_obs.drop(['index', 'Ignore', 'CP'], axis=1, inplace=True)
    return Daily_obs


def str2date2str(y, m, d):
    """
    input year, month and day to obtain the mm-dd formatted date
    :param y: year in integer
    :param m: month in integer
    :param d: day in integer
    :return: 'mm-dd' formatted date, a string
    """
    date = str(y) + '-' + str(m) + '-' + str(d)
    out = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")[5:]
    return out  # data formate: mm-dd


def FILTER_timewindow(Daily_obs,
                      Target_date={'y': 2009, 'm': 1, 'd': 5},
                      width=14
                      ):
    """
    Monthly based disaggregation procedure;
    Filter the Daily_obs (candidate days dataframe) conditioned on the date window
    Parameters:
    ----------
        Daily_obs: historical multisite daily rainfall series
        date: target day to be disaggregated, a string variable
        width: half-length of the time window,
    :return
    the filtered dataframe
    """
    Daily_obs.reset_index(inplace=True)
    del Daily_obs['index']

    date_str = str(Target_date['y']) + '-' + str(Target_date['m']) + '-' + str(Target_date['d'])
    date_str = datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime("%Y-%m-%d")  # a string, 'yyyy-mm-dd'

    date_target = datetime.date.fromisoformat(date_str)  # target day in datetime datatype
    start = date_target - datetime.timedelta(days=width)  # compute the start day of the time window
    end = date_target + datetime.timedelta(days=width)  # end day
    daterange = [start + datetime.timedelta(days=x) for x in
                 range(0, (end - start).days + 1)]  # date series of the time window
    daterange_str_md = [i.strftime("%Y-%m-%d")[5:] for i in
                        daterange]  # extract the month and day to be used as reference

    # obtain the datatime type of the historical/observed daily rainfall day
    # to match with the time window;
    # the match is based on "%m-%d"

    date_his_md = list(map(str2date2str, Daily_obs['year'], Daily_obs['month'], Daily_obs['day']))
    # date_his_md = list()  # the length of date_his_md equals the number of rows of Daily_obs
    # for i in range(0, Daily_obs.shape[0]):
    #     date = str(Daily_obs.iloc[i, 0]) + '-' + str(Daily_obs.iloc[i, 1]) + '-' + str(Daily_obs.iloc[i, 2])
    #     date_his_md.append(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")[5:])

    # match the time window and obtain the filtering Boolean series
    boolean_daterange = list()
    for i in range(0, len(date_his_md)):
        if date_his_md[i] in daterange_str_md:
            boolean_daterange.append(True)
        else:
            boolean_daterange.append(False)
    out = Daily_obs[boolean_daterange]
    return out


def FILTER_continuity(Daily_obs,
                      Daily_targets,
                      Target_date={'y': 2009, 'm': 1, 'd': 5},
                      days_continuity=1
                      ):
    """
    Filter the historical daily rainfall dataframe
        based on dry-moisture status before and after the present.
    When there are too many sites of rainfall data to be disaggregated simultaneously and
        very little hourly rainfall to be sampled from,
        then usually this condition tends to be very strict,
        which will yield few or no candidate days.
    Parameters:
         Daily_obs: historical multisite daily rainfall series
         days_continuity: 1 or 3
            days_continuity == 1, means excluding the continuity in candidate days selection
            days_continuity == 3, means considering one day before and the day after the target day
                the criteria are to some extent strict, as the status matching is carried out based on
                multiple (at least 3) vectors or a matrix, which could yield empty output ******
        Daily_targets: pd.DataFrame, multisite daily rainfall to be disaggregated
        Target_date: the date for the target day to be disaggregated
    ----------
        return the filtered (updated) Daily_obs
    """
    # --compute the dry-wet status of the historical daily rainfall dataframe---
    # rainfall > 0, assign the status 1
    Daily_obs_status = Daily_obs.copy()
    df = Daily_obs.iloc[:, 3:]
    df[df > 0] = 1  # wet <-> 1; dry <-> 0
    Daily_obs_status.iloc[:, 3:] = df
    del df

    # calculate the status of target day(s) dataframe:
    # target_df: dataframe, rainfall for the target day (or together with neighboring 2 days)
    #     so, different days_continuity values, affect the data structure of target_df (different dimensions)
    #     if days_continuity == 1, target_df should be a onw-row pandas.DataFrame;
    #     if days_continuity == 3, target_df should be a 3*n pandas.DataFrame, where n is the number of rainfall sites
    Daily_targets.reset_index(inplace=True)
    del Daily_targets['index']
    target_df = Daily_targets[(Daily_targets['year'] == Target_date['y']) &
                              (Daily_targets['month'] == Target_date['m']) &
                              (Daily_targets['day'] == Target_date['d'])]
    target_index = int(target_df.index[0])
    if (days_continuity == 1) | (target_index == 0) | (target_index == (Daily_targets.shape[0] - 1)):
        # when the target day is the first or last day of the target dataframe,
        #   days_continuity is coerced to be 1, neighboring continuity is ignored for these two special days;
        # a one-row Pandas.DataFrame;
        target_status = target_df.copy()
        df = target_df.iloc[0, 3:]
        df[df > 0] = 1
        target_status.iloc[0, 3:] = df
        del df
        # filter by one-row dry-wet status
        boolean_status = list()  # Boolean variable holder, used to filter the candidate days dataframe
        boolean_rain_sum = list()
        for i in range(0, Daily_obs_status.shape[0]):
            boolean_status.append(
                Daily_obs_status.iloc[i, 3:].equals(target_status.iloc[0, 3:])
            )
            boolean_rain_sum.append(
                sum(Daily_obs_status.iloc[i, 3:][target_status.iloc[0, 3:] > 0]) == sum(target_status.iloc[0, 3:])
            )
        if sum(boolean_status) > 0:
            # share totally the same wet-dry status vector
            out = Daily_obs[boolean_status]
        else:  # elif sum(boolean_rain_sum) > 0:
            # In this case, no matching of the rainfall status vector
            # includes the historical data with rainfall amount at the same sites greater than 0
            out = Daily_obs[boolean_rain_sum]
        return out

    elif days_continuity == 3:
        # target_df should be a Pandas.DataFrame, with 3 rows
        # when days_continuity == 3, what should be noted is that
        # target_index can't be the fist or the last row of
        # the Daily_targets DataFrame, otherwise there will be no
        # neighboring three dry-wet status vectors
        target_df = Daily_targets.iloc[(target_index - 1):(target_index + 2), :]
        target_status = target_df.copy()
        df = target_df.iloc[:, 3:]
        df[df > 0] = 1
        target_status.iloc[:, 3:] = df
        del df
        # filter by 3 rows dry-wet status dataframe
        boolean_status = list()
        boolean_status.append(False)
        for i in range(1, Daily_obs_status.shape[0] - 1):
            # exclude the first and last one row
            boolean_status.append(
                Daily_obs_status.iloc[(i - 1):(i + 1), 3:].equals(target_status.iloc[:, 3:])
                # this condition is to some extent strict, hard to be met.
                # could yield no candidates.
            )
        boolean_status.append(False)
        out = Daily_obs[boolean_status]
        return out
    else:
        print("Error: parameter 'days_continuity' can only be set as 1 or 3!")


def FILTERs(
        Daily_obs,  # first 3 parameters are expected to be transferred with pd.DataFrames
        Daily_targets,
        CPs_obs,
        cp=1, days_continuity=1,  # set default values
        season='summer',
        target_date={'y': 2000, 'm': 4, 'd': 15},
        width=0
):
    """
    filter the candidate days
    Parameters:
    ------
        Daily_obs: historical/observed multisite series
        Daily_targets:
            dataframe, multisite daily rainfall to be disaggregated
            used to extract the rainfall vector (days_continuity=1) or matrix (days_continuity=3)
        CPs_his: dataframe of circulation pattern of each day
        days_continuity: integer value (scalar); value range: [0, 1, 3]
            0: deactivate this filtering
            1: disaggregation without considering rainstorm continuity
            3: consider one day before and after the target day
        cp: integer value (scalar)
            0: disaggregation without considering circulation pattern classifications
            cp > 0: cp represents the class of the CP
        season: whether consider the seasons in disaggregation model
            character, scalar, value range: ['None', 'winter', 'summer']
            'None': disaggregation without considering the seasons
        width: integer scalar value,
            0: exclude the time window in candidate days identification
            >0: decide the half-width of the time window
    ------
    The parameters with default values should be explicitly assigned when called.
    Finally return the filtered historical references (as the potential candidates): updated/filter Daily_obs
    """
    # Hierarchical filters: the filter order is important.
    # the continuity condition should be first filtered,
    # because continuity filter should be based on original (Chronological) order of historical daily
    # rainfall dataframe, in case other filters destroy the orders.
    Daily_obs_copy = Daily_obs.copy()

    if cp > 0:
        Daily_obs = FILTER_CP(Daily_obs,
                              CPs_obs=CPs_obs, cp=cp
                              )
        # when there is no candidate day for the target day,
        # this condition is coerced invalid
        # if out.shape[0] > 0:
        #     Daily_obs = out.copy()
        #     del out
    if season.lower() in ['summer', 'winter']:
        Daily_obs = FILTER_season(Daily_obs,
                                  season=season
                                  )

    if width > 0:
        Daily_obs = FILTER_timewindow(Daily_obs,
                                      Target_date=target_date,
                                      width=width
                                      )
    if days_continuity in [1, 3]:
        out = FILTER_continuity(Daily_obs,
                                Daily_targets,
                                Target_date=target_date,
                                days_continuity=days_continuity
                                )
        if out.shape[0] == 0:
            out = FILTER_continuity(Daily_obs_copy,  # a special setting
                                    Daily_targets,
                                    Target_date=target_date,
                                    days_continuity=days_continuity
                                    )
        Daily_obs = out.copy()
        del out
    return Daily_obs


def Manhattan_distance(x1, x2):
    """
    Manhattan distance is calculated as the sum of the absolute differences between the two vectors.
    Parameters:
        x1, x2 two vectors (lists) sharing the same length
    """
    out = np.sum(np.abs(np.array(x1) - np.array(x2)))
    return out


def Euclidean_distance(x1, x2):
    """
    Parameters:
        x1, x2 two vectors (lists) sharing the same length
    """
    D = 0
    for i in range(0, len(x1)):
        D += (x1[i] - x2[i]) ^ 2
    return math.sqrt(D)


def Normalization_maxmin(x):
    """
    Min-max feature scaling
    x: a list of numeric values (a vector) to be inputted
    return a list (vector)
    """
    x = np.array(x)
    return list((x - x.min()) / (x.max() - x.min()))


def Stantardization(x):
    """
    standardization with a square root
    x: a list of numeric values (a vector or matrix);
    return a list too
    """
    x = np.array(x)
    v_mean = np.mean(x)
    v_std = np.std(x)
    out = (x - v_mean) / v_std
    return out


def Distance_kNN(filtered, target_list, method='manhattan', standardization=True):
    """
    Calculate the distance between the target and candidate day
    default: the distance is calculated with Manhattan method

    According to the distances, sort the potential candidate days in
    ascending order and then select the first k smallest distances

    Before calculating the distance, the rainfall vector needs standardization or normalization
    Parameters:
    ------
        filtered:
            historical daily multisite rainfall (or air temperature) dataframe after filtering
            should have excluded the dry status (rainfall == 0 for each site) in the filtering
            2 dimensions, number of columns should be greater than 1
        target_list: a list of the daily rainfall volumes for the multiple sites (a vector)
        method: ['Euclidean', 'Manhattan']

    ------
    """
    if standardization:
        filtered.iloc[:, 3:] = Stantardization(filtered.iloc[:, 3:])
        target = list(Stantardization(target_list))
    else:
        target = list(target_list)
    n = filtered.shape[0]
    k = math.floor(math.sqrt(n))  # select the k nearest candidates
    distances = list()
    for i in range(0, n):
        if method.lower() == 'manhattan':
            distances.append(
                Manhattan_distance(target, filtered.iloc[i, 3:])
            )
        else:
            # Euclidean distance approach
            distances.append(
                Euclidean_distance(target, filtered.iloc[i, 3:])
            )
    filtered['distance'] = distances
    filtered = filtered.dropna()  # drop rows with NA values
    # ascending, to find the first the k smallest distances
    filtered = filtered.sort_values(by=['distance'],
                                    ascending=True
                                    )  # ascending is the default sorting method
    return filtered.head(k)


def Resample_Weights(Distance_df):
    """
    Calculate the sampling weights for each candidate day;
    Sample once and obtain the candidate day
    Parameters:
    ------
        Distance_df: historical multisite daily dataframe with a "distance" column
        return a one-row dataframe, as a finally resampled candidate day
    ------
    """
    d = Distance_df['distance']
    if sum(d == 0) == 1:
        out = Distance_df.head(1)
    elif sum(d == 0) > 1:
        Candidates_distance = Distance_df[d == 0]
        out = Candidates_distance.sample(1)
    else:
        # no distance equals 0
        weights = 1 / d / sum(1 / d)
        out = Distance_df.sample(1, weights=weights)  # sample once
    return out


def Reassign_fragments(candidate_hourly, target_day_list,
                       variable='prec',
                       candidate_date={'y': 2009, 'm': 1, 'd': 9},
                       target_date={'y': 2002, 'm': 1, 'd': 15}
                       ):
    """
    Assign the fragments of the candidate day (multisite) to the target day (multisite)
    by multiplying the daily total rainfall, OR
    by add the deviation to mean daily air temperature
    Parameters:
    ------
        candidate_hourly: the dataframe for hourly multisite rainfall (temperature) data series
        target_day_list: the value vector (multisite) for the target day
            which should be a list, with the length as the number of rainfall sites
        variable: toggle button, 'prec': precipitation; 'tem': air temperature
        candidate_date: a dictionary indicating the historical day obtained from kNN resampling
    ------
    output:
        return the hourly rainfall dataframe for the target day at multisite scale.
    """
    Boolean_target = (candidate_hourly['year'] == candidate_date['y']) & \
                     (candidate_hourly['month'] == candidate_date['m']) & \
                     (candidate_hourly['day'] == candidate_date['d'])
    candidate_hourly = candidate_hourly[Boolean_target]  # nrows = 24 (hours), obtain the potential fragments
    # variable holder for disaggregated hourly rainfall dataframe
    # the year, month and day columns need updating, as those of the target day
    disagg_output = candidate_hourly.copy()
    disagg_output = disagg_output.reset_index().drop(['index'], axis=1)
    disagg_output['year'] = target_date['y']
    disagg_output['month'] = target_date['m']
    disagg_output['day'] = target_date['d']
    if variable == 'prec':
        for i in range(0, len(target_day_list)):
            if target_day_list[i] <= 0:
                # no rainfall in ith site, daily rainfall equals 0
                disagg_output.iloc[:, i + 4] = 0
            else:
                Fragments = np.array(list(candidate_hourly.iloc[:, i + 4]))
                Fragments = Fragments / Fragments.sum()
                Fragments_target = Fragments * target_day_list[i]
                disagg_output.iloc[:, i + 4] = list(Fragments_target.round(1))

    elif variable == 'tem':
        # fragments for hourly air temperature reassignment
        for i in range(0, len(target_day_list)):
            Fragments = np.array(list(candidate_hourly.iloc[:, i + 4]))
            Fragments = Fragments - Fragments.mean()
            Fragments_target = Fragments + target_day_list[i]
            disagg_output.iloc[:, i + 4] = list(Fragments_target.round(1))
    else:
        print('Error in attribute "variable"!')
    return disagg_output


def Generate_Zero_Prec_hourly(candidate_hourly, target_date={'y': 2000, 'm': 1, 'd': 15}):
    """
    The regional status is dry, meaning no rainfall was recorded at any site.
    The disaggregation procedure generates 0 for each hour in the day for each site.
    """
    out = candidate_hourly.head(24)
    out['year'] = target_date['y']
    out['month'] = target_date['m']
    out['day'] = target_date['d']
    out.iloc[:, 4:] = 0
    return out


def MOF_kNN_mul(Daily_target, Hourly_his, CPs_obs, variable='prec', CP=False, season=False, continuity=1):
    """
    Rainfall temporal disaggregation procedures:
    multisite(mul) + method of fragments(MOF) + k-nearest neighboring resampling(kNN) +
        circulation pattern(CP) based or monthly based
    :parameter
    Daily_target: daily multisite rainfall (or air temperature) series to be disaggregated
    Hourly_his: hourly observed series as candidates
    CPs_obs: circulation pattern series
    variable: toggle button, 'prec': precipitation; 'tem': air temperature
    CP: toggle button for CP;
        if True, base the disaggregation on circulation pattern classification
        if CP = False, base the disaggregation on monthly scale.
    season: toggle button for season; True reveals the disaggregation procedures take into account the season
        only VALID when CP == True.
        Optional: season (summer and winter) classification
    continuity: value range: [0,1,3]
    :return:
    we should output both the disaggregated hourly series and the intermediate results
    return a dictionary
    """
    # aggregate to obtain historical multisite daily data
    # as reference to calculate the distance between candidate days and target day
    if variable == 'prec':
        # aggregation method: sum
        Daily_his = Hourly_his.groupby(['year', 'month', 'day']).aggregate(func=sum).reset_index()
    else:
        # average/mean
        Daily_his = Hourly_his.groupby(['year', 'month', 'day']).aggregate(func=np.mean).reset_index()
    Daily_his.drop(['hour'], axis=1, inplace=True)  # remove the 'hour' column

    Target_wet_date = []  # store the date of the day with rainfall
    Target_wet_Cans_size = []  # store the corresponding number of candidates of the targeted rainy day
    for i in range(0, Daily_target.shape[0]):
        # ------------ Loop for disaggregation (each day) -------------
        Target_year = Daily_target.iloc[i, 0]
        Target_mon = Daily_target.iloc[i, 1]
        Target_day = Daily_target.iloc[i, 2]
        Target_date_str = str(Target_year) + '-' + str(Target_mon) + '-' + str(Target_day)
        # a 'yyyy-mm-dd'-formatted string (date of the target day)
        Target_date_str = datetime.datetime.strptime(Target_date_str, '%Y-%m-%d').strftime("%Y-%m-%d")
        Target_date_dic = {'y': Target_year, 'm': Target_mon, 'd': Target_day}  # a dictionary (data structure)

        # Obtain the list (a vector) of the multisite variable to be disaggregated
        target_day_list = Daily_target.iloc[i, 3:]

        if (sum(target_day_list) <= 0) & (variable == 'prec'):
            # dry status for the target day (no rain), in the disaggregation procedure of precipitation
            Hourly_target_day = Generate_Zero_Prec_hourly(
                Hourly_his.head(24),
                target_date=Target_date_dic
            )
        else:
            # -------- filtering to obtain the candidates pool ---------------
            if CP:
                # CP == True, base disaggregation on circulation pattern
                # column names of CPs_obs: year  month  day  Ignore  CP
                Target_CP = CPs_obs[(CPs_obs['year'] == Target_year) &
                                    (CPs_obs['month'] == Target_mon) &
                                    (CPs_obs['day'] == Target_day)].iloc[0, 4]  # Target_CP is an integer
                if season:
                    # season == True
                    if Target_mon in range(5, 11):
                        Target_season = 'summer'  # 5, 6, 7, 8, 9, 10; total 6 summer months
                    else:
                        Target_season = 'winter'
                else:
                    # season == False
                    Target_season = 'None'
                # wet status (target day): kNN+MOF+CPs
                # filter the candidates with the conditions: CP types, seasons (optional)
                filtered_df = FILTERs(
                    Daily_his,
                    Daily_target,
                    CPs_obs,
                    cp=Target_CP,
                    days_continuity=continuity,
                    season=Target_season,
                    target_date=Target_date_dic,
                    width=0
                )
            else:
                # CP == False, indicating that disaggregation is monthly based
                # cp and season are invalid now for filtering
                filtered_df = FILTERs(
                    Daily_his,
                    Daily_target,
                    CPs_obs,  # this also is deactivated
                    cp=0,
                    days_continuity=continuity,
                    season='None',
                    target_date=Target_date_dic,
                    width=14  # time window width with target day as the center
                )
            # ---------distance (similarity) computation and candidate sampling-------------
            Target_wet_date.append(Target_date_str)
            Target_wet_Cans_size.append(filtered_df.shape[0])  # the sample size of the candidate pools

            if filtered_df.shape[0] > 1:
                # Calculate the distance between target and candidates, Sort the distances and Sample with weights
                Bool_standardization = True if variable == 'prec' else False  # True for distance calculation.
                Distance_df = Distance_kNN(
                    filtered_df,
                    target_day_list,
                    method='manhattan',
                    standardization=Bool_standardization
                )
                if Distance_df['distance'].iloc[0] == 0:
                    # the closest distance is 0, indicating the candidate is the 'same' as the target (a special case)
                    candidate_day = Distance_df.head(1)
                else:
                    candidate_day = Resample_Weights(Distance_df)
            else:
                candidate_day = filtered_df.copy()  # only one-row DataFrame
            # ------- candidate fragments and reassignment--------
            # obtain the year, month, day of the candidate day
            Can_year = candidate_day.iloc[0, 0]  # year of the target day's date, an integer
            Can_mon = candidate_day.iloc[0, 1]  # month
            Can_day = candidate_day.iloc[0, 2]  # day
            # Obtain the fragments of the candidate day and assign to the target day
            Hourly_target_day = Reassign_fragments(
                Hourly_his, target_day_list,
                variable=variable,
                candidate_date={'y': Can_year, 'm': Can_mon, 'd': Can_day},
                target_date=Target_date_dic
            )
        print(Target_date_str + ': Done')
        # disaggregation done for one iteration.
        if i == 0:
            out = Hourly_target_day
        else:
            out = pd.concat([out, Hourly_target_day])
    out = out.reset_index().drop(['index'], axis=1)
    # ---all iteration done---
    Cans_size = pd.DataFrame.from_dict({'Date': Target_wet_date,
                                        'Size': Target_wet_Cans_size})
    results = {
        "Hourly_results": out,
        "Candidates_size": Cans_size
    }
    return results

def MOF_kNN_mul_uncertainty(Daily_target, Hourly_his, CPs_obs,
                            variable='prec', CP=False, season=False, continuity=1, runs=50
                            ):
    """
    Rainfall temporal disaggregation procedures:
    multisite(mul) + method of fragments(MOF) + k-nearest neighboring resampling(kNN) +
        circulation pattern(CP) based or monthly based
    :parameter
    Daily_target: daily multisite rainfall (or air temperature) series to be disaggregated
    Hourly_his: hourly observed series as candidates
    CPs_obs: circulation pattern series
    variable: toggle button, 'prec': precipitation; 'tem': air temperature
    CP: toggle button for CP;
        if True, base the disaggregation on circulation pattern classification
        if CP = False, base the disaggregation on monthly scale.
    season: toggle button for season; True reveals the disaggregation procedures take into account the season
        only VALID when CP == True.
        Optional: season (summer and winter) classification
    continuity: value range: [0,1,3]
    :return:
    we should output both the disaggregated hourly series and the intermediate results
    return a dictionary
    """
    # aggregate to obtain historical multisite daily data
    # as reference to calculate the distance between candidate days and target day
    if variable == 'prec':
        # aggregation method: sum
        Daily_his = Hourly_his.groupby(['year', 'month', 'day']).aggregate(func=sum).reset_index()
    else:
        # average/mean
        Daily_his = Hourly_his.groupby(['year', 'month', 'day']).aggregate(func=np.mean).reset_index()
    Daily_his.drop(['hour'], axis=1, inplace=True)  # remove the 'hour' column

    # Target_wet_date = []  # store the date of the day with rainfall
    # Target_wet_Cans_size = []  # store the corresponding number of candidates of the targeted rainy day
    for i in range(0, Daily_target.shape[0]):
        # ------------ Loop for disaggregation (each day) -------------
        Target_year = Daily_target.iloc[i, 0]
        Target_mon = Daily_target.iloc[i, 1]
        Target_day = Daily_target.iloc[i, 2]
        Target_date_str = str(Target_year) + '-' + str(Target_mon) + '-' + str(Target_day)
        # a 'yyyy-mm-dd'-formatted string (date of the target day)
        Target_date_str = datetime.datetime.strptime(Target_date_str, '%Y-%m-%d').strftime("%Y-%m-%d")
        Target_date_dic = {'y': Target_year, 'm': Target_mon, 'd': Target_day}  # a dictionary (data structure)

        # Obtain the list (a vector) of the multisite variable to be disaggregated
        target_day_list = Daily_target.iloc[i, 3:]

        if (sum(target_day_list) <= 1) & (variable == 'prec'):
            # dry status for the target day (no rain), in the disaggregation procedure of precipitation
            Hourly_target_day = Generate_Zero_Prec_hourly(
                Hourly_his.head(24),
                target_date=Target_date_dic
            )
            Hourly_target_day = pd.concat(
                [Hourly_target_day, pd.Series([0] * 24, name='run_index')],
                axis=1
            )
            out = Hourly_target_day.copy()
            for r in range(1, runs):
                Hourly_target_day['run_index'] = r
                out = pd.concat([out, Hourly_target_day], ignore_index=True, axis=0)
        else:
            # -------- filtering to obtain the candidates pool ---------------
            if CP:
                # CP == True, base disaggregation on circulation pattern
                # column names of CPs_obs: year  month  day  Ignore  CP
                Target_CP = CPs_obs[(CPs_obs['year'] == Target_year) &
                                    (CPs_obs['month'] == Target_mon) &
                                    (CPs_obs['day'] == Target_day)].iloc[0, 4]  # Target_CP is an integer
                if season:
                    # season == True
                    if Target_mon in range(5, 11):
                        Target_season = 'summer'  # 5, 6, 7, 8, 9, 10; total 6 summer months
                    else:
                        Target_season = 'winter'
                else:
                    # season == False
                    Target_season = 'None'
                # wet status (target day): kNN+MOF+CPs
                # filter the candidates with the conditions: CP types, seasons (optional)
                filtered_df = FILTERs(
                    Daily_his,
                    Daily_target,
                    CPs_obs,
                    cp=Target_CP,
                    days_continuity=continuity,
                    season=Target_season,
                    target_date=Target_date_dic,
                    width=0
                )
            else:
                # CP == False, indicating that disaggregation is monthly based
                # cp and season are invalid now for filtering
                filtered_df = FILTERs(
                    Daily_his,
                    Daily_target,
                    CPs_obs,  # this also is deactivated
                    cp=0,
                    days_continuity=continuity,
                    season='None',
                    target_date=Target_date_dic,
                    width=14  # time window width with target day as the center
                )
            # ---------distance (similarity) computation and candidate sampling-------------
            if filtered_df.shape[0] > 1:
                # Calculate the distance between target and candidates, Sort the distances and Sample with weights
                Bool_standardization = True if variable == 'prec' else False  # True for distance calculation.
                Distance_df = Distance_kNN(
                    filtered_df,
                    target_day_list,
                    method='manhattan',
                    standardization=Bool_standardization
                )
                for r in range(0, runs):
                    candidate_day = Resample_Weights(Distance_df)
                    Can_year = candidate_day.iloc[0, 0]  # year of the target day's date, an integer
                    Can_mon = candidate_day.iloc[0, 1]  # month
                    Can_day = candidate_day.iloc[0, 2]  # day
                    # Obtain the fragments of the candidate day and assign to the target day
                    Hourly_target_day = Reassign_fragments(
                        Hourly_his, target_day_list,
                        variable=variable,
                        candidate_date={'y': Can_year, 'm': Can_mon, 'd': Can_day},
                        target_date=Target_date_dic
                    )
                    Hourly_target_day = pd.concat(
                        [Hourly_target_day, pd.Series([r] * 24, name='run_index')],
                        axis=1
                    )
                    if r == 0:
                        out = Hourly_target_day
                    else:
                        out = pd.concat([out, Hourly_target_day], ignore_index=True, axis=0)
            else:
                candidate_day = filtered_df.copy()  # only one-row DataFrame
            # ------- candidate fragments and reassignment--------
            # obtain the year, month, day of the candidate day
                Can_year = candidate_day.iloc[0, 0]  # year of the target day's date, an integer
                Can_mon = candidate_day.iloc[0, 1]  # month
                Can_day = candidate_day.iloc[0, 2]  # day
                # Obtain the fragments of the candidate day and assign to the target day
                Hourly_target_day = Reassign_fragments(
                    Hourly_his, target_day_list,
                    variable=variable,
                    candidate_date={'y': Can_year, 'm': Can_mon, 'd': Can_day},
                    target_date=Target_date_dic
                )
                Hourly_target_day = pd.concat(
                    [Hourly_target_day, pd.Series([0] * 24, name='run_index')],
                    axis=1
                )
                out = Hourly_target_day.copy()
                for r in range(1, runs):
                    Hourly_target_day['run_index'] = r
                    out = pd.concat([out, Hourly_target_day], ignore_index=True, axis=0)

        print(Target_date_str + ': Done')
        # disaggregation done for one iteration.
        if i == 0:
            results = out
        else:
            results = pd.concat([results, out])
    results = results.reset_index().drop(['index'], axis=1)
    # ---all iteration done---
    return results  # a pd.DataFrame
