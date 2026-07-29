import matplotlib.pyplot as plt
import numpy as np
import os

def volatility(return_list: list):
    daily_return_vol = np.std(return_list)
    annual_return_vol = daily_return_vol * 16 # Rule of 16 to convert daily to annual

    return daily_return_vol, annual_return_vol

def autocorrelation(return_list: list, lag_time=1):
    daily_return_mean = np.mean(return_list)
    daily_return_var = np.var(return_list)
    autocovariance_numer = 0

    for idx in range(len(return_list)):
        autocovariance += ((return_list[idx] - daily_return_mean) * (return_list[idx + lag_time] - daily_return_mean))

    autocovariance = autocovariance_numer / len(return_list)

    return autocovariance/daily_return_var

def moving_average_list(return_list: list, n):
    reverse_list = return_list.reverse()
    MA_list = []
    for idx in range(len(reverse_list) - n):
        MA_list.append(sum(return_list[idx:idx+n]) / n)

    return MA_list

def MACD_plot(return_list: list, time_int_1=12, time_int_2=26):
    ...