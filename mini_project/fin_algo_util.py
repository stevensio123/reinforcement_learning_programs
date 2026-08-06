import matplotlib.pyplot as plt
import numpy as np
import os


def volatility(input_list: list):
    daily_return_vol = np.std(input_list)
    annual_return_vol = daily_return_vol * 16  # Rule of 16 to convert daily to annual

    return daily_return_vol, annual_return_vol


def autocorrelation(input_list: list, lag=1):
    """
    The autocorrelation at lag k is defined as the ratio of the autocovariance at lag k to the variance.
    """
    arr = np.array(input_list)
    mean = np.mean(arr)
    unlagged = np.array(arr[:-lag])
    lagged = np.array(arr[lag:])
    # deviations from the mean
    dev_unlagged = unlagged - mean
    dev_lagged = lagged - mean
    # sum product of deviations
    numerator = np.sum(dev_unlagged * dev_lagged) 
    # sum squared deviations of original arr
    denominator = np.sum((arr - mean) ** 2) # this is just the variance could use np.var
    return numerator / denominator


def moving_average_list(input_list: list, lag, type="simple"):
    type_list = ["simple","weighed","exponential"]

    if type in type_list:
        input_list.reverse() # To flip data in order of oldest to newest
        MA_list = []

        match type:
            case "simple":
                for idx in range(len(input_list) - lag + 1):
                    MA_list.append(sum(input_list[idx:idx + lag]) / lag)
            case "weighed":
                total_weight = 0
                total_weight += sum(i + 1 for i in range(lag))
                for idx in range(len(input_list) - lag + 1):
                    final_sum = 0
                    for weight_idx in range(lag):
                        final_sum += input_list[idx + weight_idx] * ((weight_idx + 1) / total_weight)
                    MA_list.append(final_sum)
            case "exponential":
                MA_list.append(sum(input_list[:lag]) / lag) # First data-point is a SMA as it can't get a t - 1 datapoint following formula
                smoothing = 2 / (lag + 1)
                for idx in range(len(input_list) - lag - 1):
                    MA_list.append((input_list[idx + lag] * (smoothing / (1 + lag))) + (MA_list[-1] * (1 - (smoothing / (1 + lag)))))

    else:
        print("Please insert valid MA type: Simple, Weighed or Exponential")

    return MA_list


def MACD_plot(input_list: list, time_int_1=12, time_int_2=26): 
    ...
