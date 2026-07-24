import json
import requests
import math
import statistics

"""
company = input("Company name? ")

url_search = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company}&apikey=53VSVGZSPI7P8JRT"
response_search = requests.get(url_search)
data_search = response_search.json()

if len(data_search["bestMatches"]) > 0:
    index_num = 0
    response_list = []
    for response in data_search["bestMatches"]:
        new_dict = {}
        new_dict["name"] = response["2. name"]
        new_dict["symbol"] = response["1. symbol"]
        new_dict["region"] = response["4. region"]
        print(index_num, new_dict)
        response_list.append(new_dict)
        index_num += 1
    while True:
        choice = int(input("Please enter the associated index number to the company of interest: "))
        if choice >= index_num:
            print("Invalid choice, please choose again.")
        else:
            break
    symbol = response_list[choice]["symbol"]
    url_data = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=53VSVGZSPI7P8JRT'
    response_data = requests.get(url_data)
    data = response_data.json()

    log_return_list = []
    trade_info_yesterday = 0

    lst_version_of_data = list(data["Time Series (Daily)"].items())

    for trade_info_idx in range(len(data["Time Series (Daily)"])):
        trade_info_yesterday = lst_version_of_data[trade_info_idx + 1][1]["4. close"]
        log_return_list.append(math.log(lst_version_of_data[trade_info_idx]["4. close"] - trade_info_yesterday))
        


else:
    print("No symbol found.")
"""
# Temporary for local test data as alphaframe only allow 25 requests per day
with open("info.json", "r", encoding="utf-8") as file:
    data = json.load(file)

log_daily_return_list = []
trade_info_yesterday = 0

lst_version_of_data = list(data["Time Series (Daily)"].items())

for trade_info_idx in range(len(data["Time Series (Daily)"]) - 1):
    trade_info_yesterday = float(lst_version_of_data[trade_info_idx + 1][1]["4. close"])
    trade_info_today = float(lst_version_of_data[trade_info_idx][1]["4. close"])
    log_daily_return_list.append(math.log(trade_info_today / trade_info_yesterday))

daily_return_mean = statistics.mean(log_daily_return_list)
daily_return_std = statistics.stdev(log_daily_return_list)
annual_return_std = daily_return_std * 16