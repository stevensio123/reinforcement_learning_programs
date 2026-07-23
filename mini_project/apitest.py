import json
import requests

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
    url_data = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey=53VSVGZSPI7P8JRT'
    response_data = requests.get(url_data)
    data = response_data.json()

    print(data["Meta Data"])

else:
    print("No symbol found.")