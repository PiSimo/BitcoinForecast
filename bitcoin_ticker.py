#!/usr/bin/python3

##
## run the code for about 2/3 days
##

import requests
import time

f = open("dataset.csv","a")
keys = ["price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]

while True:
  data = requests.get("https://api.coinmarketcap.com/v1/ticker/bitcoin/").json()[0]
  price = requests.get("https://btc-e.com/api/3/ticker/btc_usd").json()
  for i in data.keys():
    if i in keys:
      f.write(str(data[i])+",")
  f.write("{},{},{}".format(price["btc_usd"]["high"],price["btc_usd"]["low"],price["btc_usd"]["avg"]))
  f.write("\n")
  f.flush()
  time.sleep(5*60)
