#!/usr/bin/python3

##
## run the code for about 2/3 days
##

import requests
import time

f = open("dataset.csv","a")

while True:
  data = requests.get("https://api.blockchain.info/stats").json()
  price = requests.get("https://btc-e.com/api/3/ticker/btc_usd").json()
  for i in data.keys():
    if "{}".format(i) != "timestamp" and "{}".format(i) != "market_price_usd":
      f.write(str(data[i])+",")
  f.write("{},{},{}".format(price["btc_usd"]["high"],price["btc_usd"]["low"],price["btc_usd"]["avg"]))
  f.write("\n")
  f.flush()
  time.sleep(9*60)
