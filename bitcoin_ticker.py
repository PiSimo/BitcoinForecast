#!/usr/bin/python3

import requests
import time

f = open("dataset.csv","a")

while True:
data = requests.get("https://api.blockchain.info/stats").json()
ticked = requests.get("https://blockchain.info/ticker").json()
for i in data.keys():
if "{}".format(i) != "timestamp" and "{}".format(i) != "market_price_usd":
f.write(str(data[i])+",")
f.write("{},{},{}".format(ticked["USD"]["sell"],ticked["USD"]["buy"],ticked["USD"]["15m"]))
f.write("\n")
f.flush()
time.sleep(16*60)
