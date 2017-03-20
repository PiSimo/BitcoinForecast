import requests

maxs =[]
mins =[]

def loadData(f_name):
    data  = f_name.read().split("\n")
    data = data[:len(data)-1]
    label = []
    for i in range(len(data)):
        data[i] = data[i].split(",")
        data[i] = [float(x) for x in data[i]]
        label.append(data[i][len(data[i])-1])
        data[i] = data[i][0:len(data[i])-1]
    return data[1:],label[:len(label)-1]

def reduceVector(vec,getVal=False):
    vect = []
    mx,mn = max(vec),min(vec)
    mx = mx+mn
    mn = mn-((mx-mn)*0.4)
    for x in vec:
        vect.append((x-mn)/(mx-mn))
    if not getVal:return vect
    else:return vect,mx,mn

def reduceValue(x,mx,mn):
    return (x-mn)/(mx-mn)

def augmentValue(x,mx,mn):
    return (mx-mn)*x+mn

def reduceMatRows(data):
    l = len(data[0])
    for i in range(l):
        v = []
        for t in range(len(data)):
            v.append(data[t][i])
        v,mx,mn = reduceVector(v,getVal=True)
        maxs.append(mx)
        mins.append(mn)
        for t in range(len(data)):
            data[t][i] = v[t]

    return data
def reduceCurrent(data):
    for i in range(len(data)):
        data[i] = reduceValue(data[i],maxs[i],mins[i])
    return data

def getCurrentData(label=False):
  keys = ["price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]
  vect = []
  data = requests.get("https://api.coinmarketcap.com/v1/ticker/bitcoin/").json()[0]
  bstamp = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/").json()
  bkc = requests.get("https://blockchain.info/ticker").json()
  for i in data.keys():
    if i in keys:
      vect.append(float(data[i]))
  vect.append(float(bstamp["volume"]))
  vect.append(float(bstamp["vwap"]))
  vect.append(float(bkc["USD"]["sell"]))
  vect.append(float(bkc["USD"]["buy"]))

  if label:return vect,float(bkc["USD"]["15m"])
  else : return vect
