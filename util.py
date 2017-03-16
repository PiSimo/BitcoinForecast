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

def getCurrentData(label=False):
  data = requests.get("https://api.blockchain.info/stats").json()
  price = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/").json()
  vect = []
  for i in data.keys():
    if "{}".format(i) != "timestamp" and "{}".format(i) != "market_price_usd":
      vect.append(float(data[i]))
  vect.append(float(price["bid"]))
  vect.append(float(price["ask"]))

  if label:return vect,float(price["last"])
  else : return vect
