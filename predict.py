
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

def getslope(plot=False):
  con = sqlite3.connect('data.db')
  data = pd.read_sql_query("SELECT rowid,* from predict", con, index_col="created") 
  print(data.tail(2))

  start = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
  stop =  datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
  #print(start)
  #print(stop)

  selected = data.loc[(start < data.index) & (data.index < stop),'cex_ask']
  #print(selected)
  #selected.plot()
  #plt.show()

  try:
    coefficients, residuals, _, _, _ = np.polyfit(range(len(selected.index)),selected,1,full=True)
    mse = residuals[0]/(len(selected.index))
    nrmse = np.sqrt(mse)/(selected.max() - selected.min())
    slope =  coefficients[0]
    if plot:
      print ('Slope: '+ str(slope))
      print('NRMSE: ' + str(nrmse))
      plt.plot(selected)
      plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(selected))])
      plt.show()
  except:
    slope = 0
    nrmse = 0
  return slope,nrmse


#getslope(True)
