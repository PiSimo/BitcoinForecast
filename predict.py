
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


con = sqlite3.connect('data.db')

data = pd.read_sql_query("SELECT * from predict", con, index_col="created").tail(12) 

print(data.head())


selected = data.loc[('2018-04-14 08:36:46' < data.index) & (data.index < '2018-04-27 08:36:46'),'actual']
selected.plot()
plt.show()



import numpy as np
coefficients, residuals, _, _, _ = np.polyfit(range(len(selected.index)),selected,1,full=True)
mse = residuals[0]/(len(selected.index))
nrmse = np.sqrt(mse)/(selected.max() - selected.min())
print('Slope ' + str(coefficients[0]))
print('NRMSE: ' + str(nrmse))


plt.plot(selected)
plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(selected))])
plt.show()