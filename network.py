import util
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape

file_name = 'dataset.csv'
leng = 12

def main():
    w_init ="glorot_uniform"
    #Building network
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(12,init=w_init,input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(GRU(50,init=w_init,activation='sigmoid',return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(70,init=w_init,activation='sigmoid',return_sequences=False))
    net.add(Dense(1,init=w_init,activation='linear'))
    net.compile(optimizer='rmsprop',loss='mse')#mean_squared_logarithmic_error
    print("done!")

    #Data
    print("Loading data...",end="")
    f = open(file_name,'r')
    data,labels = util.loadData(f)
    data = util.reduceMatRows(data)
    labels,m1,m2 =util.reduceVector(labels,getVal=True)
    labels =labels
    print("{} chunk loaded!\nTraining...".format(len(labels)),end="")

    #Training dnn
    net.fit(data,labels,nb_epoch=200)

    print("trained!\nSaving...")
    net.save

    reals,preds = [],[]
    sm = 0
    for i in range(len(data)): 
        x = np.array(data[i]).reshape(1,12)
        predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
        real = util.augmentValue(labels[i],m1,m2)
        preds.append(predicted)
        reals.append(real)
        sm = abs(real-predicted)
       
    sm /= len(data)
    print("Average Likehood:{}".format(sm))

#    actual = np.array(util.reduceCurrent(util.getCurrentData())).reshape(1,12)
#    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
#    print("At {} predicted next 15m:{}$".format(time.strftime("%H:%M:%S"),pred[0]))

    ### PLOTTING
    plt.plot(reals,color='g')
    plt.plot(preds,color='r')
    plt.ylabel('BTC/USD')
    plt.xlabel("15Minute")
    plt.show()
if __name__ == '__main__':
    main()
