import util
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape

file_name = 'dataset.csv'
leng = 12

def main():

    #Building network
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(12,init="glorot_uniform",input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(GRU(35,init="glorot_uniform",activation='sigmoid',return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(60,init="glorot_uniform",activation='sigmoid',return_sequences=False))
    net.add(Dropout(0.3))
    net.add(Dense(1,init="glorot_uniform",activation='linear'))
    net.compile(optimizer=SGD(lr=.01,momentum=.9,nesterov=True),loss='mean_squared_logarithmic_error')
    print("done!")

    #Data
    print("Loading data...",end="")
    f = open(file_name,'r')
    data,labels = util.loadData(f)
    data = util.reduceMatRows(data)
    labels,m1,m2 =util.reduceVector(labels,getVal=True)
    print("{} chunk loaded!\nTraining...".format(len(labels)),end="")

    #Training dnn
    net.fit(data,labels,nb_epoch=400)

    print("trained!\nSaving...")
    net.save

    reals,preds = [],[]
    for i in range(len(data)):
        x = np.array(data[i]).reshape(1,12)
        predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
        real = util.augmentValue(labels[i],m1,m2)
        preds.append(predicted)
        reals.append(real)
        if i > (len(data)-17):print("Real:{} Predicted:{}  likehood({})".format(real,predicted,abs( math.sqrt((real-predicted)**2) )))
    actual = np.array(util.reduceCurrent(util.getCurrentData())).reshape(1,12)
    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    print("At {} predicted next 15m:{}$".format(time.strftime("%H:%M:%S"),pred[0]))

    ### PLOTTING
    plt.plot(reals,color='g')
    plt.plot(preds,color='r')
    plt.ylabel('BTC/USD')
    plt.show()
if __name__ == '__main__':
    main()
