import util
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sys import argv,exit
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape

file_name = 'dataset.csv'
leng = 12
net = None
w_init ="glorot_uniform"
act    ="tanh"
wait_time = 2*60

def predictNext15m(m1,m2,old_pred):
    print("Fetching data...",end="")
    actual,latest_p = util.getCurrentData(label=True)
    print("!")
    actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    print("[{}] Actual:{}$ Last Prediction:{}$ Next 15m:{}$".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred[0]))
    return latest_p,pred[0]

if __name__ == '__main__':
    if len(argv) != 2:
        print(argv[0]+" train/run")
        exit(-1)

    #Building network
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(12,init=w_init,input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(GRU(30,init=w_init,activation=act,return_sequences=True))
    net.add(GRU(50,init=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(90,init=w_init,activation=act,return_sequences=True))
    net.add(GRU(61,init=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(10,init=w_init,activation=act,return_sequences=False))
    net.add(Dense(1,init=w_init,activation='linear'))
    net.compile(optimizer='rmsprop',loss='mse')#mean_squared_logarithmic_error
    print("done!")


    #Loading Data (necessary also for running it to normalize data)
    print("Loading data...",end="")
    d = open(file_name,'r')
    data,labels = util.loadData(d)
    data = util.reduceMatRows(data)
    labels,m1,m2 =util.reduceVector(labels,getVal=True)
    print("{} chunk loaded!\nTraining...".format(len(labels)),end="")

    if argv[1] == 'run':
        #Loading weights
        w_name = input("Weight file:")
        net.load_weights(w_name)
        print("Starting main loop...")
        old_pred = 0
        while True:
            try:
                a,b = predictNext15m(m1,m2,old_pred)
                old_pred = b
                time.sleep(wait_time)
            except KeyboardInterrupt:
                print("[!]Closing main loop",end="")
                break
        print("!")
    elif argv[1] == 'train':
        #Training dnn
        print("training...")
        net.fit(data,labels,nb_epoch=1001,batch_size=45)
        print("trained!\nSaving...",end="")
        net.save_weights("model.h5")
        print("saved!")

        ### Predict all over the dataset to build the chart
        reals,preds = [],[]
        for i in range(len(data)):
            x = np.array(data[i]).reshape(1,12)
            predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
            real = util.augmentValue(labels[i],m1,m2)
            preds.append(predicted)
            reals.append(real)

        ### Predict Price for the next 15m
        real,hip = predictNext15m(m1,m2)
        reals.append(real)
        preds.append(hip)

        ### PLOTTING
        plt.plot(reals,color='g')
        plt.plot(preds,color='r')
        plt.ylabel('BTC/USD')
        plt.xlabel("15Minute")
        plt.show()

    else :
        print("Wrong argument")
