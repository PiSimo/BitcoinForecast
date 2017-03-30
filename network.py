import util
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sys import argv,exit
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape
from keras.layers.normalization import BatchNormalization

file_name = 'dataset.csv'
net = None
wait_time = 9*60

def buildNet(w_init="glorot_uniform",act="tanh"):
    global net
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(12,kernel_initializer=w_init,input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(BatchNormalization())
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.3))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=False))
    net.add(Dropout(0.4))
    net.add(Dense(1,kernel_initializer=w_init,activation='linear'))
    net.compile(optimizer='nadam',loss='mse') #'mean_squared_logarithmic_error')
    print("done!")

def predictFuture(m1,m2,old_pred):
    actual,latest_p = util.getCurrentData(label=True)
    actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    pred = float(int(pred*100)/100)
    print("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred[0]))
    return latest_p,pred[0]


if __name__ == '__main__':
    if len(argv) != 2:
        print(argv[0]+" train/run")
        exit(-1)

    #Assembling Net:
    buildNet()

    #Loading Data (necessary also for running it to normalize data)
    print("Loading data...",end="")
    d = open(file_name,'r')
    data,labels = util.loadData(d)
    data = util.reduceMatRows(data)
    labels,m1,m2 =util.reduceVector(labels,getVal=True)
    print("{} chunk loaded!\n".format(len(labels)),end="")

    if argv[1] == 'run':
        #Loading weights
        w_name = input("Weight file:")
        net.load_weights(w_name)
        print("Starting main loop...")
        hip = 0
        reals,preds = [],[]
        for i in range(len(data)-40,len(data)):
            x = np.array(data[i]).reshape(1,12)
            predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
            real = util.augmentValue(labels[i],m1,m2)
            preds.append(predicted)
            reals.append(real)
        while True:
            try:
                real,hip = predictFuture(m1,m2,hip)
                reals.append(real)
                preds.append(hip)
                time.sleep(wait_time)
            except KeyboardInterrupt:
                ### PLOTTING
                plt.plot(reals,color='g')
                plt.plot(preds,color='r')
                plt.ylabel('BTC/USD')
                plt.xlabel("9Minute")
                plt.savefig("run_chart.png")
                print("Chart saved!")
                break
        print("Closing..")
    elif argv[1] == 'train':
        #Training dnn
        print("training...")
        el = len(data)-10
        net.fit(data[:el],labels[:el],epochs=500,batch_size=300)
        print("trained!\nSaving...",end="")
        net.save_weights("model.h5")
        print("saved!")

        ### Predict all over the dataset to build the chart
        reals,preds = [],[]
        for i in range(len(data)-40,len(data)):
            x = np.array(data[i]).reshape(1,12)
            predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
            real = util.augmentValue(labels[i],m1,m2)
            if(i > len(data)-20):
                preds.append(predicted)
                reals.append(real)

        ### Predict Price for the future (magic)
        real,hip = predictFuture(m1,m2,0)
        reals.append(real)
        preds.append(hip)

        ### PLOTTING
        plt.plot(reals,color='g')
        plt.plot(preds,color='r')
        plt.ylabel('BTC/USD')
        plt.xlabel("9Minute")
        plt.savefig("chart.png")
        plt.show()


    else :
        print("Wrong argument")
