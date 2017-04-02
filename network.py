import util
import time
import numpy as np
import matplotlib.pyplot as plt
from   sys import argv,exit
from   keras.models import Sequential
from   keras.layers import Dense,Dropout,GRU,Reshape
from   keras.layers.normalization import BatchNormalization

file_name = 'dataset2.csv'
net = None
wait_time = 530

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
    net.compile(optimizer='nadam',loss='mse')
    print("done!")

def chart(real,predicted,show=True):
    plt.plot(real,color='g')
    plt.plot(predicted,color='r')
    plt.ylabel('BTC/USD')
    plt.xlabel("9Minutes")
    plt.savefig("chart.png")
    if show:plt.show()

def predictFuture(m1,m2,old_pred,writeToFile=False):
    actual,latest_p = util.getCurrentData(label=True,toFile=True)
    actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    pred = float(int(pred[0]*100)/100)
    if writeToFile:
        f = open("results","a")
        f.write("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$\n".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
        f.close()

    print("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
    return latest_p,pred


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
                real,hip = predictFuture(m1,m2,hip,writeToFile=True)
                reals.append(real)
                preds.append(hip)
                time.sleep(wait_time)
            except KeyboardInterrupt:
                ### PLOTTING
                chart(reals,preds,show=False)
                print("Chart saved!")
                s = input("Type yes to close the program: ")
                if s.lower() == "yes":break
                print("Resuming...")

        print("Closing..")

    elif argv[1] == 'train':
        cho = input("Fine tune model [yes/no] ?")
        if cho == 'yes':
            model_name = input('trained model\'s path:')
            net.load_weights(model_name)

        #Training dnn
        print("training...")
        el = len(data)-10     #Last ten elements are for testing
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
            preds.append(predicted)
            reals.append(real)

        ### Predict Price the next 9m price (magic)
        real,hip = predictFuture(m1,m2,0)
        reals.append(real)
        preds.append(hip)

        ### PLOTTING
        chart(reals,preds)

    else :
        print("Wrong argument")
