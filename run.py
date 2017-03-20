import util
import numpy as np,time
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape

dnn = Sequential()
def buildNet(w_init="glorot_uniform"):
    dnn.add(Dense(12,init=w_init,input_dim=12,activation='linear'))
    dnn.add(Reshape((1,12)))
    dnn.add(GRU(50,init=w_init,activation='sigmoid',return_sequences=True))
    dnn.add(Dropout(0.4))
    dnn.add(GRU(70,init=w_init,activation='sigmoid',return_sequences=False))
    dnn.add(Dense(1,init=w_init,activation='linear'))
    dnn.compile(optimizer='rmsprop',loss='mse')


if __name__ == '__main__':
    print("buildin net...",end="")
    buildNet()
    print("build!")

    #Loading min and max values from csv file for normalization
    labelNorm = []
    print("Loading minmaxs values to normalize data...",end="")
    lines = open("minmax.csv","r").read().split("\n")
    for l in lines[:len(lines)-1]:
            l = l.split(",")
            util.maxs.append(float(l[0]))
            util.mins.append(float(l[1]))
    m1 = float(lines[::-1][0].split(",")[0])
    m2 = float(lines[::-1][0].split(",")[1])
    print("loaded!")

    #Main Loop
    while True:
        try:
            actual,lprice = util.getCurrentData(label=True)
            actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
            actual = util.augmentValue(dnn.predict(actual)[0],m1,m2)
            print("[{}] Actual Price:{}$ | Next 15m:{}$".format(time.strftime("%H:%M:%S"),lprice,actual))

        except KeyboardInterrupt:
            print("Killing main loop...")
        time.sleep(7*60)
