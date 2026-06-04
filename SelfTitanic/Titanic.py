import keras
import pandas as pd
import string
import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import csv
import os
import sys

import tensorflow as tf

TrainData = pd.read_csv("train.csv")

"""if(input("clear plot?") == "y"):""
with open("plot.txt", "w") as file:
    #sys.exit(0)
    pass"""

def cleaner(arr):
    y_train = arr["Survived"]
    arr = arr.drop(["Survived", "Ticket", "Parch", "Cabin"], inplace=False, axis=1)

    """cabin = arr["Cabin"].tolist()
    print(cabin)
    cabinuni = np.unique(cabin)
    for z in range(len(cabin)): 
        cabin[z] = np.where(cabinuni == cabin[z])`
    print(cabin)
    arr["Cabin"]=cabin"""

    name = arr["Name"].tolist()
    for z in range(len(name)):
        name[z] = (name[z].split(",")[0])
    for z in range(len(name)):
        name[z] = (name.index(name[z])+1)
    arr["Name"] = name


    gender = arr["Sex"].tolist()
    for z in range(len(gender)):
        if(gender[z] == "male"):
            gender[z] = 2
        else:
            gender[z] = 1
    arr["Sex"] = gender
    embarked = arr["Embarked"].tolist()
    for z in range(len(embarked)):
        if(embarked[z] == "C"):
            embarked[z] = 3
        if(embarked[z] == "Q"):
            embarked[z] = 1
        else:
            embarked[z] = 2
    arr["Embarked"] = embarked
    SibSp = arr["SibSp"].tolist()
    for z in range(len(SibSp)):
        SibSp[z] += 1
    arr["SibSp"] = SibSp  
    
    age = arr["Age"].tolist()

    avgage = 0
    count = 0
    for z in range(len(age)):
        if(math.isnan(age[z]) == False):
            count += 1
            avgage += age[z]
    avgage = avgage/count

    for z in range(len(age)):
        if(math.isnan(age[z]) == True):
            age[z] = avgage

    arr["Age"] = age

    return arr, y_train

x_train,y_train  = cleaner(TrainData)


if not os.path.exists("plot.txt"):
    open("plot.txt", "w").close()

#load = input("load? y/n\n")
load = "y"
if(load == "y"):
    load = True
else:
    load = False

if(load):

    #i = input("enter file:\n")
    i = "zrg"
    path = "/home/gr00vingm0nkey/vscode/SelfTitanic/" + i + ".keras"
    model = tf.keras.models.load_model(path)

    EPOCHS = 100
 
    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    #i = input("train? y/n\n")
    i = "y"
    if(i=="y"):
        print(x_train["Pclass"])
        history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1) # 2 to hide the per epoch info, 1 to hide the loading bar, 0 to show all data
        model.save("model.keras")
        
        history = history.history["binary_accuracy"]
        with open("plot.txt", "a") as plot:
            county = 0
            avg = 0
            for z in range(len(history)):
                county = int(county+1)
                avg += round(history[z], 4)
                print(avg/4, int(round(history[z], 4)))
                if(county%4 == 3):
                    print(avg/4, int(round(history[z], 4)), "--")
                    plot.write(","+str(avg/4))
                    avg = 0
                    county = 0

else:
    model = keras.models.Sequential([
        keras.layers.Dense(2048, activation=keras.activations.leaky_relu),
        keras.layers.Dense(512, activation=keras.activations.leaky_relu),
        keras.layers.Dense(100, activation=keras.activations.leaky_relu),
        keras.layers.Dense(1, activation=keras.activations.sigmoid),
    ])

    EPOCHS = 100

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam())
    history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=2)
    model.save("zrg.keras")
    hi = open("plot.txt", "w")
    hi.close()

with open('plot.txt', mode='r', newline='') as file:    

    Y = file.read().split(",")
    print(Y)
    print("Y")

    X = np.linspace(0, len(Y), len(Y))
    for z in range(len(X)):
        X[z] = round(X[z],5)
    print(X)
    print("X")
    fig, ax = mpl.pyplot.subplots()
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
    ax.plot(X,Y)
    mpl.pyplot.savefig("plot.png")

vals = model(x_train)

#print(y_train[0:10])
#print(vals[0:10])

y_pred = [round(x) for x in list(vals.numpy().reshape(-1))]

#print(y_pred[0:10]) 
#print(list(vals.numpy().reshape(-1))[0:10])

print(accuracy_score(y_train, y_pred))

TestData = pd.read_csv("test.csv")
testpred = model.predict(TestData)
print(testpred)

