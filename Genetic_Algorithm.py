import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Goal = np.array(np.random.randint(low=0, high=51, size=20))

Generation = []

Children = 70
Mutations = 40
Sensitivity = 15

loss = []

for z in range(Children):
    Generation.append(np.random.randint(low=0, high=30+1, size=len(Goal)))

def Graph(values):

    df = pd.DataFrame(values)
    print(df)
    plot = df.plot(title= "Graph")
    plt.xticks(rotation=25)
    plt.savefig("plot.png")

def ratings(goal, arr):
    mse = []

    for z in range(len(arr)):
        mse.append(np.mean(np.square(np.subtract(Goal, arr[z]))))

    mse = np.array(mse)
    mse.flatten()
    mse = mse.tolist()
    return mse

def pointratings(goal, arr):
    mse = []
    
    for z in range(len(arr)):
        mse.append(np.mean(np.square(np.subtract(Goal[z], arr[z]))))

    mse = np.array(mse)
    mse.flatten()
    mse = mse.tolist()
    return mse

def copymutate(arr):
    children = []
    for i in range(Children):
        mse = ratings(Goal, arr)
        winner = mse.index(np.min(mse))
        mse[winner] += 1000000
        second = mse.index(np.min(mse))
        child = []
        for z in range(len(arr[winner])):
            low = arr[second][z] if arr[second][z]<arr[winner][z] else arr[winner][z]
            high = arr[second][z] if arr[second][z]>arr[winner][z] else arr[winner][z]
            if(low==high):
                child.append(np.array(low))
                continue
            child.append(np.random.randint(low=low, high=high+1))
        children.append(child)
    return children

def pointmutate(arr):
    mse = ratings(Goal, arr)
    arr = arr[mse.index(np.max(mse))]
    mse = pointratings(Goal, arr)
    for z in range(len(arr)):
        if(mse[z] != 0):
            arr[z] += np.random.choice([-1, 1])

def mutate(arr):
    children = []
    for i in range(Mutations):
        mse = ratings(Goal, arr)
        winner = mse.index(np.min(mse))
        mse[winner] += 1000000
        second = mse.index(np.min(mse))
        child = []
        for z in range(len(arr[winner])):
            low = arr[second][z] if arr[second][z]<arr[winner][z] else arr[winner][z]
            high = arr[second][z] if arr[second][z]>arr[winner][z] else arr[winner][z]
            if(low==high):
                child.append(np.array(low+int(np.random.randint(low=-5, high=5+1))))
                continue
            child.append(np.random.randint(low=low-5, high=high+5+1))
        children.append(child)
    return children

def prune(arr): 
    mse = ratings(Goal, arr)
    loss.append(np.min(mse))
    survived = []
    for z in range(Children):
        survived.append(arr.pop(mse.index(np.min(mse))))
        mse.pop(mse.index(np.min(mse)))

    return survived

EPOCHS = 50

for i in range(EPOCHS):
    print(i)
    New = copymutate(Generation)
    pointer = False
    if(len(loss)>=Sensitivity):
        for z in range(Sensitivity):
            if(loss[-z-1] != loss[-z-2]):
                pointer = False
                break
            elif(loss[-z-1] == loss[-z-2]):
                pointer = True
    if(Sensitivity):
        pointmutate(Generation)
    for z in New:
        Generation.append(np.array(z).flatten())
    New = mutate(Generation)
    for z in New:
        Generation.append(np.array(z).flatten())
    Generation = prune(Generation)

print("-------------Final-------------")
for z in Generation:
    print(z)
print("\n\n\n")
mse = ratings(Goal, Generation)
print(Generation[mse.index(np.min(mse))], mse[mse.index(np.min(mse))])
print(Goal)
 
Graph(loss)