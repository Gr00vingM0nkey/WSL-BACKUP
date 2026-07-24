import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras
import pandas as pd
import matplotlib as plt

@keras.utils.register_keras_serializable(package="custom_layers")
class Keras_Norm(keras.layers.Layer):
    def __init__(self, name=None, kernel_initializer=None, **kwargs):
        print("CLK__init__----")
        super().__init__(**kwargs)

        # create tensorflow variables here

    def build(self, input_shape):
        print("CLKbuild----")
        print(input_shape)
        
        # initialize shape-dependent variables here

        super().build(input_shape)

    def call(self, inputs): 
        print("CLKcall----")
        return tf_self_default_keras_normalization(inputs)
         
        # you do layer output logic here.
        # for example, a dense layer would basically do the multiplication and addition here
        # and then return a value
@keras.utils.register_keras_serializable(package="custom_layers")
class Z_score_Norm(keras.layers.Layer):
    def __init__(self, name=None, kernel_initializer=None, **kwargs):
        print("CLZ__init__----")
        super().__init__(**kwargs)

        # create tensorflow variables here

    def build(self, input_shape):
        print("CLZbuild----")
        print(input_shape)
        
        # initialize shape-dependent variables here

        super().build(input_shape)

    def call(self, inputs): 
        print("CLZcall----")
        return tf_self_z_score_normalization(inputs)
@keras.utils.register_keras_serializable(package="custom_layers")
class Min_Max_Norm(keras.layers.Layer):
    def __init__(self, name=None, kernel_initializer=None, **kwargs):
        print("CLM__init__----")
        super().__init__(**kwargs)

        # create tensorflow variables here

    def build(self, input_shape):
        print("CLMbuild----")
        print(input_shape)
        
        # initialize shape-dependent variables here

        super().build(input_shape)

    def call(self, inputs): 
        print("CLMcall----")
        return tf_self_min_max_normalization(inputs)

@keras.utils.register_keras_serializable(package="custom_layers")
class CustomModel(keras.Model):
    def __init__(self, units, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        print("CM__init__----")
        # maybe do super().__init__(units, **kwargs)
        # create individual layers here
        # create any model weights here (weights = variables)
        self.keras_norm = Keras_Norm(name="billy")
        self.z_score_norm = Z_score_Norm(name="bob")
        self.min_max_norm = Min_Max_Norm(name="joe")
        self.layer1 = keras.layers.Dense(units=2048,activation=keras.activations.leaky_relu)
        self.layer2 = keras.layers.Dense(units=1024, activation=keras.activations.leaky_relu)
        self.layer3 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.final = keras.layers.Dense(units=1, activation=keras.activations.sigmoid)

        # TODO plan of what to do TODO
        #predict Siblings/Spouses, and Parents/Children to group Last names together. combine with everything
        #
        
        pass

    def build(self, input_shape):
        print("CMbuild----")
        #super.build()

        # build all layers & weights here
        pass

    def call(self, inputs):
        print("CMcall----")
        

        #x1 = self.keras_norm(inputs)
        #x2 = self.z_score_norm(inputs)
        #x3 = self.min_max_norm(inputs)

        #x4 = tf.concat([x1, x2, x3], axis=1)
        #print("Layer1")
        x5 = self.layer1(inputs)
        #print(x5)
        #print("Layer2")
        x6 = self. layer2(x5)
        #print(x6)
        #print("Layer3")
        x7 = self.layer3(x6)
        #print("Final")
        x8 = self.final(x7)
        #print(x8)
        #print("Return")
        return x8
        
        # basically just do your model's work here. you have the freedom to make non-sequential models
        pass

"""def Get_Data_Test(file):
    TestData = pd.read_csv(file)

    def cleaner(arr):

        arr = arr.drop(["Ticket", "Parch", "Cabin"], inplace=False, axis=1)
        Pid = arr["PassengerId"].tolist()
        #cabin = arr["Cabin"].tolist()
        #print(cabin)
        #cabinuni = np.unique(cabin)
        #for z in range(len(cabin)): 
        #    cabin[z] = np.where(cabinuni == cabin[z])`
        #print(cabin)
        #arr["Cabin"]=cabin

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

        embarked = arr["Embarked"].tolist()#C, Q, S, nan
        arr = arr.drop(["Embarked"], inplace=False, axis=1)

        LeftC = []
        LeftQ = []
        LeftS = []
        for z in range(len(embarked)):
            print(embarked[z])
            if(embarked[z] == "C"):
                #print("C")
                LeftC.append(1)
                LeftQ.append(0)
                LeftS.append(0)
            if(embarked[z] == "Q"):
                #print("Q")
                LeftC.append(0)
                LeftQ.append(1)
                LeftS.append(0)
            if(embarked[z] == "S"):
                #print("S")
                LeftC.append(0)
                LeftQ.append(0)
                LeftS.append(1)
            #else:
                #print("na-----")
                #LeftC.append(0)
                #LeftQ.append(0)
                #LeftS.append(0)
        arr.insert(2,"LeftC", LeftC)
        arr.insert(2,"LeftQ", LeftQ)
        arr.insert(2,"LeftS", LeftS)

        SibSp = arr["SibSp"].tolist()
        for z in range(len(SibSp)):
            SibSp[z] += 1
        arr["SibSp"] = SibSp  
        
        age = arr["Age"].tolist()

        avgage = 0
        count = 0
        for z in range(len(age)):
            if(np.isnan(age[z]) == False):
                count += 1
                avgage += age[z]
        avgage = avgage/count

        for z in range(len(age)):
            if(np.isnan(age[z]) == True):
                age[z] = avgage

        arr["Age"] = age

        return arr, Pid

    return cleaner(TestData)
def Get_Data_Train(file):
    TrainData = pd.read_csv(file)

    def cleaner(arr):

        y = arr["Survived"]
        arr = arr.drop(["Survived", "Ticket", "Parch", "Cabin"], inplace=False, axis=1)

        #cabin = arr["Cabin"].tolist()
        #print(cabin)
        #cabinuni = np.unique(cabin)
        #for z in range(len(cabin)): 
        #    cabin[z] = np.where(cabinuni == cabin[z])`
        3print(cabin)
        #arr["Cabin"]=cabin

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

        embarked = arr["Embarked"].tolist()#C, Q, S, nan
        arr = arr.drop(["Embarked"], inplace=False, axis=1)

        LeftC = []
        LeftQ = []
        LeftS = []
        for z in range(len(embarked)):
            print(embarked[z])
            if(embarked[z] == "C"):
                #print("C")
                LeftC.append(1)
                LeftQ.append(0)
                LeftS.append(0)
            if(embarked[z] == "Q"):
                #print("Q")
                LeftC.append(0)
                LeftQ.append(1)
                LeftS.append(0)
            if(embarked[z] == "S"):
                #print("S")
                LeftC.append(0)
                LeftQ.append(0)
                LeftS.append(1)
            #else:
                #print("na-----")
                #LeftC.append(0)
                #LeftQ.append(0)
                #LeftS.append(0)
        LeftC.append(0)
        LeftC.append(0)
        LeftQ.append(0)
        LeftQ.append(0)
        LeftS.append(0)
        LeftS.append(0)
        arr.insert(2,"LeftC", LeftC)
        arr.insert(2,"LeftQ", LeftQ)
        arr.insert(2,"LeftS", LeftS)

        SibSp = arr["SibSp"].tolist()
        for z in range(len(SibSp)):
            SibSp[z] += 1
        arr["SibSp"] = SibSp  
        
        age = arr["Age"].tolist()

        avgage = 0
        count = 0
        for z in range(len(age)):
            if(np.isnan(age[z]) == False):
                count += 1
                avgage += age[z]
        avgage = avgage/count

        for z in range(len(age)):
            if(np.isnan(age[z]) == True):
                age[z] = avgage

        arr["Age"] = age

        return arr, y

    return cleaner(TrainData)"""
def Get_Data(file, is_train=True):
    data = pd.read_csv("Titanic/"+file)

    y = None
    Pid = data["PassengerId"].tolist()

    if is_train:
        y = data["Survived"]
        data = data.drop(["Name", "Survived", "Ticket", "Cabin"], axis=1)
    else:
        data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

    """name = data["Name"].tolist()
    for z in range(len(name)):
        name[z] = name[z].split(",")[0]
    for z in range(len(name)):
        name[z] = name.index(name[z]) + 1
    data["Name"] = name"""

    fare = data["Fare"].tolist()
    fare = self_min_max_normalization(fare)
    data["Fare"] = fare

    gender = data["Sex"].tolist()
    for z in range(len(gender)):
        gender[z] = 2 if gender[z] == "male" else 1
    data["Sex"] = gender

    embarked = data["Embarked"].fillna("S").tolist()
    data = data.drop(["Embarked"], axis=1)

    LeftC, LeftQ, LeftS = [], [], []
    for x in embarked:
        if x == "C":
            LeftC.append(1)
            LeftQ.append(0)
            LeftS.append(0)
        elif x == "Q":
            LeftC.append(0)
            LeftQ.append(1)
            LeftS.append(0)
        elif x == "S":
            LeftC.append(0)
            LeftQ.append(0)
            LeftS.append(1)
        else:
            print(x)
            print("LALALALLALALALALLALALALALLALALALALAL")
        

    data.insert(2, "LeftC", LeftC)
    data.insert(2, "LeftQ", LeftQ)
    data.insert(2, "LeftS", LeftS)

    age = data["Age"].tolist()
    avgage = np.nanmean(age)
    age = [avgage if np.isnan(a) else a for a in age]
    age = self_z_score_normalization(age)
    data["Age"] = age

    return (data, y) if is_train else (data, Pid)
def Find_Dif(input1, input2):
    count = 0
    for z in range(len(input1)):
        count += 0 if input1[z]==input2[z] else 1
    return f"Total difference:{count}, Incount:{len(input1)}, Outcount:{len(input2)}"
def self_default_keras_normalization(input):
    ray = []
    mean = np.mean(input)
    var = np.var(input)
    for z in range(len(input)):
        ray.append((input[z]-mean)/np.sqrt(var))
    return ray
def tf_self_default_keras_normalization(input:tf.Tensor):
    mean = tf.math.reduce_mean(input)
    sqrtvar = tf.math.sqrt(tf.math.reduce_variance(input))
    return (input-mean)/sqrtvar
def self_z_score_normalization(input):
    ray = []
    mean = np.mean(input)
    std = np.std(input)
    for z in range(len(input)):
        ray.append((input[z]-mean)/std)
    return ray
def tf_self_z_score_normalization(input:tf.Tensor):
    mean = tf.math.reduce_mean(input)
    std = tf.math.reduce_std(input)
    return (input-mean)/std
def tf_self_min_max_normalization(input:tf.Tensor):
    min = tf.math.reduce_min(input)
    max = tf.math.reduce_max(input)
    return (input-min)/(max-min)
def self_min_max_normalization(input):
    min = np.min(input)
    max = np.max(input)
    return (input-min)/(max-min)
def normalization_things():
    #def tf_self_
    x_train,y_train = Get_Data()
    def float(x):
        float(x)
    data = np.array(x_train["SibSp"].tolist(), dtype="float64")
    #print(data)
    normal = data
    #normal = tf_self_z_score_normalization(normal)
    #normal = tf_self_default_keras_normalization(normal)
    normal = tf_self_min_max_normalization(normal)
    #normal.build((len(data),))
    #print(normal.adapt(data))
    #(normal.adapt(data))
    print(np.var(data))
    print(np.mean(data))
    print(np.var(data)/np.mean(data))
    print(np.array(data).dtype)
    #normal.adapt(data)
    n = tf.convert_to_tensor(normal)
    """for z in range(len(data)):
        print(data[z], n[z])"""
    h=n
    print(np.array(h))
    print(np.array(h).dtype)
    print("----------"+Find_Dif(np.array(data), np.array(n))+"----------")
def Example_Data_Set():
    x_train = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10]
    ]
    y_train = []
    for z in range(10):
        y.append([x[z][0]+10])
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train)
    X = [
        [40],
        [41],
        [42],
        [43],
        [44],
        [45],
        [46],
        [47],
        [49],
        [50]
    ]   

x_train, y_train = Get_Data("train.csv", is_train=True)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
print(x_train, y_train)
x_test, Pid = Get_Data("test.csv", is_train=False)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
print(x_test)
model = CustomModel(2)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
    loss=keras.losses.BinaryCrossentropy, #           
)

EPOCHS = 50000

print("---START---")
history = model.fit(x_train,y_train, epochs=EPOCHS, verbose=1, validation_split=0.15)

def Plot_Loss():
    fig, ax = plt.pyplot.subplots()
    ax.yaxis.set_major_locator(plt.ticker.MaxNLocator(10))
    #print(history.history)
    ploty = history.history["loss"]
    plotx = np.linspace(0, len(ploty), len(ploty))
    ax.plot(plotx,ploty)

    ploty = history.history["val_loss"]
    ax.plot(plotx,ploty)

    ax.set_yscale("log")
    #ax.set_xscale("log")
    plt.pyplot.savefig("plot.png")
Plot_Loss()#----------Can be seen through plot.png

prediction = model.predict(x_test)
print("---Prediction---")
prediction = np.round(prediction)
#print(prediction)
prediction = prediction.flatten()

def Submission(Pid, Prediction):

    Submission = pd.DataFrame({
        "PassengerId": Pid,
        "Survived": Prediction
        })
    Submission.fillna(0, inplace=True)
    print(Submission["Survived"].tolist())
    Submission = Submission.astype("int64")

    Submission.to_csv("submission.csv", index=False)

Submission(Pid, prediction)