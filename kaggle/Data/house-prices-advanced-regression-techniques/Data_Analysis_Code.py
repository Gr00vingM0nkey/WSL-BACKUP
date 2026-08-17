import keras
import pandas as pd
import numpy as np
# import math
import matplotlib.pyplot as plt
# import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler




def Data(train, test):
    train = train.copy()
    test = test.copy()

    # Save target
    y_train = np.log1p(train.pop("SalePrice").astype("int32"))

    # Remove IDs
    train.drop(columns=["Id"], inplace=True)
    test.drop(columns=["Id"], inplace=True)

    # Combine so train and test get identical columns
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Fill missing numeric values
    numeric_cols = combined.select_dtypes(include=["number"]).columns
    combined[numeric_cols] = combined[numeric_cols].fillna(
        combined[numeric_cols].median()
    )

    # Scale numeric values
    min_max = MinMaxScaler()
    combined[numeric_cols] = np.round(
        min_max.fit_transform(combined[numeric_cols]), 4
    )

    # Fill missing categorical values
    categorical_cols = combined.select_dtypes(include=["object"]).columns
    combined[categorical_cols] = combined[categorical_cols].fillna("Missing")

    # One-hot encode
    combined = pd.get_dummies(combined, dtype="int32")

    # Convert to float32
    combined = combined.astype("float32")

    # Split back
    x_train = combined.iloc[:len(train)].reset_index(drop=True)
    x_test = combined.iloc[len(train):].reset_index(drop=True)

    return x_train, x_test, y_train


def CustomAddition(train, test):
    train = train.copy()
    test = test.copy()

    features = [
        "BsmtFinSF2",
        "LowQualFinSF",
        "2ndFlrSF",
        "3SsnPorch",
        "BsmtHalfBath",
        "EnclosedPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "WoodDeckSF",
        "OpenPorchSF",
        "MasVnrArea",
        "LotArea"
    ]

    for feature in features:
        train[f"has_{feature}"] = (train[feature] > 0).astype("int32")
        test[f"has_{feature}"] = (test[feature] > 0).astype("int32")

    return train, test

@keras.utils.register_keras_serializable(package="custom_layers")
class Normal_Model1(keras.Model):
    def __init__(self, units, **kwargs):
        super(Normal_Model1, self).__init__(**kwargs)
        # print("CM__init__----")

        self.layer1 = keras.layers.Dense(units=2048,activation=keras.activations.leaky_relu)
        self.layer2 = keras.layers.Dense(units=1024, activation=keras.activations.leaky_relu)
        self.layer3 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.layer4 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.final = keras.layers.Dense(units=1, activation=keras.activations.linear)
        pass

    def build(self, input_shape):
        # print("CMbuild----")
        #super.build()
        pass

    def call(self, inputs):
        # print("CMcall----")
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.final(x4)
        return x5

@keras.utils.register_keras_serializable(package="custom_layers")
class Normal_Model2(keras.Model):
    def __init__(self, units, **kwargs):
        super(Normal_Model2, self).__init__(**kwargs)
        # print("CM__init__----")

        self.layer1 = keras.layers.Dense(units=2048,activation=keras.activations.leaky_relu)
        self.layer2 = keras.layers.Dense(units=1024, activation=keras.activations.leaky_relu)
        self.layer3 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.layer4 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.final = keras.layers.Dense(units=1, activation=keras.activations.linear)
        pass

    def build(self, input_shape):
        # print("CMbuild----")
        #super.build()
        pass

    def call(self, inputs):
        # print("CMcall----")
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.final(x4)
        return x5

def unscaled_rmse(y_true, y_pred):
    y_true_orig = tf.math.expm1(y_true)
    y_pred_orig = tf.math.expm1(y_pred)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true_orig - y_pred_orig)))

def Plot_Loss(Data, Title, split):
    n_groups = max(split) + 1
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 4 * n_groups))
    if n_groups == 1:
        axes = [axes]

    for z in range(len(Data)):
        ax = axes[split[z]]
        ploty = Data[z]
        skip  = len(ploty) // 10
        ploty = ploty[skip:]
        plotx = np.arange(skip, skip + len(ploty))
        ax.plot(plotx, ploty, label=Title[z])

    for ax in axes:
        ax.yaxis.set_major_locator(plt.ticker.MaxNLocator(10))
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.savefig("plot.png", bbox_inches="tight")
    plt.show()