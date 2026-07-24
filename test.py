
import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow as tf

def half(arr):
    n = tf.shape(arr)[0]
    keep_n = n - n // 2
    scores = tf.random.uniform([n])
    keep_indices = tf.math.top_k(scores, k=keep_n).indices
    keep_indices = tf.sort(keep_indices)

    return tf.gather(arr, keep_indices)

population = tf.Variable(
            tf.random.normal((10,5)),
            trainable=False,
            dtype=tf.float32,
        )

import random

randnum = random.randint(0,100)

tries = 5
tries = int(input("How many tries do you want?"))

for z in range(tries):
    inp = int(input("What is your guess?"))

    if(randnum==inp):
        print("!!!You Won!!!")
        break

    elif(randnum<inp):
        print("The number is smaller, TRY AGAIN")

    elif(randnum>inp):
        print("The number is bigger, TRY AGAIN")

if(randnum!=inp):
    print(f"You lost, The number was {randnum}")