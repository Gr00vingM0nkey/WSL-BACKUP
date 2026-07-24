import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt 
#from tqdm import tqdm
import os
import gym
from Monster_Chase_env import MC
import time
import sys

MAPSIZE = 5

env = MC(Map_Size=MAPSIZE)


n_observations = env.observation_space
n_actions = env.action_space


@keras.utils.register_keras_serializable(package="custom_layers")
class CustomModel(keras.Model):
    def __init__(self, units, input_dim, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        print("CM__init__----")
        # maybe do super().__init__(units, **kwargs)
        # create individual layers here
        # create any model weights here (weights = variables)
        self.layer1 = keras.layers.Input(shape=(input_dim,))
        self.layer1 = keras.layers.Dense(units=2048,activation=keras.activations.leaky_relu)
        self.layer2 = keras.layers.Dense(units=1024, activation=keras.activations.leaky_relu)
        self.layer3 = keras.layers.Dense(units=512, activation=keras.activations.leaky_relu)
        self.final = keras.layers.Dense(units=4, activation=keras.activations.relu)

        # TODO plan of what to do TODO
        #predict Siblings/Spouses, and Parents/Children to group Last names together. combine with everything
        #
        

    def build(self, input_shape):
        print("CMbuild----")
        #super.build()

        # build all layers & weights here

    def call(self, inputs):
        #print("CMcall----")
        

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

def choose_action(model, observation, single=True):
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    observation = np.expand_dims(observation, axis=0)
    logits = model(observation, training=False)
    action = tf.random.categorical(logits, num_samples=1)
    return int(action.numpy().flatten()[0])

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

memory = Memory()

def aggregate_memory(memories):
    batch_memory = Memory()
    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)
    return batch_memory

def discounted_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return discounted_rewards.astype(np.float32)

def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)

        # model.trainable_variables = model nuerons
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
input_dimensions = tf.convert_to_tensor(env.observation_space)
input_dimensions = input_dimensions.shape[1]
Action_Space = 4#Action space
Runner_model = CustomModel(input_dimensions, Action_Space)#initialize model
frames = []#holding [[frame], [frame]] of the game
avgspg = []#Average Steps per game

NUM_EPISODES = 100

for i_episode in range(NUM_EPISODES):
    print(i_episode)

    #__________AVGSPG
    count = 0

    #__________Reset the game__________MAIN
    observation = env.reset()
    memory.clear()
    while True:

        #__________Adding one to get steps in game__________AVGSPG
        count += 1

        #__________Agent and game action, with state of game__________MAIN
        action = choose_action(Runner_model, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        memory.add_to_memory(observation, action, reward)

        #__________Getting frames to make record__________FRAMES
        #frame = np.array(next_observation).reshape(MAPSIZE,MAPSIZE).tolist()
        #print(reward)
        #for z in frame:
        #    print(z)
        #print()
        #frames.append(frame)
        
        if done:

            #__________Pushing Steps to memory__________AVGSPG
            avgspg.append(count)

            #__________Training model of data__________MAIN
            total_reard = sum(memory.rewards)
            train_step(Runner_model, optimizer,
                       observations=np.vstack(memory.observations),
                       actions=np.array(memory.actions),
                       discounted_rewards=discounted_rewards(memory.rewards))            
            break
        observation = next_observation

def Clean_Print():
    for z in frames:
        for x in z:
            print(x)
        for x in range(5):
            sys.stdout.write("\033[F")
        time.sleep(2)
    
print(avgspg)    

def Plot_Loss():
    fig, ax = plt.subplots()
    ploty = avgspg
    plotx = np.linspace(0, len(avgspg), len(avgspg))
    ax.plot(plotx,ploty)

    ax.set_yscale("log")
    #ax.set_xscale("log")
    plt.savefig("plot.png")

Plot_Loss()
plt.show()
print("Done------------------------------------\n\n\n\n")