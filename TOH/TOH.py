import numpy as np
import tensorflow as tf
from tensorflow import keras
from TOH_env import Tower_of_Hanoi

env = Tower_of_Hanoi()

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

def create__model(input_dim, n_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions)
    ])
    return model

def choose_action(model, observation, single=True):
    observation = np.expand_dims(observation, axis=0)
    print(observation)
    logits = model.predict(observation, verbose=0)
    print(f"this is logits: {logits}")
    input()
    logits = logits[0]
    action = tf.random.categorical(logits, num_samples=1)

    action = action.numpy().flatten()
    return action[0] if single else action

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
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

observation, _, _, _, _ = env.reset()
print((observation))

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
print(type(observation))
input_dim = observation.shape
n_actions = env.action_space.n
TOH_Agent = create__model(input_dim, n_actions)

NUM_EPISODES = 10

for i_episode in range(NUM_EPISODES):
    observation, _, _, _, _ = env.reset()
    memory.clear()
    while True:
        print(observation)
        input()
        action = choose_action(TOH_Agent, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        memory.add_to_memory(observation, action, reward)
        if done:
            total_reward = sum(memory.rewards)
            train_step(TOH_Agent, optimizer,
                       observations=np.vstack(memory.observations),
                       actions=np.array(memory.actions),
                       discounted_rewards=discounted_rewards(memory.rewards))
            
            memory.clear()
            break
        observation = next_observation