"""
DQN Environment

Reference: 
1. https://tiewkh.github.io/blog/deepqlearning-openaitaxi/
2. https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
import sys
sys.path.insert(1, '/home/nalindas9/Documents/courses/spring_2020/enpm690-robot-learning/github/ADRURL_BOT_Autonomous_Delivery_Robot_Using_Reinforcment_Learning/qlearning')
import algo
import matplotlib.pyplot as plt

# Agent class
class Agent():
  def __init__(self, state_size, action_size):
    self.weight_backup = "trained_weights.h5"
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=10000)
    self.learning_rate = 0.001
    self.gamma = 1.0
    self.exploration_rate = 1.0
    self.exploration_min = 0.02
    self.exploration_decay = 0.995
    self.brain = self._build_model()
  
  # Function to build the DQN
  def _build_model(self):
    # Choosing Sequential Model here
    model = Sequential()
    # Adding the two 24x1 hidden layers here and one output layer
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(self.action_size, activation = 'linear')) 
    
    model.compile(loss = 'mse', optimizer=Adam(lr=self.learning_rate))
    
    if os.path.isfile(self.weight_backup):
      print('Using trained weights! ...')
      model.load_weights(self.weight_backup)
      self.exploration_rate = self.exploration_min
    return model
  
  def save_model(self):
    self.brain.save(self.weight_backup)
    
  def act(self, state):
    if np.random.rand() <= self.exploration_rate: 
      return random.randrange(self.action_size)
    act_values = self.brain.predict(np.array([hash(state)]))
    return np.argmax(act_values[0])
    
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    
  def replay(self, sample_batch_size):
    if len(self.memory) < sample_batch_size:
      return
    sample_batch = random.sample(self.memory, sample_batch_size)
    for state, action, reward, next_state, done in sample_batch:
      target = reward
      if not done:
        #print('Next State:', hash(next_state))
        target = reward + self.gamma*np.amax(self.brain.predict(np.array([hash(next_state)])))
        #print('Target:', target)
      target_f = self.brain.predict(np.array([hash(state)]))
      target_f[0][action] = target
      self.brain.fit(np.array([hash(state)]), target_f, epochs=1, verbose=0)
    if self.exploration_rate > self.exploration_min:
      self.exploration_rate *= self.exploration_decay
    #print('Exploration rate:', self.exploration_rate)
      
# Adrurlbot Class
class Adrurlbot():
  def __init__(self, start_state):
    self.sample_batch_size = 64
    self.episodes = 20000
    self.state_size = 500
    self.action_size = 6
    self.agent = Agent(self.state_size, self.action_size)
    self.start_state = start_state
    
  def run(self):
    try:
      for index_episode in range(self.episodes):
        state = self.start_state
        done = False
        total_reward = 0
        steps = 0
        picked_up = False
        while not done:
          action = self.agent.act(state)
          next_state, reward, done, picked_up = algo.act(state, action, picked_up)
          total_reward += reward
          self.agent.remember(state, action, reward, next_state, done)
          state = next_state
          steps += 1
        
        print('Episode: {:},  Reward: {:}, Steps: {:}, len(self.memory): {:}, Exploration Rate: {:}'.format(index_episode, total_reward, steps, len(self.agent.memory),  self.agent.exploration_rate))  
        plt.scatter(index_episode, total_reward, color='r')
        plt.draw()
        self.agent.replay(self.sample_batch_size)
        plt.pause(0.001)
      plt.show()
    finally:
      self.agent.save_model()
      
      
         
    
         
