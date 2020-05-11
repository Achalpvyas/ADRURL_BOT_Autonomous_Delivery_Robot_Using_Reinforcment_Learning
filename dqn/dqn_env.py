import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
sys.path.insert(1, '/home/nalindas9/Documents/courses/spring_2020/enpm690-robot-learning/github/ADRURL_BOT_Autonomous_Delivery_Robot_Using_Reinforcment_Learning/qlearning')
import algo

# Agent class
class Agent():
  def __init__(self, state_size, action_size):
    self.weight_backup = "trained_weights.h5"
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
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
    model.add(Dense(24, input_dim = self.state_size, activation = 'relu'))
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
    act_values = self.brain.predict(state)
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
        target = reward + self.gamma*np.amax(self.brain.predict(next_state)[0])
      target_f[0][action] = target
      self.brain.fit(state, target_f, epochs=1, verbose=0)
    if self.exploration_rate > exploration_min:
      self.exploration_rate *= self.exploration_decay
      
# Adrurlbot Class
class Adrurlbot():
  def __init__(self, start_state):
    self.sample_batch_size = 32
    self.episodes = 10000
    self.state_size = 500
    self.action_size = 6
    self.agent = Agent(self.state_size, self.action_size)
    self.start_state = start_state
    
  def run(self):
    try:
      for index_episode in range(self.episodes):
        state = self.start_state
        
        done = False
        picked_up = False
        while not done:
          action = self.agent.act(state)
          next_state, reward, done, picked_up = algo.act(state, action, picked_up)
          self.agent.remember(state, action, reward, next_state, done)
          state = next_state
        
        print('Episode: {:}'.format(index_episode))  
        self.agent.replay(self.sample_batch_size)
    finally:
      self.agent.save_model()
      
      
         
    
         
