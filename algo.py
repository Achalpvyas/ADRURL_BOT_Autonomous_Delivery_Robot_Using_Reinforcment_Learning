"""
Q Learning algorithm

Reference: https://medium.com/@curiousily/solving-an-mdp-with-q-learning-from-scratch-deep-reinforcement-learning-for-hackers-part-1-45d1d360c120

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import random
import env
import numpy as np
from copy import deepcopy

# Defining the map
ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

# Random map generator
grid_elements = [ZOMBIE, ZOMBIE, EMPTY, EMPTY, EMPTY, EMPTY]
grid = []
for i in range(5):
  if i == 0:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), ICE_CREAM])
  elif i == 4:
    grid.append([CAR, random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements)])  
  else:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements)])

# Defining the action set
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

start_state = env.State(grid=grid, car_pos=[4, 0])
N_STATES = 25
N_EPISODES = 40

MAX_EPISODE_STEPS = 100

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0

# Function to action on the environment
def act(state, action):
  def new_car_pos(state, action):
    p = deepcopy(state.car_pos)
    if action == UP:
      p[0] = max(0, p[0] - 1)
    elif action == DOWN:
      p[0] = min(len(state.grid) - 1, p[0] + 1)
    elif action == LEFT:
      p[1] = max(0, p[1] - 1)
    elif action == RIGHT:
      p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
    else:
      raise ValueError(f"Unknown action {action}")
    return p

  p = new_car_pos(state, action)
  grid_item = state.grid[p[0]][p[1]]

  new_grid = deepcopy(state.grid)
  
  if grid_item == ZOMBIE:
      reward = -100
      is_done = True
      new_grid[p[0]][p[1]] += CAR
  elif grid_item == ICE_CREAM:
      reward = 1000
      is_done = True
      new_grid[p[0]][p[1]] += CAR
  elif grid_item == EMPTY:
      reward = -1
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] = CAR
  elif grid_item == CAR:
      reward = -1
      is_done = False
  else:
      raise ValueError(f"Unknown grid item {grid_item}")
  
  for row in new_grid:
    print(' '.join(row))
        
  return env.State(grid=new_grid, car_pos=p), reward, is_done
      
def train():
  # Running the algo for N Episodes
  for e in range(N_EPISODES):
      
      state = start_state
      total_reward = 0
      alpha = alphas[e]
      
      itr = 0
      # Iterating for Max episode steps in each episode
      for _ in range(MAX_EPISODE_STEPS):
          print('itr:', itr)
          for row in grid:
            print(' '.join(row))
          action = env.choose_action(state)
          next_state, reward, done = act(state, action)
          total_reward += reward
          
          env.q(state)[action] = env.q(state, action) + \
                  alpha * (reward + gamma *  np.max(env.q(next_state)) - env.q(state, action))
          state = next_state
          itr = itr+1
          if done:
              break
      print(f"Episode {e + 1}: total reward -> {total_reward}")
  return env.q_table
  
def test(q_table):
  itr = 0
  state = start_state
  for _ in range(MAX_EPISODE_STEPS):
    print('itr:', itr)
    grid = state.grid
    for row in grid:
      print(' '.join(row))
    print('')
    action = np.argmax(q_table[state])     
    next_state, reward, done = act(state, action)
    state = next_state
    itr = itr+1
    if done:
      break
