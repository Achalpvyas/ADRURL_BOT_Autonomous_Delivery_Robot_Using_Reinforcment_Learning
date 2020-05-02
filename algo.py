"""
Q Learning algorithm

Reference: 
1. https://medium.com/@curiousily/solving-an-mdp-with-q-learning-from-scratch-deep-reinforcement-learning-for-hackers-part-1-45d1d360c120
2. https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

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
WALL = "|"
ROBOT = "R"
DESTINATION = "D"
SOURCE = "S"
PACKAGE = "p"
EMPTY = "*"

# Random map generator
grid_elements = [WALL, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
pick_drop_elements = [SOURCE, DESTINATION, EMPTY, EMPTY]
random.shuffle(pick_drop_elements)

grid = []
for i in range(5):
  if i == 0:
    grid.append([pick_drop_elements[0], random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), pick_drop_elements[1]])
  elif i == 2:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), ROBOT, random.choice(grid_elements), random.choice(grid_elements)]) 
  elif i == 4:
    grid.append([pick_drop_elements[2], random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), pick_drop_elements[3]])
  else:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements)])

# Defining the action set
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
PICKUP = 4
DROPOFF = 5

ACTIONS = [UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF]

start_state = env.State(grid=grid, car_pos=[2, 2])
N_STATES = 500
N_EPISODES = 250

MAX_EPISODE_STEPS = 200

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0

# Function to take action on the environment
def act(state, action, picked_up):
  def new_car_pos(state, action):
    p = deepcopy(state.car_pos)
    pick_drop = 7
    if action == UP:
      p[0] = max(0, p[0] - 1)
    elif action == DOWN:
      p[0] = min(len(state.grid) - 1, p[0] + 1)
    elif action == LEFT:
      p[1] = max(0, p[1] - 1)
    elif action == RIGHT:
      p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
    elif action == PICKUP:
      p[0], p[1] = p[0], p[1]
      pick_drop = True
    elif action == DROPOFF:
      p[0], p[1] = p[0], p[1]
      pick_drop = False
    else:
      raise ValueError(f"Unknown action {action}")
    return p, pick_drop

  p, pick_drop = new_car_pos(state, action)
  grid_item = state.grid[p[0]][p[1]]

  new_grid = deepcopy(state.grid)
  
  
  if grid_item == WALL:
      reward = -100
      is_done = True
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] += ROBOT
  elif grid_item == SOURCE:
      reward = -1
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] += ROBOT
      print('Reached Pickup Point!')
  elif grid_item == EMPTY and state.grid[state.car_pos[0]][state.car_pos[1]] == SOURCE+ROBOT:
      reward = -100
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = SOURCE
      new_grid[p[0]][p[1]] = ROBOT
  elif grid_item == DESTINATION:
      reward = -1
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] += ROBOT
      print('Reached Dropoff Point!')
  elif grid_item == EMPTY and state.grid[state.car_pos[0]][state.car_pos[1]] == DESTINATION+ROBOT:
      reward = -100
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = DESTINATION
      new_grid[p[0]][p[1]] = ROBOT
  elif grid_item == SOURCE+ROBOT and pick_drop == True:
      reward = 1000
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] = ROBOT
      picked_up = True
      print('Pickup successfull!')
  elif grid_item != SOURCE and pick_drop == True:
      reward = -500
      is_done = False
      print('Tried picking up. Failed :(')
  elif grid_item == DESTINATION+ROBOT and pick_drop == False and picked_up == True:
      reward = 1000
      is_done = True
      new_grid[p[0]][p[1]] += SOURCE
      print('Dropoff successfull!')
  elif grid_item == DESTINATION+ROBOT and pick_drop == False and picked_up == False:
      reward = -500
      is_done = True
      new_grid[p[0]][p[1]] += SOURCE
      print('Dropoff successfull!')
  elif grid_item != DESTINATION and pick_drop == False:
      reward = -500
      is_done = True
      print('Tried dropping off. Failed :(')
  elif grid_item == EMPTY:
      reward = -1
      is_done = False
      old = state.car_pos
      new_grid[old[0]][old[1]] = EMPTY
      new_grid[p[0]][p[1]] = ROBOT
  elif grid_item == SOURCE+ROBOT:
      reward = -1
      is_done = False
  elif grid_item == DESTINATION+ROBOT:
      reward = -1
      is_done = False
  elif grid_item == ROBOT:
      reward = -1
      is_done = False
  else:
      raise ValueError(f"Unknown grid item {grid_item}")
  
  for row in new_grid:
    print(' '.join(row))
        
  return env.State(grid=new_grid, car_pos=p), reward, is_done, picked_up
      
def train():
  # Running the algo for N Episodes
  for e in range(N_EPISODES):
      
      state = start_state
      total_reward = 0
      alpha = alphas[e]
      
      itr = 0
      picked_up = False
      # Iterating for Max episode steps in each episode
      for _ in range(MAX_EPISODE_STEPS):
          print('itr:', itr)
          for row in grid:
            print(' '.join(row))
          print('')
          action = env.choose_action(state)
          next_state, reward, done, picked_up = act(state, action, picked_up)
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
  picked_up = False
  for _ in range(MAX_EPISODE_STEPS):
    print('itr:', itr)
    grid = state.grid
    for row in grid:
      print(' '.join(row))
    print('')
    action = np.argmax(q_table[state])     
    next_state, reward, done, picked_up = act(state, action, picked_up)
    state = next_state
    itr = itr+1
    if done:
      break
