"""
Main script for DQN

Reference: 
1. https://tiewkh.github.io/blog/deepqlearning-openaitaxi/
2. https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import dqn_env
import random
import numpy as np
import sys
sys.path.insert(1, '/home/nalindas9/Documents/courses/spring_2020/enpm690-robot-learning/github/ADRURL_BOT_Autonomous_Delivery_Robot_Using_Reinforcment_Learning/qlearning')
import env

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
    
    
start_state = env.State(grid=grid, car_pos=[2, 2])

def main():
  adrurlbot = dqn_env.Adrurlbot(start_state)
  adrurlbot.run()

if __name__ == "__main__":
  main()
