"""
Python Environment for Q Learning

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import random
import numpy as np
eps = 0.3
q_table = dict()
# Defining the action set
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
PICKUP = 4
DROPOFF = 5

ACTIONS = [UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF]

# State class inititalized with current grid and the car position in the grid
class State:
    
    def __init__(self, grid, car_pos):
        self.grid = grid
        self.car_pos = car_pos
        
    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.car_pos == other.car_pos
    
    def __hash__(self):
        return hash(str(self.grid) + str(self.car_pos))
    
    def __str__(self):
        return f"State(grid={self.grid}, car_pos={self.car_pos})"

# Function to fetch the Q value for given state and action from the Q table
def q(state, action=None):
    
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
        
    if action is None:
        return q_table[state]
    
    return q_table[state][action]

# Function to choose an action given a state
def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS) 
    else:
        return np.argmax(q(state))
