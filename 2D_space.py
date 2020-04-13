import env

# Defining Initial State
ZOMBIE = 'z'
CAR = 'c'
ICE_CREAM = 'i'
EMPTY = '*'

grid = [[ICE_CREAM, EMPTY],[ZOMBIE, CAR]]

for row in grid:
  print(''.join(row))
  
# Defining actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

start_state = env.State(grid, [1,1])


