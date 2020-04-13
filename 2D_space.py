import 

# Defining Initial State
ZOMBIE = 'z'
CAR = 'c'
ICE_CREAM = 'i'
EMPTY = '*'

grid = [[ICE_CREAM, EMPTY],[ZOMBIE, CAR]]

for row in grid:
  print(''.join(row))
  
  
