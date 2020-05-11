import random

# Defining the map
ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

grid_elements = [ZOMBIE, EMPTY, EMPTY, EMPTY, EMPTY]
grid = []
for i in range(5):
  if i == 0:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), ICE_CREAM])
  elif i == 4:
    grid.append([CAR, random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements)])  
  else:
    grid.append([random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements), random.choice(grid_elements)])
  
for row in grid:
  print(' '.join(row))
      

