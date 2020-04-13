# Environment class

class State:
  def __init__(self, grid, car_pos):
    self.grid = grid
    self.car_pos = car_pos
    
  def __eq__(self, other):
    return isinstance(other,State) and self.grid == other.grid and self.car_pos == other.car_pos
    
  def __hash__(self):
    return hash(str(self.grid) + str(self.car_pos))
    
  def __str__(self):
    return f"State(grid = {self.grid}, car_pos={self.car_pos})"
    
  
