"""
Q Learning

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import env
import algo
import random

# Defining the map
WALL = "|"
ROBOT = "R"
DESTINATION = "D"
SOURCE = "S"
PACKAGE = "p"
EMPTY = "*"

# Main Function
def main():
  print('*************************** Training started ******************************')
  q_table = algo.train()
  print('*************************** Training finished ******************************')
  print('The final Q table after training is:', q_table)
  print('*************************** Testing started ******************************')
  algo.test(q_table)
  print('*************************** Testing finished ******************************')
if __name__ == '__main__':
  main()
