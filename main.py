"""
Q Learning

Authors: 
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import env
import algo

# Main Function
def main():
  q_table = algo.train()
  print('The final Q table after training is:', q_table)
  algo.test(q_table)
if __name__ == '__main__':
  main()
