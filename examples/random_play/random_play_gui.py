################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 random_play.py -g <game>                                                                             #                                                              #
################################################################################################################
import random, numpy, argparse
from minatar import Environment
from minatar.gui import GUI

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

env = Environment(args.game)
gui = GUI(env.game_name(), env.n_channels)

e = 0
returns = []
num_actions = env.num_actions()

# Initialize the environment
env.reset()
terminated = False
G = 0

def func():
    gui.display_state(env.state())
    # Select an action uniformly at random
    action = random.randrange(num_actions)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime, unused by random agent, but included for illustration
    s_prime = env.state()

    # G += reward
    
    gui.update(50, func)
    if terminated:
        gui.set_message("Game over! Score: " + str(G))
        print("Final Score: "+str(G))
        gui.quit()


gui.update(0, func)
gui.run()
