import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from rlglue.rl_glue import RLGlue
import main_agent
import ten_arm_env
import test_env

def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        if q_values[i] > top_value:
            top_value = q_values[i]
            ties = [i]
        elif q_values[i] == top_value:
            ties.append(i)
        
    return np.random.choice(ties)


if __name__ == "__main__":
    test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert argmax(test_array) == 8,

    np.random.seed(0)
    test_array = [1, 0, 0, 1]

    assert argmax(test_array) == 0