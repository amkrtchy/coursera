import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from rlglue.rl_glue import RLGlue
import main_agent
import ten_arm_env
import test_env

plt.ion()

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


class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation=None):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        self.arm_count[self.last_action] = self.arm_count[self.last_action] + 1 
        step_size = 1/self.arm_count[self.last_action]
        
        self.q_values[self.last_action] = self.q_values[self.last_action] + step_size * (reward - self.q_values[self.last_action])
        
        
        current_action = argmax(self.q_values)
        self.last_action = current_action
        
        return current_action

class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        
        self.arm_count[self.last_action] = self.arm_count[self.last_action] + 1 
        step_size = 1/self.arm_count[self.last_action]
        
        self.q_values[self.last_action] = self.q_values[self.last_action] + step_size * (reward - self.q_values[self.last_action])
     
        rnd = np.random.random()
        if rnd < self.epsilon:
            current_action = np.random.randint(0, len(self.arm_count))
        else:
            current_action = argmax(self.q_values)
        
        self.last_action = current_action
        
        return current_action

class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        
        self.arm_count[self.last_action] = self.arm_count[self.last_action] + 1 
        
        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size * (reward - self.q_values[self.last_action])
     
        rnd = np.random.random()
        if rnd < self.epsilon:
            current_action = np.random.randint(0, len(self.arm_count))
        else:
            current_action = argmax(self.q_values)
        
        
        self.last_action = current_action
        
        return current_action

def evaluate_agent(agent):
    num_runs = 200                    # The number of times we run the experiment
    num_steps = 1000                  # The number of pulls of each arm the agent takes
    env = ten_arm_env.Environment     # We set what environment we want to use to test
    agent_info = {"num_actions": 10}  # We pass the agent the information it needs. Here how many arms there are.
    env_info = {}                     # We pass the environment the information it needs. In this case nothing.

    rewards = np.zeros((num_runs, num_steps))
    average_best = 0
    for run in range(num_runs):           # tqdm is what creates the progress bar below
        np.random.seed(run)
        
        rl_glue = RLGlue(env, agent)          # Creates a new RLGlue experiment with the env and agent we chose above
        rl_glue.rl_init(agent_info, env_info) # We pass RLGlue what it needs to initialize the agent and environment
        rl_glue.rl_start()                    # We start the experiment

        average_best += np.max(rl_glue.environment.arms)
        
        for i in range(num_steps):
            reward, _, action, _ = rl_glue.rl_step() # The environment and agent take a step and return
                                                    # the reward, and action taken.
            rewards[run, i] = reward

    scores = np.mean(rewards, axis=0)
    # Create a new figure with a unique number based on the agent name
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k', num=f"{agent.__name__}")
    plt.plot([average_best / num_runs for _ in range(num_steps)], linestyle="--")
    plt.plot(scores)
    plt.legend(["Best Possible", "Agent"])
    plt.title(f"Average Reward of {agent.__name__}")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.draw()
    plt.pause(0.1)

if __name__ == "__main__":
   evaluate_agent(GreedyAgent)
   evaluate_agent(EpsilonGreedyAgent)
   evaluate_agent(EpsilonGreedyAgentConstantStepsize)