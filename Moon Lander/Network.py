# Import required libraries
import numpy as np
from RLGlue import RLGlue
from LunarLanderEnvironment import LunarLanderEnvironment
from copy import deepcopy
from tqdm import tqdm
import os 
import shutil
from Agent import Agent

class ActionValueNetwork :
    
    def __init__(self, network_config) :
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        
        self.layer_sizes = [self.state_dim, self.num_hidden_units, self.num_actions]
        
        # Initialization of the weights that transform one layer matrix to the next
        self.weights = [dict() for i in range(len(self.layer_sizes) - 1)]
        for i in range(len(self.layer_sizes) - 1) :
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i+1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i+1]))
    
    # We need to get the values of all actions given the current state
    def get_action_values(self, s) :
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        # Given the state we compute action values through short neural network with relu activation
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0) # relu activation
        
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        action_values = np.dot(x, W1) + b1
        
        return action_values
    
    def get_TD_update(self, s, delta_mat) :
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat  
            correspond to one state in the batch. Each row has only one non-zero element 
            which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """
        
        # Get weights to update
        W0, b0 = self.weights[0]['W'], self.weights[1]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)
        
        # Array of Dictionaries to update weights
        td_update = [dict() for i in range(len(self.weights))]
        
        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        
        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v)
        
        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis = 0, keepdims=True) * 1. / s.shape[0]
        
        return td_update
    
    def init_saxe(self, rows, cols) :
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols :
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph
        
        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)
    
    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)
        
def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """

    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)

    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    exp_preferences = np.exp(preferences - reshaped_max_preference)

    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences

    action_probs = action_probs.squeeze()
    
    return action_probs



def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau) :
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """
    
    q_next_mat = current_q.get_action_values(next_states)
    
    probs_mat = softmax(q_next_mat, tau)
    
    v_next_vec = np.sum(q_next_mat * probs_mat, axis=1) * (1 - terminals)
    
    target_vec = rewards + discount * v_next_vec
    
    q_mat = network.get_action_values(states)
    
    batch_indices = np.arange(q_mat.shape[0])
    
    q_vec = q_mat[batch_indices, actions]
    
    delta_vec = target_vec - q_vec
    
    return delta_vec


def optimize_network(experiences, discount, optimizer, network, current_q, tau) :
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions, 
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    """
    
    # Get states, action, rewards, terminals, and next_states from experiences
    states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]
    
    delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)
    
    batch_indices = np.arange(batch_size)
    
    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec
    
    td_update = network.get_TD_update(states, delta_mat)

    weights = optimizer.update_weights(network.get_weights(), td_update)

    network.set_weights(weights)
    
            
def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters) :
    
    rl_glue = RLGlue(environment, agent)
    
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], experiment_parameters["num_episodes"]))
    
    env_info = {}

    agent_info = agent_parameters
    
    for run in range(1, experiment_parameters["num_runs"]+1) :
        agent_info["seed"] = run
        agent_info["network_configuration"]["seed"] = run
        env_info["seed"] = run
        
        rl_glue.rl_init(agent_info, env_info)
        
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)) :
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
            
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results') :
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)
    shutil.make_archive('results', 'zip', 'results')

# Experiment parameters
experiment_parameters = {
    "num_runs" : 1,
    "num_episodes" : 300,
    "timeout" : 1000
}

# Environment parameters
environment_parameters = {}

current_env = LunarLanderEnvironment

# Agent parameters
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
current_agent = Agent

# Run experiment
run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

