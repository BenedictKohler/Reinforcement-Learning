from ReplayBuffer import ReplayBuffer
from Network import ActionValueNetwork, softmax, optimize_network
from AdamOptimizer import Adam
from copy import deepcopy
import numpy as np


class Agent() :
    
    def __init__(self) :
        self.name = "expected_sarsa_agent"
    
    def agent_init(self, agent_config) :
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0
    
    def policy(self, state) :
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action
    
    def agent_start(self, state) :
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action
    
    def agent_step(self, reward, state) :
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1
        
        state = np.array([state])
        
        action = self.policy(state)
        
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Replay some steps
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size :
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay) :
                experiences = self.replay_buffer.sample()
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
        
        self.last_state = state
        self.last_action = action
        
        return action
    
    def agent_end(self, reward) :
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1
        
        state = np.zeros_like(self.last_state)
        
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Replay some steps
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                experiences = self.replay_buffer.sample()
                
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
        
    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
            
            

    