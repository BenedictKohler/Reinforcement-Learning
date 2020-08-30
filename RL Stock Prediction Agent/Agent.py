# Description: An agent that learns to maximise profits based on features (states) inputted

from Model import Model
from Environment import Environment
from tensorflow import keras
import random
import numpy as np

class Agent :
    
    def __init__(self, stateDim, closePrice, features, hiddenLayers, modelName) :
        """
        Arguments: 
            stateDim: Number of features as a tuple
            closePrice: 1D array of close prices
            features: N-D array of features which are states
            hiddenLayers: Neural Network architecture
        """
        if modelName != None :
            try :
                self.model = keras.models.load_model(modelName)
            except :
                print("The Model name you gave was not recognized")
        else :
            self.model = Model(stateDim, 3, hiddenLayers)
        self.environment = Environment(closePrice, features)
        self.epsilon = 0.2 # Used for exploration
        self.gamma = 0.9 # Discounts future rewards, because want immediate rewards to be more important
        self.decayEpsilon = 0.95 # Reduces exploration over time
        self.timestep = 0
        self.totalReward = 0
        self.currentReward = 0
        self.memoryBuffer = [] # Used for experience replay
        self.memoryMax = 500
        self.lastAction = 1 # Start by doing nothing
        self.count = 0 # How long we have held current position
        self.possibleRewards = [0, 0, 0]
        self.allRewards = []
        self.initialState = None
    
    def updatePossibleRewards(self) :
        """
        To speed up learning, we update the model in a way that simulates taking multiple actions
        """
        rBuy = self.environment.getReward(0, self.timestep) * self.gamma**self.count
        rHold = self.environment.getReward(1, self.timestep) * self.gamma**self.count
        rSell = self.environment.getReward(2, self.timestep) * self.gamma**self.count
        
        self.possibleRewards[0] += rBuy
        self.possibleRewards[1] += rHold
        self.possibleRewards[2] += rSell
        
    def act(self) :
        """
        This is where the agent does its decision making
        Given the state it takes an action and gets a reward
        If it takes an action that closes current position then
        the episode is complete.
        
        Returns:
            state: current state
            episodeCompleted: Whether a position has been closed
        """

        state = self.environment.getState(self.timestep)
        episodeCompleted = False
        
        # Used for memory buffer
        if self.count == 0:
            self.initialState = state

        actions = self.policy(state)
        
        if random.random() < self.epsilon :
            action = random.choice(range(len(actions[0]))) # Select a random action
        
        else :
            action = np.random.choice(range(len(actions[0])), p=actions[0]) # Choose action based on probabilities
        
        if self.lastAction == 0 and (action == 0 or action == 1) : # Continue to hold long position
            self.lastAction = 0
            self.currentReward += self.environment.getReward(0, self.timestep) * self.gamma**self.count
            self.updatePossibleRewards()
            self.count += 1
            self.timestep += 1
            return state, episodeCompleted
        
        elif self.lastAction == 2 and (action == 2 or action == 1) : # Continue to hold short position
            self.lastAction = 2
            self.currentReward += self.environment.getReward(2, self.timestep) * self.gamma**self.count
            self.updatePossibleRewards()
            self.count += 1
            self.timestep += 1
            return state, episodeCompleted
        
        elif self.lastAction == 1 and action == 1 : # No positions open
            self.lastAction = 1
            self.currentReward += self.environment.getReward(1, self.timestep) * self.gamma**self.count
            self.updatePossibleRewards()
            self.count += 1
            self.timestep += 1
            return state, episodeCompleted
        
        elif self.lastAction == 1 and action == 0 : # Open a new long position
            self.lastAction = 0
            self.currentReward += self.environment.getReward(0, self.timestep) * self.gamma**self.count
            self.updatePossibleRewards()
            self.count += 1
            self.timestep += 1
            return state, episodeCompleted
        
        elif self.lastAction == 1 and action == 2 : # Open a new short position
            self.lastAction = 2
            self.currentReward += self.environment.getReward(2, self.timestep) * self.gamma**self.count
            self.updatePossibleRewards()
            self.count += 1
            self.timestep += 1
            return state, episodeCompleted
        
        elif self.lastAction == 0 and action == 2 : # Close long position
            episodeCompleted = True
            return state, episodeCompleted
        
        elif self.lastAction == 2 and action == 0 : # Close short position
            episodeCompleted = True
            return state, episodeCompleted
        
        else :
            raise Exception("Executed unreachable statement in act")
        
    def testAct(self) :
        """
        This is for testing, the model
        is not updated
        """
        state = self.environment.getState(self.timestep)

        actions = self.model.predict(np.array(state).reshape(1, len(state)))

        action = np.random.choice(range(len(actions[0])), p=actions[0]) # Choose action based on probabilities
        
        if self.lastAction == 0 and (action == 0 or action == 1) : # Continue to hold long position
            self.totalReward += self.environment.getTestReward("Long", self.timestep)
            self.lastAction = 0
            self.timestep += 1
        
        elif self.lastAction == 2 and (action == 2 or action == 1) : # Continue to hold short position
            self.totalReward += self.environment.getTestReward("Short", self.timestep)
            self.lastAction = 2
            self.timestep += 1
        
        elif self.lastAction == 1 and action == 1 : # No positions open
            self.lastAction = 1
            self.timestep += 1
        
        elif self.lastAction == 1 and action == 0 : # Open a new long position
            self.totalReward += self.environment.getTestReward("Long", self.timestep)
            self.lastAction = 0
            self.timestep += 1
        
        elif self.lastAction == 1 and action == 2 : # Open a new short position
            self.totalReward += self.environment.getTestReward("Short", self.timestep)
            self.lastAction = 2
            self.timestep += 1
        
        elif self.lastAction == 0 and action == 2 : # Close long position
            self.lastAction = 1
        
        elif self.lastAction == 2 and action == 0 : # Close short position
            self.lastAction = 1
        
        else :
            raise Exception("Executed unreachable statement in testAct")
        
        self.allRewards.append(self.totalReward)
    
    def resetTest(self) :
        """
        Used for the test run
        Resets necessary values
        """
        self.allRewards = []
        self.lastAction = 1
        self.timestep = 0
        
        
    
    def updateWeights(self, state) :
        """
        Given state and rewards it updates a model
        Called at end of episode
        """
        self.model.update(state, self.possibleRewards)
    
    def endEpisode(self, state, expReplay=False, epochs=50) :
        """
        Called when a position is closed or no more training examples
        """
        # Update and reset
        self.updateWeights(state)
        if len(self.memoryBuffer) >= self.memoryMax :
            del(self.memoryBuffer[0])
            self.memoryBuffer.append([self.initialState, self.possibleRewards])
        self.count = 0
        self.epsilon *= self.decayEpsilon
        self.initialState = None
        self.possibleRewards = [0, 0, 0]
        self.lastAction = 1
        self.totalReward += self.currentReward
        self.currentReward = 0
        self.allRewards.append(self.totalReward)
        
        if expReplay and len(self.memoryBuffer) > 100:
            self.runExpReplay(epochs)
    
    def getRewards(self) :
        """
        Returns list of rewards used for plotting
        """
        return self.allRewards
        
    
    def reset(self, display=True, save=True) :
        """
        Called after training is complete
        Saves and resets agent
        """
        
        if display :
            self.displayResults()
            
        self.timestep = 0
        self.memoryBuffer = []
        
        if save :
            self.model.saveWeights()
        
        
    def displayResults(self) :
        print("Timesteps:", self.timestep)
        print("Total Rewards:", self.totalReward)
    
    def runExpReplay(self, epochs) :
        """
        Experience replay used to make better
        use of our data
        """
        for i in range(epochs) :
            batch = random.choice(self.memoryBuffer)
            self.model.update(batch[0], batch[1])
    
            
    def policy(self, state) :
        """
        Returns a probability distribution
        of actions to take
        """
        return self.model.makePrediction(state)
            
        