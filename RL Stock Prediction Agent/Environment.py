# Description: The environment simulates a stock chart

class Environment :
    
    def __init__(self, closePrices, features) :
        """
        Arguments: 
            Close price to be used on each timestep
            Features so that the model can learn
        """
        self.features = features
        self.close = closePrices
        
    def getReward(self, action, timestep) :
        """
        Given current state and action
        it returns the reward
        """
        # Rewards come from previous price and current price
        if action == 0 : # Buy
            return (self.close[timestep+1] - self.close[timestep])
        
        elif action == 1 : # Hold
            return - abs( (self.close[timestep+1] - self.close[timestep]) ) * 0.2 # Penalize agent for missing an opportunity
        
        elif action == 2 : # Sell
            return (self.close[timestep] - self.close[timestep+1])
    
    def getTestReward(self, decision, timestep) :
        """
        This is used for testing the model
        It generatates actual profits or losses
        """
        if decision == "Long" :
            return (self.close[timestep+1] - self.close[timestep])
        
        elif decision == "Short" :
            return (self.close[timestep] - self.close[timestep+1])
        
        else :
            raise Exception("Executed unreachable statement in getTestReward")
            
        
    def getState(self, timestep) :
        """
        State is feautures of current timestep
        """
        return self.features.iloc[timestep, :]
