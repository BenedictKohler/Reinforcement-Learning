# Description: The model converts states into a probability distribution of actions to take.
#              It therefore learns action values.

from tensorflow import keras
import numpy as np

class Model :
    
    def __init__(self, state_dim, output_dim, hidden_layers=[5, 5]) :
        """
        Arguments: 
            state_dim: tuple for number of features
            output_dim: number of actions the agent can take
            hidden_layers: Neural network architecture
        """
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.inputs = keras.Input(shape=state_dim)
        
        # Network is designed to be shallow DNN so model can learn quicker
        x = self.inputs
        for i in range(len(hidden_layers)) :
            x = keras.layers.Dense(hidden_layers[i], kernel_initializer='random_normal', bias_initializer='zeros', activation='relu')(x)
        
        self.outputs = keras.layers.Dense(output_dim, kernel_initializer='random_normal', bias_initializer='zeros', activation='softmax')(x)
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    def printSummary(self) :
        """
        Displays model architecture
        """
        print(self.model.summary())
    
    def update(self, state, reward) :
        """
        Updates the weights of the model
        """
        self.model.fit(np.array(state).reshape(1, len(state)), np.array(reward).reshape(1, 3), batch_size=1, epochs=1)

    def saveWeights(self) :
        """
        Save to continue learning later
        """
        self.model.save("RL_model_weights")
    
    def makePrediction(self, state) :
        """
        Given state, model makes a prediction
        """
        return self.model.predict(np.array(state).reshape(1, len(state)))
