import pandas as pd
import pylab as py
from Agent import Agent
import random
from tqdm import tqdm

def getFile() :
    """
    Gets a csv file entered by the user
    and returns it in a pandas dataframe
    """
    df = None
    print("\nFile Format: Close, feature_1, ..., feature_n")
    while True :
        userFile = input("Enter the csv file to be used, eg file.csv or type c to cancel: ")
        if userFile.lower() == 'c' :
            break
        try :
            df = pd.read_csv(userFile)
            break
        except FileNotFoundError :
            print("The file you entered doesn't exist, re-enter please")
            continue
    
    return df


def buildData(df) :
    """
    Arguments: A pandas dataframe
    Returns: A training and test set
    """
    count = df.shape[0] # number of data points
    t1 = int(count*0.8)
    train, test = df.iloc[:t1, :], df.iloc[t1:, :]
    
    return train, test

def trainModel(trainingData, hiddenLayers=[5, 5], numTrials=5) :
    """
    Arguments:
        trainingData: pandas df containing prices and features
        hiddenLayers: NN archictecture
        numTrials: Number of times model should run on trainingData
    """
    stateDim = trainingData.shape[1] - 1
    closePrice = trainingData.iloc[:, 0]
    features = trainingData.iloc[:, 1:]
    agent = Agent((stateDim,), closePrice, features, hiddenLayers, None)
    
    for i in range(numTrials) :
        for t in tqdm(range(len(closePrice)-1)) :
            state, episodeCompleted = agent.act()
            
            if episodeCompleted :
                replay = False
                # Ocassionally run experience replay
                if random.random() < 0.1 :
                    replay = True
                agent.endEpisode(state, replay)
        
        # Plot results
        py.title('Cumulative Rewards over time')
        py.xlabel('Timesteps')
        py.ylabel('Rewards')
        py.plot(agent.getRewards())
        
        agent.reset(True, False)
    
    agent.reset() # Automatically saves model

def testModel(trainingData, hiddenLayers=[5, 5], numTrials=5, modelName=None) :
    """
    Arguments:
        trainingData: pandas df containing prices and features
        hiddenLayers: NN archictecture
        numTrials: Number of times model should run on trainingData
    """
    stateDim = trainingData.shape[1] - 1
    closePrice = trainingData.iloc[:, 0]
    features = trainingData.iloc[:, 1:]
    agent = Agent((stateDim,), closePrice, features, hiddenLayers, modelName)
    
    for i in range(numTrials) :
        for t in tqdm(range(len(closePrice)-1)) :
            agent.testAct()
        
        # Plot results
        py.title('Profit over time')
        py.xlabel('Timesteps')
        py.ylabel('Profit')
        py.plot(agent.getRewards())
        py.savefig("TestResults")
        
        agent.resetTest()


def run() :
    df = getFile()
    try :
        if df == None :
            print("Cancelled")
            return
    except :
        pass
    
    train, test = buildData(df.iloc[:, 1:])
    trainModel(train, numTrials=2)
    #testModel(train, numTrials=1, modelName="RL_model_weights")

run()
    