# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state,action)]


    def setQValue(self, state, action, v):
        self.qvalues[(state, action)] = v 


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          Only access Q values by calling getQValue
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0 : return 0
        qs = [self.getQValue(state, a) for a in actions]
        return max(qs)



    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          Only access Q values by calling getQValue
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0 : return None
        qs = [self.getQValue(state, a) for a in actions]
        mq = max(qs)
        bestIndices = [index for index in range(len(qs)) if qs[index] == mq]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return actions[chosenIndex]


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        bestAction = self.computeActionFromQValues(state)
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        return bestAction


    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        nextNextActions = self.getLegalActions(nextState)
        qs = [self.getQValue(nextState, a) for a in nextNextActions]
        v = self.discount * max(qs) if len(qs) > 0 else 0 
        q = self.getQValue(state, action)
        self.setQValue(state, action, (1 -self.alpha) * q + 
          self.alpha * (reward + v))

        return


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is. -- 4/4
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter() 

    def getWeights(self):
        return self.weights

    def setWeights(self, k, v):
        self.weights[k] = v

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        s = 0 
        w = self.getWeights()
        # dict; key -> v; key can be string or (state, action) etc.
        features = self.featExtractor.getFeatures(state, action)
        for k, v in features.items():
            s += w[k] * v
        return s


    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        nextNextActions = self.getLegalActions(nextState)
        qs = [self.getQValue(nextState, a) for a in nextNextActions]
        v = self.discount * max(qs) if len(qs) > 0 else 0 
        q = self.getQValue(state, action)
        diff = reward + v - q

        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)
        for k, v in weights.items():
          u = self.alpha * diff * features[k]
          self.setWeights(k, v + u)
        
        return


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.getWeights())
