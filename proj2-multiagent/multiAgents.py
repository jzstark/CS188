# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def _avg_manhattan(self, xy, lst):
        dist = [manhattanDistance(xy, xy2) for xy2 in lst]
        return sum(dist) / len(dist) if len(dist) != 0 else 0
    

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() # (1, 2)
        # Food: Boolean[][]; use function asList
        newFood = successorGameState.getFood() 
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        curGhostStates = currentGameState.getGhostStates()
        curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
        "*** YOUR CODE HERE ***"

        """ Achieve score: 3/4 """
        # distance to all foods
        curPos = currentGameState.getPacmanPosition()
        curFood = currentGameState.getFood()
        df_c_sum = self._avg_manhattan(curPos, curFood.asList()) 
        df_sum = self._avg_manhattan(newPos, newFood.asList())
        # distance to all ghosts 
        ghostPos = currentGameState.getGhostPositions()
        newGhostPos = successorGameState.getGhostPositions()
        dg_c_sum = self._avg_manhattan(curPos, ghostPos)
        dg_sum = self._avg_manhattan(newPos, newGhostPos)
        # distance to capsules 
        curCapPos = currentGameState.getCapsules()
        newCapPos = successorGameState.getCapsules()
        dc_c = self._avg_manhattan(curPos, curCapPos)
        dc = self._avg_manhattan(newPos, newCapPos)

        # if the new position has food or Ghost; if closer to food/capsule; if far from ghost
        # reduce point if repeat steps
        point =  (newPos in curFood.asList()) * 10 - (newPos in newGhostPos) * 100000 \
            + (newPos in curCapPos) * 30 \
            + (df_c_sum - df_sum) * 1.5 \
            - (dg_c_sum - dg_sum) * 1 \
            + (dc_c - dc) * 2 \
            + (successorGameState.getScore() - currentGameState.getScore())
        
        if min(curScaredTimes) > manhattanDistance(curPos, ghostPos[0]):
            point += (dg_c_sum - dg_sum) * 5
        # NOTE: changing the last parameter 8 something large would stop 
        # the agent from eating the capsules somehow in the mediumClassic layout 
        if len(newFood.asList()) <= 5:
            point += (random.random() - 0.5) * 3

        return point


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        # layer counter in the tree 
        self.current_layer = 1

# A big problem: "current_depth" needs to backtrack!

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getMaxValue(self, state: GameState):
        self.current_layer += 1

        v = -100000000
        pacman_actions = state.getLegalActions(0)
        vs = [0] * len(pacman_actions)
        for i, action in enumerate(pacman_actions):
            pacman_state = state.generateSuccessor(0, action)
            vs[i] = self.getValue(pacman_state, 1) # value from the first ghost
        
        # Backtrack layer counter 
        self.current_layer -= 1
        return max(v, max(vs))
        

    def getMinValue(self, state: GameState, gIndex):
        self.current_layer += 1

        if gIndex + 1 < state.getNumAgents():
            nextAgentIdx = gIndex + 1 
        else:
            nextAgentIdx = 0
        
        v = 100000000
        ghost_actions = state.getLegalActions(gIndex)
        vs = [0] * len(ghost_actions)
        for i, action in enumerate(ghost_actions):
            ghost_state = state.generateSuccessor(gIndex, action)
            vs[i] = self.getValue(ghost_state, nextAgentIdx)
        
        self.current_layer -= 1
        return min(v, min(vs))


    def getValue(self, state: GameState, index):
        n = state.getNumAgents()
        if self.current_layer >= self.depth * n or state.isWin() or state.isLose():
            return self.evaluationFunction(state) 

        if index == 0 : return self.getMaxValue(state)
        return self.getMinValue(state, index)


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Reset layer counter for each move 
        self.current_layer = 1 

        legalMoves = gameState.getLegalActions(0)
        pacStates = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.getValue(s, 1) for s in pacStates] # starting from the first ghost

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        
        return legalMoves[chosenIndex]

        # NOTE: do NOT use 'getScore' in print when evaluating; 
        # the count of getScore function matters!


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    alpha: MAX's best option on a path to root
    beta : MIN's best option on a path to root

    NOTE: how alpha-beta works in multiple MINs could be a big problem
    """

    def getMaxValue(self, state: GameState, alpha, beta):
        self.current_layer += 1

        v = -100000000
        vAction = Directions.STOP

        pacman_actions = state.getLegalActions(0)
        for action in pacman_actions:
            pacman_state = state.generateSuccessor(0, action)
            
            u, _uAction = self.getValue(pacman_state, 1, alpha, beta)
            if v < u: v = u ; vAction = action

            if v > beta :
                self.current_layer -= 1
                return v, action
            alpha = max(alpha, v)
        
        self.current_layer -= 1

        # No early exit; return the action of the largest value
        return v, vAction
        

    def getMinValue(self, state: GameState, gIndex, alpha, beta):
        self.current_layer += 1

        if gIndex + 1 < state.getNumAgents():
            nextAgentIdx = gIndex + 1 
        else:
            nextAgentIdx = 0
        
        v = 100000000
        vAction = Directions.STOP
        
        ghost_actions = state.getLegalActions(gIndex)
        for action in ghost_actions:
            ghost_state = state.generateSuccessor(gIndex, action)
            #v = min(v, self.getValue(ghost_state, nextAgentIdx, alpha, beta))
            u, _uAction = self.getValue(ghost_state, nextAgentIdx, alpha, beta)
            if v > u:
                # "I don't care about the direction my child choose, i.e. uAction; 
                # only its value, so that I can choose my own direction"
                v = u; vAction = action

            if v < alpha:
                self.current_layer -= 1
                return v, action
            beta = min(beta, v)
        
        self.current_layer -= 1
        return v, vAction


    # RETURN: (1) the best value of my children; (2) the direction of that child
    def getValue(self, state: GameState, index, alpha, beta):
        n = state.getNumAgents()
        if self.current_layer >= self.depth * n or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP

        if index == 0 : return self.getMaxValue(state, alpha, beta)
        return self.getMinValue(state, index, alpha, beta)


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialise
        self.current_layer = 1 
        alpha = -100000000
        beta = 100000000

        # NOTE: do NOT copy the previous minmax agent code -- 
        # the other layers should be aware of the top MAX node!

        _bestScore, bestAction = self.getValue(gameState, 0, alpha, beta)
        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
