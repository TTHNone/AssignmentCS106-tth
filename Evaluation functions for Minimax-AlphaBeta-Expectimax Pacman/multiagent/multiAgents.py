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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        #util.raiseNotDefined()
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    #The max function take a gameState, depth of tree and agentIndex.
    #It comoutes the maximum value in AlphaBeta pruning algorithm for max-palyer.

    def max_fun(self, gameState, depth, agentIndex, alpha, beta):
        #The index for pacman is 0.
        pacmanIndex = 0
        #pacman_actions denotes the legal actions that it can take.
        pacman_actions = gameState.getLegalActions(pacmanIndex)
        #ubound denotes the negative infinity or high value for Minimax algorithm.
        ubound = -100000
        #Terminal test to Check if we have reach the cut-off state or leaf node.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #Loop to generate successors.
        for action in pacman_actions:
            #Removing Directions.STOP from legal actions as given in question.
            if action != Directions.STOP:
                #Generate successor for the pacman using action from actions.
                next_node = gameState.generateSuccessor(pacmanIndex, action)
                #Minimize next agent.
                ghostIndex = pacmanIndex+1
                value = self.min_fun(next_node, depth,ghostIndex, alpha, beta)
                if value > beta:
                    #Update value to remove the unvisited branch of tree.
                    return value
                #Check if value is greater than negative infinity.
                if value > ubound: # and action!= Directions.STOP:
                    #Update value of negative infinity
                    ubound = max(value,ubound)
                    #Update the action taken by max-player.
                    max_result = action
                    #Update alpha as per algorithm
                    alpha = max(alpha, ubound)
            #Return ation taken for depth being 1.
            #Else return the new value of negative infinity
        return(ubound, max_result) [depth ==1]


    #The min_fun take a gameState, depth of tree and agentIndex.
    #It computes the minimum value in AlphaBeta algorithm for min-player.

    def min_fun(self, gameState, depth, agentIndex, alpha, beta):
        #Ghost actions denotes legal action the ghost agent can take.
        ghost_actions = gameState.getLegalActions(agentIndex)
        #lbound denotes the positive inifinity value of MinMax algorithm.
        lbound = 100000
        #agent_count dentoes the total number of enemy agents in game.
        agent_count = gameState.getNumAgents()
        #Terminal test to check if the state is terminal state so as to cut-off.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #Loop for every action in legal ghost/agent actions.
        for action in ghost_actions:
            #Remove action from legal actions according to question.
            if action!= Directions.STOP:
                next_node = gameState.generateSuccessor(agentIndex, action)
                #Decrement the agent_count to check if ghost/agent left.
                if agentIndex == agent_count-1 :
                #Check if leaf node reached.
                    if depth == self.depth:
                        value = self.evaluationFunction(next_node)
            #Else call max_fun to maximize value in next ply.
                    else:
                        pacmanIndex = 0
                        #Maximize for pacman.
                        value = self.max_fun(next_node,depth+1,pacmanIndex, alpha, beta)
                else:
                    #For remaining ghosts, minimize the value.
                    value = self.min_fun(next_node, depth, agentIndex+1, alpha, beta)
            #Update value to remove the unvisited branch of tree.
        if value < alpha:
            return value
        #Update the value of positive infinity
        if value < lbound: # and action!= Directions.STOP:
            lbound = min(value,lbound)
            min_result = action
            #Update beta as per algorithm
            beta = min(beta, value)
        return lbound

    #The minmax function computes the action taken to maximize the value for max player.
    def minmax(self, gameState):
        depth = 0
        depth += 1
        pacmanIndex = 0
        max_result = self.max_fun(gameState, depth, pacmanIndex, -100000, 100000)
        return max_result


    def getAction(self, gameState):

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
        """
      return self.minmax(gameState)

      #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState)
        #util.raiseNotDefined()

    def max_fun(self, gameState, depth, agentIndex):
        #The index for pacman is 0.
        pacmanIndex = 0
        #pacman_actions denotes the legal actions that it can take.
        pacman_actions = gameState.getLegalActions(pacmanIndex)
        #ubound denotes the negative infinity or high value for Minimax algorithm.
        ubound = -100000
        #Terminal test to Check if we have reach the cut-off state or leaf node.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #Loop to generate successors.
        for action in pacman_actions:
        #Removing Directions.STOP from legal actions as given in question.
            if action != Directions.STOP:
                #Generate successor for the pacman using action from actions.
                next_node = gameState.generateSuccessor(pacmanIndex, action)
                #Minimize ghost-agent.
                ghostIndex = pacmanIndex+1
                value = self.min_fun(next_node, depth, ghostIndex)
                #Check if value is greater than negative infinity.
                if value > ubound: # and action!= Directions.STOP:
                    #Update value of negative infinity
                    ubound = max(value,ubound)
                    #Update the action taken by max-player.
                    max_result = action
            #Return ation taken for depth being 1.
            #Else return the new value of negative infinity
        return(ubound, max_result) [depth ==1]

    def min_fun(self, gameState, depth, agentIndex):
        #Ghost actions denotes legal action the ghost agent can take.
        ghost_actions = gameState.getLegalActions(agentIndex)
        #lbound denotes the positive inifinity value of MinMax algorithm.
        lbound = 0
        #agent_count dentoes the total number of enemy agents in game.
        agent_count = gameState.getNumAgents()
        #Terminal test to check if the state is terminal state so as to cut-off.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #Calculating expected value for next ply.
        #expected_value = 1.0 / (1+len(ghost_actions))
        expected_value = 1.0 / len(ghost_actions)
        #Loop for every action in legal ghost/agent actions.
        for action in ghost_actions:
            #Remove action from legal actions according to question.
            if action!= Directions.STOP:
                next_node = gameState.generateSuccessor(agentIndex, action)
                #Decrement the agent_count to check if ghost/agent left.
                if agentIndex == agent_count-1 :
                    #Check if leaf node reached.
                    if depth == self.depth:
                        value = self.evaluationFunction(next_node)
                    #Else call max_fun to maximize value in next ply.
                    else:
                        #Maximize for pacman.
                        pacmanIndex = 0
                        value = self.max_fun(next_node,depth+1,pacmanIndex)
                else:
                    #For remaining ghosts, minimize the value.
                    value = self.min_fun(next_node, depth, agentIndex+1)
                    effective_value=value*expected_value
                    lbound=lbound+effective_value
        return lbound


    def expectimax(self, gameState):
        depth = 0
        depth += 1
        pacmanIndex = 0
        max_result = self.max_fun(gameState, depth, pacmanIndex)
        return max_result

# def betterEvaluationFunction(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function (question 5).

#     DESCRIPTION: <write something here so we know what you did>
#     """
#     "*** YOUR CODE HERE ***"
#     #util.raiseNotDefined()
#     newPos = currentGameState.getPacmanPosition()
#     newFood = currentGameState.getFood()
#     newGhostStates = currentGameState.getGhostStates()
#     newCapsules = currentGameState.getCapsules()
#     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

#     closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
#     if newCapsules:
#         closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
#     else:
#         closestCapsule = 0

#     if closestCapsule:
#         closest_capsule = -3 / closestCapsule
#     else:
#         closest_capsule = 100

#     if closestGhost:
#         ghost_distance = -2 / closestGhost
#     else:
#         ghost_distance = -500

#     foodList = newFood.asList()
#     if foodList:
#         closestFood = min([manhattanDistance(newPos, food) for food in foodList])
#     else:
#         closestFood = 0

#     return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    capsules = currentGameState.getCapsules()   
    newFoodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pacman_pos = currentGameState.getPacmanPosition()
    current_score = currentGameState.getScore()
    ghost_score = 0
    cap_score = 0

    #Calculate the distance between pacman and capsules in game using Manhattan distance.
    #Check if capsule list is not empty.
    if(len(capsules) != 0):
        #Use manhattan distance formula
        for capsule in capsules:
        cap_dis = min([manhattanDistance(capsule, pacman_pos)])
            if cap_dis == 0 :
                cap_score = float(1)/cap_dis
            else:
                cap_score = -100
        
    #Calculate distance between ghosts and pacman using Manhattan distance.  
    for ghost in ghostStates:
        ghost_x = (ghost.getPosition()[0])
        ghost_y = (ghost.getPosition()[1])
        ghost_pos = ghost_x,ghost_y
        ghost_dis = manhattanDistance(pacman_pos, ghost_pos)

    #Evaluation function return following scores.
    return current_score  - (1.0/1+ghost_dis)  + (1.0/1+cap_score)


# Abbreviation
better = betterEvaluationFunction
