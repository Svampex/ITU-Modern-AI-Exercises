# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from search import *

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    currentNode = problem.getStartState()
    explored = []
    childParent = {}
    while True:
        newNodes = problem.getSuccessors(currentNode)
        for node, action, cost in newNodes:
            if node not in explored:
                frontier.push((node, action, currentNode)) #destination node, action to get to it, the source node
        explored.append(currentNode)
        oldNode = currentNode
        (currentNode, action, prevnode) = frontier.pop()
        childParent[currentNode] = (prevnode, action)
        if problem.isGoalState(currentNode):
            break
    path = []

    #path.append(action)
    while True:
        (node, action) = childParent[currentNode]
        path.append(action)
        currentNode = node
        if currentNode == problem.getStartState():
            break

    path.reverse()
    print "Start:", problem.getStartState()
    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()
    currentNode = problem.getStartState()
    explored = []
    currentCost = 0
    childParent = {}
    while True:
        newNodes = problem.getSuccessors(currentNode)
        for node, action, cost in newNodes:
            if node not in explored:
                frontier.push((node, action, currentNode, currentCost + cost), currentCost + cost)  # destination node, action to get to it, the source node
        explored.append(currentNode)
        oldNode = currentNode
        (currentNode, action, prevnode, currentCost) = frontier.pop()
        childParent[currentNode] = (prevnode, action)
        if problem.isGoalState(currentNode):
            break
    path = []

    # path.append(action)
    while True:
        (node, action) = childParent[currentNode]
        path.append(action)
        currentNode = node
        if currentNode == problem.getStartState():
            break

    path.reverse()
    print "Start:", problem.getStartState()
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()
    currentNode = problem.getStartState()
    explored = []
    currentCost = 0
    childParent = {}
    while True:
        newNodes = problem.getSuccessors(currentNode)
        for node, action, cost in newNodes:
            if node not in explored:
                frontier.push((node, action, currentNode, currentCost + cost), currentCost + cost + heuristic(currentNode, problem))  # destination node, action to get to it, the source node
        explored.append(currentNode)
        oldNode = currentNode
        (currentNode, action, prevnode, currentCost) = frontier.pop()
        childParent[currentNode] = (prevnode, action)
        if problem.isGoalState(currentNode):
            break
    path = []

    # path.append(action)
    while True:
        (node, action) = childParent[currentNode]
        path.append(action)
        currentNode = node
        if currentNode == problem.getStartState():
            break

    path.reverse()
    print "Start:", problem.getStartState()
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
