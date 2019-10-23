    # searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

import copy
import numpy as np

import behaviourTree as BT

import random
from searchAgents import *

def is_dangerous(direction, state):
    """ YOUR CODE HERE!"""
    ghost_pos = state.getGhostPositions()
    pac_pos = state.getPacmanPosition()
    for ghost in state.getGhostStates():
        t = ghost.scaredTimer
        if ghost.scaredTimer > 0:
            return False
    for pos in ghost_pos:
        rng = 3
        distX = pac_pos[0] - pos[0]
        distY = pac_pos[1] - pos[1]
        if direction == "West":
            if rng > distX > 0 and rng > distY > -rng:
                return True
        if direction == "East":
            if 0 > distX > -rng and rng > distY > -rng:
                return True
        if direction == "North":
            if 0 > distY > -rng and rng > distX > -rng:
                return True
        if direction == "South":
            if rng > distY > 0 and rng > distX > -rng:
                return True
    return False

dirs = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]



# Agent Using Behaviour Trees and Path finding
class BehaviourTreeAgent(SearchAgent):
    root = None

    def mkTree(self):
        self.actions = None

        def takeRandomAction():
            self.actions = None
            randomDirs = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

            while True:
                action = random.choice(randomDirs)
                randomDirs.remove(action)
                if action in self.state.getLegalActions() and not is_dangerous(action, self.state):
                    return action
                if len(randomDirs) == 0:
                    return Directions.STOP

        def findPath():
            problem = AnyFoodSearchProblem(self.state)
            self.actions = self.searchFunction(problem)

        def danger(dir, state):
            return is_dangerous(dir, state)

        qPathExists = BT.QueryLeaf(lambda : self.actions is None)
        aFindPath = BT.ActionLeaf(lambda: findPath())
        qSafePath = BT.QueryLeaf(lambda : not danger(self.actions[0], self.state))
        aMove = BT.ActionLeaf(lambda : self.popNextAction())
        aRun = BT.ActionLeaf(lambda : takeRandomAction())

        seqFindPath = BT.Sequence()
        seqFindPath.addChild(qPathExists)
        seqFindPath.addChild(aFindPath)


        seqMove = BT.Sequence()
        seqMove.addChild(qSafePath)
        seqMove.addChild(aMove)

        seqFindPath.addChild(seqMove)

        self.root = BT.Selector()
        self.root.addChild(seqFindPath)
        self.root.addChild(seqMove)
        self.root.addChild(aRun)

    def popNextAction(self):
        if len(self.actions) > 0:
            a = self.actions.pop(0)
            if len(self.actions) == 0:
                self.actions = None
            return a
        else:
            return Directions.STOP

    def getAction(self, state):
        self.state = state
        if self.root is None:
            self.mkTree()
        return self.root.execute()[0]

#Simple Path finding agent
class PFAgent(SearchAgent):


    def popNextAction(self):
        if len(self.actions) > 0:
            a = self.actions.pop(0)
            return a
        else:
            return Directions.STOP

    def getAction(self, state):

        return self.popNextAction()