import numpy as np
import random

opposites = {
    "North": "South",
    "South": "North",
    "West": "East",
    "East": "West",
}


class Sequence:
    """ Continues until one failure is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        for child in self.children:
            status, result = child.__call__(state)
            if not status:
                return False, None
        return True, result
        # raise NotImplementedError


class Selector:
    """ Continues until one success is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        for child in self.children:
            status, result = child.__call__(state)
            if status:
                return True, result
        return False, None
        # raise NotImplementedError


class CheckValid:
    """ Check whether <direction> is a valid action for PacMan
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        if self.direction not in state.getLegalPacmanActions():
            return False, None
        else:
            return True, None
        # raise NotImplementedError


class CheckDanger:
    """ Check whether there is a ghost in <direction>, or any of the adjacent fields.
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        return self.is_dangerous(state)
        # raise NotImplementedError

    def is_dangerous(self, state):
        """ YOUR CODE HERE!"""
        ghost_pos = state.getGhostPositions()
        pac_pos = state.getPacmanPosition()

        for pos in ghost_pos:
            if self.direction == "West":
                if pos[0] - pac_pos[0] < 2 and pos[1] == pac_pos[1]:
                    return False, None
            if self.direction == "East":
                if pos[0] - pac_pos[0] < 2 and pos[1] == pac_pos[1]:
                    return False, None
            if self.direction == "North":
                if pos[1] - pac_pos[1] < 2 and pos[0] == pac_pos[0]:
                    return False, None
            if self.direction == "South":
                if pos[1] - pac_pos[1] < 2 and pos[0] == pac_pos[0]:
                    return False, None
        return True, None


class ActionGo:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self, direction="Random"):
        self.direction = direction

    def __call__(self, state):
        if self.direction == "random":
            directions = state.getLegalPacmanActions()  # get all legal directions

            return True, random.choice(directions)
        else:
            return True, self.direction
        #raise NotImplementedError


class ActionGoNot:
    """ Go in a random direction that isn't <direction>
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        directions = state.getLegalPacmanActions()  # get all legal directions
        print directions
        directions.remove(self.direction)  # remove current direction

        return True, random.choice(directions)
        #raise NotImplementedError


class DecoratorInvert:
    def __call__(self, arg):
        return not arg[0], arg[1]


def parse_node(genome, parent=None):
    if len(genome) == 0:
        return

    if isinstance(genome[0], list):
        parse_node(genome[0], parent)
        parse_node(genome[1:], parent)

    elif genome[0] == "SEQ":
        if parent is not None:
            node = parent.add_child(Sequence(parent))
        else:
            node = Sequence(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0] == 'SEL':
        if parent is not None:
            node = parent.add_child(Selector(parent))
        else:
            node = Selector(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0].startswith("Valid"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckValid(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Danger"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckDanger(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("GoNot"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGoNot(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Go"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGo(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0] == ("Invert"):
        arg = genome[0].split('.')[-1]
        parent.add_child(DecoratorInvert(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    else:
        print("Unrecognized in ")
        raise Exception

    return parent