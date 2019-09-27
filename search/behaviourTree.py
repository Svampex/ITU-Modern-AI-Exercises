from abc import abstractmethod


class Node:
    children = []

    def addChild(self, child):
        self.children.append(child)

    @abstractmethod
    def execute(self):
        pass


class QueryLeaf(Node):
    def __init__(self, query):
        self.query = query

    def execute(self):
        return self.query()


class ActionLeaf(Node):
    def __init__(self, action):
        self.action = action

    def execute(self):
        self.action()
        return True


class Sequence(Node):
    def execute(self):
        for child in self.children:
            if not child.execute():
                return False


class Selector(Node):
    def execute(self):
        for child in self.children:
            if child.execute():
                return True

