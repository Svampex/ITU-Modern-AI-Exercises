from abc import abstractmethod


class Node:

    def __init__(self):
        self.children = []



    def addChild(self, child):
        self.children.append(child)

    @abstractmethod
    def execute(self):
        pass


class QueryLeaf(Node):
    def __init__(self, query):
        self.query = query

    def execute(self):
        q = self.query()
        return None, q


class ActionLeaf(Node):
    def __init__(self, action):
        self.action = action

    def execute(self):
        return self.action(), True


class Sequence(Node):
    def execute(self):
        result = (None, False)
        for child in self.children:
            result = child.execute()
            if not result[1]:
                return None, False
        return result

class Selector(Node):
    def execute(self):
        for child in self.children:
            result = child.execute()
            if result[1]:
                return result

