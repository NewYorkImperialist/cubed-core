from collections import Counter
class Node():
    def __init__(self, type, id, adj=None):
        self.type = type
        self.id = id
        if not adj:
            self.adj = set()
        else:
            self.adj = adj

    def add_edge(self, node):
        self.adj.add(node)

    def __str__(self):
        res = "Type: {type}\nID: {id}\nAdj: {{\n".format(type=self.type, id=self.id)
        for adj_node in self.adj:
            res += "\tType: {type}, ID: {id}\n".format(type=adj_node.type, id=adj_node.id)
        res += "}"
        return res
    
    def equals(self, node):
        if self.type != node.type:
            return False
        cube = Counter([_.type for _ in self.adj])
        sub = Counter([_.type for _ in node.adj])
        for c in sub:
            if c not in cube or sub[c] > cube[c]:
                return False
        return True