from core.face import Face
from core.node import Node
class Cube():
    FACE_MAP = {0: "up", 1: "right", 2: "front", 3: "down", 4: "left", 5: "back"}
    COORD_MAP = {
        # up
        (0, 0): {(0, 1), (0, 3), (4, 0), (5, 2)},
        (0, 1): {(0, 0), (0, 2), (0, 4), (5, 1)},
        (0, 2): {(0, 1), (0, 5), (5, 0), (1, 2)},
        (0, 3): {(0, 0), (0, 4), (0, 6), (4, 1)},
        (0, 4): {(0, 1), (0, 3), (0, 5), (0, 7)},
        (0, 5): {(0, 2), (0, 4), (0, 8), (1, 1)},
        (0, 6): {(0, 3), (0, 7), (4, 2), (2, 0)},
        (0, 7): {(0, 4), (0, 6), (0, 8), (2, 1)},
        (0, 8): {(0, 5), (0, 7), (1, 0), (2, 2)},
        # right
        (1, 0): {(1, 1), (1, 3), (2, 2), (0, 8)},
        (1, 1): {(1, 0), (1, 2), (1, 4), (0, 5)},
        (1, 2): {(1, 1), (1, 5), (5, 0), (0, 2)},
        (1, 3): {(1, 0), (1, 4), (1, 6), (2, 5)},
        (1, 4): {(1, 1), (1, 3), (1, 5), (1, 7)},
        (1, 5): {(1, 2), (1, 4), (1, 8), (5, 3)},
        (1, 6): {(1, 3), (1, 7), (2, 8), (3, 2)},
        (1, 7): {(1, 4), (1, 6), (1, 8), (2, 1)},
        (1, 8): {(1, 5), (1, 7), (1, 0), (3, 5)},
        # front
        (2, 0): {(2, 1), (2, 3), (4, 2), (0, 6)},
        (2, 1): {(2, 0), (2, 2), (2, 4), (0, 7)},
        (2, 2): {(2, 1), (2, 5), (1, 0), (0, 8)},
        (2, 3): {(2, 0), (2, 4), (2, 6), (4, 5)},
        (2, 4): {(2, 1), (2, 3), (2, 5), (2, 7)},
        (2, 5): {(2, 2), (2, 4), (2, 8), (1, 3)},
        (2, 6): {(2, 3), (2, 7), (4, 8), (3, 0)},
        (2, 7): {(2, 4), (2, 6), (2, 8), (3, 5)},
        (2, 8): {(2, 5), (2, 7), (1, 6), (3, 8)},
        # down
        (3, 0): {(3, 1), (3, 3), (4, 8), (2, 6)},
        (3, 1): {(3, 0), (3, 2), (3, 4), (2, 7)},
        (3, 2): {(3, 1), (3, 5), (2, 8), (1, 6)},
        (3, 3): {(3, 0), (3, 4), (3, 6), (4, 7)},
        (3, 4): {(3, 1), (3, 3), (3, 5), (3, 7)},
        (3, 5): {(3, 2), (3, 4), (3, 8), (1, 7)},
        (3, 6): {(3, 3), (3, 7), (4, 6), (5, 8)},
        (3, 7): {(3, 4), (3, 6), (3, 8), (5, 7)},
        (3, 8): {(3, 5), (3, 7), (1, 8), (5, 6)},
        # left
        (4, 0): {(4, 1), (4, 3), (0, 0), (5, 2)},
        (4, 1): {(4, 0), (4, 2), (4, 4), (0, 3)},
        (4, 2): {(4, 1), (4, 5), (0, 6), (2, 0)},
        (4, 3): {(4, 0), (4, 4), (4, 6), (5, 5)},
        (4, 4): {(4, 1), (4, 3), (4, 5), (4, 7)},
        (4, 5): {(4, 2), (4, 4), (4, 8), (2, 3)},
        (4, 6): {(4, 3), (4, 7), (5, 8), (3, 6)},
        (4, 7): {(4, 4), (4, 6), (4, 8), (3, 3)},
        (4, 8): {(4, 5), (4, 7), (2, 6), (3, 0)},
        # back
        (5, 0): {(5, 1), (5, 3), (1, 2), (0, 2)},
        (5, 1): {(5, 0), (5, 2), (5, 4), (0, 1)},
        (5, 2): {(5, 1), (5, 5), (0, 0), (4, 0)},
        (5, 3): {(5, 0), (5, 4), (5, 6), (1, 5)},
        (5, 4): {(5, 1), (5, 3), (5, 5), (5, 7)},
        (5, 5): {(5, 2), (5, 4), (5, 8), (4, 3)},
        (5, 6): {(5, 3), (5, 7), (4, 8), (3, 8)},
        (5, 7): {(5, 4), (5, 6), (5, 8), (3, 7)},
        (5, 8): {(5, 5), (5, 7), (4, 6), (3, 6)},
    }

    def __init__(self, state=None):
        if not state:
            self.state = {
                'up':['white','white','white','white','white','white','white','white','white'],
                'right':['red','red','red','red','red','red','red','red','red'],
                'front':['green','green','green','green','green','green','green','green','green'],
                'down':['yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow'],
                'left':['orange','orange','orange','orange','orange','orange','orange','orange','orange'],
                'back':['blue','blue','blue','blue','blue','blue','blue','blue','blue']
            }
        else:
            self.state = state
        self.slice_map = {"up": "e", "right": "m", "front": "s", "down": "e", "left": "m", "back": "s"}
        self.graph = self.state_to_graph()


    def move(self, side):
        main = self.state[side]
        front = self.state['front']
        left = self.state['left']
        right = self.state['right']
        up = self.state['up']
        down = self.state['down']
        back = self.state['back']
        if side == 'front':
            left[2],left[5],left[8],up[6],up[7],up[8],right[0],right[3],right[6],down[0],down[1],down[2] = down[0],down[1],down[2],left[8],left[5],left[2],up[6],up[7],up[8],right[6],right[3],right[0] 
        elif side == 'up':
            left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2] = front[0],front[1],front[2],left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2]
        elif side == 'down':
            left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8] = back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8],left[6],left[7],left[8]
        elif side == 'back':
            left[0],left[3],left[6],up[0],up[1],up[2],right[2],right[5],right[8],down[6],down[7],down[8] = up[2],up[1],up[0],right[2],right[5],right[8],down[8],down[7],down[6],left[0],left[3],left[6] 
        elif side == 'left':
            front[0],front[3],front[6],down[0],down[3],down[6],back[2],back[5],back[8],up[0],up[3],up[6] = up[0],up[3],up[6],front[0],front[3],front[6],down[6],down[3],down[0],back[8],back[5],back[2]
        elif side == 'right':
            front[2],front[5],front[8],down[2],down[5],down[8],back[0],back[3],back[6],up[2],up[5],up[8] = down[2],down[5],down[8],back[6],back[3],back[0],up[8],up[5],up[2],front[2],front[5],front[8]

        main[0],main[1],main[2],main[3],main[4],main[5],main[6],main[7],main[8] = main[6],main[3],main[0],main[7],main[4],main[1],main[8],main[5],main[2]


    def move_prime(self, side):
        main = self.state[side]
        front = self.state['front']
        left = self.state['left']
        right = self.state['right']
        up = self.state['up']
        down = self.state['down']
        back = self.state['back']
        if side == 'front':
            left[2],left[5],left[8],up[6],up[7],up[8],right[0],right[3],right[6],down[0],down[1],down[2] = up[8],up[7],up[6],right[0],right[3],right[6],down[2],down[1],down[0],left[2],left[5],left[8]
        elif side == 'up':
            left[0],left[1],left[2],back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2] = back[0],back[1],back[2],right[0],right[1],right[2],front[0],front[1],front[2],left[0],left[1],left[2]
        elif side == 'down':
            left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8],front[6],front[7],front[8] = front[6],front[7],front[8],left[6],left[7],left[8],back[6],back[7],back[8],right[6],right[7],right[8]
        elif side == 'back':
            left[0],left[3],left[6],up[0],up[1],up[2],right[2],right[5],right[8],down[6],down[7],down[8] = down[6],down[7],down[8],left[6],left[3],left[0],up[0],up[1],up[2],right[8],right[5],right[2] 
        elif side == 'left':
            front[0],front[3],front[6],down[0],down[3],down[6],back[2],back[5],back[8],up[0],up[3],up[6] = down[0],down[3],down[6],back[8],back[5],back[2],up[6],up[3],up[0],front[0],front[3],front[6]
        elif side == 'right':
            front[2],front[5],front[8],down[2],down[5],down[8],back[0],back[3],back[6],up[2],up[5],up[8] = up[2],up[5],up[8],front[2],front[5],front[8],down[8],down[5],down[2],back[6],back[3],back[0]

        main[0],main[1],main[2],main[3],main[4],main[5],main[6],main[7],main[8] = main[2],main[5],main[8],main[1],main[4],main[7],main[0],main[3],main[6]


    def move_two(self, side):
        self.move(side)
        self.move(side)


    def move_slice(self, slice):
        front = self.state['front']
        left = self.state['left']
        right = self.state['right']
        up = self.state['up']
        down = self.state['down']
        back = self.state['back']
        if slice == "m":
            front[1],front[4],front[7],up[1],up[4],up[7],back[1],back[4],back[7],down[1],down[4],down[7] = up[1],up[4],up[7],back[7],back[4],back[1],down[7],down[4],down[1],front[1],front[4],front[7]
        elif slice == "s":
            up[3],up[4],up[5],right[1],right[4],right[7],down[3],down[4],down[5],left[1],left[4],left[7] = left[7],left[4],left[1],up[3],up[4],up[5],right[7],right[4],right[1],down[3],down[4],down[5]
        elif slice == "e":
            front[3],front[4],front[5],right[3],right[4],right[5],back[3],back[4],back[5],left[3],left[4],left[5] = left[3],left[4],left[5],front[3],front[4],front[5],right[3],right[4],right[5],back[3],back[4],back[5]


    def move_slice_prime(self, slice):
        front = self.state['front']
        left = self.state['left']
        right = self.state['right']
        up = self.state['up']
        down = self.state['down']
        back = self.state['back']
        if slice == "m":
            front[1],front[4],front[7],up[1],up[4],up[7],back[1],back[4],back[7],down[1],down[4],down[7] = down[1],down[4],down[7],front[1],front[4],front[7],up[7],up[4],up[1],back[7],back[4],back[1]
        elif slice == "s":
            up[3],up[4],up[5],right[1],right[4],right[7],down[3],down[4],down[5],left[1],left[4],left[7] = right[1],right[4],right[7],down[5],down[4],down[3],left[1],left[4],left[7],up[5],up[4],up[3]
        elif slice == "e":
            front[3],front[4],front[5],right[3],right[4],right[5],back[3],back[4],back[5],left[3],left[4],left[5] = right[3],right[4],right[5],back[3],back[4],back[5],left[3],left[4],left[5],front[3],front[4],front[5]


    def move_slice_two(self, slice):
        self.move_slice(slice)
        self.move_slice(slice)


    # M follows L, E follows D, S follows F — so wide moves on the
    # opposite faces (r/u/b) take the slice in the prime direction.
    SLICE_OPPOSED = {"right", "up", "back"}

    def wide_move(self, side):
        self.move(side)
        if side in self.SLICE_OPPOSED:
            self.move_slice_prime(self.slice_map[side])
        else:
            self.move_slice(self.slice_map[side])


    def wide_move_prime(self, side):
        self.move_prime(side)
        if side in self.SLICE_OPPOSED:
            self.move_slice(self.slice_map[side])
        else:
            self.move_slice_prime(self.slice_map[side])


    def wide_move_two(self, side):
        self.wide_move(side)
        self.wide_move(side)

    @staticmethod
    def _cw(f):
        return [f[6],f[3],f[0],f[7],f[4],f[1],f[8],f[5],f[2]]

    @staticmethod
    def _ccw(f):
        return [f[2],f[5],f[8],f[1],f[4],f[7],f[0],f[3],f[6]]

    @staticmethod
    def _rev(f):
        return f[::-1]

    def rotate(self, move):
        s = self.state
        u, r, f, d, l, b = s['up'][:], s['right'][:], s['front'][:], s['down'][:], s['left'][:], s['back'][:]
        if move == "x":
            s['up'], s['front'], s['down'], s['back'] = f, d, self._rev(b), self._rev(u)
            s['right'], s['left'] = self._cw(r), self._ccw(l)
        elif move == "x'":
            s['up'], s['front'], s['down'], s['back'] = self._rev(b), u, f, self._rev(d)
            s['right'], s['left'] = self._ccw(r), self._cw(l)
        elif move == "y":
            s['front'], s['right'], s['back'], s['left'] = r, b, l, f
            s['up'], s['down'] = self._cw(u), self._ccw(d)
        elif move == "y'":
            s['front'], s['right'], s['back'], s['left'] = l, f, r, b
            s['up'], s['down'] = self._ccw(u), self._cw(d)
        elif move == "z":
            s['up'], s['right'], s['down'], s['left'] = self._cw(l), self._cw(u), self._cw(r), self._cw(d)
            s['front'], s['back'] = self._cw(f), self._ccw(b)
        elif move == "z'":
            s['up'], s['right'], s['down'], s['left'] = self._ccw(r), self._ccw(d), self._ccw(l), self._ccw(u)
            s['front'], s['back'] = self._ccw(f), self._cw(b)


    def normalize(self):
        """Rotate cube to canonical orientation: white center up, green center front.

        This makes the state orientation-independent: R and wide l
        produce identical normalized states since they differ only
        by a whole-cube rotation.
        """
        if self.state['up'][4] == 'white' and self.state['front'][4] == 'green':
            return
        import copy
        top_rots = [[], ["x"], ["x", "x"], ["x'"], ["z"], ["z'"]]
        for top_rot in top_rots:
            for n_y in range(4):
                c = copy.deepcopy(self)
                for r in top_rot:
                    c.rotate(r)
                for _ in range(n_y):
                    c.rotate("y")
                if c.state['up'][4] == 'white' and c.state['front'][4] == 'green':
                    self.state = c.state
                    return

    def cube_to_faces(self):
        res = {}
        for face in self.state:
            color = self.state[face][4] #center
            res[color] = Face(color, self.state[face])
        return res


    def get_state(self):
        return self.state

    def state_to_graph(self):
        graph = {"white": set(), "red": set(), "green": set(), "yellow": set(), "orange": set(), "blue": set()}
        node_store = {}
        for i in range(6):
            for j in range(9):
                if (i, j) not in node_store:
                    curr_node = Node(self.state[self.FACE_MAP[i]][j], (i, j))
                    node_store[(i, j)] = curr_node
                else:
                    curr_node = node_store[(i, j)]
                for coord in self.COORD_MAP[(i, j)]:
                    if coord not in node_store:
                        temp = Node(self.state[self.FACE_MAP[coord[0]]][coord[1]], (coord[0], coord[1]))
                        curr_node.add_edge(temp)
                        node_store[coord] = temp
                    else:
                        curr_node.add_edge(node_store[coord])
        for k in node_store:
            graph[node_store[k].type].add(node_store[k])
        return graph

    def orientation_equals(self, other):
        """Check if two cube states are equal up to whole-cube rotation.

        Tries all 24 orientations of `other` and returns True if any
        produces the same state dict as `self`. This correctly identifies
        R and wide l as equivalent (they differ only by a cube rotation).
        """
        import copy
        # 6 ways to pick which face points up × 4 y-rotations each = 24
        top_rotations = [[], ["x"], ["x", "x"], ["x'"], ["z"], ["z'"]]
        for top_rot in top_rotations:
            for n_y in range(4):
                c = copy.deepcopy(other)
                for r in top_rot:
                    c.rotate(r)
                for _ in range(n_y):
                    c.rotate("y")
                if self.state == c.state:
                    return True
        return False

    def __str__(self):
        sides = ["up", "right", "front", "down", "left", "back"]
        colors = ["white", "red", "green", "yellow", "orange", "blue"]
        matrix = "Matrix:\n"+ "\n".join(["{s}:".format(s=s) + str(self.state[s]) for s in sides]) + "\n"
        graph = "Graph:\n"
        for c in colors:
            graph+="\n".join([str(n) for n in self.graph[c]])
        return matrix+"\n"+graph

if __name__ == "__main__":
    c = Cube()
    print(c)
    # c.move("front")
    # print(c)
    # c.move_slice_prime("m")
    # print(c)
    # c.move_slice_prime("m")
    # c.move("front")
    # print(c)
    # c.move("right")
    # c.rotate("x")
    # print(c.state)
    # print(c)
    # faces = c.cube_to_faces()
    # for f in faces:
    #     print(faces[f])