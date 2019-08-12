STUDENT = 0
CLASS = 1


class Graph:
    """Representation of a simple graph using an adjacency map."""

    # ------------------------- nested Vertex class -------------------------
    class Vertex:
        """Lightweight vertex structure for a graph."""
        __slots__ = '_element', '_type', "_children", "_root", "_parent", "_next", "_pre"

        def __init__(self, x, t):
            """Do not call constructor directly. Use Graph's insert_vertex(x)."""
            self._element = x
            self._type = t
            self._children = []
            self._root = None
            self._next = None
            self._pre = None
            self._parent = None

        def children(self):
            return self._children

        def add_children(self, u):
            self._children.append(u)

        def delete_children(self, u):
            assert isinstance(u, Graph.Vertex)
            self._children.remove(u)

        def type(self):
            return self._type

        def element(self):
            """Return element associated with this vertex."""
            return self._element

        def element_and_type(self):
            return self.element(), self.type()

        def parent(self):
            return self._parent

        def set_parent(self, u):
            if u is not None:
                assert isinstance(u, Graph.Vertex)
            self._parent = u

        def set_root(self, u):
            if u is not None:
                assert isinstance(u, Graph.Vertex)
            self._root = u

        def set_next(self, u):
            if u is not None:
                assert isinstance(u, Graph.Edge)
            self._next = u

        def set_pre(self, u):
            if u is not None:
                assert isinstance(u, Graph.Edge)
            self._pre = u

        def root(self):
            return self._root

        def pre(self):
            return self._pre

        def next(self):
            return self._next

        def __hash__(self):  # will allow vertex to be a map/set key
            return hash(id(self))

        def __str__(self):
            return str((self._element, self._type))

        def __repr__(self):
            return str((self._element, self._type))

    # ------------------------- nested Edge class -------------------------
    class Edge:
        """Lightweight edge structure for a graph."""
        __slots__ = '_origin', '_destination', '_element'

        def __init__(self, u, v, x=None):
            """Do not call constructor directly. Use Graph's insert_edge(u,v,x)."""
            self._origin = u
            self._destination = v
            self._element = x

        def coordinate(self):
            if self._origin.type() == STUDENT:
                return (self._origin.element(), self._destination.element())
            elif self._origin.type() == CLASS:
                return (self._destination.element(), self._origin.element())

        def endpoints(self):
            """Return (u,v) tuple for vertices u and v."""
            return (self._origin, self._destination)

        def opposite(self, v):
            """Return the vertex that is opposite v on this edge."""
            if not isinstance(v, Graph.Vertex):
                raise TypeError('v must be a Vertex')
            return self._destination if v is self._origin else self._origin
            raise ValueError('v not incident to edge')

        def element(self):
            """Return element associated with this edge."""
            return self._element

        def __hash__(self):  # will allow edge to be a map/set key
            return hash((self._origin, self._destination))

        def __str__(self):
            return '({0},{1},{2})'.format(self._origin, self._destination, self._element)

        def __repr__(self):
            return '({0},{1},{2})'.format(self._origin, self._destination, self._element)

    # ------------------------- Graph methods -------------------------
    def __init__(self, directed=False):
        """Create an empty graph (undirected, by default).

        Graph is directed if optional paramter is set to True.
        """
        self._all_heads = {}
        self._cur_root = None
        self._cur_tail = None
        self._vertex_dic = {}
        self._outgoing = {}
        # only create second map for directed graph; use alias for undirected
        self._incoming = {} if directed else self._outgoing

    def cur_tail(self):
        return self._cur_tail

    def all_roots(self):
        return list(self._all_heads.keys())

    def _validate_vertex(self, v):
        """Verify that v is a Vertex of this graph."""
        if not isinstance(v, self.Vertex):
            raise TypeError('Vertex expected')
        if v not in self._outgoing:
            raise ValueError('Vertex does not belong to this graph.')

    def is_directed(self):
        """Return True if this is a directed graph; False if undirected.

        Property is based on the original declaration of the graph, not its contents.
        """
        return self._incoming is not self._outgoing  # directed if maps are distinct

    def vertex_count(self):
        """Return the number of vertices in the graph."""
        return len(self._vertex_dic)

    def vertices(self):
        """Return an iteration of all vertices of the graph."""
        return self._vertex_dic.values()

    def edge_count(self):
        """Return the number of edges in the graph."""
        total = sum(len(self._outgoing[v]) for v in self._outgoing)
        # for undirected graphs, make sure not to double-count edges
        return total if self.is_directed() else total // 2

    def edges(self):
        """Return a set of all edges of the graph."""
        result = set()  # avoid double-reporting edges of undirected graph
        for secondary_map in self._outgoing.values():
            result.update(secondary_map.values())  # add edges to resulting set
        return result

    def get_edge(self, u, v):
        """Return the edge from u to v, or None if not adjacent."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self._outgoing[u].get(v)  # returns None if v not adjacent

    def degree(self, v, outgoing=True):
        """Return number of (outgoing) edges incident to vertex v in the graph.

        If graph is directed, optional parameter used to count incoming edges.
        """
        self._validate_vertex(v)
        adj = self._outgoing if outgoing else self._incoming
        return len(adj[v])

    def incident_edges(self, v, outgoing=True):
        """Return all (outgoing) edges incident to vertex v in the graph.

        If graph is directed, optional parameter used to request incoming edges.
        """
        self._validate_vertex(v)
        adj = self._outgoing if outgoing else self._incoming
        for edge in adj[v].values():
            yield edge

    def insert_vertex(self, x, t):
        """Insert and return a new Vertex with element x."""
        if self._cur_tail is not None and self._cur_tail.type() == t:
            raise ValueError("Cannot have two adjacent vertices with the same type")
        if (x, t) not in self._vertex_dic.keys():
            new = self.Vertex(x, t)
            self._vertex_dic[(x, t)] = new
            self._outgoing[new] = {}
            if self.is_directed():
                self._incoming[new] = {}  # need distinct map for incoming edges

            if self._cur_root is None:
                self._cur_root = new
                self._cur_tail = new
                self._all_heads[new] = [new]
                new.set_root(new)

            else:
                self._cur_tail.add_children(new)
                self.insert_edge(self._cur_tail, new)
                new.set_root(self._cur_root)
                new.set_pre(self.get_edge(self._cur_tail, new))
                self._cur_tail.set_next(self.get_edge(self._cur_tail, new))
                self._cur_tail = new
        else:

            peer = self._vertex_dic[(x, t)]
            self.insert_edge(self._cur_tail, peer)
            if peer.root() == self._cur_tail.root():
                print("form a cycle")
                cycle = []
                cur = self._cur_root
                count = 0
                while True:
                    if count > 1000:
                        raise RuntimeError("Exceed maximum loops.")
                    next_edge = cur.next()
                    if next_edge is None:
                        break
                    cycle.append(next_edge.coordinate())
                    cur = next_edge.opposite(cur)
                    count += 1

                cycle.append(self.get_edge(self.cur_tail(), peer).coordinate())

                cur = peer
                count = 0
                while True:

                    if count > 1000:
                        raise RuntimeError("Exceed maximum loops.")
                    next_edge = cur.pre()

                    if next_edge is None or next_edge.coordinate() in cycle:
                        if next_edge is not None:
                            cycle = cycle[cycle.index(next_edge.coordinate()) + 1:]
                        break
                    print(cur, cur.pre(), cur.next())
                    cycle.append(next_edge.coordinate())
                    next_node = next_edge.opposite(cur)
                    # print(cur, next_node, next_node.next())
                    if next_node.next() is None or next_node.next().coordinate() not in cycle:
                        print("here", cur, next_node, next_node.next(), next_edge)

                        self.print_all_vertices()
                        print(cycle)
                        if next_node.next() is not None:
                            old = next_node.next().opposite(next_node)

                            self._all_heads[self._cur_root].append(old)
                        self._all_heads[self._cur_root].remove(next_edge.opposite(next_node))
                        next_node.set_next(next_edge)
                    cur = next_node

                self._cur_tail.add_children(peer)
                self._cur_tail.set_next(self.get_edge(self._cur_tail, peer))

                if peer.next() is None or peer.next().coordinate() not in cycle:
                    if peer.next() is not None:
                        old = peer.next().opposite(peer)
                        self._all_heads[self._cur_root].append(old)

                    peer.set_next(self.get_edge(self._cur_tail, peer))

                peer.add_children(self._cur_tail)

                print(cycle)
                return cycle
            else:
                print("join trees")

                self._cur_tail.set_next(self.get_edge(self._cur_tail, peer))
                self._cur_tail.add_children(peer)
                # print(self._all_heads)
                tmp_root = peer.root()
                print("tmp_root", tmp_root, "current", self.cur_tail().root(), self._cur_root, self._all_heads)
                l = self._all_heads.pop(tmp_root)
                l.remove(tmp_root)
                self._all_heads[self._cur_root].extend(l)

                # print(peer, peer.pre(), peer.next())
                # cur = peer.pre().opposite(cur)
                print(peer.children())
                cur = peer
                if cur.pre() is not None:
                    cur = cur.pre().opposite(cur)
                    self._all_heads[self._cur_root].append(cur)
                else:
                    cur = None

                if cur is not None:
                    while True:
                        next_edge = cur.pre()

                        self.flip_node(cur)
                        # print(cur, cur.pre(), cur.next())
                        if next_edge is None:
                            if cur.next() is None and len(cur.children()) > 0:
                                cur.set_next(self.get_edge(cur, cur.children()[0]))
                            break
                        next_node = next_edge.opposite(cur)
                        cur = next_node
                if peer.pre() is not None:
                    peer.add_children(peer.pre().opposite(peer))
                print(peer.children())
                # print("peer", peer, self.get_edge(self._cur_tail, peer), peer.children())
                peer.set_pre(self.get_edge(self._cur_tail, peer))

                cur = peer
                while True:
                    next_edge = cur.next()
                    if next_edge is None:
                        self._cur_tail = cur
                        break
                    cur = next_edge.opposite(cur)

    def insert_edge(self, u, v, x=None):
        """Insert and return a new Edge from u to v with auxiliary element x.

        Raise a ValueError if u and v are not vertices of the graph.
        Raise a ValueError if u and v are already adjacent.
        """
        if self.get_edge(u, v) is not None:  # includes error checking
            raise ValueError('u and v are already adjacent')
        e = self.Edge(u, v, x)
        self._outgoing[u][v] = e
        self._incoming[v][u] = e

    def remove_edge(self, coordinate, peer=None, path=None, cycled=True):
        # To do
        assert isinstance(coordinate, tuple)
        if peer is not None:
            peer = self._vertex_dic[peer]
        u = (coordinate[0], STUDENT)
        v = (coordinate[1], CLASS)
        u = self._vertex_dic[u]
        v = self._vertex_dic[v]
        assert isinstance(u, Graph.Vertex)
        assert isinstance(v, Graph.Vertex)
        deleted_edge = self._outgoing[u][v]
        self._outgoing[u].pop(v)
        self._incoming[v].pop(u)
        print(u, v, peer, self.cur_tail())
        if cycled:
            if (u == peer and v == self.cur_tail()) or (u == self.cur_tail() and v == peer):
                if u.pre() != deleted_edge and u.next() != deleted_edge:
                    v.set_next(None)
                elif v.pre() != deleted_edge and v.next() != deleted_edge:
                    u.set_next(None)
                else:
                    v.set_next(None)
                    u.set_next(None)
                u.delete_children(v)
                v.delete_children(u)

            else:
                if coordinate == path[-1] and cycled:
                    if u.pre() != deleted_edge and u.next() != deleted_edge:
                        v.set_pre(None)
                        u.delete_children(v)
                    elif v.pre() != deleted_edge and v.next() != deleted_edge:
                        u.set_pre(None)
                        v.delete_children(u)
                    else:
                        raise ValueError()

                else:
                    if u.pre() == deleted_edge:
                        u.set_pre(None)

                    elif u.next() == deleted_edge:
                        u.set_next(None)
                        u.delete_children(v)

                    else:
                        print(deleted_edge, u, u.next(), u.pre())
                        raise ValueError("Deleted edge is neither next nor pre of u.")

                    if v.pre() == deleted_edge:
                        v.set_pre(None)

                    elif v.next() == deleted_edge:
                        v.set_next(None)
                        v.delete_children(u)

                    else:
                        print(deleted_edge, v, v.next(), v.pre())
                        raise ValueError("Deleted edge is neither next nor pre of v.")
                # delete an edge from a cycle
                if cycled:
                    cur = self._cur_root
                    need_flip = False
                    while True:
                        # print(cur, cur.pre(), cur.next())
                        if need_flip:
                            self.flip_node(cur)
                        next_edge = cur.next()
                        if next_edge is None:
                            self._cur_tail = cur
                            break
                        next_node = next_edge.opposite(cur)
                        if next_node.next() == next_edge:
                            need_flip = True
                        cur = next_node
                    if coordinate != path[-1]:
                        cur = u if self.cur_tail() == v else v
                        print(cur)
                        need_flip = False
                        while True:
                            print(cur, cur.pre(), cur.next())
                            if cur.pre() is None:
                                need_flip = True

                            if need_flip:
                                self.flip_node(cur)
                            next_edge = cur.pre()
                            if next_edge.coordinate() == path[-1]:
                                if cur not in self._all_heads[self._cur_root]:
                                    self._all_heads[self._cur_root].append(cur)
                                break
                            next_node = next_edge.opposite(cur)
                            # print(need_flip, cur, cur.pre(), cur.next(), next_node)

                            if need_flip and next_node.next().opposite(next_node) == cur:
                                need_flip = False
                            cur = next_node

                    if len(self.cur_tail().children()) >= 1 and self.cur_tail().next() is None:
                        new = self.cur_tail().children()[0]
                        self.cur_tail().set_next(self.get_edge(self.cur_tail(), new))

                        self._all_heads[self._cur_root].remove(new)

                        cur = self._cur_tail
                        while True:
                            next_edge = cur.next()
                            if next_edge is None:
                                self._cur_tail = cur
                                break
                            cur = next_edge.opposite(cur)

        else:
            print("not cycled")
            new_root = u if u.pre() == deleted_edge else v
            new_tail = u if new_root == v else v
            new_root.set_pre(None)
            if new_tail.next() == deleted_edge:
                new_tail.set_next(None)
            print(new_root, new_tail)
            new_tail.delete_children(new_root)

            new_heads = []

            for v in self.all_child_nodes(new_root):
                v.set_root(new_root)
                if v in self._all_heads[self._cur_root]:
                    new_heads.append(v)
                    self._all_heads[self._cur_root].remove(v)
            self._all_heads[new_root] = new_heads
            if self.cur_tail().root() != self._cur_root:
                self._cur_tail = new_tail
                cur = self._cur_tail
                while True:
                    if cur.next() is None and len(cur.children()) == 0:
                        break
                    if cur.next() is None:
                        cur.set_next(self.get_edge(cur, cur.children()[0]))
                    cur = cur.next().opposite(cur)

    def main_path(self):
        result = []
        cur = self._cur_root
        while True:
            next_edge = cur.next()
            if next_edge is None:
                break
            result.append(next_edge.coordinate())
            cur = next_edge.opposite(cur)
        return result

    def all_child_nodes(self, u):
        if isinstance(u, tuple):
            u = self._vertex_dic[u]
        else:
            assert isinstance(u, Graph.Vertex)
        l = [u]
        result = [u]
        while len(l) > 0:
            cur = l.pop()
            print(result)
            result.extend(cur.children())
            l.extend(cur.children())
        return result

    def remove_vertex(self, v):
        """ Remove a vertex from the graph. Remember to delete any edge that has this vertex.
            @v: the vertex to be removed.
        """
        # To do
        self._outgoing.pop(v)
        if self.is_directed():
            self._incoming.pop(v)

    def flip_edge(self, u, v):
        e = self._outgoing[u][v]
        self._outgoing[u].pop(v)
        self._outgoing[v][u] = e
        self._incoming[v].pop(u)
        self._incoming[u][v] = e

    def flip_node(self, u):
        assert isinstance(u, Graph.Vertex)
        pre_edge = u.pre()
        next_edge = u.next()
        if pre_edge is not None:
            u.add_children(pre_edge.opposite(u))
        if next_edge is not None:
            u.delete_children(next_edge.opposite(u))
        u.set_next(pre_edge)
        u.set_pre(next_edge)

    def flip_main_path(self):
        print("flip main path")
        cur = self._cur_root
        while True:
            tmp_edge = cur.next()
            self.flip_node(cur)
            if tmp_edge is None:
                break
            next_node = tmp_edge.opposite(cur)
            cur = next_node
        l = self._all_heads.pop(self._cur_root)
        l.remove(self._cur_root)
        l.append(self._cur_tail)
        self._all_heads[self._cur_tail] = l
        self._cur_root, self._cur_tail = self._cur_tail, self._cur_root
        if len(self.cur_tail().children()) > 0:
            edge = self.get_edge(self._cur_tail, self.cur_tail().children()[0])
            self._cur_tail.set_next(edge)
            self.cur_tail().children()[0].set_pre(edge)

        cur = self._cur_tail
        while True:
            next_edge = cur.next()
            if next_edge is None:
                self._cur_tail = cur
                break
            cur = next_edge.opposite(cur)

        for node in self.all_child_nodes(self._cur_root):
            node.set_root(self._cur_root)

        print(self._all_heads.keys())

    def all_heads(self):
        result = []
        for i in self._all_heads.values():
            result.extend(i)
        return result

    def print_all_vertices(self):
        for v in self.vertices():
            print("node is:", v, "pre is:", v.pre(), "next is:", v.next(), "has children:", v.children(), "root is:", v.root())

    def print_all_children(self):
        for v in self.vertices():
            print(v, v.children())

    def check_valid(self):
        for i in self.vertices():
            if i.next() is not None and i.next().opposite(i).pre() != i.next():
                print(i)
                return False
            if i.pre() is not None and i not in i.pre().opposite(i).children():
                print(i)
                return False

        for i in self.vertices():
            for c in i.children():
                if c.pre().opposite(c) != i:
                    print(i, c)
                    return False

        return True

    def __str__(self):
        print("The tree is:")
        result = []
        for root in self._all_heads.keys():
            tmp_heads = self._all_heads[root]
            result.append("Current root is: {}".format(root))

            for head in tmp_heads:

                tmp_path = []
                if head == root:
                    tmp_path.append("Main path:")

                cur = head
                # print("head: ", head, head.next())
                count = 0
                while True:
                    if count > 10000:

                        raise RuntimeError("Exceed maximum loops for head {}.".format(head))
                    next_edge = cur.next()
                    # print(cur, cur.pre(), cur.next())
                    if next_edge is None or next_edge.opposite(cur) == root:
                        break
                    tmp_path.append(str(next_edge.coordinate()))
                    next_node = next_edge.opposite(cur)
                    cur = next_node
                    count += 1

                if len(tmp_path) != 0:
                    result.append(" ".join(tmp_path))
            result.append("")

        return "\n".join(result)


def main():
    # T = Graph()  # Mark True so the graph is directed.
    # print(1)
    # T.insert_vertex(0, STUDENT)
    # T.insert_vertex(2, CLASS)
    # T.insert_vertex(2, STUDENT)
    # T.insert_vertex(7, CLASS)
    # path = T.insert_vertex(0, STUDENT)
    # T.remove_edge((2, 7), path=path)
    #
    # print(2)
    # T.insert_vertex(8, CLASS)
    # T.insert_vertex(3, STUDENT)
    # T.insert_vertex(1, CLASS)
    # T.insert_vertex(1, STUDENT)
    # T.insert_vertex(0, CLASS)
    # T.insert_vertex(5, STUDENT)
    # T.insert_vertex(4, CLASS)
    # T.insert_vertex(10, STUDENT)
    # path = T.insert_vertex(1, CLASS)
    # T.remove_edge((5, 0), path=path)
    #
    # print(3)
    # T.insert_vertex(6, STUDENT)
    # path = T.insert_vertex(2, CLASS)
    # T.remove_edge((3, 8), path=path)
    # T.print_all_vertices()
    # print(4)
    # path = T.insert_vertex(5, STUDENT)
    # T.remove_edge((5, 8), path=path)
    # print(T)
    #
    # T.insert_vertex(7, STUDENT)
    # path = T.insert_vertex(0, CLASS)
    # T.remove_edge((2, 2), path=path)
    # print(T)
    #
    # T.insert_vertex(16, CLASS)
    # path = T.insert_vertex(0, STUDENT)
    # T.remove_edge((0, 2), path=path)
    # print(T)
    #
    # path = T.insert_vertex(3, STUDENT)
    # T.remove_edge((3, 1), path=path)
    # print(T)
    #
    # path = T.insert_vertex(2, CLASS)
    # T.remove_edge((6, 0), path=path)
    # print(T)
    #
    # T.insert_vertex(5, CLASS)
    # path = T.insert_vertex(1, STUDENT)
    # T.remove_edge((3, 2), path=path)
    # # T.print_all_vertices()
    # print(T)
    #
    # T.insert_vertex(13, CLASS)
    # path = T.insert_vertex(1, STUDENT)
    # T.remove_edge((3, 13), path=path)
    # print(T.all_heads())
    # print(T)
    # # T.print_all_vertices()
    #
    # T.insert_vertex(15, CLASS)
    # path = T.insert_vertex(0, STUDENT)
    # T.remove_edge((3, 15), path=path)
    #
    # T.insert_vertex(19, CLASS)
    # path = T.insert_vertex(0, STUDENT)
    # T.remove_edge((3, 19), path=path)
    #
    # T.insert_vertex(20, CLASS)
    # path = T.insert_vertex(0, STUDENT)
    # T.remove_edge((0, 7), path=path)
    # T.print_all_vertices()
    # print(T)
    #
    # path = T.insert_vertex(7, STUDENT)
    # T.remove_edge((7, 13), path=path)
    # print(T)

    # basic test
    T = Graph()
    T.insert_vertex(1, STUDENT)
    T.insert_vertex(2, CLASS)
    T.insert_vertex(3, STUDENT)
    T.insert_vertex(4, CLASS)
    T.insert_vertex(5, STUDENT)
    T.insert_vertex(6, CLASS)

    path = T.insert_vertex(1, STUDENT)
    T.remove_edge((3, 2), peer=(1, STUDENT), path=path)
    print(T)

    T.insert_vertex(7, STUDENT)
    T.insert_vertex(8, CLASS)
    path = T.insert_vertex(5, STUDENT)
    T.remove_edge((7, 2), peer=(5, STUDENT), path=path)
    print(T.cur_tail())
    print(T)
    T.remove_edge((1, 6), path=path, cycled=False)
    print(T)
    print(T._all_heads)
    T.insert_vertex(5, STUDENT)
    print(T)
    T.print_all_vertices()
    print(T._all_heads)

    # T.print_all_vertices()
    # T.insert_vertex(5, STUDENT)
    # T.insert_vertex(6, CLASS)
    # print(T)
    # # # print(T.edges())
    # path = T.insert_vertex(3, STUDENT)
    #
    # T.remove_edge((5, 6), path=path)
    # # print(T.edges())
    # print(T)
    # T.flip_main_path()
    # print(T)
    # #
    # T.insert_vertex(7, STUDENT)
    # T.insert_vertex(8, CLASS)
    # print(T)
    # T.print_all_vertices()
    # path = T.insert_vertex(1, STUDENT)
    # T.remove_edge((7, 6), path=path)
    # print(T)
    #
    # path = T.insert_vertex(6, CLASS)
    # T.remove_edge((3, 6), path=path)
    # # T.print_all_vertices()
    # #
    # T.remove_edge((1, 7), cycled=False)
    # # T.print_all_vertices()
    # print(T)
    #
    # # print(T.all_child_nodes((6, CLASS)))
    # # T.insert_vertex(9, CLASS)
    # # print(T)
    # T.insert_vertex(6, CLASS)
    # print(T)
    # T.insert_vertex(10, STUDENT)
    # T.insert_vertex(11, CLASS)
    # print(T)
    # path = T.insert_vertex(1, STUDENT)
    #
    # T.remove_edge((10, 11), path=path)
    # T.print_all_vertices()
    # print(T)
    # T.insert_vertex(12, CLASS)
    # print(T)
    # T.print_all_vertices()
    #
    # print(T)
    #
    # print(T.all_child_nodes((1, STUDENT)))


if __name__ == "__main__":
    main()
