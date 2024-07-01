import numpy as np
from heapq import *
from sklearn.metrics import pairwise_distances
from union_find import UnionFind
import pandas as pd
from ordered_set import OrderedSet
import gudhi
import networkx

from typing import Tuple, Dict, List, Iterable

class MergeTree:
    def __init__(self, elements: Iterable, delta: float = 0.5, metric: str = 'cosine'):
        self.elements = elements
        self.distance_matrix = pairwise_distances(elements, metric=metric)
        rips_complex = gudhi.RipsComplex(distance_matrix=self.distance_matrix,
                                 max_edge_length=0.8) # TODO What value should this be?
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)


    
    def calculate_persistence(self):
        return self.simplex_tree.persistence()

    def persistence_pairs(self):
        self.simplex_tree.compute_persistence()
        return self.simplex_tree.persistence_pairs()


class HierarchyNodeH2:

    def __init__(self, elements, metric='cosine', use_distances=False):
        if use_distances:
            self.distance_matrix = elements
        else:
            self.distance_matrix = pairwise_distances(elements, metric=metric)
        self.persistence = []
        self.size = self.distance_matrix.shape[0]
        self.edges = self.initialize_edges(self.distance_matrix)
        self.edge_matrix = np.zeros((self.size, self.size), dtype=int)
        self.union_find = UnionFind()
        self.edge_counter = 0
        self.sentence_sets = []
        self.h_nodes = {}
        self.h_nodes_adj = {}
        self.h_node_id = 0
        self.birth_time = {}
        self.death_time = {}
        self.presenter = {}
        self.root = 0
        self.n_leaves = 0
        self.node_persistence = {}
        n = self.distance_matrix.shape[0]
        self.h_sentence_to_node = [0] * n
        self.persistance_pairs_h1 = {}

        # Make disjoint union sets 
        # on time t=0
        for i in range(n):
            self.union_find.make_set(i, 0)
            self.node_birth(i)

    def initialize_edges(self, matrix):
        """ Look at the upper diagonal of `matrix`
            and push to the PQ `edges` a tuple with 
            `dist, (i, j)` where `i` and `j` are the
            indices of the elements in `matrix` that
            have a distance greater than 0.
        """
        edges = []
        n = matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] > 0:
                    heappush(edges, (matrix[i, j], (i, j)))
        return edges

    def connect_new_edge(self, e: Tuple[int, int]): 
        """ Connect a new edge in the edge matrix

        """
        x, y = e
        self.edge_matrix[x, y] = 1
        self.edge_matrix[y, x] = 1

    def connect_new_edges(self):
        """ Get the smallest edge in the graph

        """
        # If there are no edges, then do nothing
        if len(self.edges) == 0:
            return []
        new_edges = []
        # Pop the smallest elemment from the heap
        # that is, the smallest distance `length`
        # and the tuple `(i, j)` named `e`
        length, e = heappop(self.edges)
        # Connect this edge in the edge_matrix
        self.connect_new_edge(e)
        # Add as `new_edge`
        new_edges.append((length, e))

        # If this was the last edge, then return
        # a list with this last edge
        if len(self.edges) == 0:
            return new_edges

        # Get the smallest value
        l, e = nsmallest(1, self.edges)[0]

        # While the next smallest edge is smaller
        # TODO Should l be always bigger since length is by definition
        # the smallest edge?
        while l <= length:
            # Pop the next smallest edge
            l, e = heappop(self.edges)
            # Add this edge to the edge matrix
            self.connect_new_edge(e)
            # Add this edge to the new edges
            new_edges.append((l,e))
            if len(self.edges) == 0:
                break
            l, e = nsmallest(1, self.edges)[0]
        return new_edges

    def step(self):
        """ A timestep in the persitence calculation
        
        """
        new_edges = self.connect_new_edges()

        # If there are no new edges to connect
        # we are done
        if len(new_edges) == 0:
            return False

        for edge in new_edges:
            self.edge_counter += 1
            l, e = edge
            x, y = e

            p_x = self.union_find.find_set(x)
            p_y = self.union_find.find_set(y)
            parent = self.union_find.union(x, y, self.distance_matrix[x, y])
            self.merge_two_components(p_x, p_y, parent, l)
        return True

    def calculate_persistence(self):
        """ Calculate the persitent homology of the
            hierarchy tree.
        """
        b = self.step()
        while b:
            b = self.step()
        ###
        self.root = np.max(list(self.h_nodes_adj.keys()))
        self.n_leaves = np.min(list(self.h_nodes_adj.keys()))
        self.fill_presenters()
        self.find_death(self.root, self.root+1)

        ids = list(range(self.n_leaves, self.root+1))
        p = []
        for id in ids:
            p.append(self.death_time[id] - self.birth_time[id])

        self.node_persistence = pd.DataFrame({'id': ids, 'persistence': p})
        self.node_persistence = self.node_persistence.sort_values('persistence', ascending=False)
        return self.node_persistence


    def get_node_persistence(self):
        self.fill_presenters()
        self.find_death(self.root, self.root + 1)
        ids = list(range(self.n_leaves, self.root + 1))
        p = []
        for id in ids:
            p.append(self.death_time[id] - self.birth_time[id])

        self.node_persistence = pd.DataFrame({'id': ids, 'persistence': p})
        self.node_persistence = self.node_persistence.sort_values('persistence', ascending=False)
        return self.node_persistence

    def get_sentence_groups(self):
        sentence_sets = []
        for i in range(len(self.union_find.death)):
            if self.union_find.parent[i] == i:
                s = OrderedSet()
                x, y, z = self.triangles[i]
                s.add(x)
                s.add(y)
                s.add(z)
                all_children = self.union_find.children[i].copy()
                while len(all_children) != 0:
                    c = all_children[0]
                    all_children.remove(c)
                    addition = self.union_find.children[c].copy()
                    all_children += addition
                    x, y, z = self.triangles[c]
                    s.add(x)
                    s.add(y)
                    s.add(z)
                sentence_sets.append(s)
        return sentence_sets

    def get_sentence_groups_of_parents(self, parents):
        sentence_sets = []
        for i in parents:
            s = OrderedSet()
            s.add(i)
            all_children = self.union_find.children[i].copy()
            while len(all_children) != 0:
                c = all_children[0]
                all_children.remove(c)
                s.add(c)
            sentence_sets.append(s)
        return sentence_sets

    def update_node_sentences(self, t1):
        n = self.h_sentence_to_node[t1]
        self.h_nodes[n] |= self.get_sentence_groups_of_parents([t1])[0]

    def merge_two_components(self, t1, t2, p, length):
        h1 = self.h_sentence_to_node[t1]
        h2 = self.h_sentence_to_node[t2]
        if h1 != h2:
            merged = self.h_nodes[h1].copy()
            merged |= self.h_nodes[h2]
            self.h_nodes[self.h_node_id] = merged
            self.h_nodes_adj[self.h_node_id] = [h1, h2]
            self.h_sentence_to_node[p] = self.h_node_id
            self.birth_time[self.h_node_id] = length
            self.h_node_id += 1

    def relevant_merge(self, t1, t2, death):
        b1 = self.union_find.birth[t1]
        b2 = self.union_find.birth[t2]
        if b1 < death and b2 < death and t1 != t2:
            return True
        return False

    def node_birth(self, t):
        """ Add a node birth to time `t`
        """
        s = OrderedSet()
        s.add(t)
        # Set the h_node_id-th node to a set containing i
        self.h_nodes[self.h_node_id] = s
        # Set the i-th node to the h_node_id-th node id
        self.h_sentence_to_node[t] = self.h_node_id

        self.h_node_id += 1

    def fill_presenters(self):
        n_leaves = np.min(list(self.h_nodes_adj.keys()))
        n_nodes = np.max(list(self.h_nodes_adj.keys())) + 1
        for n in range(n_leaves):
            self.presenter[n] = n
        for id in range(n_leaves, n_nodes):
            r, l = self.h_nodes_adj[id]
            r_leaf = r < n_leaves
            l_leaf = l < n_leaves
            if r_leaf and l_leaf:
                self.presenter[id] =  min(self.presenter[r], self.presenter[l])
                continue
            if not r_leaf and l_leaf:
                self.presenter[id] = self.presenter[r]
                continue
            if not l_leaf and r_leaf:
                self.presenter[id] = self.presenter[l]
                continue
            bottom_of_r = r
            bottom_of_l = l
            xr = self.presenter[r]
            xl = self.presenter[l]

            while True:
                n1, n2 = self.h_nodes_adj[bottom_of_r]
                if n1 < n_leaves and n2 < n_leaves:
                    break
                if self.presenter[n1] == xr:
                    bottom_of_r = n1
                else:
                    bottom_of_r = n2

            while True:
                n1, n2 = self.h_nodes_adj[bottom_of_l]
                if n1 < n_leaves and n2 < n_leaves:
                    break
                if self.presenter[n1] == xl:
                    bottom_of_l = n1
                else:
                    bottom_of_l = n2

            if bottom_of_r < bottom_of_l:
                self.presenter[id] = xr
            else:
                self.presenter[id] = xl

    def find_death(self, node, parent):
        if node < self.n_leaves:
            self.death_time[node] = self.birth_time[parent]
            return
        if node == self.root:
            self.death_time[node] = 1
        elif self.presenter[parent] == self.presenter[node]:
            self.death_time[node] = self.death_time[parent]
        else:
            self.death_time[node] = self.birth_time[parent]
        r, l = self.h_nodes_adj[node]
        self.find_death(r, node)
        self.find_death(l, node)

