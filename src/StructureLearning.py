import sys
import time
import copy
from ConditionalIndependence import ConditionalIndependence
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle


class StructureLearning(ConditionalIndependence):
    def __init__(self, datafile):
        super().__init__(datafile)
        self.find_skeleton()

    def find_skeleton(self):
        g = nx.complete_graph(self.rand_vars)
        max_degree = len(g.nodes) - 1
        l = 0
        sep_set = []
        # var_set = set(self.rand_vars)
        while l < max_degree - 1:
            print(f"level: {l}")
            nx.draw_circular(g, labels={i: i for i in g.nodes}, font_weight='bold')
            plt.savefig(f"./plots/structure_learning_level_{l}.png")
            plt.clf()
            g_copy = g.copy()
            if l == 0:
                for edge in g_copy.edges:
                    var_set = set(g.neighbors(edge[0]))
                    other_vars = var_set.difference(edge)
                    combinations = it.combinations(other_vars, l)
                    for condition_set in combinations:
                        p = self.compute_p_value(edge[0], edge[1], list(condition_set))
                        # if P does not pass hypothesis remove edge from g
                        if p > 0.05:
                            print(f"Removing edge between {edge[0]} and {edge[1]}")
                            g.remove_edge(edge[0], edge[1])
                            break
            else:
                for node in g_copy.nodes:
                    neighbors = list(g.neighbors(node))
                    for n in neighbors:
                        var_set = set(g_copy.neighbors(node))
                        other_vars = var_set.difference([node, n])
                        combinations = it.combinations(other_vars, l)
                        for condition_set in combinations:
                            p = self.compute_p_value(node, n, list(condition_set))
                            if p > 0.05:
                                print(f"Removing edge between {node} and {n}")
                                g.remove_edge(node, n)
                                break
            l += 1

        # directed = nx.DiGraph()
        # directed.add_edges_from([shuffle(list(e)) for e in g.edges])
        nx.draw_circular(g, labels={i: i for i in g.nodes}, font_weight='bold')
        plt.show()
        print(f"Edges {g.edges}")


if len(sys.argv) != 2:
    print("USAGE: StructureLearning.py [training_file.csv]")
    print("EXAMPLE> StructureLearning.py data_banknote_authentication-train.csv")
    exit(0)
else:
    datafile = sys.argv[1]
    StructureLearning(datafile)
