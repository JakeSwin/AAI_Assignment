import random
import sys
import time
import copy
import numpy as np
from ConditionalIndependence import ConditionalIndependence
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from CPT_Generator import CPT_Generator
from ModelEvaluator import ModelEvaluator
from random import shuffle


class StructureLearning(ConditionalIndependence):
    show_graph = True

    def __init__(self, datafile, testfile, configfile):
        self.datafile = datafile
        self.testfile = testfile
        self.configfile = configfile
        super().__init__(datafile)
        self.g = self.find_skeleton()
        if self.show_graph:
            nx.draw_circular(self.g, labels={i: i for i in self.g.nodes}, font_weight='bold')
            plt.show()
        self.directed_graph = self.make_directed(self.g)
        self.structure_string = self.directed_to_string_structure(self.directed_graph)
        print(self.structure_string)
        config_name = datafile.split("_")[0].split("/")[-1].title()
        self.brute_force_test_structure(config_name)

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
                        if p > 0.01:
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
        return g

    @staticmethod
    def make_directed(g):
        while True:
            cycles = False
            directed_graph = nx.DiGraph()
            for node in g.nodes:
                neighbors = list(g.neighbors(node))
                if len(neighbors) > 0:
                    for n in neighbors:
                        np_e = np.array([node, n])
                        np.random.shuffle(np_e)
                        directed_graph.add_edge(np_e[0], np_e[1])
                else:
                    directed_graph.add_node(node)
            for cycle in nx.simple_cycles(directed_graph):
                cycles = True
                print("Cycle found:" + str(cycle))
            if cycles is False:
                break
        return directed_graph

    @staticmethod
    def directed_to_string_structure(directed_graph):
        dg_sorted = nx.topological_sort(directed_graph)
        structure = "structure:"
        for node in dg_sorted:
            var = f"P({node}"
            predecessors = list(directed_graph.predecessors(node))
            if len(predecessors) > 0:
                var += "|"
                for p in predecessors:
                    var += p + ","
            var = var.rstrip(',')
            var += ");"
            structure += var
            print(f"Node: {node} Predecessors {list(directed_graph.predecessors(node))}")
        return structure.rstrip(';')

    def brute_force_test_structure(self, name):
        # results = []
        results = {}
        # Generate Structure
        # Write to file
        for i in range(5):
            while self.structure_string in results:
                self.directed_graph = self.make_directed(self.g)
                self.structure_string = self.directed_to_string_structure(self.directed_graph)
            rand_vars = []
            for rv in self.rand_vars:
                label = ''.join([c for c in rv if c.isupper()])
                rand_vars.append(label + "(" + rv + ")")
            rand_vars = str(rand_vars)
            rand_vars = str(rand_vars).replace('[', '').replace(']', '')
            rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';')

            with open(self.configfile, "w") as cfg_file:
                cfg_file.write("name:" + str(name))
                cfg_file.write('\n')
                cfg_file.write('\n')
                cfg_file.write("random_variables:" + str(rand_vars))
                cfg_file.write('\n')
                cfg_file.write('\n')
                cfg_file.write(str(self.structure_string))
                cfg_file.write('\n')
                cfg_file.write('\n')

            # Run CPT Generator
            CPT_Generator(self.configfile, self.datafile)

            # Run Model Evaluator
            me = ModelEvaluator(self.configfile, self.datafile, self.testfile)
            results[self.structure_string] = {
                "Balanced Accuracy": me.bal_acc,
                "F1 Score": me.f1,
                "Area Under Curve": me.auc,
                "Brier Score": me.brier,
                "KL Divergence": me.kl_div
            }

        for k, v in sorted(results.items(), key=lambda e: e[1]["Balanced Accuracy"], reverse=True):
            print(v, k)


if len(sys.argv) != 4:
    print("USAGE: StructureLearning.py [training_file.csv] [testing_file.csv] [output_structure.txt]")
    print("EXAMPLE> StructureLearning.py ./data/assignment/diabetes_data-discretized-train.csv"
          "./data/assignment/diabetes_data-discretized-test.csv ./config/config-diabetes-learnt-gen.txt")
    exit(0)
else:
    datafile = sys.argv[1]
    testfile = sys.argv[2]
    configfile = sys.argv[3]
    StructureLearning(datafile, testfile, configfile)
