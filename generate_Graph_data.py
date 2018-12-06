from igraph import *
from model_function import create_network
import networkx as nx
import numpy as np
import pickle
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

deg = 'strong'
new_graph = False
if not new_graph:
    print("old graph")
    graph = pickle.load(open("graph_data.pkl", 'rb'))
    graph, friend_pref, adopt_nodes = create_network(node_num=400,
                                                     graph=graph,
                                                     p=0.1,
                                                     pref_range=10,
                                                     pref_dim=2,
                                                     homo_degree=deg)
else:
    graph, friend_pref, adopt_nodes = create_network(node_num=400,
                                                     p=0.1,
                                                     pref_range=10,
                                                     pref_dim=2,
                                                     homo_degree=deg)

edges = graph.edges()
vertices = graph.nodes()
susceptible = {}
influence = {}

node_pref = nx.get_node_attributes(graph, 'pref')

product = np.array([-1, 1])
for node, pref in node_pref.items():
    susceptible[node] = np.dot(pref,product)
    influence[node] = 1-sigmoid(susceptible[node])

g = Graph(vertex_attrs={"label":vertices, "susceptible": list(susceptible.values()),
                        "influence": list(influence.values())}, edges=edges, directed=True)

save(g, '../inf-max-code-release/graph.gml')
pickle.dump(graph,open("graph_data.pkl", 'wb'))