from Graphing import Graph
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

if __name__ == "__main__":


    adjacency = np.array([[1,0,0],[3,1,0],[9,6,7]])
    
    print(adjacency)
    print(adjacency.shape)

    g = nx.Graph()

    for i in range(adjacency.shape[0]):
        for j in range(i+1):
            g.add_edge(i,j, weight= adjacency[i,j])
    
    # for i in range(g.)


    print(g[0])
    print(g[1])
    print(g[2])




    # print(g.)
    print(g.degree)
    sorted(g.degree, key=lambda x: x[1], reverse=True)
    print(g.degree)

    pos = nx.spring_layout(g, seed=7)

    elarge = [(u,v) for (u,v,d) in g.edges(data=True)]

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos, edgelist=elarge)

    nx.draw_networkx_labels(g, pos)


    edgelabels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, pos, edgelabels)

    plt.show()
    plt.clf()


    
    




    # nx.draw_networkx()
