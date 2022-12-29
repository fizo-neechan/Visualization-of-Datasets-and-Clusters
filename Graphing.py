import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


class Graph:
    """Class containing all the graphing utilities.

    This allows the user to make graphs, print them using networkx
    and make and visualizeclusters from the graph

    Attributes:
    -----------------------------------------
    adjacency: The adjacency matrix of the weighted graph

    graph: The networkx graph produced from the adjacency matrix

    Methods:
    -----------------------------------------
    __initNodes:
        initialize nodes and edges from the adjacency matrix

    __initEdges:
        intialize edges and weights from the adjacency matrix

    __makeGraph:
        produce a graph from the provided adjacency matrix

    makeClusters:
        make clusters from the graph's adjacency matrix 
        and return all the cluster produced

    printGraph:
        plot and display the graph using networkx and matplotlib.pyplot

    VisualizeClusters:
        plot and display the graph along with its produced clusters
        using networkx and matplotlib.pyplot

    """    



    def __init__(self, adjacency: np.ndarray) -> None:
        """constructor

        asks user for the threshhold value for making the graph

        args:
            adjacency:np.ndarray
            adjacency matrix for the graph

        returns:
            None
        """    


        if adjacency.shape[0] != adjacency.shape[1]:
            raise Exception("Adjacency matrix does not have proper shape")
        elif len(adjacency.shape) != 2:
            raise Exception("Adjacency matrix is not properly flattened")
        

        thresh = float(input("enter thresholding value: "))
        adjacency[adjacency <= thresh] = 0

        self.adjacency = adjacency
        self.graph = nx.Graph()
        self.__initNodes(self.adjacency)

    def __initNodes(self, adjacency: np.ndarray) -> None:
        """initialize nodes and edges from the adjacency matrix

        0 is not considered as an egde

        args:
            adjacency:np.ndarray
            adjacency matrix for the graph

        returns:
            None
        """   


        for i in range(adjacency.shape[0]):
            for j in range(i):
                if adjacency[i,j] != 0:
                    self.graph.add_edge(i,j, weigh= adjacency[i,j])
        

    def __initEdges(self, adjacency: np.ndarray) -> None:
        """intialize edges and weights from the adjacency matrix

        args:
            adjacency:np.ndarray
            adjacency matrix for the graph

        returns:
            None
        """


        for i in range(adjacency.shape[0]):
            for j in range(i+1):
                self.graph.add_weighted_edges_from((i, j, adjacency[i,j]))


    def __makeGraph(self, adjacency:np.ndarray) -> nx.Graph:
        """produce a graph from the provided adjacency matrix

        args:
            adjacency:np.ndarray
            adjacency matrix for the graph

        returns:
            nx.Graph
            new graph that is produced
        """

        
        g = nx.Graph()
        for i in range(adjacency.shape[0]):
            for j in range(i):
                if adjacency[i,j] != 0:
                    g.add_edge(i,j, weigh= adjacency[i,j])
        

    def makeClusters(self, adjacency: np.ndarray) -> list:
        """make clusters from the graph's adjacency matrix 
        and return all the clusters produced

        0 weight of an edge is not considered an edge

        args:
            adjacency:np.ndarray
            adjacency matrix for the graph

        returns:
            list of all the clusters that are produced
        """


        clusters = []
        newAdjacency = adjacency.copy()
        numClusters = 0
        while(np.max(newAdjacency) != 0):
            
            numClusters += 1

            maxNode = np.argmax(np.sum(newAdjacency, axis=1))
            clusterAdjacency = newAdjacency[maxNode]
            clusterlist = [i for i, n in enumerate(clusterAdjacency) if n != 0]

            cluster = nx.Graph()
            for i, n in enumerate(clusterlist):
                cluster.add_edge(maxNode, n, weight= clusterAdjacency[i])
                newAdjacency[n,:] = 0            
                newAdjacency[:,n] = 0            

            clusters.append(cluster)
        
        return clusters


    def printGraph(self, graph: nx.Graph, title: str) -> None:
        """plot and display the graph using networkx and matplotlib.pyplot

        args:
            graph:nx.Graph
            Graph to be printed

            title:str
            title of the graph

        returns:
            None
        """
        
        plt.title(title)
        nx.draw_circular(graph, with_labels=True)
        plt.show()
        plt.clf()


    def VisualizeClusters(self) -> None:
        """plot and display the graph along with its produced clusters
        using networkx and matplotlib.pyplot

        args:
            None

        returns:
            None
        """

        clusters = self.makeClusters(self.adjacency)

        fig = plt.figure(figsize=(12,5))
        fig.suptitle("Visualization of different clusters formed")

        ax = fig.add_subplot(1, len(clusters)+1, 1)
        nx.draw_circular(self.graph, with_labels= True)
        ax.set_title("Original graph")

        for i, c in enumerate(clusters):
            ax = fig.add_subplot(1, len(clusters)+1, i+2)
            nx.draw_circular(c, with_labels= True)
            ax.set_title(f"Cluster#{i+1}")
        
        plt.show()
        plt.clf()

