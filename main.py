from dataProcessing import dataProcessing
from Graphing import Graph

# driver code
if __name__ == "__main__":

# get file path from user
    filepath = input("enter the filepath of the data file: ")

# initialize the dataProcessing object
    d = dataProcessing(filepath)

# visualize the data for task1
    d.VisualizeData()

# visualize the data for task2
    d.VisualizeSignatureTechnique()

# initialize the Graph object with the correlation matrix as adjacency matrix
    g = Graph(d.corrMat)

# visualize the clusers for task3
    g.VisualizeClusters()
