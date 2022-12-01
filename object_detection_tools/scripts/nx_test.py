import networkx as nx
from matplotlib import pyplot as plt

##### Test program
if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6])
    G.add_edges_from([(1,2), (2,3), (2,4), (3,4), (1,5), (3,5), (3,6)])
    nx.draw(G, node_size=400, node_color='red', with_labels=True, font_weight='bold')
    plt.show()
    # ここで、図を消せば次に進む

    G.add_nodes_from([7,8])
    G.add_edges_from([(1,7), (2,7), (3,7), (1,8), (2,8), (3,8), (7,8)])
    nx.draw(G, node_size=400, node_color='red', with_labels=True, font_weight='bold')
    plt.show()