# Visualise the graph using NetworkX and Matplotlib
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, image_shape):
    plt.figure(figsize=(10, 10))

    pos = {}
    for node in G.nodes(data = True):
        bbox = node[1]['bbox']

        # Calculate the node position as the center of the bounding box
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        # Normalise now
        pos[node[0]] = (x_center / image_shape[1], (image_shape[0] - y_center) / image_shape[0])

    # Draw the graph with labels
    nx.draw(G, pos, with_labels = True, node_color = 'skyblue', node_size = 700, font_size = 10, edge_color = 'gray')
    plt.gca().invert_yaxis()
    plt.show()