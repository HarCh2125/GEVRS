from detector import ObjectDetector
from utils import *
from graph_constructor import construct_graph
from graph_visualisation import visualize_graph

def main(image_path):
    # Load the image
    image = load_image(image_path)

    # Preprocess the image
    tensor_image = preprocess_image(image)

    # Make a prediction
    detector = ObjectDetector()
    predictions = detector.predict(tensor_image)

    # Plot the detections
    plot_detections(image, predictions)

    # Construct a graph from the detected objects
    graph = construct_graph(predictions, iou_threshold = 0.3)

    # Visualise the graph
    visualize_graph(graph, image.shape)

    # Overlay the graph on the image
    visualize_graph_on_image(graph, image_path)

if __name__ == '__main__':
    image_path = 'data/sample1.jpg'
    main(image_path)