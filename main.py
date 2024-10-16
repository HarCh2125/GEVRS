from detector import ObjectDetector
from utils import load_image, preprocess_image, plot_detections

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

if __name__ == '__main__':
    image_path = 'data/sample1.jpg'
    main(image_path)