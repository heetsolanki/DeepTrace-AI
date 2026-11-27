import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np

# Create the detector once (it loads a small model into memory)
detector = MTCNN()

def extract_face(image_path):
    """
    Takes an image file path.
    Returns a PIL Image of the detected face, or None if no face is found.
    """
    # Load image with OpenCV, convert BGR to RGB
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ask MTCNN to find faces
    results = detector.detect_faces(img)

    if not results:
        return None

    # Grab the first face found
    x, y, width, height = results[0]['box']

    # Crop just that face region
    face_crop = img[y : y + height, x : x + width]

    return Image.fromarray(face_crop)