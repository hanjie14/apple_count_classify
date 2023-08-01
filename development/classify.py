from imageai.Detection import ObjectDetection
from pathlib import Path
import cv2
import numpy as np

def check_folder(folder_path_list):
    if isinstance(folder_path_list, list) :
        for folder_path in folder_path_list:
            folder_path = Path(folder_path)
            if not folder_path.is_dir():
                folder_path.mkdir(parents=True)

def find_next_filename(filename):
    name, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(os.path.join(output_dir, filename)):
        i += 1
        filename = f"{name}_{i}{ext}"
    return filename


def find_dominant_color(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red, green, and yellow
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    
    green_lower = np.array([30, 40, 40])
    green_upper = np.array([90, 255, 255])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    
    # Create masks for each color
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    
    # Count the number of pixels in each mask
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    # Find the dominant color
    max_pixels = max(red_pixels, green_pixels, yellow_pixels)
    if max_pixels == red_pixels:
        return "red"
    elif max_pixels == green_pixels:
        return "green"
    else:
        return "yellow"


def check_dir(folder_path_list):
    if isinstance(folder_path_list, list) :
        for folder_path in folder_path_list:
            folder_path = Path(folder_path)

            if not folder_path.is_dir():
                folder_path.mkdir(parents=True)



def main(model_path, input_image, output_image,minimum_percentage_probability): 
    
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    returned_image, detections, extracted_objects   = detector.detectObjectsFromImage(input_image, output_image_path=output_image, 
                                                                                     output_type="array", 
                                                                                     extract_detected_objects=True, 
                                                                                     minimum_percentage_probability=minimum_percentage_probability)

    print("The number "+ str(len(detections))+ " is the number of apples that are in the image.")
    print("extracted_objects")
    # print("Object detection complete!")
    return returned_image, detections, extracted_objects