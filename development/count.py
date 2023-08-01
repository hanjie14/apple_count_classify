from imageai.Detection import ObjectDetection
from pathlib import Path
import configparser

def check_folder(folder_path_list):
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

    detections   = detector.detectObjectsFromImage(input_image, output_image_path=output_image, 
                                                                                     minimum_percentage_probability=minimum_percentage_probability)
    
    print("The number "+ str(len(detections))+ " is the number of apples that are in the image.")
