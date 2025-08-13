import logging
import os
import time
import cv2
from YOLOv8_ONNX import YOLOv8

logger = logging.getLogger()

yolo_model_1 = YOLOv8("160_437.onnx", conf_thres=0.2, iou_thres=0.7)
yolo_model_2 = YOLOv8("best50.onnx", conf_thres=0.2, iou_thres=0.7)

image_folder = "images"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

total_time_1 = 0
num_images = len(image_files)
start_time_1 = time.time()

for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yolo_model_1.run(img)

end_time_1 = time.time()
execution_time_1 = end_time_1 - start_time_1

if num_images > 0:
    average_time_1 = execution_time_1 / num_images
    logger.log(logging.WARNING, f"Среднее время выполнения для {num_images} изображений с моделью 1: {average_time_1:.6f} секунд, FPS - {1 / average_time_1}")

total_time_2 = 0
start_time_2 = time.time()

for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yolo_model_2.run(img)

end_time_2 = time.time()
execution_time_2 = end_time_2 - start_time_2

if num_images > 0:
    average_time_2 = execution_time_2 / num_images
    logger.log(logging.WARNING, f"Среднее время выполнения для {num_images} изображений с моделью 2: {average_time_2:.6f} секунд, FPS - {1 / average_time_2}")
