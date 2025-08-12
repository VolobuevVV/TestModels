import logging
import os
import time

import cv2
from YOLOv8_ONNX import YOLOv8
logger = logging.getLogger()

yolo = YOLOv8("/kaggle/working/TestModels/best50.onnx", conf_thres=0.2, iou_thres=0.7)

image_folder = "/kaggle/working/TestModels/images"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

total_time = 0
num_images = len(image_files)
start_time = time.time()
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yolo.run(img)

end_time = time.time()
execution_time = end_time - start_time

if num_images > 0:
    average_time = execution_time / num_images
    logger.log(logging.WARNING, f"Среднее время выполнения для {num_images} изображений: {average_time:.6f} секунд, FPS - {1 / average_time}")
