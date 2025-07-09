import os
import time
import logging
from ultralytics import YOLO  # Замените на правильный импорт

logger = logging.getLogger()
image_folder = "images"
models_folder = "models"

total_time = 0
count = 0

for model_file in os.listdir(models_folder):
    count = 0
    total_time = 0
    model_path = os.path.join(models_folder, model_file)
    onnx_model = YOLO(model_path)
    logger.log(logging.WARNING, f'Model: {model_file}')

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            start_time = time.time()
            results = onnx_model(image_path, verbose=False)
            total_time += time.time() - start_time
            count += 1
    if count > 0:
        average_time = total_time / count
        logger.log(logging.WARNING, f'Average time: {average_time:.4f} seconds')
        logger.log(logging.WARNING, f'Inferences per second: {1 / average_time:.2f}\n')

