import os
from ultralytics import YOLO
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)d'
)
logger = logging.getLogger()

image_folder = "TestImages"

onnx_model = YOLO("best.onnx")
logger.log(logging.WARNING, 'ONNX')

for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        results = onnx_model(image_path)
